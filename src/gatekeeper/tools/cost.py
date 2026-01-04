"""Deterministic resource estimation utilities (RAM / runtime / cost).

Purpose
-------
This module provides *tool-grade*, deterministic estimates that the Resource
Manager (and Synthesizer) can rely on as evidence. We avoid LLM-dependent math
and keep all formulas explicit and testable.

Data sources
------------
We implement the rules-of-thumb / formulas found in `simulation_formulas.json`:
- RAM_GB ≈ (cell_count / 1e6) * 1.5
- Work units (million-cells·iteration):
    work_units = (cell_count / 1e6) * (steps * iterations_per_step)
- Cost (USD):
    cost_usd = work_units * cost_per_m_cells_per_iter
- Runtime (hours):
    runtime_h = work_units * hours_per_m_cells_per_iter

Notes
-----
- `cost_per_m_cells_per_iter` and `hours_per_m_cells_per_iter` are org-specific
  coefficients. They must be injected by the caller (from formulas/env/CLI),
  not hard-coded here.
- We compute LOW/HIGH bounds deterministically when an iterations-per-step range
  is available.
- We also provide conservative single-value estimates (equal to HIGH) for
  backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Handbook constants (deterministic)
# -----------------------------
RAM_GB_PER_MCELL: float = 1.5


@dataclass(frozen=True)
class CostEstimate:
    """Deterministic resource estimates (with LOW/HIGH bounds)."""

    # Primary extracted inputs
    cell_count: float
    steps: Optional[float]
    iters_per_step_min: Optional[float]
    iters_per_step_max: Optional[float]

    # Derived counts
    total_solver_iterations_low: Optional[float]
    total_solver_iterations_high: Optional[float]

    # Derived resource metrics
    ram_gb: float

    # Work units: million-cells·iteration
    work_units_low: Optional[float]
    work_units_high: Optional[float]

    # Cost/runtime bounds (None if coefficients missing)
    estimated_cost_usd_low: Optional[float]
    estimated_cost_usd_high: Optional[float]
    estimated_runtime_hours_low: Optional[float]
    estimated_runtime_hours_high: Optional[float]

    # Backward-compatible conservative single-value estimates (equal to HIGH)
    estimated_cost_usd: Optional[float]
    estimated_runtime_hours: Optional[float]

    # For auditability / debug traces
    missing_inputs: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, bool):
            return None
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_cell_count(mesh_report: Dict[str, Any]) -> Optional[float]:
    """Extract cell_count from known mesh_report schemas."""
    topo = mesh_report.get("topology")
    if isinstance(topo, dict):
        v = _safe_float(topo.get("cell_count"))
        if v is not None:
            return v

    v = _safe_float(mesh_report.get("cell_count"))
    if v is not None:
        return v

    qual = mesh_report.get("quality")
    if isinstance(qual, dict):
        v = _safe_float((qual.get("topology") or {}).get("cell_count"))
        if v is not None:
            return v

    return None


def extract_steps(sim_config: Dict[str, Any]) -> Optional[float]:
    """Extract transient step count from sim_config (deterministic)."""
    numerics = sim_config.get("numerics")
    if isinstance(numerics, dict):
        ts = numerics.get("time_step")
        if isinstance(ts, dict):
            v = _safe_float(ts.get("count"))
            if v is not None:
                return v
    return None


def extract_iters_per_step_range(sim_config: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Extract iterations-per-step range.

    Expected schema (from your current sim_config):
      sim_config.numerics.convergence_controls.min_iterations_per_step
      sim_config.numerics.convergence_controls.max_iterations_per_step
    """
    numerics = sim_config.get("numerics")
    if isinstance(numerics, dict):
        cc = numerics.get("convergence_controls")
        if isinstance(cc, dict):
            mn = _safe_float(cc.get("min_iterations_per_step"))
            mx = _safe_float(cc.get("max_iterations_per_step"))
            return mn, mx
    return None, None


def estimate_ram_gb(cell_count: float) -> float:
    """RAM_GB ≈ (cell_count / 1e6) * 1.5 (handbook rule-of-thumb)."""
    mcell = cell_count / 1_000_000.0
    ram = mcell * RAM_GB_PER_MCELL
    return round(float(ram), 6)


def _round_opt(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 6)


def compute_resource_estimates(
    *,
    mesh_report: Dict[str, Any],
    sim_config: Dict[str, Any],
    cost_per_m_cells_per_iter: Optional[float] = None,
    hours_per_m_cells_per_iter: Optional[float] = None,
) -> CostEstimate:
    """Compute deterministic RAM/cost/runtime estimates (LOW/HIGH bounds)."""

    missing: List[str] = []
    notes: List[str] = []

    cell_count = extract_cell_count(mesh_report)
    if cell_count is None:
        missing.append("$.mesh_report.topology.cell_count")
        return CostEstimate(
            cell_count=0.0,
            steps=None,
            iters_per_step_min=None,
            iters_per_step_max=None,
            total_solver_iterations_low=None,
            total_solver_iterations_high=None,
            ram_gb=0.0,
            work_units_low=None,
            work_units_high=None,
            estimated_cost_usd_low=None,
            estimated_cost_usd_high=None,
            estimated_runtime_hours_low=None,
            estimated_runtime_hours_high=None,
            estimated_cost_usd=None,
            estimated_runtime_hours=None,
            missing_inputs=tuple(missing),
            notes=("cell_count missing; cannot estimate resources",),
        )

    steps = extract_steps(sim_config)
    if steps is None:
        missing.append("$.sim_config.numerics.time_step.count")

    it_min, it_max = extract_iters_per_step_range(sim_config)
    if it_min is None:
        missing.append("$.sim_config.numerics.convergence_controls.min_iterations_per_step")
    if it_max is None:
        missing.append("$.sim_config.numerics.convergence_controls.max_iterations_per_step")

    ram = estimate_ram_gb(float(cell_count))

    # Work units (million-cells·iteration)
    mcell = float(cell_count) / 1_000_000.0

    total_low = None
    total_high = None
    wu_low = None
    wu_high = None

    if steps is not None and it_min is not None:
        total_low = float(steps) * float(it_min)
        wu_low = mcell * total_low
    if steps is not None and it_max is not None:
        total_high = float(steps) * float(it_max)
        wu_high = mcell * total_high

    # Cost bounds
    cost_low = None
    cost_high = None
    if cost_per_m_cells_per_iter is None:
        notes.append("cost coefficient missing (cost_per_m_cells_per_iter)")
    else:
        if wu_low is not None:
            cost_low = wu_low * float(cost_per_m_cells_per_iter)
        if wu_high is not None:
            cost_high = wu_high * float(cost_per_m_cells_per_iter)

    # Runtime bounds
    rt_low = None
    rt_high = None
    if hours_per_m_cells_per_iter is None:
        notes.append("runtime coefficient missing (hours_per_m_cells_per_iter)")
    else:
        if wu_low is not None:
            rt_low = wu_low * float(hours_per_m_cells_per_iter)
        if wu_high is not None:
            rt_high = wu_high * float(hours_per_m_cells_per_iter)

    # Conservative single-value estimates (equal to HIGH)
    cost_single = cost_high
    rt_single = rt_high

    return CostEstimate(
        cell_count=float(cell_count),
        steps=float(steps) if steps is not None else None,
        iters_per_step_min=float(it_min) if it_min is not None else None,
        iters_per_step_max=float(it_max) if it_max is not None else None,
        total_solver_iterations_low=_round_opt(total_low),
        total_solver_iterations_high=_round_opt(total_high),
        ram_gb=float(ram),
        work_units_low=_round_opt(wu_low),
        work_units_high=_round_opt(wu_high),
        estimated_cost_usd_low=_round_opt(cost_low),
        estimated_cost_usd_high=_round_opt(cost_high),
        estimated_runtime_hours_low=_round_opt(rt_low),
        estimated_runtime_hours_high=_round_opt(rt_high),
        estimated_cost_usd=_round_opt(cost_single),
        estimated_runtime_hours=_round_opt(rt_single),
        missing_inputs=tuple(missing),
        notes=tuple(notes),
    )