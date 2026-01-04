# src/gatekeeper/tools/similarity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _rel_sim(a: float, b: float, eps: float = 1e-12) -> float:
    denom = max(abs(a), abs(b), eps)
    rel = abs(a - b) / denom
    s = 1.0 - min(1.0, rel)
    return max(0.0, min(1.0, s))


def _cat_sim(a: Optional[str], b: Optional[str]) -> float:
    if a is None or b is None:
        return 0.0
    return 1.0 if str(a) == str(b) else 0.0


def _get_current_fields(sim_config: Dict[str, Any]) -> Dict[str, Any]:
    physics = sim_config.get("physics_setup") or {}
    tm = physics.get("turbulence_model") or {}
    solver = sim_config.get("solver_settings") or {}
    relax = (solver.get("relaxation_factors") or {})

    numerics = sim_config.get("numerics") or {}
    ts = (numerics.get("time_step") or {})

    return {
        "turbulence_model": tm.get("model", None),
        "time_step": ts.get("size", None),
        "relax_pressure": relax.get("pressure", None),
        "relax_momentum": relax.get("momentum", None)
    }


def _get_past_fields(past_run: Dict[str, Any]) -> Dict[str, Any]:
    cfg = past_run.get("config") or {}
    relax = cfg.get("relaxation_factors") or {}
    return {
        "turbulence_model": cfg.get("turbulence_model", None),
        "time_step": cfg.get("time_step", None),
        "relax_pressure": relax.get("pressure", None),
        "relax_momentum": relax.get("momentum", None)
    }


def _weighted_similarity(cur: Dict[str, Any], past: Dict[str, Any]) -> Tuple[float, List[str], bool]:
    weights = {
        "turbulence_model": 0.35,
        "time_step": 0.25,
        "relax_pressure": 0.20,
        "relax_momentum": 0.20,
    }

    diffs: List[str] = []
    score_sum = 0.0
    weight_sum = 0.0

    w = weights["turbulence_model"]
    s = _cat_sim(cur.get("turbulence_model"), past.get("turbulence_model"))
    score_sum += w * s
    weight_sum += w
    if s < 1.0:
        diffs.append("turbulence_model")

    w = weights["time_step"]
    c_ts = cur.get("time_step")
    p_ts = past.get("time_step")
    if isinstance(c_ts, (int, float)) and isinstance(p_ts, (int, float)):
        s = _rel_sim(float(c_ts), float(p_ts))
        score_sum += w * s
        weight_sum += w
        if s < 0.90:
            diffs.append("time_step")
    else:
        score_sum += 0.0
        weight_sum += w
        diffs.append("time_step_missing")

    w = weights["relax_pressure"]
    c_rp = cur.get("relax_pressure")
    p_rp = past.get("relax_pressure")
    if isinstance(c_rp, (int, float)) and isinstance(p_rp, (int, float)):
        s = _rel_sim(float(c_rp), float(p_rp))
        score_sum += w * s
        weight_sum += w
        if s < 0.90:
            diffs.append("relaxation_factors.pressure")
    else:
        score_sum += 0.0
        weight_sum += w
        diffs.append("relax_pressure_missing")

    w = weights["relax_momentum"]
    c_rm = cur.get("relax_momentum")
    p_rm = past.get("relax_momentum")
    if isinstance(c_rm, (int, float)) and isinstance(p_rm, (int, float)):
        s = _rel_sim(float(c_rm), float(p_rm))
        score_sum += w * s
        weight_sum += w
        if s < 0.90:
            diffs.append("relaxation_factors.momentum")
    else:
        score_sum += 0.0
        weight_sum += w
        diffs.append("relax_momentum_missing")

    similarity = score_sum / max(weight_sum, 1e-12)

    knobs_changed = False
    if isinstance(c_ts, (int, float)) and isinstance(p_ts, (int, float)):
        knobs_changed |= (_rel_sim(float(c_ts), float(p_ts)) < 0.95)
    else:
        knobs_changed = True

    if isinstance(c_rp, (int, float)) and isinstance(p_rp, (int, float)):
        knobs_changed |= (_rel_sim(float(c_rp), float(p_rp)) < 0.95)
    else:
        knobs_changed = True

    if isinstance(c_rm, (int, float)) and isinstance(p_rm, (int, float)):
        knobs_changed |= (_rel_sim(float(c_rm), float(p_rm)) < 0.95)
    else:
        knobs_changed = True

    diffs = sorted(set(diffs))
    return similarity, diffs, knobs_changed


def retrieve_similar_runs(
    sim_config: Dict[str, Any],
    past_runs: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    cur = _get_current_fields(sim_config)
    hits: List[Dict[str, Any]] = []

    for r in past_runs:
        past = _get_past_fields(r)
        sim, diffs, knobs_changed = _weighted_similarity(cur, past)

        hits.append(
            {
                "job_id": r.get("job_id"),
                "similarity_score": round(float(sim), 6),
                "outcome": r.get("status"),
                "failure_reason": r.get("failure_reason", None),
                "key_differences": diffs,
                "stabilization_knobs_changed": bool(knobs_changed),
            }
        )

    hits.sort(key=lambda x: (-x["similarity_score"], str(x.get("job_id") or "")))
    return hits[: max(0, int(top_k))]