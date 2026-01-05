# src/gatekeeper/graph/nodes.py
from __future__ import annotations

import json
import re
import time
import os
from pathlib import Path
from typing import Any, Dict, List

from gatekeeper.schemas import (
    AgentName,
    VoteType,
    VerdictStatus,
    AgentVote,
    DebateMessage,
    FinalVerdict,
    DebateStance,
)
from gatekeeper.tools.cfl import compute_cfl_from_handbook
from gatekeeper.tools.similarity import retrieve_similar_runs
from gatekeeper.tools.cost import compute_resource_estimates
from gatekeeper.llm import call_gemini

# -----------------------------
# Utilities: timebox + JSON parsing
# -----------------------------
def _ensure_time(state: Dict[str, Any], margin_s: float = 1.0) -> None:
    deadline = float(state.get("deadline_epoch_s", 0.0) or 0.0)
    if deadline <= 0:
        return
    if time.time() + margin_s > deadline:
        raise TimeoutError("TIMEBOX_EXCEEDED")


def _strip_line_comments(text: str) -> str:
    # Minimal deterministic stripper for // comments (assumes no // inside JSON strings)
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"//.*$", "", line))
    return "\n".join(lines)


def _load_json_with_comments(text: str) -> Dict[str, Any]:
    cleaned = _strip_line_comments(text)
    return json.loads(cleaned)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _trace_append(state: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    trace = dict(state.get("trace") or {})
    trace[key] = value
    return {"trace": trace}


def _get_agent_prompt(agent_name: str) -> str:
    # Map AgentName (upper) to filename (lower)
    # e.g. PHYSICIST -> physicist.system.txt
    root = Path(__file__).parent.parent / "prompts"
    path = root / f"{agent_name.lower()}.system.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"You are {agent_name}. You are a strict gatekeeper."


# -----------------------------
# Node 0: load inputs
# -----------------------------
def load_inputs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    paths = state.get("input_paths") or {}
    required = ["mesh_report", "sim_config", "past_runs", "formulas"]
    missing = [k for k in required if k not in paths]
    if missing:
        raise ValueError(f"Missing input_paths keys: {missing}")

    raw = {
        "mesh_report": _read_text(paths["mesh_report"]),
        "sim_config": _read_text(paths["sim_config"]),
        "past_runs": _read_text(paths["past_runs"]),
        "formulas": _read_text(paths["formulas"]),
    }

    updates: Dict[str, Any] = {"raw": raw}
    updates |= _trace_append(state, "load_inputs", {"loaded": True, "files": dict(paths)})
    return updates


# -----------------------------
# Node 1: parse + normalize
# -----------------------------
def parse_and_normalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    raw = state.get("raw") or {}
    parsed = {
        "mesh_report": _load_json_with_comments(raw["mesh_report"]),
        "sim_config": _load_json_with_comments(raw["sim_config"]),
        "past_runs": _load_json_with_comments(raw["past_runs"]),
        "formulas": _load_json_with_comments(raw["formulas"]),
    }

    # Minimal normalize defaults
    parsed["sim_config"].setdefault("budget_usd", None)
    parsed["sim_config"].setdefault("max_runtime_hours", None)

    updates: Dict[str, Any] = {"parsed": parsed}
    updates |= _trace_append(state, "parse_and_normalize", {"ok": True})
    return updates


# -----------------------------
# Node 2: CFL tool node (DO NOT crash if missing fields)
# -----------------------------
def compute_cfl_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    parsed = state["parsed"]
    mesh = parsed["mesh_report"]
    sim = parsed["sim_config"]

    topo = mesh.get("topology") or {}
    inlet = (sim.get("boundary_conditions") or {}).get("inlet_main") or {}
    ts = (sim.get("numerics") or {}).get("time_step") or {}

    missing = []
    if "cell_count" not in topo:
        missing.append("$.mesh_report.topology.cell_count")
    if "velocity_magnitude" not in inlet:
        missing.append("$.sim_config.boundary_conditions.inlet_main.velocity_magnitude")
    if "hydraulic_diameter" not in inlet:
        missing.append("$.sim_config.boundary_conditions.inlet_main.hydraulic_diameter")
    if "size" not in ts:
        missing.append("$.sim_config.numerics.time_step.size")

    derived = dict(state.get("derived_metrics") or {})

    if missing:
        derived.update(
            {
                "courant_number": None,
                "courant_source": "MISSING_FIELDS",
                "missing_cfl_fields": missing,
            }
        )
        return {
            "derived_metrics": derived,
            **_trace_append(state, "compute_cfl", {"missing": missing}),
        }

    cell_count = float(topo["cell_count"])
    velocity = float(inlet["velocity_magnitude"])
    characteristic_length = float(inlet["hydraulic_diameter"])
    dt = float(ts["size"])

    courant, dx_eff = compute_cfl_from_handbook(
        velocity_m_per_s=velocity,
        time_step_s=dt,
        characteristic_length_m=characteristic_length,
        cell_count=cell_count,
    )

    derived.update(
        {
            "u_used_m_per_s": velocity,
            "dt_used_s": dt,
            "characteristic_length_m": characteristic_length,
            "cell_count": cell_count,
            "dx_eff_m": dx_eff,
            "courant_number": courant,
            "courant_source": "TOOL:compute_cfl_from_handbook",
        }
    )

    return {
        "derived_metrics": derived,
        "trace": {
            **(state.get("trace") or {}),
            "compute_cfl": {"courant_number": courant, "dx_eff_m": dx_eff},
        },
    }

# -----------------------------
# Node 3: deterministic metrics (cost/runtime placeholders removed; DO NOT overwrite CFL)
# -----------------------------
def precompute_metrics_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    parsed = state["parsed"]
    mesh = parsed["mesh_report"]
    sim = parsed["sim_config"]
    formulas = parsed.get("formulas") or {}

    derived = dict(state.get("derived_metrics") or {})

    # -----------------------------
    # Cost / runtime coefficients (deterministic extraction)
    # -----------------------------
    cost_coeff = None
    runtime_coeff = None

    cost_model = formulas.get("cost_model") or {}
    runtime_model = formulas.get("runtime_model") or {}

    for k in (
        "cost_per_m_cells_per_iter",
        "usd_per_m_cells_per_iter",
        "usd_per_mcell_iter",
        "usd_per_mcell_step",
    ):
        v = cost_model.get(k)
        if isinstance(v, (int, float)):
            cost_coeff = float(v)
            break

    for k in (
        "hours_per_m_cells_per_iter",
        "hours_per_mcell_iter",
        "hours_per_mcell_step",
        "hours_per_mcell_per_step",
    ):
        v = runtime_model.get(k)
        if isinstance(v, (int, float)):
            runtime_coeff = float(v)
            break

    # -----------------------------
    # Compute deterministic resource estimates (tool-grade)
    # -----------------------------
    est = compute_resource_estimates(
        mesh_report=mesh,
        sim_config=sim,
        cost_per_m_cells_per_iter=cost_coeff,
        hours_per_m_cells_per_iter=runtime_coeff,
    )

    # Do NOT clobber CFL results; only set/augment resource-related metrics.
    if est.cell_count and est.cell_count > 0:
        derived["cell_count"] = float(est.cell_count)
    else:
        derived.setdefault("cell_count", 0.0)

    derived["ram_gb"] = float(est.ram_gb)

    # Conservative single-value estimates (equal to HIGH)
    derived["estimated_cost_usd"] = float(est.estimated_cost_usd) if est.estimated_cost_usd is not None else None
    derived["estimated_runtime_hours"] = (
        float(est.estimated_runtime_hours) if est.estimated_runtime_hours is not None else None
    )

    # LOW/HIGH bounds (preferred fields for gating and for minimum required budget)
    derived["steps"] = float(est.steps) if est.steps is not None else None
    derived["iters_per_step_min"] = float(est.iters_per_step_min) if est.iters_per_step_min is not None else None
    derived["iters_per_step_max"] = float(est.iters_per_step_max) if est.iters_per_step_max is not None else None

    derived["total_solver_iterations_low"] = (
        float(est.total_solver_iterations_low) if est.total_solver_iterations_low is not None else None
    )
    derived["total_solver_iterations_high"] = (
        float(est.total_solver_iterations_high) if est.total_solver_iterations_high is not None else None
    )

    derived["work_units_low"] = float(est.work_units_low) if est.work_units_low is not None else None
    derived["work_units_high"] = float(est.work_units_high) if est.work_units_high is not None else None

    derived["estimated_cost_usd_low"] = (
        float(est.estimated_cost_usd_low) if est.estimated_cost_usd_low is not None else None
    )
    derived["estimated_cost_usd_high"] = (
        float(est.estimated_cost_usd_high) if est.estimated_cost_usd_high is not None else None
    )

    derived["estimated_runtime_hours_low"] = (
        float(est.estimated_runtime_hours_low) if est.estimated_runtime_hours_low is not None else None
    )
    derived["estimated_runtime_hours_high"] = (
        float(est.estimated_runtime_hours_high) if est.estimated_runtime_hours_high is not None else None
    )

    # Audit metadata for deterministic debugging and LLM anchoring.
    derived["resource_missing_inputs"] = list(est.missing_inputs) if est.missing_inputs else []
    derived["resource_notes"] = list(est.notes) if est.notes else []

    updates: Dict[str, Any] = {"derived_metrics": derived}
    updates |= _trace_append(
        state,
        "precompute_metrics",
        {
            "courant_number": derived.get("courant_number"),
            "cell_count": derived.get("cell_count"),
            "ram_gb": derived.get("ram_gb"),
            "estimated_cost_usd": derived.get("estimated_cost_usd"),
            "estimated_runtime_hours": derived.get("estimated_runtime_hours"),
            "steps": derived.get("steps"),
            "iters_per_step_min": derived.get("iters_per_step_min"),
            "iters_per_step_max": derived.get("iters_per_step_max"),
            "work_units_low": derived.get("work_units_low"),
            "work_units_high": derived.get("work_units_high"),
            "estimated_cost_usd_low": derived.get("estimated_cost_usd_low"),
            "estimated_cost_usd_high": derived.get("estimated_cost_usd_high"),
            "resource_missing_inputs": derived.get("resource_missing_inputs"),
        },
    )
    return updates

def retrieve_similar_runs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    parsed = state["parsed"]
    sim = parsed["sim_config"]
    past_runs = parsed["past_runs"]

    derived = dict(state.get("derived_metrics") or {})
    hits = retrieve_similar_runs(sim_config=sim, past_runs=past_runs, top_k=3)

    derived["similar_runs"] = hits
    return {
        "derived_metrics": derived,
        **_trace_append(state, "retrieve_similar_runs", {"top_k": len(hits), "hits": hits}),
    }
# -----------------------------
# Node 4: Phase 1 - rule-based votes (swap to Gemini later)
# -----------------------------
def phase1_agents_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    parsed = state["parsed"]
    mesh = parsed["mesh_report"]
    sim = parsed["sim_config"]
    formulas = parsed["formulas"]
    derived = state.get("derived_metrics") or {}

    votes: List[Dict[str, Any]] = []

    # GEOMETER
    qm = mesh.get("quality_metrics") or {}
    skew_max = float((qm.get("skewness") or {}).get("max", 0.0) or 0.0)
    ortho_min = float((qm.get("orthogonal_quality") or {}).get("min", 1.0) or 1.0)

    if skew_max >= 0.90:
        v = AgentVote(
            agent=AgentName.GEOMETER,
            vote=VoteType.REJECT,
            reason=f"Mesh skewness.max={skew_max} >= 0.90 (hard reject).",
            hard_constraints_triggered=["MESH_SKEWNESS_TOO_HIGH"],
            modifications_required=[],
        )
    elif ortho_min < 0.10:
        v = AgentVote(
            agent=AgentName.GEOMETER,
            vote=VoteType.REJECT,
            reason=f"Mesh orthogonal_quality.min={ortho_min} < 0.10 (hard reject).",
            hard_constraints_triggered=["MESH_ORTHO_QUALITY_TOO_LOW"],
            modifications_required=[],
        )
    else:
        v = AgentVote(
            agent=AgentName.GEOMETER,
            vote=VoteType.APPROVE,
            reason="Mesh quality metrics are within acceptable ranges.",
            hard_constraints_triggered=[],
            modifications_required=[],
        )
    votes.append(v.model_dump(mode="json"))

    # PHYSICIST (enum check)
    allowed = {"k-epsilon", "k-omega", "Spalart-Allmaras"}
    inlet_u = float((sim.get("boundary_conditions") or {}).get("inlet_main", {}).get("velocity_magnitude", 0.0) or 0.0)
    physics = sim.get("physics_setup") or {}
    tm = physics.get("turbulence_model") or {}
    model = tm.get("model", None)
    flow_regime = str(physics.get("flow_regime", "")).lower()

    if model not in allowed:
        v = AgentVote(
            agent=AgentName.PHYSICIST,
            vote=VoteType.REJECT,
            reason=f"Invalid turbulence_model enum: {model}. Allowed: {sorted(allowed)}.",
            hard_constraints_triggered=["INVALID_TURBULENCE_MODEL_ENUM"],
            modifications_required=[],
        )
    elif flow_regime == "laminar" and inlet_u >= 100.0:
        v = AgentVote(
            agent=AgentName.PHYSICIST,
            vote=VoteType.REJECT,
            reason="Laminar at >=100 m/s is disallowed by policy.",
            hard_constraints_triggered=["LAMINAR_AT_HIGH_SPEED"],
            modifications_required=[],
        )
    else:
        v = AgentVote(
            agent=AgentName.PHYSICIST,
            vote=VoteType.APPROVE,
            reason="Physics model passes policy checks.",
            hard_constraints_triggered=[],
            modifications_required=[],
        )
    votes.append(v.model_dump(mode="json"))

    # RESOURCE_MANAGER (budget / ROI gating using deterministic cost tool outputs)
    budget = sim.get("budget_usd", None)

    # Prefer HIGH bound for conservative gating
    est_cost_high = derived.get("estimated_cost_usd_high", None)
    est_cost = derived.get("estimated_cost_usd", None)
    est_runtime_high = derived.get("estimated_runtime_hours_high", None)
    est_runtime = derived.get("estimated_runtime_hours", None)

    # pick conservative values
    cost_used = est_cost_high if est_cost_high is not None else est_cost
    runtime_used = est_runtime_high if est_runtime_high is not None else est_runtime

    if budget is None:
        # No budget => cannot do ROI gate deterministically
        v = AgentVote(
            agent=AgentName.RESOURCE_MANAGER,
            vote=VoteType.MODIFY,
            reason="Missing budget_usd; cannot perform ROI gating.",
            hard_constraints_triggered=[],
            modifications_required=[
                {
                    "field": "$.budget_usd",
                    "proposed_value": 50,
                    "rationale": "Budget is required for deterministic ROI gating.",
                    "priority": "HIGH",
                }
            ],
        )
    elif cost_used is None:
        # Budget exists but we cannot compute cost (most likely missing coefficient in formulas)
        v = AgentVote(
            agent=AgentName.RESOURCE_MANAGER,
            vote=VoteType.MODIFY,
            reason="Budget present, but estimated_cost_usd is unavailable (missing cost model coefficient or iterations inputs).",
            hard_constraints_triggered=[],
            modifications_required=[
                {
                    "field": "$.formulas.cost_model.cost_per_m_cells_per_iter",
                    "proposed_value": 0.00002,
                    "rationale": "Provide cost_per_m_cells_per_iter to enable deterministic cost estimation.",
                    "priority": "HIGH",
                }
            ],
        )
    else:
        budget_f = float(budget)
        cost_f = float(cost_used)
        
        # New Logic: Tolerance window
        # 1. Cost > Budget * 1.2 => Hard Reject (>20% over)
        # 2. Budget < Cost <= Budget * 1.2 => Modify (0-20% over)
        # 3. Cost <= Budget => Approve

        if cost_f > budget_f * 1.2:
            # Hard reject: significantly over budget
            v = AgentVote(
                agent=AgentName.RESOURCE_MANAGER,
                vote=VoteType.REJECT,
                reason=f"Estimated cost ${cost_f:.6g} exceeds budget ${budget_f:.6g} by >20%.",
                hard_constraints_triggered=["BUDGET_EXCEEDED"],
                modifications_required=[],
            )
        elif cost_f > budget_f:
            # Modify: minor overrun (<= 20%)
            v = AgentVote(
                agent=AgentName.RESOURCE_MANAGER,
                vote=VoteType.MODIFY,
                reason=f"Estimated cost ${cost_f:.6g} slightly exceeds budget ${budget_f:.6g} (<=20%). Consider increasing budget or optimizing.",
                hard_constraints_triggered=[],
                modifications_required=[
                    {
                        "field": "$.budget_usd",
                        "proposed_value": round(cost_f * 1.05, 2),
                        "rationale": "Slight budget increase required to cover conservative estimate.",
                        "priority": "MEDIUM",
                    }
                ],
            )
        else:
            # Within budget => approve
            v = AgentVote(
                agent=AgentName.RESOURCE_MANAGER,
                vote=VoteType.APPROVE,
                reason=f"Estimated cost ${cost_f:.6g} is within budget ${budget_f:.6g}.",
                hard_constraints_triggered=[],
                modifications_required=[],
            )

    votes.append(v.model_dump(mode="json"))

    # HISTORIAN (needs similar_runs; placeholder => MODIFY)
    similar = derived.get("similar_runs") or []

    if len(similar) == 0:
        v = AgentVote(
            agent=AgentName.HISTORIAN,
            vote=VoteType.MODIFY,
            reason="No similar_runs produced; ensure similarity tool runs and past_runs is loaded.",
            hard_constraints_triggered=[],
            modifications_required=[
                {
                    "field": "$.derived_metrics.similar_runs",
                    "proposed_value": "enable",
                    "rationale": "Historian requires similarity evidence for gating.",
                    "priority": "MEDIUM",
                }
            ],
        )
    else:
        # Find the most similar crashed run, if any
        crashed = [h for h in similar if str(h.get("outcome", "")).upper() == "CRASHED"]
        top_crash = crashed[0] if crashed else None

        if top_crash and float(top_crash.get("similarity_score", 0.0)) >= 0.95 and (not bool(top_crash.get("stabilization_knobs_changed"))):
            v = AgentVote(
                agent=AgentName.HISTORIAN,
                vote=VoteType.REJECT,
                reason=f"High similarity to past crash {top_crash.get('job_id')} (>=0.95) with no stabilization knob changes.",
                hard_constraints_triggered=["SIMILAR_TO_PAST_CRASH"],
                modifications_required=[],
            )
        elif top_crash and float(top_crash.get("similarity_score", 0.0)) >= 0.85:
            # Require stabilization changes
            v = AgentVote(
                agent=AgentName.HISTORIAN,
                vote=VoteType.MODIFY,
                reason=f"Moderate/high similarity to past crash {top_crash.get('job_id')} (>=0.85). Recommend stabilization changes.",
                hard_constraints_triggered=[],
                modifications_required=[
                    {
                        "field": "$.solver_settings.relaxation_factors.momentum",
                        "proposed_value": 0.5,
                        "rationale": "Reduce momentum relaxation to mitigate divergence risk seen in similar crash.",
                        "priority": "HIGH",
                    },
                    {
                        "field": "$.numerics.time_step.size",
                        "proposed_value": "decrease",
                        "rationale": "Smaller dt can improve stability when similar setups diverged.",
                        "priority": "MEDIUM",
                    },
                ],
            )
        else:
            v = AgentVote(
                agent=AgentName.HISTORIAN,
                vote=VoteType.APPROVE,
                reason="No sufficiently similar past crash pattern detected in top-k retrieval.",
                hard_constraints_triggered=[],
                modifications_required=[],
            )
    votes.append(v.model_dump(mode="json"))

    # STABILIZER (must rely on tool-computed courant_number)
    cfl = derived.get("courant_number", None)
    max_allowed = None
    if isinstance((formulas.get("cfl") or {}).get("max_allowed"), (int, float)):
        max_allowed = float(formulas["cfl"]["max_allowed"])

    if cfl is None:
        v = AgentVote(
            agent=AgentName.STABILIZER,
            vote=VoteType.MODIFY,
            reason="Missing tool-computed courant_number.",
            hard_constraints_triggered=[],
            modifications_required=[
                {
                    "field": "$.derived_metrics.courant_number",
                    "proposed_value": "compute",
                    "rationale": "CFL check requires tool-computed Courant number.",
                    "priority": "HIGH",
                }
            ],
        )
    elif max_allowed is not None:
        cfl_f = float(cfl)
        
        # New Logic: Tolerance for CFL
        # 1. CFL > Max * 2.0 => Hard Reject (Way too unstable)
        # 2. Max < CFL <= Max * 2.0 => Modify (Risky, maybe implicit solver can handle it)
        # 3. CFL <= Max => Approve
        
        if cfl_f > max_allowed * 2.0:
            v = AgentVote(
                agent=AgentName.STABILIZER,
                vote=VoteType.REJECT,
                reason=f"CFL violation: Courant={cfl_f:.2f} > 2x max_allowed ({max_allowed}). Unstable.",
                hard_constraints_triggered=["CFL_VIOLATION"],
                modifications_required=[],
            )
        elif cfl_f > max_allowed:
            v = AgentVote(
                agent=AgentName.STABILIZER,
                vote=VoteType.MODIFY,
                reason=f"CFL {cfl_f:.2f} exceeds max_allowed {max_allowed} but is within 2x tolerance. Recommend reducing time-step.",
                hard_constraints_triggered=[],
                modifications_required=[
                    {
                        "field": "$.numerics.time_step.size",
                        "proposed_value": "decrease",
                        "rationale": "Reduce dt to bring Courant number below max_allowed.",
                        "priority": "HIGH",
                    }
                ],
            )
        else:
            v = AgentVote(
                agent=AgentName.STABILIZER,
                vote=VoteType.APPROVE,
                reason="CFL check passes under current thresholds.",
                hard_constraints_triggered=[],
                modifications_required=[],
            )
    else:
        # No max_allowed defined? Approve by default or Warn? Let's approve for now.
        v = AgentVote(
            agent=AgentName.STABILIZER,
            vote=VoteType.APPROVE,
            reason="No max_allowed CFL defined in formulas.",
            hard_constraints_triggered=[],
            modifications_required=[],
        )
    votes.append(v.model_dump(mode="json"))

    return {
        "phase1_votes": votes,
        **_trace_append(state, "phase1_agents", {"votes": [x["vote"] for x in votes]}),
    }


def needs_debate_router(state: Dict[str, Any]) -> str:
    _ensure_time(state)

    votes = state.get("phase1_votes") or []
    if not votes:
        return "synthesize_verdict"

    unique = sorted({v["vote"] for v in votes})
    return "synthesize_verdict" if len(unique) == 1 else "debate_one_round"


def debate_one_round_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    votes = state.get("phase1_votes") or []
    approve = [v for v in votes if v["vote"] == VoteType.APPROVE.value]
    reject = [v for v in votes if v["vote"] == VoteType.REJECT.value]
    modify = [v for v in votes if v["vote"] == VoteType.MODIFY.value]

    # Deterministically pick support: try Approve -> Modify -> Reject (consensus fallback)
    # We just need ONE defender to stand for the "Yes" or "Maybe" side.
    if approve:
        support_vote = approve[0]
    elif modify:
        support_vote = modify[0]
    else:
        # If everyone rejects, pick the first one as a nominal "defender" just to enable loop,
        # or better: return early because there's no debate (everyone agrees to kill it).
        # But if we must debate, we pick the first.
        # Actually, if everyone rejects, debate is moot. But let's assume router sent us here.
        support_vote = votes[0]

    support_agent = support_vote["agent"]
    support_prompt_sys = _get_agent_prompt(support_agent)
    
    # Identify ALL opponents (Rejectors and Modifiers who disagree with Support)
    # If Support is Approve, opponents are Reject + Modify
    # If Support is Modify, opponents are Reject (and maybe other Modifies? simplicity: just Reject)
    opponents = []
    if support_vote["vote"] == VoteType.APPROVE.value:
        opponents = reject + modify
    elif support_vote["vote"] == VoteType.MODIFY.value:
        opponents = reject
    
    # Fallback: if no clear opponents found but we are here, maybe it's Modify vs Modify?
    # Let's just grab anyone who isn't the support agent and has a different vote/reason.
    if not opponents:
        opponents = [v for v in votes if v["agent"] != support_agent and v["vote"] != support_vote["vote"]]

    # Sort opponents deterministically by AgentName to ensure stable order
    opponents.sort(key=lambda x: x["agent"])

    # If still empty (e.g. all Approved), return empty
    if not opponents:
        return {"debate_messages": []}

    messages: List[Dict[str, Any]] = []

    # 1. Support agent speaks ONCE to set the stage
    # "I vote APPROVE because X. I see some of you disagree."
    p1 = f"""
    You are in a debate. You voted {support_vote['vote']} because: "{support_vote['reason']}".
    You are facing {len(opponents)} opponent(s): {', '.join([op['agent'] for op in opponents])}.
    
    Produce a short, sharp argument (max 40 words) defending your position and challenging the opponents.
    Output plain text only.
    """
    try:
        msg1_text = call_gemini(support_prompt_sys + "\n\n" + p1, temperature=0.0).strip()
    except Exception as e:
        msg1_text = f"(AI Error: {e}) My vote stands based on {support_vote['reason']}."

    m1 = DebateMessage(
        agent=AgentName(support_agent),
        stance=DebateStance.DEFEND,
        message=msg1_text,
        targets=[AgentName(op["agent"]) for op in opponents][:3], # max 3 targets in schema
    )
    messages.append(m1.model_dump(mode="json"))

    # 2. EACH Opponent rebuts the Support agent independently (Parallel Logic in loop)
    # In a real async system we'd fire these in parallel. Here we loop sequentially but it's fast.
    for op_vote in opponents:
        op_agent = op_vote["agent"]
        op_prompt_sys = _get_agent_prompt(op_agent)
        
        p2 = f"""
        You are in a debate. You voted {op_vote['vote']} because: "{op_vote['reason']}".
        The supporter {support_agent} just said: "{msg1_text}".
        
        Produce a short, sharp rebuttal (max 40 words) explaining why they are wrong and your constraint is non-negotiable.
        Output plain text only.
        """
        try:
            msg2_text = call_gemini(op_prompt_sys + "\n\n" + p2, temperature=0.0).strip()
        except Exception as e:
            msg2_text = f"(AI Error: {e}) I disagree. {op_vote['reason']}"
            
        m2 = DebateMessage(
            agent=AgentName(op_agent),
            stance=DebateStance.ATTACK,
            message=msg2_text,
            targets=[AgentName(support_agent)],
        )
        messages.append(m2.model_dump(mode="json"))

    return {
        "debate_messages": messages,
        **_trace_append(state, "debate_one_round", {
            "support": support_agent, 
            "opponents": [op["agent"] for op in opponents]
        }),
    }


def synthesize_verdict_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state, margin_s=0.2)

    votes = state.get("phase1_votes") or []
    debate_msgs = state.get("debate_messages") or []
    
    # 1. Deterministic Fallback Checks
    has_hard_reject = any(
        (v["vote"] == VoteType.REJECT.value) and (len(v.get("hard_constraints_triggered") or []) > 0)
        for v in votes
    )
    all_approve = len(votes) > 0 and all(v["vote"] == VoteType.APPROVE.value for v in votes)

    # If all approve, no need for LLM synthesis (save time/cost)
    if all_approve:
        verdict = FinalVerdict(status=VerdictStatus.APPROVED, confidence=1.0, modifications_required=[])
        return {
            "final_verdict": verdict.model_dump(mode="json"),
            **_trace_append(state, "synthesize_verdict", {"method": "unanimous_approve"}),
        }

    # 2. LLM Synthesis
    # Prepare context
    synth_prompt = _get_agent_prompt("SYNTHESIZER")
    
    # Minimal serialization for prompt context
    votes_json = json.dumps(votes, indent=2, ensure_ascii=False)
    debate_json = json.dumps(debate_msgs, indent=2, ensure_ascii=False)
    
    user_prompt = f"""
    PHASE 1 VOTES:
    {votes_json}
    
    DEBATE LOGS:
    {debate_json}
    
    Based on the above, produce the Final Verdict JSON.
    """
    
    try:
        response_text = call_gemini(synth_prompt + "\n\n" + user_prompt, json_mode=True, temperature=0.0).strip()
        # Strip markdown code blocks if present
        if response_text.startswith("```"):
            response_text = re.sub(r"^```[a-z]*\n", "", response_text)
            response_text = re.sub(r"\n```$", "", response_text)
            
        final_dict = json.loads(response_text)
        verdict = FinalVerdict.model_validate(final_dict)
        
        # Safety check: if hard reject exists, status MUST be REJECTED
        if has_hard_reject and verdict.status == VerdictStatus.APPROVED:
            # Overrule LLM hallucination
            verdict = FinalVerdict(
                status=VerdictStatus.REJECTED, 
                confidence=verdict.confidence,
                modifications_required=[
                    {
                        "field": "$.status", 
                        "proposed_value": "REJECTED", 
                        "rationale": "Hard constraints triggered; LLM approval overruled.",
                        "priority": "HIGH",
                        "owner": AgentName.SYNTHESIZER
                    }
                ]
            )
            
        return {
            "final_verdict": verdict.model_dump(mode="json"),
            **_trace_append(state, "synthesize_verdict", {"method": "llm_synthesis"}),
        }

    except Exception as e:
        # Fallback to rule-based logic if LLM fails
        # Re-use the original rule-based logic for safety
        merged_mods: List[Dict[str, Any]] = []
        seen = set()
        for v in votes:
            for m in v.get("modifications_required") or []:
                key = (m["field"], json.dumps(m.get("proposed_value", None), sort_keys=True, ensure_ascii=False))
                if key in seen:
                    continue
                seen.add(key)
                merged_mods.append(
                    {
                        "field": m["field"],
                        "proposed_value": m.get("proposed_value", None),
                        "rationale": m.get("rationale", ""),
                        "priority": m.get("priority", "MEDIUM"),
                        "owner": v["agent"],
                    }
                )

        conf = 1.0
        for v in votes:
            if v["vote"] == VoteType.REJECT.value:
                conf -= 0.25
            elif v["vote"] == VoteType.MODIFY.value:
                conf -= 0.10
        conf = max(0.0, min(1.0, conf))

        if not merged_mods:
            merged_mods = [
                {
                    "field": "$.action",
                    "proposed_value": "provide_missing_information",
                    "rationale": "Non-unanimous votes; provide required info or changes (Fallback Logic).",
                    "priority": "HIGH",
                    "owner": AgentName.SYNTHESIZER.value,
                }
            ]
        verdict = FinalVerdict(status=VerdictStatus.REJECTED, confidence=conf, modifications_required=merged_mods)
        
        return {
            "final_verdict": verdict.model_dump(mode="json"),
            **_trace_append(state, "synthesize_verdict", {"method": "fallback_rule_based", "error": str(e)}),
        }
