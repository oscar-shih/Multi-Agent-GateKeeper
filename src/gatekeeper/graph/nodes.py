# src/gatekeeper/graph/nodes.py
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from gatekeeper.schemas import (
    AgentName,
    VoteType,
    VerdictStatus,
    AgentVote,
    DebateMessage,
    FinalVerdict,
)
from gatekeeper.tools.cfl import compute_cfl_from_handbook
from gatekeeper.tools.similarity import retrieve_similar_runs

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
# Node 3: cost + similarity (MVP placeholders; DO NOT overwrite CFL)
# -----------------------------
def precompute_metrics_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    parsed = state["parsed"]
    mesh = parsed["mesh_report"]

    derived = dict(state.get("derived_metrics") or {})

    # Keep CFL fields intact; only add missing placeholders
    cell_count = float((mesh.get("topology") or {}).get("cell_count", 0) or 0)
    derived.setdefault("cell_count", cell_count)

    derived.setdefault("estimated_cost_usd", None)
    derived.setdefault("estimated_runtime_hours", None)
    derived.setdefault("similar_runs", [])

    updates: Dict[str, Any] = {"derived_metrics": derived}
    updates |= _trace_append(
        state,
        "precompute_metrics",
        {"courant_number": derived.get("courant_number")},
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

    # RESOURCE_MANAGER (budget required)
    budget = sim.get("budget_usd", None)
    if budget is None:
        v = AgentVote(
            agent=AgentName.RESOURCE_MANAGER,
            vote=VoteType.MODIFY,
            reason="Missing budget_usd; cannot assess ROI.",
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
    else:
        v = AgentVote(
            agent=AgentName.RESOURCE_MANAGER,
            vote=VoteType.APPROVE,
            reason="Budget present (cost model not enabled yet).",
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
    elif max_allowed is not None and float(cfl) > max_allowed:
        v = AgentVote(
            agent=AgentName.STABILIZER,
            vote=VoteType.REJECT,
            reason=f"CFL violation: Courant={cfl} > max_allowed={max_allowed}.",
            hard_constraints_triggered=["CFL_VIOLATION"],
            modifications_required=[],
        )
    else:
        v = AgentVote(
            agent=AgentName.STABILIZER,
            vote=VoteType.APPROVE,
            reason="CFL check passes under current thresholds.",
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

    support = approve[0]["agent"] if approve else (modify[0]["agent"] if modify else votes[0]["agent"])
    oppose = reject[0]["agent"] if reject else (modify[0]["agent"] if modify else votes[0]["agent"])

    m1 = DebateMessage(
        agent=AgentName(support),
        stance="DEFEND",
        message="My vote reflects my domain constraints; propose minimal changes to satisfy other stakeholders.",
        targets=[AgentName(oppose)],
    )
    m2 = DebateMessage(
        agent=AgentName(oppose),
        stance="ATTACK",
        message="Your position ignores a hard risk in my domain; adjust flagged items before running.",
        targets=[AgentName(support)],
    )

    return {
        "debate_messages": [m1.model_dump(mode="json"), m2.model_dump(mode="json")],
        **_trace_append(state, "debate_one_round", {"support": support, "oppose": oppose}),
    }


def synthesize_verdict_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state, margin_s=0.2)

    votes = state.get("phase1_votes") or []

    has_hard_reject = any(
        (v["vote"] == VoteType.REJECT.value) and (len(v.get("hard_constraints_triggered") or []) > 0)
        for v in votes
    )
    all_approve = len(votes) > 0 and all(v["vote"] == VoteType.APPROVE.value for v in votes)

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

    if all_approve:
        verdict = FinalVerdict(status=VerdictStatus.APPROVED, confidence=conf, modifications_required=[])
    else:
        if not merged_mods:
            merged_mods = [
                {
                    "field": "$.action",
                    "proposed_value": "provide_missing_information",
                    "rationale": "Non-unanimous votes; provide required info or changes.",
                    "priority": "HIGH",
                    "owner": AgentName.SYNTHESIZER.value,
                }
            ]
        verdict = FinalVerdict(status=VerdictStatus.REJECTED, confidence=conf, modifications_required=merged_mods)

    return {
        "final_verdict": verdict.model_dump(mode="json"),
        **_trace_append(state, "synthesize_verdict", {"final_status": verdict.status}),
    }