from __future__ import annotations

import time
from typing import Any, Dict, List

from gatekeeper.schemas import (
    AgentName,
    VoteType,
    VerdictStatus,
    DebateMessage,
    DebateStance,
)
from gatekeeper.tools.cfl import compute_cfl_from_handbook
from gatekeeper.tools.similarity import retrieve_similar_runs
from gatekeeper.tools.cost import compute_resource_estimates
from gatekeeper.llm import call_gemini

# Import Agents
from gatekeeper.agents.geometer import GeometerAgent
from gatekeeper.agents.physicist import PhysicistAgent
from gatekeeper.agents.resource_manager import ResourceManagerAgent
from gatekeeper.agents.historian import HistorianAgent
from gatekeeper.agents.stabilizer import StabilizerAgent
from gatekeeper.agents.synthesizer import SynthesizerAgent
from gatekeeper.agents.base import BaseAgent

# Import IO utilities
from gatekeeper.io.load_json import load_json_with_comments
from gatekeeper.io.normalize import normalize_sim_config

# -----------------------------
# Utilities: timebox
# -----------------------------
def _ensure_time(state: Dict[str, Any], margin_s: float = 1.0) -> None:
    deadline = float(state.get("deadline_epoch_s", 0.0) or 0.0)
    if deadline <= 0:
        return
    if time.time() + margin_s > deadline:
        raise TimeoutError("TIMEBOX_EXCEEDED")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _trace_append(state: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    trace = dict(state.get("trace") or {})
    trace[key] = value
    return {"trace": trace}

def _get_agent_instance(name_str: str) -> BaseAgent:
    name = AgentName(name_str)
    if name == AgentName.GEOMETER:
        return GeometerAgent()
    elif name == AgentName.PHYSICIST:
        return PhysicistAgent()
    elif name == AgentName.RESOURCE_MANAGER:
        return ResourceManagerAgent()
    elif name == AgentName.HISTORIAN:
        return HistorianAgent()
    elif name == AgentName.STABILIZER:
        return StabilizerAgent()
    elif name == AgentName.SYNTHESIZER:
        return SynthesizerAgent()
    else:
        raise ValueError(f"Unknown agent: {name_str}")

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
    
    # Use new helper from load_json.py
    parsed = {
        "mesh_report": load_json_with_comments(raw["mesh_report"]),
        "sim_config": load_json_with_comments(raw["sim_config"]),
        "past_runs": load_json_with_comments(raw["past_runs"]),
        "formulas": load_json_with_comments(raw["formulas"]),
    }

    # Use new helper from normalize.py
    parsed["sim_config"] = normalize_sim_config(parsed["sim_config"])

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
# Node 3: deterministic metrics
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
# Node 4: Phase 1 - rule-based votes (REFACTORED)
# -----------------------------
def phase1_agents_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state)

    votes: List[Dict[str, Any]] = []

    # Instantiate and run each agent
    agents = [
        GeometerAgent(),
        PhysicistAgent(),
        ResourceManagerAgent(),
        HistorianAgent(),
        StabilizerAgent(),
    ]

    for agent in agents:
        v = agent.vote(state)
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

    # 1. Grouping: Defenders vs Challengers
    defenders = []
    challengers = []

    if approve:
        defenders = approve
        challengers = reject + modify
    elif modify:
        defenders = modify
        challengers = reject
    else:
        # Everyone rejected (consensus reject) -> No debate needed
        return {"debate_messages": []}

    # Sort for deterministic order
    defenders.sort(key=lambda x: x["agent"])
    challengers.sort(key=lambda x: x["agent"])

    if not challengers:
        # Everyone approved (consensus approve) -> No debate needed
        return {"debate_messages": []}

    messages: List[Dict[str, Any]] = []

    # 2. Round 1: Defenders Speak (Parallel)
    # Each defender sees the opposition exists and states their case.
    challenger_names = [c["agent"] for c in challengers]
    
    # We collect all defender messages first
    defender_msgs_obj = []

    for d_vote in defenders:
        d_agent_name = d_vote["agent"]
        d_agent = _get_agent_instance(d_agent_name)
        d_prompt_sys = d_agent.load_system_prompt()
        
        # Context: Identify who is challenging and why (briefly)
        opp_summary = "\n".join([f"- {c['agent']}: {c['vote']} ({c['reason']})" for c in challengers])
        
        p1 = f"""
        You are in a debate. You voted {d_vote['vote']} because: "{d_vote['reason']}".
        The following agents oppose you (Challengers):
        {opp_summary}
        
        Produce a strong, concise argument (max 40 words) defending your position against these challenges.
        Output ONLY plain text. DO NOT use markdown, JSON blocks, or any formatting.
        """
        
        try:
            msg_text = call_gemini(d_prompt_sys + "\n\n" + p1, temperature=0.0).strip()
        except Exception as e:
            msg_text = f"(AI Error: {e}) My vote stands based on {d_vote['reason']}."
            
        m = DebateMessage(
            agent=AgentName(d_agent_name),
            stance=DebateStance.DEFEND,
            message=msg_text,
            targets=[AgentName(n) for n in challenger_names], # target all challengers
        )
        defender_msgs_obj.append(m)
        messages.append(m.model_dump(mode="json"))

    # 3. Round 2: Challengers Rebut (Parallel)
    # Each challenger sees ALL defender messages and rebuts the collective defense.
    
    # Format defender arguments for the context
    defender_context = "\n".join([f"- {m.agent}: {m.message}" for m in defender_msgs_obj])
    defender_names = [d["agent"] for d in defenders]

    for c_vote in challengers:
        c_agent_name = c_vote["agent"]
        c_agent = _get_agent_instance(c_agent_name)
        c_prompt_sys = c_agent.load_system_prompt()
        
        p2 = f"""
        You are in a debate. You voted {c_vote['vote']} because: "{c_vote['reason']}".
        The Defenders have stated their case:
        {defender_context}
        
        Produce a concise rebuttal (max 40 words) addressing these points and reaffirming why your constraint is critical.
        Output ONLY plain text. DO NOT use markdown, JSON blocks, or any formatting.
        """
        
        try:
            msg_text = call_gemini(c_prompt_sys + "\n\n" + p2, temperature=0.0).strip()
        except Exception as e:
            msg_text = f"(AI Error: {e}) I disagree. {c_vote['reason']}"
            
        m = DebateMessage(
            agent=AgentName(c_agent_name),
            stance=DebateStance.ATTACK,
            message=msg_text,
            targets=[AgentName(n) for n in defender_names], # target all defenders
        )
        messages.append(m.model_dump(mode="json"))

    return {
        "debate_messages": messages,
        **_trace_append(state, "debate_one_round", {
            "defenders": defender_names, 
            "challengers": challenger_names
        }),
    }


def synthesize_verdict_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_time(state, margin_s=0.2)
    
    synthesizer = SynthesizerAgent()
    verdict = synthesizer.synthesize(state)

    method = "llm_synthesis"
    if verdict.confidence == 1.0 and not verdict.modifications_required and verdict.status == VerdictStatus.APPROVED:
         method = "unanimous_approve" # Simple heuristic for tracing
    
    return {
        "final_verdict": verdict.model_dump(mode="json"),
        **_trace_append(state, "synthesize_verdict", {"method": method}),
    }
