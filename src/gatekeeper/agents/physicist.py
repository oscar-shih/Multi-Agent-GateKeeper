from typing import Any, Dict
from gatekeeper.agents.base import BaseAgent
from gatekeeper.schemas import AgentName, AgentVote, VoteType

class PhysicistAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.PHYSICIST)

    def vote(self, state: Dict[str, Any]) -> AgentVote:
        parsed = state["parsed"]
        sim = parsed["sim_config"]
        
        allowed = {"k-epsilon", "k-omega", "Spalart-Allmaras"}
        inlet_u = float((sim.get("boundary_conditions") or {}).get("inlet_main", {}).get("velocity_magnitude", 0.0) or 0.0)
        physics = sim.get("physics_setup") or {}
        tm = physics.get("turbulence_model") or {}
        model = tm.get("model", None)
        flow_regime = str(physics.get("flow_regime", "")).lower()

        if model not in allowed:
            return AgentVote(
                agent=self.name,
                vote=VoteType.REJECT,
                reason=f"Invalid turbulence_model enum: {model}. Allowed: {sorted(allowed)}.",
                hard_constraints_triggered=["INVALID_TURBULENCE_MODEL_ENUM"],
                modifications_required=[],
            )
        elif flow_regime == "laminar" and inlet_u >= 100.0:
            return AgentVote(
                agent=self.name,
                vote=VoteType.REJECT,
                reason="Laminar at >=100 m/s is disallowed by policy.",
                hard_constraints_triggered=["LAMINAR_AT_HIGH_SPEED"],
                modifications_required=[],
            )
        else:
            return AgentVote(
                agent=self.name,
                vote=VoteType.APPROVE,
                reason="Physics model passes policy checks.",
                hard_constraints_triggered=[],
                modifications_required=[],
            )

