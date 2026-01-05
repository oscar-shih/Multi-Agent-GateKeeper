from typing import Any, Dict
from gatekeeper.agents.base import BaseAgent
from gatekeeper.schemas import AgentName, AgentVote, VoteType

class GeometerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.GEOMETER)

    def vote(self, state: Dict[str, Any]) -> AgentVote:
        parsed = state["parsed"]
        mesh = parsed["mesh_report"]
        
        qm = mesh.get("quality_metrics") or {}
        skew_max = float((qm.get("skewness") or {}).get("max", 0.0) or 0.0)
        ortho_min = float((qm.get("orthogonal_quality") or {}).get("min", 1.0) or 1.0)

        if skew_max >= 0.90:
            return AgentVote(
                agent=self.name,
                vote=VoteType.REJECT,
                reason=f"Mesh skewness.max={skew_max} >= 0.90 (hard reject).",
                hard_constraints_triggered=["MESH_SKEWNESS_TOO_HIGH"],
                modifications_required=[],
            )
        elif ortho_min < 0.10:
            return AgentVote(
                agent=self.name,
                vote=VoteType.REJECT,
                reason=f"Mesh orthogonal_quality.min={ortho_min} < 0.10 (hard reject).",
                hard_constraints_triggered=["MESH_ORTHO_QUALITY_TOO_LOW"],
                modifications_required=[],
            )
        else:
            return AgentVote(
                agent=self.name,
                vote=VoteType.APPROVE,
                reason="Mesh quality metrics are within acceptable ranges.",
                hard_constraints_triggered=[],
                modifications_required=[],
            )

