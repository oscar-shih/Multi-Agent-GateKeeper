from typing import Any, Dict
from gatekeeper.agents.base import BaseAgent
from gatekeeper.schemas import AgentName, AgentVote, VoteType

class GeometerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.GEOMETER)

    def vote(self, state: Dict[str, Any]) -> AgentVote:
        parsed = state["parsed"]
        mesh = parsed["mesh_report"]
        topo = mesh.get("topology") or {}
        
        qm = mesh.get("quality_metrics") or {}
        skew_max = float((qm.get("skewness") or {}).get("max", 0.0) or 0.0)
        ortho_min = float((qm.get("orthogonal_quality") or {}).get("min", 1.0) or 1.0)
        
        # Check for Negative Volume (Fatal)
        # Assuming min_volume might be present in quality metrics
        min_vol = float(qm.get("min_volume", 1e-9))
        
        cell_count = float(topo.get("cell_count", 0))

        if cell_count <= 0:
             return AgentVote(
                agent=self.name,
                vote=VoteType.REJECT,
                reason=f"Invalid cell count: {cell_count}. Mesh must have cells.",
                hard_constraints_triggered=["INVALID_MESH_TOPOLOGY"],
                modifications_required=[],
            )

        if min_vol <= 0:
            return AgentVote(
                agent=self.name,
                vote=VoteType.REJECT,
                reason=f"Negative or zero cell volume detected: {min_vol:.3e}. Mesh is invalid.",
                hard_constraints_triggered=["NEGATIVE_VOLUME_DETECTED"],
                modifications_required=[],
            )

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
