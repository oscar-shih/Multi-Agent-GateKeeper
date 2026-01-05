from typing import Any, Dict
from gatekeeper.agents.base import BaseAgent
from gatekeeper.schemas import AgentName, AgentVote, VoteType

class StabilizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.STABILIZER)

    def vote(self, state: Dict[str, Any]) -> AgentVote:
        parsed = state["parsed"]
        formulas = parsed["formulas"]
        derived = state.get("derived_metrics") or {}
        
        cfl = derived.get("courant_number", None)
        max_allowed = None
        if isinstance((formulas.get("cfl") or {}).get("max_allowed"), (int, float)):
            max_allowed = float(formulas["cfl"]["max_allowed"])

        if cfl is None:
            return AgentVote(
                agent=self.name,
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
            
            # Logic: Tolerance for CFL
            # 1. CFL > Max * 2.0 => Hard Reject (Way too unstable)
            # 2. Max < CFL <= Max * 2.0 => Modify (Risky, maybe implicit solver can handle it)
            # 3. CFL <= Max => Approve
            
            if cfl_f > max_allowed * 2.0:
                return AgentVote(
                    agent=self.name,
                    vote=VoteType.REJECT,
                    reason=f"CFL violation: Courant={cfl_f:.2f} > 2x max_allowed ({max_allowed}). Unstable.",
                    hard_constraints_triggered=["CFL_VIOLATION"],
                    modifications_required=[],
                )
            elif cfl_f > max_allowed:
                return AgentVote(
                    agent=self.name,
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
                return AgentVote(
                    agent=self.name,
                    vote=VoteType.APPROVE,
                    reason="CFL check passes under current thresholds.",
                    hard_constraints_triggered=[],
                    modifications_required=[],
                )
        else:
            return AgentVote(
                agent=self.name,
                vote=VoteType.APPROVE,
                reason="No max_allowed CFL defined in formulas.",
                hard_constraints_triggered=[],
                modifications_required=[],
            )

