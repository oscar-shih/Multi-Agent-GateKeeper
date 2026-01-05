from typing import Any, Dict
from gatekeeper.agents.base import BaseAgent
from gatekeeper.schemas import AgentName, AgentVote, VoteType

class ResourceManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.RESOURCE_MANAGER)

    def vote(self, state: Dict[str, Any]) -> AgentVote:
        parsed = state["parsed"]
        sim = parsed["sim_config"]
        derived = state.get("derived_metrics") or {}
        
        budget = sim.get("budget_usd", None)

        # Prefer HIGH bound for conservative gating
        est_cost_high = derived.get("estimated_cost_usd_high", None)
        est_cost = derived.get("estimated_cost_usd", None)
        
        # pick conservative values
        cost_used = est_cost_high if est_cost_high is not None else est_cost

        if budget is None:
            # No budget => cannot do ROI gate deterministically
            return AgentVote(
                agent=self.name,
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
            # Budget exists but we cannot compute cost
            return AgentVote(
                agent=self.name,
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
            
            # Logic: Tolerance window
            # 1. Cost > Budget * 1.2 => Hard Reject (>20% over)
            # 2. Budget < Cost <= Budget * 1.2 => Modify (0-20% over)
            # 3. Cost <= Budget => Approve

            if cost_f > budget_f * 1.2:
                # Hard reject: significantly over budget
                return AgentVote(
                    agent=self.name,
                    vote=VoteType.REJECT,
                    reason=f"Estimated cost ${cost_f:.6g} exceeds budget ${budget_f:.6g} by >20%.",
                    hard_constraints_triggered=["BUDGET_EXCEEDED"],
                    modifications_required=[],
                )
            elif cost_f > budget_f:
                # Modify: minor overrun (<= 20%)
                return AgentVote(
                    agent=self.name,
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
                return AgentVote(
                    agent=self.name,
                    vote=VoteType.APPROVE,
                    reason=f"Estimated cost ${cost_f:.6g} is within budget ${budget_f:.6g}.",
                    hard_constraints_triggered=[],
                    modifications_required=[],
                )

