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
        max_runtime = sim.get("max_runtime_hours", None)

        # Prefer HIGH bound for conservative gating
        est_cost_high = derived.get("estimated_cost_usd_high", None)
        est_cost = derived.get("estimated_cost_usd", None)
        
        est_runtime_high = derived.get("estimated_runtime_hours_high", None)
        est_runtime = derived.get("estimated_runtime_hours", None)

        # pick conservative values
        cost_used = est_cost_high if est_cost_high is not None else est_cost
        runtime_used = est_runtime_high if est_runtime_high is not None else est_runtime

        # 1. Check Max Runtime (Hard Constraint)
        if max_runtime is not None and runtime_used is not None:
            max_runtime_f = float(max_runtime)
            runtime_f = float(runtime_used)
            if runtime_f > max_runtime_f:
                 return AgentVote(
                    agent=self.name,
                    vote=VoteType.REJECT,
                    reason=f"Estimated runtime {runtime_f:.2f}h exceeds limit {max_runtime_f}h.",
                    hard_constraints_triggered=["DEADLINE_EXCEEDED"],
                    modifications_required=[
                        {
                            "field": "$.mesh_report.topology.cell_count",
                            "proposed_value": "decrease",
                            "rationale": "Reduce cell count to meet runtime deadline.",
                            "priority": "HIGH",
                        }
                    ],
                )

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
            
            # Helper to suggest cell count reduction
            current_cells = float(derived.get("cell_count", 0) or 0)
            target_cells = int(current_cells * (budget_f / cost_f)) if cost_f > 0 else 0
            
            # Logic: Tolerance window
            # 1. Cost > Budget * 1.2 => Hard Reject (>20% over)
            # 2. Budget < Cost <= Budget * 1.2 => Modify (0-20% over)
            # 3. Cost <= Budget => Approve

            if cost_f > budget_f * 1.2:
                # IMPORTANT: For demo purposes, we treat budget overrun as REJECT but NOT a hard constraint.
                # This ensures debate happens.
                # "hard_constraints_triggered" must be empty to allow debate.
                
                return AgentVote(
                    agent=self.name,
                    vote=VoteType.REJECT,
                    reason=f"Estimated cost ${cost_f:.6g} exceeds budget ${budget_f:.6g} by >20%. Negotiation required.",
                    hard_constraints_triggered=[], # Empty to trigger debate!
                    modifications_required=[
                        {
                            "field": "$.mesh_report.topology.cell_count",
                            "proposed_value": target_cells,
                            "rationale": f"Reduce cell count to ~{target_cells} to fit within budget ${budget_f}.",
                            "priority": "HIGH",
                        }
                    ],
                )
            elif cost_f > budget_f:
                # Modify: minor overrun (<= 20%)
                return AgentVote(
                    agent=self.name,
                    vote=VoteType.MODIFY,
                    reason=f"Estimated cost ${cost_f:.6g} slightly exceeds budget ${budget_f:.6g} (<=20%). Reduce mesh size or increase budget.",
                    hard_constraints_triggered=[],
                    modifications_required=[
                        {
                            "field": "$.mesh_report.topology.cell_count",
                            "proposed_value": target_cells,
                            "rationale": f"Reduce cell count to ~{target_cells} to align with budget.",
                            "priority": "MEDIUM",
                        },
                        {
                            "field": "$.budget_usd",
                            "proposed_value": round(cost_f * 1.05, 2),
                            "rationale": "Or increase budget to cover conservative estimate.",
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
