from typing import Any, Dict
from gatekeeper.agents.base import BaseAgent
from gatekeeper.schemas import AgentName, AgentVote, VoteType

class HistorianAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.HISTORIAN)

    def vote(self, state: Dict[str, Any]) -> AgentVote:
        derived = state.get("derived_metrics") or {}
        similar = derived.get("similar_runs") or []

        if len(similar) == 0:
            return AgentVote(
                agent=self.name,
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
                return AgentVote(
                    agent=self.name,
                    vote=VoteType.REJECT,
                    reason=f"High similarity to past crash {top_crash.get('job_id')} (>=0.95) with no stabilization knob changes.",
                    hard_constraints_triggered=["SIMILAR_TO_PAST_CRASH"],
                    modifications_required=[],
                )
            elif top_crash and float(top_crash.get("similarity_score", 0.0)) >= 0.85:
                # Require stabilization changes
                return AgentVote(
                    agent=self.name,
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
                return AgentVote(
                    agent=self.name,
                    vote=VoteType.APPROVE,
                    reason="No sufficiently similar past crash pattern detected in top-k retrieval.",
                    hard_constraints_triggered=[],
                    modifications_required=[],
                )

