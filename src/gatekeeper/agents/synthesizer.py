import json
import re
from typing import Any, Dict, List, Set, Tuple

from gatekeeper.agents.base import BaseAgent
from gatekeeper.schemas import AgentName, AgentVote, VoteType, FinalVerdict, VerdictStatus
from gatekeeper.llm import call_gemini

class SynthesizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.SYNTHESIZER)

    def vote(self, state: Dict[str, Any]) -> AgentVote:
        raise NotImplementedError("Synthesizer does not vote in Phase 1.")

    def synthesize(self, state: Dict[str, Any]) -> FinalVerdict:
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
            return FinalVerdict(status=VerdictStatus.APPROVED, confidence=1.0, modifications_required=[])

        # 2. LLM Synthesis
        synth_prompt = self.load_system_prompt()
        
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
            return verdict

        except Exception as e:
            # Fallback to rule-based logic if LLM fails
            return self._fallback_verdict(votes, str(e))

    def _fallback_verdict(self, votes: List[Dict[str, Any]], error_msg: str) -> FinalVerdict:
        merged_mods: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
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
        
        return FinalVerdict(status=VerdictStatus.REJECTED, confidence=conf, modifications_required=merged_mods)

