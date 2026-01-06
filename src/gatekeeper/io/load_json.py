from __future__ import annotations

import json
import re
from typing import Any, Dict

from pydantic import BaseModel

from gatekeeper.schemas import AgentVote, DebateMessage, FinalVerdict

def _strip_line_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"//.*$", "", line))
    return "\n".join(lines)

def load_json_with_comments(text: str) -> Dict[str, Any]:
    cleaned = _strip_line_comments(text)
    return json.loads(cleaned)

def canonical_json(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def parse_agent_vote(obj: Any) -> AgentVote:
    return AgentVote.model_validate(obj)

def parse_debate_message(obj: Any) -> DebateMessage:
    return DebateMessage.model_validate(obj)

def parse_final_verdict(obj: Any) -> FinalVerdict:
    return FinalVerdict.model_validate(obj)
