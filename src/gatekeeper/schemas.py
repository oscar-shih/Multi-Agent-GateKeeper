from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, conlist, confloat, constr, model_validator, JsonValue

class AgentName(str, Enum):
    GEOMETER = "GEOMETER"
    PHYSICIST = "PHYSICIST"
    RESOURCE_MANAGER = "RESOURCE_MANAGER"
    HISTORIAN = "HISTORIAN"
    STABILIZER = "STABILIZER"
    SYNTHESIZER = "SYNTHESIZER"

class VoteType(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MODIFY = "MODIFY"

class Priority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class DebateStance(str, Enum):
    DEFEND = "DEFEND"
    ATTACK = "ATTACK"

class VerdictStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

JsonPath = constr(min_length=1, max_length=200)  # jsonpath-like or dotted path
ShortReason = constr(min_length=1, max_length=800)
ShortRationale = constr(min_length=1, max_length=200)
DebateText = constr(min_length=1, max_length=600)
ConstraintCode = constr(min_length=1, max_length=64)

class ModificationRequired(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: JsonPath = Field(..., description="JSONPath-like field locator, e.g., $.solver.dt")
    proposed_value: JsonValue = Field(..., description="Proposed value to apply to the config.")
    rationale: ShortRationale = Field(..., description="Short justification for the modification.")
    priority: Priority = Field(..., description="Priority of the modification.")

class VerdictModificationRequired(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: JsonPath
    proposed_value: JsonValue
    rationale: constr(min_length=1, max_length=400) = Field(
        ..., description="Justification. Synthesizer may consolidate and expand rationale slightly."
    )
    priority: Priority
    owner: AgentName

class AgentVote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: AgentName
    vote: VoteType
    reason: ShortReason

    hard_constraints_triggered: List[ConstraintCode] = Field(
        default_factory=list,
        description="Machine-readable constraint codes (hard failures).",
    )

    modifications_required: List[ModificationRequired] = Field(
        default_factory=list,
        description="Requested changes. Should be empty for APPROVE.",
    )

    @model_validator(mode="after")
    def _enforce_vote_consistency(self) -> "AgentVote":
        if self.vote == VoteType.APPROVE:
            if self.hard_constraints_triggered:
                raise ValueError("APPROVE vote cannot include hard_constraints_triggered.")
            if self.modifications_required:
                raise ValueError("APPROVE vote cannot include modifications_required.")

        if self.vote == VoteType.MODIFY and len(self.modifications_required) == 0:
            raise ValueError("MODIFY vote must include at least one modifications_required item.")

        return self

class DebateMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: AgentName
    stance: DebateStance
    message: DebateText
    targets: conlist(AgentName, min_length=1, max_length=3) = Field(
        ..., description="Agents this message is responding to."
    )

class FinalVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VerdictStatus
    confidence: confloat(ge=0.0, le=1.0) = Field(
        ..., description="Deterministic confidence in [0,1]."
    )
    modifications_required: List[VerdictModificationRequired] = Field(
        default_factory=list,
        description="If REJECTED, must explain required changes; if APPROVED, should be empty.",
    )

    @model_validator(mode="after")
    def _enforce_verdict_consistency(self) -> "FinalVerdict":
        if self.status == VerdictStatus.REJECTED and len(self.modifications_required) == 0:
            raise ValueError("REJECTED verdict must include at least one modifications_required item.")
        return self
