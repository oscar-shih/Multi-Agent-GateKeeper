from __future__ import annotations

import operator
from typing import Any, Dict, List
from typing_extensions import Annotated, TypedDict

class GatekeeperState(TypedDict, total=False):
    start_epoch_s: float
    deadline_epoch_s: float

    input_paths: Dict[str, str]
    raw: Dict[str, str]
    parsed: Dict[str, Dict[str, Any]]
    derived_metrics: Dict[str, Any]

    phase1_votes: Annotated[List[Dict[str, Any]], operator.add]
    debate_messages: Annotated[List[Dict[str, Any]], operator.add]
    final_verdict: Dict[str, Any]

    trace: Dict[str, Any]