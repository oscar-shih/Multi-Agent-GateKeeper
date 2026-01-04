# src/gatekeeper/graph/state.py
from __future__ import annotations

import operator
from typing import Any, Dict, List
from typing_extensions import Annotated, TypedDict


class GatekeeperState(TypedDict, total=False):
    # Timebox control
    start_epoch_s: float
    deadline_epoch_s: float

    # Input paths (CLI supplies)
    input_paths: Dict[str, str]  # keys: mesh_report, sim_config, past_runs, formulas

    # Raw texts (loaded from disk)
    raw: Dict[str, str]  # same keys as input_paths

    # Parsed JSON dicts (normalized)
    parsed: Dict[str, Dict[str, Any]]  # mesh_report/sim_config/past_runs/formulas

    # Derived metrics (pure code; deterministic)
    derived_metrics: Dict[str, Any]

    # Phase 1 outputs
    phase1_votes: Annotated[List[Dict[str, Any]], operator.add]

    # Phase 2 outputs
    debate_messages: Annotated[List[Dict[str, Any]], operator.add]

    # Phase 3 output (final JSON)
    final_verdict: Dict[str, Any]

    # Optional trace for Loom demo
    trace: Dict[str, Any]