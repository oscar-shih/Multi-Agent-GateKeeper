from __future__ import annotations

from dataclasses import dataclass
import os
from typing import FrozenSet, Optional, Dict, Any

# Allowed turbulence model enums for Physicist hallucination control.
ALLOWED_TURBULENCE_MODELS: FrozenSet[str] = frozenset(
    {
        "k-epsilon",
        "k-omega",
        "Spalart-Allmaras",
    }
)

@dataclass(frozen=True)
class Settings:
    # Global timebox for the full chain.
    timebox_seconds: float = 60.0

    # Gemini configuration.
    # Set GEMINI_API_KEY in your environment.
    gemini_api_key: str = ""
    # Model string is intentionally user-overridable because Google model IDs evolve.
    gemini_model: str = "gemini-2.5-flash"

    # Historian similarity tool settings.
    similarity_top_k: int = 3
    similarity_reject_threshold: float = 0.95
    similarity_modify_threshold: float = 0.85

    # Resource Manager suggestions when constraints are missing.
    default_budget_suggestion_usd: float = 50.0
    default_runtime_suggestion_hours: float = 24.0


def load_settings() -> Settings:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    # Allow overriding the timebox, but keep a safe default.
    timebox_s = os.getenv("GATEKEEPER_TIMEBOX_SECONDS", "60").strip()
    try:
        timebox = float(timebox_s)
    except ValueError:
        timebox = 60.0

    # Similarity knobs
    top_k_s = os.getenv("GATEKEEPER_SIM_TOP_K", "3").strip()
    try:
        top_k = int(top_k_s)
    except ValueError:
        top_k = 3

    reject_th_s = os.getenv("GATEKEEPER_SIM_REJECT_TH", "0.95").strip()
    modify_th_s = os.getenv("GATEKEEPER_SIM_MODIFY_TH", "0.85").strip()
    try:
        reject_th = float(reject_th_s)
    except ValueError:
        reject_th = 0.95
    try:
        modify_th = float(modify_th_s)
    except ValueError:
        modify_th = 0.85

    # Suggestions
    bud_s = os.getenv("GATEKEEPER_DEFAULT_BUDGET_USD", "50").strip()
    run_s = os.getenv("GATEKEEPER_DEFAULT_MAX_RUNTIME_HOURS", "24").strip()
    try:
        bud = float(bud_s)
    except ValueError:
        bud = 50.0
    try:
        run = float(run_s)
    except ValueError:
        run = 24.0

    return Settings(
        timebox_seconds=timebox,
        gemini_api_key=api_key,
        gemini_model=model,
        similarity_top_k=top_k,
        similarity_reject_threshold=reject_th,
        similarity_modify_threshold=modify_th,
        default_budget_suggestion_usd=bud,
        default_runtime_suggestion_hours=run,
    )


def merge_resource_overrides(
    sim_config: Dict[str, Any],
    *,
    budget_usd_override: Optional[float] = None,
    max_runtime_hours_override: Optional[float] = None,
) -> Dict[str, Any]:

    out = dict(sim_config)

    if out.get("budget_usd") is None and budget_usd_override is not None:
        out["budget_usd"] = float(budget_usd_override)

    if out.get("max_runtime_hours") is None and max_runtime_hours_override is not None:
        out["max_runtime_hours"] = float(max_runtime_hours_override)

    return out