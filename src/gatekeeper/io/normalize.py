from typing import Any, Dict

def normalize_sim_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = config.copy()
    
    normalized.setdefault("budget_usd", None)
    normalized.setdefault("max_runtime_hours", None)
    
    return normalized