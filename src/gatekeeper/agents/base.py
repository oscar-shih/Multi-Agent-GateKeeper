from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from gatekeeper.schemas import AgentName, AgentVote


class BaseAgent(ABC):
    def __init__(self, name: AgentName):
        self.name = name

    def load_system_prompt(self) -> str:
        """Loads the system prompt from the prompts directory."""
        root = Path(__file__).parent.parent / "prompts"
        path = root / f"{self.name.value.lower()}.system.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"You are {self.name.value}. You are a strict gatekeeper."

    @abstractmethod
    def vote(self, state: Dict[str, Any]) -> AgentVote:
        """Phase 1: Analyze inputs and cast a vote."""
        pass

