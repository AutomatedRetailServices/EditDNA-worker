"""Small, serializable execution observability for the clean worker."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict


@dataclass
class ExecutionTrace:
    stages: Dict[str, object] = field(default_factory=dict)

    def complete(self, stage: str, **details: object) -> None:
        self.stages[stage] = {"status": "complete", **details}

    def degraded(self, stage: str, *, reason: str, **details: object) -> None:
        self.stages[stage] = {"status": "degraded", "reason": reason, **details}

    def fail(self, stage: str, *, reason: str) -> None:
        self.stages[stage] = {"status": "failed", "reason": reason}

    def as_dict(self) -> Dict[str, object]:
        return dict(self.stages)
