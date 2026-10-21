"""Zero-cost telemetry for CutSell Hybrid Editorial Brain.

Pure bookkeeping only: no SDKs, HTTP, secrets, or provider calls. Workers can record
why the semantic gate fired, whether a provider was actually requested, token estimates,
and whether the final Best Take changed. This gives us the data needed to tune cost vs.
quality before enabling paid inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class HybridDecisionEvent:
    session_id: str
    candidate_count: int
    local_confidence: float
    conflict_score: float
    gate_requested: bool
    provider_requested: bool
    provider_available: bool
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    local_winner_clip_id: str | None = None
    final_winner_clip_id: str | None = None
    provider: str = "none"
    model: str = "none"

    @property
    def winner_changed(self) -> bool:
        return bool(
            self.local_winner_clip_id
            and self.final_winner_clip_id
            and self.local_winner_clip_id != self.final_winner_clip_id
        )


@dataclass
class HybridTelemetry:
    events: list[HybridDecisionEvent] = field(default_factory=list)

    def record(self, event: HybridDecisionEvent) -> None:
        self.events.append(event)

    def snapshot(self) -> dict[str, int | float]:
        total = len(self.events)
        gated = sum(1 for event in self.events if event.gate_requested)
        requested = sum(1 for event in self.events if event.provider_requested)
        available = sum(1 for event in self.events if event.provider_available)
        changed = sum(1 for event in self.events if event.winner_changed)
        input_tokens = sum(max(0, int(event.estimated_input_tokens)) for event in self.events)
        output_tokens = sum(max(0, int(event.estimated_output_tokens)) for event in self.events)
        return {
            "sessions": total,
            "gate_requested_sessions": gated,
            "provider_requested_sessions": requested,
            "provider_available_sessions": available,
            "winner_changed_sessions": changed,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "gate_rate": round(gated / total, 4) if total else 0.0,
            "provider_request_rate": round(requested / total, 4) if total else 0.0,
            "winner_change_rate": round(changed / total, 4) if total else 0.0,
        }


def summarize_hybrid_events(events: Iterable[HybridDecisionEvent]) -> dict[str, int | float]:
    telemetry = HybridTelemetry(list(events))
    return telemetry.snapshot()
