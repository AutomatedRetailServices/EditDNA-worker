"""Provider-neutral adapter boundary for CutSell Hybrid Editorial Brain.

This file intentionally has no HTTP/SDK implementation. A future vendor-specific
transport must be injected explicitly. The adapter performs preflight budgeting,
constructs the compact payload, and converts a strict dict response into the stable
EditorialJudge contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Any

from .hybrid_editorial import (
    EditorialDecision,
    EditorialJudgeResult,
    EditorialSession,
    HybridGatePolicy,
)
from .hybrid_payload import HybridCostPolicy, build_compact_editorial_payload, preflight_hybrid_call


Transport = Callable[[Mapping[str, Any], int], Mapping[str, Any]]


@dataclass(frozen=True)
class TransportEditorialJudge:
    provider_name: str
    model_name: str
    transport: Transport
    gate_policy: HybridGatePolicy = HybridGatePolicy()
    cost_policy: HybridCostPolicy = HybridCostPolicy()

    def judge(self, session: EditorialSession) -> EditorialJudgeResult:
        preflight = preflight_hybrid_call(
            session,
            self.gate_policy,
            cost_policy=self.cost_policy,
        )
        if not preflight["allowed"]:
            return EditorialJudgeResult(
                decisions=(),
                provider=self.provider_name,
                model=self.model_name,
                requested=False,
                available=False,
                estimated_input_tokens=int(preflight["estimated_input_tokens"]),
                estimated_output_tokens=0,
            )

        payload = build_compact_editorial_payload(session, cost_policy=self.cost_policy)
        raw = self.transport(payload, int(preflight["max_output_tokens"]))
        raw_decisions = raw.get("decisions")
        if not isinstance(raw_decisions, (list, tuple)):
            raise ValueError("hybrid provider response missing decisions array")

        decisions = []
        for item in raw_decisions:
            if not isinstance(item, Mapping):
                raise ValueError("hybrid provider decision must be an object")
            decisions.append(EditorialDecision(
                clip_id=str(item.get("clip_id") or ""),
                label=str(item.get("label") or ""),
                confidence=float(item.get("confidence", -1.0)),
                reason_code=str(item.get("reason_code") or ""),
            ))

        output_tokens = int(raw.get("output_tokens") or 0)
        if output_tokens < 0:
            raise ValueError("hybrid provider output token count invalid")
        return EditorialJudgeResult(
            decisions=tuple(decisions),
            provider=self.provider_name,
            model=self.model_name,
            requested=True,
            available=True,
            estimated_input_tokens=int(preflight["estimated_input_tokens"]),
            estimated_output_tokens=output_tokens,
        )


@dataclass
class BudgetLedger:
    """Simple in-process guard used by tests and future workers before paid transport."""

    max_calls: int
    max_estimated_input_tokens: int
    calls: int = 0
    estimated_input_tokens: int = 0

    def reserve(self, estimated_input_tokens: int) -> bool:
        tokens = max(0, int(estimated_input_tokens))
        if self.calls + 1 > self.max_calls:
            return False
        if self.estimated_input_tokens + tokens > self.max_estimated_input_tokens:
            return False
        self.calls += 1
        self.estimated_input_tokens += tokens
        return True
