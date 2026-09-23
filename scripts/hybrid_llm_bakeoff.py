from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HARD_CAP_USD = 0.50
MAX_OUTPUT_TOKENS = 600

PRICES = {
    "groq:gpt-oss-20b": (0.075, 0.30),
    "gemini:3.5-flash-lite": (0.30, 2.50),
    "gemini:3.6-flash": (1.50, 7.50),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip_id": {"type": "string"},
                    "label": {"type": "string", "enum": ["winner", "alternate", "failed", "bts", "uncertain", "keep"]},
                    "confidence": {"type": "number"},
                    "reason_code": {"type": "string"},
                },
                "required": ["clip_id", "label", "confidence", "reason_code"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}

CASES = [
    {
        "id": "b10_crop_retry",
        "candidates": [
            ("a", "The popular crop black denim jeans are back in stock anything with pockets is a win for me", "winner"),
            ("b", "the popular croc croc", "failed"),
            ("c", "Crop popular crop popular crop popular", "failed"),
            ("d", "the popular croc", "failed"),
            ("e", "The popular croc", "failed"),
        ],
    },
    {
        "id": "b10_lip_serum_retry",
        "candidates": [
            ("a", "Dr. Malasin sent me this limp", "failed"),
            ("b", "Dr. Malasin sent me this lip plumping serum to try out", "winner"),
            ("c", "Dr. Malasin sent me this limp pumper", "failed"),
            ("d", "Dr. Malasin sent me this", "failed"),
            ("e", "Dr. Malasin", "failed"),
        ],
    },
    {
        "id": "b14_election_word_search",
        "candidates": [
            ("a", "As a content creator, if you do not have this election,", "failed"),
            ("b", "election, electric suction phone holder,", "failed"),
            ("c", "election, suction phone holder, election. Oh my God.", "failed"),
        ],
    },
    {
        "id": "b14_launch_retry",
        "candidates": [
            ("a", "just released their blowout bundle with their thermal blowout brush.", "winner"),
            ("b", "Launch just launched, just released their thermal launch,", "failed"),
            ("c", "Launch just released their launch,", "failed"),
        ],
    },
    {
        "id": "b14_gift_hack_retry",
        "candidates": [
            ("a", "I'm literally all about finding the best hacks when it comes to gift giving this year.", "winner"),
            ("b", "I'm literally all,", "failed"),
            ("c", "I am all about finding the best hacks.", "alternate"),
            ("d", "I am literally all about finding the best hacks when it comes to gift", "failed"),
        ],
    },
    {
        "id": "b11_recording_process_meta",
        "candidates": [
            ("a", "miss out on this i'm on the cozy cardigan train why do i keep saying that it's stupid", "bts"),
            ("b", "it's stupid yeah yeah it's stupid oh i don't know how to end tiktok shot videos like i hate", "bts"),
            ("c", "saying the length below i hate saying don't miss out on this deal i hate being like", "bts"),
            ("d", "want it in every color done cool everything else says you need to call the action call the action", "bts"),
            ("e", "salesy some people don't end them they just stop saying it they're like i love it i", "bts"),
        ],
    },
    {
        "id": "b11_break_character",
        "candidates": [
            ("a", "What the fuck is happening?", "bts"),
            ("b", "Okay, what the frig okay?", "bts"),
            ("c", "What just happened?", "bts"),
            ("d", "Okay, anyway", "bts"),
            ("e", "side. What are you doing with your hands fuck", "bts"),
        ],
    },
    {
        "id": "b14_self_review",
        "candidates": [
            ("a", "What did I just say? And then they have black, what?", "bts"),
        ],
    },
]


def estimated_tokens(text: str) -> int:
    return max(1, (len(text) + 2) // 3)


def prompt_for(case: dict) -> str:
    lines = [
        "You are CutSell's editorial judge. Classify recording attempts from ONE already-bounded creator mini-session.",
        "A winner is the strongest complete intended delivery. Alternate is usable but not the best. Failed is a stumble, false start, word-search, incomplete or broken delivery. BTS is creator self-talk, recording-process commentary, frustration, self-review or breaking character. Keep is valid speech when winner/alternate ranking is not applicable. Use uncertain only when evidence is insufficient.",
        "Do not invent clip IDs. Classify every supplied clip exactly once. Do not infer relationships outside this session.",
        f"session_id={case['id']}",
    ]
    for clip_id, text, _expected in case["candidates"]:
        lines.append(f"clip_id={clip_id} | text={text}")
    return "\n".join(lines)


@dataclass
class Spend:
    reserved: float = 0.0
    actual: float = 0.0

    def reserve(self, amount: float) -> None:
        if self.reserved + amount > HARD_CAP_USD + 1e-12:
            raise RuntimeError(f"hard budget cap would be exceeded: ${self.reserved + amount:.6f} > ${HARD_CAP_USD:.2f}")
        self.reserved += amount


def max_call_cost(provider_key: str, prompt: str) -> float:
    inp, out = PRICES[provider_key]
    return estimated_tokens(prompt) / 1_000_000 * inp + MAX_OUTPUT_TOKENS / 1_000_000 * out


def http_json(url: str, headers: dict[str, str], payload: dict) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": "CutSell-Hybrid-Bakeoff/1.0",
            "Accept": "application/json",
            **headers,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode("utf-8")), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def call_groq(prompt: str, spend: Spend) -> tuple[dict, dict]:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    provider_key = "groq:gpt-oss-20b"
    reserve = max_call_cost(provider_key, prompt)
    spend.reserve(reserve)
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "low",
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": {"name": "cutsell_editorial", "strict": True, "schema": SCHEMA}},
    }
    raw, latency = http_json("https://api.groq.com/openai/v1/chat/completions", {"Authorization": f"Bearer {key}"}, payload)
    text = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens") or estimated_tokens(prompt))
    out_tok = int(usage.get("completion_tokens") or estimated_tokens(text))
    inp, out = PRICES[provider_key]
    actual = in_tok / 1_000_000 * inp + out_tok / 1_000_000 * out
    spend.actual += actual
    return json.loads(text), {"latency_sec": latency, "input_tokens": in_tok, "output_tokens": out_tok, "actual_cost_usd": actual, "reserved_cost_usd": reserve}


def call_gemini(prompt: str, spend: Spend, model: str) -> tuple[dict, dict]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing")
    short = "3.5-flash-lite" if model == "gemini-3.5-flash-lite" else "3.6-flash"
    provider_key = f"gemini:{short}"
    reserve = max_call_cost(provider_key, prompt)
    spend.reserve(reserve)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "thinkingConfig": {"thinkingLevel": "minimal"},
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
        },
    }
    raw, latency = http_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key},
        payload,
    )
    parts = raw["candidates"][0]["content"].get("parts") or []
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    if not text.strip():
        finish = raw.get("candidates", [{}])[0].get("finishReason")
        raise ValueError(f"gemini returned empty structured text; finishReason={finish}")
    usage = raw.get("usageMetadata") or {}
    in_tok = int(usage.get("promptTokenCount") or estimated_tokens(prompt))
    out_tok = int(usage.get("candidatesTokenCount") or estimated_tokens(text))
    inp, out = PRICES[provider_key]
    actual = in_tok / 1_000_000 * inp + out_tok / 1_000_000 * out
    spend.actual += actual
    return json.loads(text), {"latency_sec": latency, "input_tokens": in_tok, "output_tokens": out_tok, "actual_cost_usd": actual, "reserved_cost_usd": reserve}


def normalize(result: dict, expected_ids: set[str]) -> dict[str, str]:
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("missing decisions")
    labels = {}
    for item in decisions:
        clip_id = str(item.get("clip_id") or "")
        label = str(item.get("label") or "").lower()
        if clip_id not in expected_ids or clip_id in labels:
            raise ValueError("invalid/duplicate clip id")
        if label not in {"winner", "alternate", "failed", "bts", "uncertain", "keep"}:
            raise ValueError("invalid label")
        labels[clip_id] = label
    if set(labels) != expected_ids:
        raise ValueError("omitted clip ids")
    return labels


def score_case(case: dict, result: dict) -> tuple[int, int, bool | None]:
    expected = {clip_id: label for clip_id, _text, label in case["candidates"]}
    predicted = normalize(result, set(expected))
    exact = sum(predicted[k] == v for k, v in expected.items())
    winners = [k for k, v in expected.items() if v == "winner"]
    winner_ok = None if not winners else all(predicted[k] == "winner" for k in winners)
    return exact, len(expected), winner_ok


def run_model(name: str, fn, cases: list[dict], spend: Spend) -> dict:
    rows = []
    correct = total = 0
    winner_correct = winner_total = 0
    for case in cases:
        prompt = prompt_for(case)
        try:
            result, meta = fn(prompt, spend)
            exact, count, winner_ok = score_case(case, result)
            status = "ok"
            correct += exact
            total += count
            if winner_ok is not None:
                winner_total += 1
                winner_correct += int(winner_ok)
        except Exception as exc:
            result, meta = {}, {}
            exact, count, winner_ok = 0, len(case["candidates"]), None
            status = f"error:{exc.__class__.__name__}:{str(exc)[:180]}"
            total += count
        rows.append({"case_id": case["id"], "status": status, "correct_labels": exact, "label_count": count, "winner_ok": winner_ok, "result": result, **meta})
    return {
        "model": name,
        "label_accuracy": (correct / total) if total else 0.0,
        "winner_accuracy": (winner_correct / winner_total) if winner_total else None,
        "correct_labels": correct,
        "label_count": total,
        "winner_correct": winner_correct,
        "winner_total": winner_total,
        "rows": rows,
    }


def label_signature(row: dict, case: dict) -> dict[str, str] | None:
    if row.get("status") != "ok":
        return None
    expected_ids = {clip_id for clip_id, _text, _expected in case["candidates"]}
    try:
        return normalize(row.get("result") or {}, expected_ids)
    except Exception:
        return None


def main() -> None:
    missing = [name for name in ("GROQ_API_KEY", "GEMINI_API_KEY") if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing required secrets: " + ", ".join(missing))

    spend = Spend()
    reports = []
    reports.append(run_model("groq/openai-gpt-oss-20b", call_groq, CASES, spend))
    reports.append(run_model("gemini-3.5-flash-lite", lambda p, s: call_gemini(p, s, "gemini-3.5-flash-lite"), CASES, spend))

    groq_rows = {r["case_id"]: r for r in reports[0]["rows"]}
    lite_rows = {r["case_id"]: r for r in reports[1]["rows"]}
    escalation_cases = []
    for case in CASES:
        gr = groq_rows[case["id"]]
        lr = lite_rows[case["id"]]
        if label_signature(gr, case) != label_signature(lr, case):
            escalation_cases.append(case)
    if escalation_cases:
        reports.append(run_model("gemini-3.6-flash", lambda p, s: call_gemini(p, s, "gemini-3.6-flash"), escalation_cases, spend))

    out = {
        "benchmark": "cutsell_hybrid_llm_bakeoff_v1",
        "hard_cap_usd": HARD_CAP_USD,
        "reserved_cost_usd": round(spend.reserved, 8),
        "actual_estimated_cost_usd": round(spend.actual, 8),
        "case_count": len(CASES),
        "escalation_case_count": len(escalation_cases),
        "models": reports,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/hybrid-llm-bakeoff.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "hard_cap_usd": out["hard_cap_usd"],
        "reserved_cost_usd": out["reserved_cost_usd"],
        "actual_estimated_cost_usd": out["actual_estimated_cost_usd"],
        "escalation_case_count": out["escalation_case_count"],
        "models": [{"model": r["model"], "label_accuracy": r["label_accuracy"], "winner_accuracy": r["winner_accuracy"]} for r in reports],
    }, indent=2))


if __name__ == "__main__":
    main()
