"""Conservative, contiguous speech-edge edits grounded in complete word timing."""

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class SpeechEdgeProposal:
    keep_start: int
    keep_end: int  # exclusive word index
    confidence: float
    reason: str


def word_tokens(clip):
    return tuple(str(w.get("word", "")).strip() for w in clip.get("words", ()))


def _normalized(text):
    return re.findall(r"\w+(?:['’]\w+)*", text.casefold())


def apply_speech_edge(clip, proposal):
    """Apply only a validated edge proposal; return an observable outcome.

    Never join disjoint word ranges or infer timing for missing words. A small
    separation between adjacent words is required at each edited boundary.
    The original clip and its metadata remain intact on every rejection.
    """
    if proposal is None:
        return "not_requested"
    if (type(proposal.keep_start) is not int or type(proposal.keep_end) is not int
            or not math.isfinite(proposal.confidence) or proposal.confidence < .9
            or proposal.confidence > 1
            or proposal.reason not in {"production_talk", "false_start", "verbal_fumble"}):
        return "invalid_or_uncertain_proposal"
    words = clip.get("words") or []
    left, right = proposal.keep_start, proposal.keep_end
    if not (0 <= left < right <= len(words)) or (left == 0 and right == len(words)):
        return "invalid_range"
    if _normalized(" ".join(word_tokens(clip))) != _normalized(clip.get("text", "")):
        return "transcript_timing_mismatch"
    try:
        start, end = float(clip["start"]), float(clip["end"])
        times = [(float(w["start"]), float(w["end"])) for w in words]
        if (not all(math.isfinite(t) for t in (start, end)) or end <= start
                or any(not math.isfinite(a) or not math.isfinite(b) or b <= a
                       or a < start or b > end for a, b in times)
                or any(b > c for (_, b), (c, _) in zip(times, times[1:]))
                or any(not token for token in word_tokens(clip))):
            return "invalid_word_timing"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "invalid_word_timing"
    if ((left > 0 and times[left][0] - times[left - 1][1] < .08)
            or (right < len(words) and times[right][0] - times[right - 1][1] < .08)):
        return "no_safe_boundary_gap"
    kept = words[left:right]
    # Keep original exterior handles; pad only the newly cut boundary in its gap.
    new_start = start if left == 0 else max(start, times[left][0] - .04)
    new_end = end if right == len(words) else min(end, times[right - 1][1] + .04)
    audit = {"version": 1, "original_start": start, "original_end": end,
             "original_text": clip["text"], "keep_start": left, "keep_end": right,
             "confidence": proposal.confidence, "reason": proposal.reason,
             "evidence": "transcript_and_word_timing", "status": "applied"}
    clip.update(start=new_start, end=new_end, words=kept,
                text=" ".join(str(w["word"]).strip() for w in kept))
    clip["meta"]["speech_edge_edit"] = audit
    return "applied"
