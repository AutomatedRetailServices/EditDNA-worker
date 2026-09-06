"""Final draft-level retry authority for structures that only become visible after Best Take.

Hybrid, deterministic grouping, and Best Take deliberately fail open when evidence is
locally ambiguous. Two real long-form structures can therefore survive until the final
draft even though the *combined* evidence is decisive:

1) selected attempt -> discarded failed restart bridge -> selected clean continuation;
   if the failed bridge repeats the opening of the first attempt, the later take covers
   the same message, and the source shows a dense physical reset sequence, the earlier
   attempt is superseded;
2) selected failed/open prefix -> selected continuation, while an earlier alternate in
   the continuation's already-proven retry group covers the combined later attempt. The
   failed prefix and its continuation are one losing delivery, so the earlier peer is
   promoted rather than leaving a Frankenstein close.

This pass never creates words or timestamps, never invents retry groups, and fails open
unless the already-existing draft structure plus semantic/temporal evidence is strong.
"""
from __future__ import annotations

from dataclasses import replace
import re
import unicodedata
from typing import Iterable

from .contracts import DraftClip

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_OPEN_TAIL = frozenset({
    "a", "al", "and", "as", "at", "because", "but", "by", "con", "como", "cuando",
    "de", "del", "el", "en", "for", "from", "if", "in", "into", "la", "las", "los",
    "o", "of", "on", "or", "para", "pero", "por", "porque", "que", "si", "sin", "so",
    "than", "that", "the", "to", "un", "una", "unos", "unas", "when", "which", "while",
    "who", "with", "without", "y",
})
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "cuando",
    "de", "del", "el", "en", "es", "esta", "este", "for", "from", "fue", "in", "is",
    "it", "la", "las", "lo", "los", "me", "mi", "mis", "of", "on", "or", "para",
    "pero", "por", "porque", "que", "se", "si", "so", "su", "sus", "that", "the",
    "this", "to", "un", "una", "was", "we", "with", "y", "yo",
})
_NEGATION = frozenset({
    "no", "not", "never", "nunca", "sin", "without", "nadie", "ningun", "ningún",
    "ninguna", "ninguno", "nobody", "none", "neither",
})
_RESET_KINDS = frozenset({
    "hand_motion_reset_candidate",
    "body_reset_candidate",
    "camera_disengagement_candidate",
    "facial_expression_shift_candidate",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _concept(token: str) -> str:
    value = "".join(
        char for char in unicodedata.normalize("NFKD", str(token or "").casefold())
        if not unicodedata.combining(char)
    )
    if len(value) >= 7 and value.endswith("es"):
        value = value[:-2]
    elif len(value) >= 6 and value.endswith("s"):
        value = value[:-1]
    return value


def _content(text: str) -> set[str]:
    out = set()
    for token in _tokens(text):
        if len(token) < 3 or token in _STOP:
            continue
        concept = _concept(token)
        if concept:
            out.add(concept)
    return out


def _coverage(left_text: str, right_text: str) -> tuple[int, float, float]:
    left = _content(left_text)
    right = _content(right_text)
    if not left or not right:
        return 0, 0.0, 0.0
    shared = len(left & right)
    return shared, shared / max(1, len(left)), shared / max(1, len(right))


def _critical(text: str) -> set[str]:
    raw = str(text or "")
    out = {f"num:{value}" for value in _NUMBER_RE.findall(raw)}
    if any(token in _NEGATION for token in _tokens(raw)):
        out.add("__negation__")
    return out


def _is_open_text(text: str) -> bool:
    tokens = _tokens(text)
    if not tokens:
        return False
    if tokens[-1] in _OPEN_TAIL:
        return True
    return str(text or "").strip().endswith((",", ":", ";", "-", "–", "—"))


def _semantic_failures(diagnostics: dict) -> dict[str, float]:
    """Keep the strongest failed/BTS evidence across overlapping Hybrid windows."""
    failures: dict[str, float] = {}
    for chunk in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(chunk, dict):
            continue
        for item in chunk.get("decisions") or ():
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip().lower()
            if label not in {"failed", "bts"}:
                continue
            cid = str(item.get("clip_id") or "")
            if not cid:
                continue
            failures[cid] = max(failures.get(cid, 0.0), float(item.get("confidence") or 0.0))
    return failures


def _reset_count_between(left: DraftClip, right: DraftClip, diagnostics: dict) -> tuple[int, float]:
    start = float(left.end) - 0.40
    end = float(right.start) + 0.30
    count = 0
    best = 0.0
    whole = diagnostics.get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if not isinstance(source, dict) or source.get("source_asset_id") != left.source_asset_id:
            continue
        for event in source.get("events") or ():
            if not isinstance(event, dict):
                continue
            kind = str(event.get("kind") or "").strip().lower().replace("-", "_").replace(" ", "_")
            confidence = float(event.get("confidence") or 0.0)
            if kind not in _RESET_KINDS or confidence < 0.80:
                continue
            if float(event.get("end") or 0.0) < start or float(event.get("start") or 0.0) > end:
                continue
            count += 1
            best = max(best, confidence)
        break
    return count, best


def suppress_selected_attempt_before_failed_bridge(
    selected: Iterable[DraftClip],
    discarded: Iterable[DraftClip],
    diagnostics: dict,
    *,
    maximum_gap_sec: float = 20.0,
) -> tuple[tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[dict, ...]]:
    """Collapse selected -> failed bridge -> selected when the later delivery supersedes."""
    selected_list = list(selected)
    discarded_list = list(discarded)
    failures = _semantic_failures(diagnostics)
    removed_ids: set[str] = set()
    audit: list[dict] = []

    for index, earlier in enumerate(selected_list):
        if earlier.clip_id in removed_ids:
            continue
        for later in selected_list[index + 1 :]:
            if later.clip_id in removed_ids or later.source_asset_id != earlier.source_asset_id:
                continue
            gap = float(later.start) - float(earlier.end)
            if gap < 0:
                continue
            if gap > maximum_gap_sec:
                break

            bridges = [
                item for item in discarded_list
                if item.source_asset_id == earlier.source_asset_id
                and float(item.start) >= float(earlier.end)
                and float(item.end) <= float(later.start)
                and failures.get(item.clip_id, 0.0) >= 0.80
            ]
            if not bridges:
                continue

            shared_pair, _, later_cov = _coverage(earlier.text, later.text)
            if shared_pair < 3 or later_cov < 0.55:
                continue

            best_bridge = None
            for bridge in bridges:
                shared_bridge, bridge_cov, _ = _coverage(bridge.text, earlier.text)
                if shared_bridge < 3 or bridge_cov < 0.60:
                    continue
                candidate = (bridge_cov, shared_bridge, failures.get(bridge.clip_id, 0.0), bridge)
                if best_bridge is None or candidate[:3] > best_bridge[:3]:
                    best_bridge = candidate
            if best_bridge is None:
                continue

            bridge_cov, bridge_shared, bridge_failure, bridge = best_bridge
            reset_count, reset_conf = _reset_count_between(earlier, later, diagnostics)
            # Dense physical reset is independent evidence that these are recording
            # attempts, not two intentional neighboring sentences. A very strong failed
            # bridge can substitute only when it almost entirely repeats the first take.
            if reset_count < 2 and not (bridge_failure >= 0.90 and bridge_cov >= 0.80):
                continue

            removed_ids.add(earlier.clip_id)
            audit.append({
                "reason": "selected_attempt_yields_across_failed_restart_bridge",
                "removed_clip_id": earlier.clip_id,
                "failed_bridge_clip_id": bridge.clip_id,
                "winner_clip_id": later.clip_id,
                "bridge_failure_confidence": round(bridge_failure, 4),
                "bridge_coverage": round(bridge_cov, 4),
                "bridge_shared_content_tokens": bridge_shared,
                "later_coverage": round(later_cov, 4),
                "reset_event_count": reset_count,
                "best_reset_confidence": round(reset_conf, 4),
                "removed_text": earlier.text,
                "bridge_text": bridge.text,
                "winner_text": later.text,
            })
            break

    if not removed_ids:
        return tuple(selected_list), tuple(discarded_list), ()
    removed = [clip for clip in selected_list if clip.clip_id in removed_ids]
    survivors = tuple(clip for clip in selected_list if clip.clip_id not in removed_ids)
    existing_discarded = {clip.clip_id for clip in discarded_list}
    discarded_out = tuple(discarded_list + [replace(clip, selected=False) for clip in removed if clip.clip_id not in existing_discarded])
    return survivors, discarded_out, tuple(audit)


def promote_group_peer_over_failed_selected_chain(
    selected: Iterable[DraftClip],
    alternates: Iterable[DraftClip],
    discarded: Iterable[DraftClip],
    diagnostics: dict,
    *,
    maximum_continuation_gap_sec: float = 3.5,
    maximum_prior_gap_sec: float = 45.0,
) -> tuple[tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[dict, ...]]:
    """Promote an earlier same-group peer when a selected failed prefix owns the winner suffix."""
    selected_list = list(selected)
    alternate_list = list(alternates)
    discarded_list = list(discarded)
    failures = _semantic_failures(diagnostics)
    audit: list[dict] = []
    replacements: dict[int, DraftClip] = {}
    remove_indexes: set[int] = set()
    promoted_ids: set[str] = set()
    moved_to_discard: list[DraftClip] = []

    for index in range(len(selected_list) - 1):
        if index in remove_indexes or index + 1 in remove_indexes:
            continue
        prefix = selected_list[index]
        continuation = selected_list[index + 1]
        if prefix.source_asset_id != continuation.source_asset_id:
            continue
        gap = float(continuation.start) - float(prefix.end)
        if gap < 0 or gap > maximum_continuation_gap_sec:
            continue
        failure_conf = failures.get(prefix.clip_id, 0.0)
        if failure_conf < 0.80 or not _is_open_text(prefix.text):
            continue
        group_id = continuation.take_group_id
        if not group_id:
            continue

        options = []
        combined_text = f"{prefix.text} {continuation.text}".strip()
        combined_critical = _critical(combined_text)
        for peer in alternate_list:
            if peer.clip_id in promoted_ids or peer.take_group_id != group_id:
                continue
            if peer.source_asset_id != prefix.source_asset_id or float(peer.end) > float(prefix.start):
                continue
            prior_gap = float(prefix.start) - float(peer.end)
            if prior_gap > maximum_prior_gap_sec:
                continue
            shared, combined_cov, peer_cov = _coverage(combined_text, peer.text)
            if shared < 7 or combined_cov < 0.35 or peer_cov < 0.35:
                continue
            if not combined_critical.issubset(_critical(peer.text)):
                continue
            options.append((combined_cov, peer_cov, shared, -prior_gap, peer))
        if not options:
            continue

        combined_cov, peer_cov, shared, _, peer = max(options, key=lambda item: item[:4])
        promoted = replace(peer, selected=True)
        replacements[index] = promoted
        remove_indexes.add(index + 1)
        promoted_ids.add(peer.clip_id)
        moved_to_discard.extend((prefix, continuation))
        audit.append({
            "reason": "promote_prior_group_peer_over_failed_selected_retry_chain",
            "failed_prefix_clip_id": prefix.clip_id,
            "continuation_clip_id": continuation.clip_id,
            "promoted_clip_id": peer.clip_id,
            "take_group_id": group_id,
            "failed_prefix_confidence": round(failure_conf, 4),
            "combined_shared_content_tokens": shared,
            "combined_coverage": round(combined_cov, 4),
            "promoted_coverage": round(peer_cov, 4),
            "failed_prefix_text": prefix.text,
            "continuation_text": continuation.text,
            "promoted_text": peer.text,
        })

    if not audit:
        return tuple(selected_list), tuple(alternate_list), tuple(discarded_list), ()

    selected_out = []
    for index, clip in enumerate(selected_list):
        if index in replacements:
            selected_out.append(replacements[index])
            continue
        if index in remove_indexes:
            continue
        # The original failed prefix is replaced at its own index above.
        if any(item.get("failed_prefix_clip_id") == clip.clip_id for item in audit):
            continue
        selected_out.append(clip)

    alternates_out = tuple(clip for clip in alternate_list if clip.clip_id not in promoted_ids)
    existing_discarded = {clip.clip_id for clip in discarded_list}
    discarded_out = tuple(discarded_list + [
        replace(clip, selected=False)
        for clip in moved_to_discard
        if clip.clip_id not in existing_discarded
    ])
    return tuple(selected_out), alternates_out, discarded_out, tuple(audit)


def install_final_draft_retry_integrity() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_final_draft_retry_integrity", False):
        return

    def build_with_final_draft_retry_integrity(*args, **kwargs):
        result = original(*args, **kwargs)
        draft = result.draft
        diagnostics = dict(draft.diagnostics or {})

        selected, alternates, discarded, chain_diag = promote_group_peer_over_failed_selected_chain(
            draft.selected,
            draft.alternates,
            draft.discarded,
            diagnostics,
        )
        selected, discarded, bridge_diag = suppress_selected_attempt_before_failed_bridge(
            selected,
            discarded,
            diagnostics,
        )
        if not chain_diag and not bridge_diag:
            return result

        diagnostics["final_draft_retry_integrity"] = [*list(chain_diag), *list(bridge_diag)]
        repaired_draft = replace(
            draft,
            selected=selected,
            alternates=alternates,
            discarded=discarded,
            diagnostics=diagnostics,
        )
        return replace(result, draft=repaired_draft)

    build_with_final_draft_retry_integrity._cutsell_final_draft_retry_integrity = True
    pipeline.build_flow_b_draft = build_with_final_draft_retry_integrity
