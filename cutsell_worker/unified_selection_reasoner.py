"""Unified whole-video Selection authority for CutSell Universal Clean Cut.

The legacy pipeline may produce useful local evidence, retry groups, Hybrid votes,
and provisional Selected/SWAP/Discarded buckets.  None of those buckets are final
when a UnifiedSelectionReasoner is active.  The reasoner sees the complete candidate
universe for the source at once and returns one editorial plan before Selection freeze.

This module is provider-neutral.  It owns validation and safe application only; it
contains no HTTP, vendor SDK, benchmark timestamp, phrase, or Human Gold rule.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
import unicodedata
from typing import Protocol

from .contracts import DraftClip, DraftTimeline

_ALLOWED_ACTIONS = frozenset({"select", "swap", "discard"})
_ALLOWED_RELATIONS = frozenset({
    "independent",
    "retry_winner",
    "retry_alternate",
    "composite_piece",
    "continuation",
    "failed",
    "bts",
    "uncertain",
})
_ALLOWED_REASONS = frozenset({
    "best_complete_take",
    "independent_story_coverage",
    "composite_best_take_piece",
    "necessary_continuation",
    "usable_alternate",
    "redundant_retry",
    "failed_delivery",
    "recording_process_bts",
    "uncertain_preserve",
})

_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_CONTENT_STOPWORDS = frozenset("""
a al an and are as at con de del el ella en es esta este esto for from he her here i in is it
la las lo los me mi mira more más ni no of on or para pero por que se she si sin su sus te than
that the their this to tu tú un una was we with y ya yo you your
""".split())


def _content_tokens(text: str) -> set[str]:
    raw = unicodedata.normalize("NFKD", str(text or "").casefold())
    plain = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return {token for token in _CONTENT_TOKEN_RE.findall(plain)
            if len(token) >= 3 and token not in _CONTENT_STOPWORDS}


@dataclass(frozen=True)
class UnifiedSelectionDecision:
    clip_id: str
    action: str
    relation: str
    confidence: float
    family_index: int
    reason_code: str
    # V2-only global story placement. Legacy callers omit it and retain
    # natural source order exactly as before.
    sequence_index: int | None = None
    # V2-only optional semantic proposal; independently verified pre-Freeze.
    trailing_recording_word_count: int = 0
    trailing_recording_confidence: float | None = None
    trailing_recording_kind: str = "recording_aside"
    trailing_replacement_clip_id: str | None = None


@dataclass(frozen=True)
class UnifiedTakeCompetition:
    """Model-proposed comparison of complete deliveries, before Freeze.

    The winning side may be a composite. Covered candidates are removable
    only after all winners are selected and the model confirms high-confidence
    equivalent delivery with no material information or action lost.
    """
    winner_clip_ids: tuple[str, ...]
    covered_clip_ids: tuple[str, ...]
    material_unique_clip_ids: tuple[str, ...] = ()
    relation: str = "independent"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class UnifiedSelectionPlan:
    decisions: tuple[UnifiedSelectionDecision, ...]
    provider: str
    model: str
    requested: bool = True
    available: bool = True
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    take_competitions: tuple[UnifiedTakeCompetition, ...] = ()


class UnifiedSelectionReasoner(Protocol):
    def reason(self, draft: DraftTimeline) -> UnifiedSelectionPlan: ...


def _all_clips(draft: DraftTimeline) -> tuple[DraftClip, ...]:
    """Return every unique semantic candidate in natural source order."""
    by_id: dict[str, DraftClip] = {}
    for clip in (*draft.selected, *draft.alternates, *draft.discarded):
        by_id.setdefault(str(clip.clip_id), clip)
    return tuple(sorted(
        by_id.values(),
        key=lambda clip: (clip.source_order, float(clip.start), float(clip.end), clip.clip_id),
    ))


def _bucket_map(draft: DraftTimeline) -> dict[str, str]:
    out = {clip.clip_id: "discard" for clip in draft.discarded}
    out.update({clip.clip_id: "swap" for clip in draft.alternates})
    out.update({clip.clip_id: "select" for clip in draft.selected})
    return out


def validate_unified_selection_plan(
    draft: DraftTimeline,
    plan: UnifiedSelectionPlan,
) -> UnifiedSelectionPlan:
    expected = {clip.clip_id for clip in _all_clips(draft)}
    seen: set[str] = set()
    normalized: list[UnifiedSelectionDecision] = []

    if not plan.available:
        raise ValueError("unified selection reasoner unavailable")
    if plan.estimated_input_tokens < 0 or plan.estimated_output_tokens < 0:
        raise ValueError("unified selection token estimates must be non-negative")

    for raw in plan.decisions:
        clip_id = str(raw.clip_id)
        if clip_id not in expected:
            raise ValueError("unified selection returned unknown clip id")
        if clip_id in seen:
            raise ValueError("unified selection returned duplicate clip id")
        action = str(raw.action).strip().lower()
        relation = str(raw.relation).strip().lower()
        reason_code = str(raw.reason_code).strip().lower()
        confidence = float(raw.confidence)
        family_index = int(raw.family_index)
        if action not in _ALLOWED_ACTIONS:
            raise ValueError("unified selection returned invalid action")
        if relation not in _ALLOWED_RELATIONS:
            raise ValueError("unified selection returned invalid relation")
        if reason_code not in _ALLOWED_REASONS:
            raise ValueError("unified selection returned invalid reason code")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("unified selection confidence outside 0..1")
        if family_index < 0:
            raise ValueError("unified selection family index must be non-negative")
        if type(raw.trailing_recording_word_count) is not int or not 0 <= raw.trailing_recording_word_count <= 8:
            raise ValueError("invalid trailing recording word count")
        if raw.trailing_recording_confidence is not None and (
            type(raw.trailing_recording_confidence) not in (int, float)
            or not 0 <= raw.trailing_recording_confidence <= 1
        ):
            raise ValueError("invalid trailing recording confidence")
        if raw.trailing_recording_kind not in {"recording_aside", "abandoned_restart"}:
            raise ValueError("invalid trailing recording kind")
        if raw.trailing_replacement_clip_id is not None and raw.trailing_replacement_clip_id not in expected:
            raise ValueError("unknown trailing replacement clip")
        normalized.append(UnifiedSelectionDecision(
            clip_id=clip_id,
            action=action,
            relation=relation,
            confidence=confidence,
            family_index=family_index,
            reason_code=reason_code,
            sequence_index=(
                None if raw.sequence_index is None else int(raw.sequence_index)
            ),
            trailing_recording_word_count=raw.trailing_recording_word_count,
            trailing_recording_confidence=raw.trailing_recording_confidence,
            trailing_recording_kind=raw.trailing_recording_kind,
            trailing_replacement_clip_id=raw.trailing_replacement_clip_id,
        ))
        seen.add(clip_id)

    if seen != expected:
        raise ValueError("unified selection reasoner omitted candidates")

    checked_competitions = []
    for competition in plan.take_competitions:
        winners = tuple(competition.winner_clip_ids)
        covered = tuple(competition.covered_clip_ids)
        unique = tuple(competition.material_unique_clip_ids)
        if (not winners or not covered or
                any(not group or len(group) != len(set(group)) or not set(group) <= expected
                    for group in (winners, covered)) or
                len(unique) != len(set(unique)) or not set(unique) <= expected or
                set(winners) & set(covered) or set(unique) & set(covered)):
            raise ValueError("invalid whole-take competition membership")
        if competition.relation not in {"equivalent_take", "complementary", "independent"}:
            raise ValueError("invalid whole-take competition relation")
        if not math.isfinite(float(competition.confidence)) or not 0 <= competition.confidence <= 1:
            raise ValueError("invalid whole-take competition confidence")
        checked_competitions.append(competition)

    return UnifiedSelectionPlan(
        decisions=tuple(normalized),
        provider=str(plan.provider or "unknown")[:80],
        model=str(plan.model or "unknown")[:120],
        requested=bool(plan.requested),
        available=True,
        estimated_input_tokens=int(plan.estimated_input_tokens),
        estimated_output_tokens=int(plan.estimated_output_tokens),
        take_competitions=tuple(checked_competitions),
    )


def _apply_v2_take_competitions(
    clips: tuple[DraftClip, ...],
    decisions: dict[str, UnifiedSelectionDecision],
    actions: list[str],
    overrides: list[str | None],
    competitions: tuple[UnifiedTakeCompetition, ...],
) -> list[dict]:
    """Resolve explicit cross-family equivalence, including lexical rescues.

    Do not deduce equivalence from topical words or chronology. A disputed
    candidate survives if the comparison is missing, low confidence, names a
    material exception, or does not have its complete winner selected.
    """
    index_by_id = {clip.clip_id: i for i, clip in enumerate(clips)}
    # Preserve any material exception named against the same winning delivery,
    # regardless of the order in which comparisons arrived.
    protected = {(tuple(sorted(c.winner_clip_ids)), clip_id)
                 for c in competitions for clip_id in c.material_unique_clip_ids}
    proposed_covered = {clip_id for contest in competitions if contest.relation == "equivalent_take"
                        for clip_id in contest.covered_clip_ids}
    disputed = set()
    for left in competitions:
        for right in competitions:
            if left is right:
                continue
            if (set(left.winner_clip_ids) & set(right.covered_clip_ids) or
                    set(right.winner_clip_ids) & set(left.covered_clip_ids)):
                disputed.update((id(left), id(right)))
    audit = []
    for contest in competitions:
        reason = "advisory_only"
        if id(contest) in disputed:
            reason = "contradictory_competitions_preserved"
        elif contest.relation != "equivalent_take":
            reason = "different_or_complementary"
        elif contest.confidence < .90:
            reason = "low_confidence"
        elif not all(actions[index_by_id[clip_id]] == "select" and
                     decisions[clip_id].action == "select" and
                     not decisions[clip_id].trailing_recording_word_count
                     for clip_id in contest.winner_clip_ids):
            reason = "winner_not_confirmed_selected"
        else:
            source_ids = {clips[index_by_id[clip_id]].source_asset_id for clip_id in
                          (*contest.winner_clip_ids, *contest.covered_clip_ids)}
            if len(source_ids) != 1:
                reason = "cross_source_conflict"
            else:
                for clip_id in contest.covered_clip_ids:
                    i = index_by_id[clip_id]
                    winner_has_cta = any(_has_purchase_action(clips[index_by_id[w]].text)
                                         for w in contest.winner_clip_ids)
                    selected_cta_elsewhere = any(
                        j != i and clips[j].clip_id not in proposed_covered and
                        actions[j] == "select" and decisions[clips[j].clip_id].action == "select" and
                        not decisions[clips[j].clip_id].trailing_recording_word_count and
                        clips[j].source_asset_id == clips[i].source_asset_id and
                        _has_purchase_action(clips[j].text) and
                        _purchase_destination(clips[j].text) == _purchase_destination(clips[i].text)
                        for j in range(len(clips)))
                    if (_has_purchase_action(clips[i].text) and not winner_has_cta and
                            not selected_cta_elsewhere):
                        if (decisions[clip_id].reason_code not in {"failed_delivery", "recording_process_bts"}
                                and decisions[clip_id].relation not in {"failed", "bts"}):
                            actions[i] = "select"
                            overrides[i] = "purchase_action_not_covered_by_winner"
                        reason = "purchase_action_coverage_conflict"
                        continue
                    if (tuple(sorted(contest.winner_clip_ids)), clip_id) in protected:
                        reason = "conflicting_material_unique_preserved"
                        continue
                    if actions[i] == "select":
                        actions[i] = "discard"
                        overrides[i] = "whole_take_equivalent_covered"
                if reason not in {"conflicting_material_unique_preserved", "purchase_action_coverage_conflict"}:
                    reason = "covered_alternates_removed"
        audit.append({
            "winners": list(contest.winner_clip_ids),
            "covered": list(contest.covered_clip_ids),
            "material_unique": list(contest.material_unique_clip_ids),
            "relation": contest.relation,
            "confidence": contest.confidence,
            "decision": reason,
            "reason": contest.reason[:240],
        })
    return audit


def _has_purchase_action(value: str) -> bool:
    """Recognize an explicit buying direction, not a product/store mention."""
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("’", "'").replace("don't", "dont").replace("didn't", "didnt")
    for clause in re.split(r"[.!?;]+", normalized):
        action = re.search(
            r"\b(?:puedes?|pueden|podras?)\s+(?:encontrar|comprar|conseguir)(?:lo|la)?\b"
            r"|\b(?:compra|compralo|pidelo|encuentralo|adquierelo)\b"
            r"|\b(?:buy now|order now|shop now|tap|click|find it|get yours)\b", clause)
        destination = re.search(r"\b(?:carrito|enlace|link|bio|tienda|cart|store|shop|checkout)\b", clause)
        if not action or not destination:
            continue
        prefix = clause[:action.start()]
        if re.search(r"\b(?:no|nunca|jamas|not|never|dont|didnt)\b(?:\W+\w+){0,3}\W*$", prefix):
            continue
        return True
    return False


def _purchase_destination(value: str) -> str | None:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    destinations = {"carrito": "cart", "cart": "cart", "enlace": "link", "link": "link",
                    "bio": "bio", "tienda": "store", "store": "store", "shop": "store",
                    "checkout": "checkout"}
    for token in re.findall(r"\b\w+\b", normalized):
        if token in destinations:
            return destinations[token]
    return None


def _effective_action(decision: UnifiedSelectionDecision, current_bucket: str) -> tuple[str, str | None]:
    """Fail open on uncertainty without turning uncertainty into destructive deletion."""
    # A reason_code is the model's own explanation for a decision, and some
    # reason codes are self-describing about which tier they belong to per
    # the editorial contract itself: "usable_alternate" is SWAP-tier by
    # definition ("a usable alternative...that should not play by default"),
    # and "failed_delivery" is DISCARD-tier by definition ("failed/abandoned
    # delivery"). A `select` action paired with either directly contradicts
    # the model's own stated reason -- checked first, ahead of confidence, so
    # a high-confidence self-contradiction is still caught. General, not
    # Video00-specific: it depends only on the fixed reason_code vocabulary.
    if decision.action == "select" and decision.reason_code == "failed_delivery":
        return "discard", "failed_delivery_reason_overrides_select_action"
    if decision.action == "select" and decision.reason_code == "usable_alternate":
        return "swap", "usable_alternate_reason_overrides_select_action"
    if decision.relation == "uncertain" or decision.confidence < 0.70:
        if current_bucket == "select":
            return "select", "uncertain_preserved_current_selected"
        return "swap", "uncertain_preserved_as_swap"
    if decision.action == "discard" and decision.confidence < 0.80:
        return "swap", "low_confidence_discard_demoted_to_swap"
    return decision.action, None


def _enforce_single_retry_family_winner(
    clips: tuple[DraftClip, ...],
    decisions: dict[str, UnifiedSelectionDecision],
    actions: list[str],
    overrides: list[str | None],
) -> None:
    """Within one retry family, a genuine retry contest -- relation
    retry_winner or retry_alternate, i.e. candidates the model itself framed
    as competing takes of the same moment -- must produce at most one SELECT.
    More than one surviving SELECT there is always a policy error, never a
    legitimate composite: composites are relation composite_piece/
    continuation and are untouched by this pass, as is every independent
    story beat. Mutates `actions`/`overrides` in place; keeps the
    highest-confidence contender, demotes the rest to SWAP (never DISCARD --
    an alternate that was good enough to reach SELECT stays available for
    manual replacement, it is not thrown away)."""
    by_family: dict[int, list[int]] = {}
    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        if decision.relation in ("retry_winner", "retry_alternate"):
            by_family.setdefault(decision.family_index, []).append(index)

    for indices in by_family.values():
        select_indices = [i for i in indices if actions[i] == "select"]
        if len(select_indices) <= 1:
            continue
        winner = max(select_indices, key=lambda i: decisions[clips[i].clip_id].confidence)
        for i in select_indices:
            if i != winner:
                actions[i] = "swap"
                overrides[i] = "retry_family_single_winner_enforced"


def _preserve_retry_alternates_with_unique_information(
    clips: tuple[DraftClip, ...],
    decisions: dict[str, UnifiedSelectionDecision],
    actions: list[str],
    overrides: list[str | None],
) -> None:
    """Prevent a false retry family from deleting materially distinct beats.

    Gemini may call an earlier hook or continuation a usable retry alternate
    even when the chosen later take does not contain much of its information.
    V2 has no manual SWAP bucket after resolution, so preserve a clean usable
    alternate when at least three meaningful tokens and 40% of its content
    vocabulary are absent from the complete selected story. Failed delivery
    and BTS never qualify.
    """
    selected_tokens: set[str] = set()
    for index, clip in enumerate(clips):
        if actions[index] == "select":
            selected_tokens.update(_content_tokens(clip.text))
    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        if actions[index] not in {"swap", "discard"}:
            continue
        if decision.relation != "retry_alternate" or decision.reason_code not in {
            "usable_alternate", "redundant_retry",
        }:
            continue
        tokens = _content_tokens(clip.text)
        unique = tokens - selected_tokens
        # Three surface words alone can describe an alternate aesthetic of
        # the same product without a distinct claim. A model-confirmed
        # redundant retry needs stronger independent information before a
        # lexical safety rescue can override it.
        minimum_unique = 4 if decision.reason_code == 'redundant_retry' else 3
        if len(unique) < minimum_unique or len(unique) / max(1, len(tokens)) < 0.40:
            continue
        actions[index] = "select"
        overrides[index] = "unique_retry_information_preserved"
        selected_tokens.update(tokens)


def _high_confidence_audience_spans(draft: DraftTimeline) -> dict[str, tuple[tuple[float, float], ...]]:
    spans: dict[str, tuple[tuple[float, float], ...]] = {}
    whole = (draft.diagnostics or {}).get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if not isinstance(source, dict):
            continue
        try:
            evidence = json.loads(str(source.get("audiovisual_evidence") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        rows = []
        for region in evidence.get("regions") or ():
            try:
                start, end = float(region["start"]), float(region["end"])
                confidence = float(region.get("confidence", 0))
                if (region.get("role") == "audience" and .90 <= confidence <= 1
                        and math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    rows.append((start, end))
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        if rows:
            merged = []
            for start, end in sorted(rows):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            spans[str(source.get("source_asset_id") or "")] = tuple(merged)
    return spans


def _preserve_unique_content_when_av_contradicts_failed(
    draft: DraftTimeline,
    clips: tuple[DraftClip, ...],
    decisions: dict[str, UnifiedSelectionDecision],
    actions: list[str],
    overrides: list[str | None],
) -> None:
    """Do not let a semantic failure label erase AV-verified clean content."""
    audience = _high_confidence_audience_spans(draft)
    selected_tokens = set().union(*(
        _content_tokens(clip.text) for i, clip in enumerate(clips) if actions[i] == "select"
    )) if any(action == "select" for action in actions) else set()
    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        if actions[index] != "discard" or decision.reason_code != "failed_delivery":
            continue
        # ASR candidates can combine a clean audience delivery with a short
        # fumble/reset tail. Preserve unique content when high-confidence AV
        # audience evidence covers a clear majority of the candidate; clips
        # that are primarily failed/BTS still cannot pass this gate.
        duration = max(.001, float(clip.end) - float(clip.start))
        audience_overlap = sum(
            max(0.0, min(float(clip.end), end) - max(float(clip.start), start))
            for start, end in audience.get(clip.source_asset_id, ())
        )
        if audience_overlap / duration < .65:
            continue
        tokens = _content_tokens(clip.text)
        unique = tokens - selected_tokens
        # A long delivery may share most of its vocabulary with the story
        # while retaining an uncovered opening/fact. Positive AV conflicts
        # with the failure label: preserve the existing three-token evidence
        # floor without making preservation depend on total clip length.
        if len(unique) < 3:
            continue
        actions[index] = "select"
        overrides[index] = "av_audience_unique_content_overrides_failed_label"
        selected_tokens.update(tokens)


def _preserve_continuous_demonstration(draft, clips, decisions, actions, overrides):
    """Do not equate a repeated instruction with a repeated visual action.

    A low-confidence redundancy proposal cannot delete an instruction inside
    one AV-observed demonstration leading directly to its selected explanation.
    This is inclusion-only, before Freeze; it never deletes a competing take.
    """
    whole = (draft.diagnostics or {}).get("whole_video_context") or {}
    demos = {}
    for source in whole.get("sources") or ():
        try:
            evidence = json.loads(source.get("audiovisual_evidence") or "{}")
        except (TypeError, ValueError):
            continue
        for region in evidence.get("regions") or ():
            try:
                start, end = float(region["start"]), float(region["end"])
                confidence = float(region.get("confidence", 0))
            except (KeyError, TypeError, ValueError):
                continue
            description = " ".join(str(region.get(k) or "") for k in (
                "visual_observation", "reason",
            )).casefold()
            if (region.get("role") == "audience" and .90 <= confidence <= 1
                    and math.isfinite(start) and math.isfinite(end) and 0 <= start < end
                    and any(term in description for term in ("demonstrat", "mixing", "pouring", "applying"))
                    and not any(term in description for term in ("retry", "restart", "fumble", "abandon"))):
                demos.setdefault(source.get("source_asset_id"), []).append((start, end))
    for i, left in enumerate(clips[:-1]):
        d = decisions[left.clip_id]
        right = clips[i + 1]
        if (actions[i] not in {"swap", "discard"} or d.reason_code != "redundant_retry"
                or d.relation != "retry_alternate" or d.confidence >= .90
                or actions[i + 1] != "select" or left.source_asset_id != right.source_asset_id
                or left.source_order != right.source_order or not 0 <= right.start - left.end <= 10
                or not _content_tokens(left.text).intersection(_content_tokens(right.text))):
            continue
        if not any(start <= left.start and end > right.start
                   for start, end in demos.get(left.source_asset_id, ())):
            continue
        if _content_tokens(left.text) <= _content_tokens(right.text):
            continue
        actions[i] = "select"
        overrides[i] = "av_continuous_demonstration_preserved"


def apply_unified_selection_reasoner(
    draft: DraftTimeline,
    reasoner: UnifiedSelectionReasoner | None,
) -> DraftTimeline:
    """Apply one whole-video semantic plan. Provider errors leave the draft untouched."""
    if reasoner is None:
        return draft

    diagnostics = dict(draft.diagnostics or {})
    try:
        plan = validate_unified_selection_plan(draft, reasoner.reason(draft))
    except Exception as exc:
        diagnostics["unified_selection_reasoner"] = {
            "status": "provider_error_fail_open",
            "error": f"{exc.__class__.__name__}: {str(exc)[:240]}",
        }
        return replace(draft, diagnostics=diagnostics)

    clips = _all_clips(draft)
    current = _bucket_map(draft)
    decisions = {decision.clip_id: decision for decision in plan.decisions}
    v2_request = bool((draft.diagnostics or {}).get("editorial_engine_v2_request"))
    if v2_request:
        sequence = [decision.sequence_index for decision in plan.decisions]
        if any(index is None or int(index) < 0 for index in sequence):
            raise ValueError("Editorial Engine V2 requires a non-negative sequence_index for every candidate")
        if len({int(index) for index in sequence}) != len(sequence):
            raise ValueError("Editorial Engine V2 requires unique sequence_index values")

    actions: list[str] = []
    overrides: list[str | None] = []
    for clip in clips:
        decision = decisions[clip.clip_id]
        action, safety_override = _effective_action(decision, current.get(clip.clip_id, "swap"))
        actions.append(action)
        overrides.append(safety_override)

    _enforce_single_retry_family_winner(clips, decisions, actions, overrides)
    if v2_request:
        _preserve_retry_alternates_with_unique_information(clips, decisions, actions, overrides)
        _preserve_unique_content_when_av_contradicts_failed(
            draft, clips, decisions, actions, overrides,
        )
        _preserve_continuous_demonstration(draft, clips, decisions, actions, overrides)
        competition_audit = _apply_v2_take_competitions(
            clips, decisions, actions, overrides, plan.take_competitions,
        )

    selected: list[DraftClip] = []
    alternates: list[DraftClip] = []
    discarded: list[DraftClip] = []
    audit: list[dict] = []
    tail_audit: list[dict] = []

    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        action = actions[index]
        normalized_clip = replace(clip, selected=(action == "select"))
        if v2_request and action == "select" and decision.trailing_recording_word_count:
            from .v2_recording_tail import trim_recording_tail
            coverage_clips = []
            for other_index, other in enumerate(clips):
                other_decision = decisions[other.clip_id]
                if actions[other_index] != "select" or other_decision.action != "select":
                    continue
                count = other_decision.trailing_recording_word_count
                coverage_clips.append(replace(other, text=" ".join(w.text for w in other.words[:-count]))
                                      if count else other)
            normalized_clip, tail_row = trim_recording_tail(
                normalized_clip, decision, diagnostics, selected_clips=coverage_clips)
            tail_audit.append(tail_row)
        if action == "select":
            selected.append(normalized_clip)
        elif action == "swap":
            alternates.append(normalized_clip)
        else:
            discarded.append(normalized_clip)
        audit.append({
            "clip_id": clip.clip_id,
            "previous_bucket": current.get(clip.clip_id),
            "model_action": decision.action,
            "effective_action": action,
            "relation": decision.relation,
            "confidence": round(decision.confidence, 4),
            "family_index": decision.family_index,
            "reason_code": decision.reason_code,
            "safety_override": overrides[index],
            "sequence_index": decision.sequence_index,
        })

    if v2_request:
        selected.sort(key=lambda clip: int(decisions[clip.clip_id].sequence_index))

    diagnostics["unified_selection_reasoner"] = {
        "status": "applied",
        "provider": plan.provider,
        "model": plan.model,
        "candidate_count": len(clips),
        "selected_count": len(selected),
        "swap_count": len(alternates),
        "discarded_count": len(discarded),
        "estimated_input_tokens": plan.estimated_input_tokens,
        "estimated_output_tokens": plan.estimated_output_tokens,
        "decisions": audit,
    }
    if tail_audit:
        diagnostics["v2_recording_tail"] = tail_audit
    if v2_request:
        diagnostics["v2_take_competitions"] = competition_audit
    return replace(
        draft,
        selected=tuple(selected),
        alternates=tuple(alternates),
        discarded=tuple(discarded),
        diagnostics=diagnostics,
    )
