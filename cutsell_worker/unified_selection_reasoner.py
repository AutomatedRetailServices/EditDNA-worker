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
    # Source-verified dependencies between adjacent pieces of one utterance.
    continuation_links: tuple[tuple[str, str], ...] = ()
    continuation_evidence: tuple[dict, ...] = ()
    competition_review: dict | None = None
    candidate_intervals: tuple[dict, ...] = ()


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

    by_id = {clip.clip_id: clip for clip in _all_clips(draft)}
    checked_links = []
    for left_id, right_id in plan.continuation_links:
        if left_id not in by_id or right_id not in by_id or left_id == right_id:
            raise ValueError("invalid continuation dependency membership")
        left, right = by_id[left_id], by_id[right_id]
        if (left.source_asset_id != right.source_asset_id or
                not left.start < right.start or abs(left.end - right.start) > .2 or
                not left.words or not right.words):
            raise ValueError("continuation dependency contradicts source alignment")
        ordering = {decision.clip_id: decision.sequence_index for decision in normalized}
        if ordering[left_id] is None or ordering[right_id] is None or ordering[left_id] >= ordering[right_id]:
            raise ValueError("continuation dependency contradicts story order")
        checked_links.append((left_id, right_id))

    return UnifiedSelectionPlan(
        decisions=tuple(normalized),
        provider=str(plan.provider or "unknown")[:80],
        model=str(plan.model or "unknown")[:120],
        requested=bool(plan.requested),
        available=True,
        estimated_input_tokens=int(plan.estimated_input_tokens),
        estimated_output_tokens=int(plan.estimated_output_tokens),
        take_competitions=tuple(checked_competitions),
        continuation_links=tuple(checked_links),
        continuation_evidence=tuple(plan.continuation_evidence),
        competition_review=(dict(plan.competition_review)
                            if isinstance(plan.competition_review, dict) else None),
        candidate_intervals=tuple(dict(row) for row in plan.candidate_intervals),
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
                    covered_clip = clips[i]
                    # A recording-process reset separates attempts. A short
                    # pre-reset winner cannot prove the later, longer complete
                    # retake redundant merely by repeating its product/CTA.
                    # Retain the later take for editorial resolution when its
                    # purchase destination matches and its delivery is not
                    # itself marked failed. This does not assume every later
                    # take wins, nor erase the earlier one here.
                    earlier_winners = [clips[index_by_id[w]] for w in contest.winner_clip_ids
                                       if clips[index_by_id[w]].end <= covered_clip.start]
                    post_reset = bool(earlier_winners and
                        decisions[clip_id].reason_code == "redundant_retry" and
                        any(decisions[reset.clip_id].reason_code == "recording_process_bts" and
                            reset.source_asset_id == covered_clip.source_asset_id and
                            any(w.end <= reset.start < covered_clip.start for w in earlier_winners)
                            for reset in clips) and
                        covered_clip.end - covered_clip.start >=
                            max(w.end - w.start for w in earlier_winners) and
                        _has_purchase_action(covered_clip.text) and
                        any(_has_purchase_action(w.text) and
                            _purchase_destination(w.text) == _purchase_destination(covered_clip.text)
                            for w in earlier_winners))
                    if post_reset:
                        actions[i] = "select"
                        overrides[i] = "post_reset_complete_take_preserved"
                        reason = "post_reset_complete_take_preserved"
                        continue
                    winning_text = ' '.join(clips[index_by_id[w]].text for w in contest.winner_clip_ids)
                    covered_norm = unicodedata.normalize('NFKD', covered_clip.text.casefold())
                    covered_norm = ''.join(ch for ch in covered_norm if not unicodedata.combining(ch))
                    winning_norm = unicodedata.normalize('NFKD', winning_text.casefold())
                    winning_norm = ''.join(ch for ch in winning_norm if not unicodedata.combining(ch))
                    # Preserve an amount together with its adjacent unit or
                    # subject. Repeating the same numeral with another unit
                    # is not coverage of the original claim.
                    amounts = lambda value: set(re.findall(
                        r'\b(\d+(?:[.,]\d+)?)\s*([a-z]+)?', value))
                    critical_amounts = amounts(covered_norm)
                    winning_amounts = amounts(winning_norm)
                    condition = re.search(r'\b(?:si|if)\s+(?:(?:tu\s+)?(?:estas\s+)?|you\s+are\s+)'
                                          r'(?:usando|utilizando|using|taking)\s+([\w-]+)', covered_norm)
                    condition_topic_missing = bool(condition and
                        condition.group(1) not in _content_tokens(winning_text))
                    if (critical_amounts - winning_amounts or condition_topic_missing):
                        # A global equivalence label is insufficient proof of
                        # coverage if the winning take omits a stated amount
                        # or audience condition. Preserve the actual source
                        # claim for independent editorial resolution.
                        reason = 'material_claim_not_covered_by_winner'
                        continue
                    if overrides[i] == "focused_av_clean_second_start_preserved":
                        reason = "focused_av_clean_action_preserved"
                        continue
                    if (covered_clip.audio_muted and not covered_clip.words
                            and not covered_clip.text.strip()
                            and not any(
                                clips[index_by_id[w]].source_asset_id == covered_clip.source_asset_id
                                and clips[index_by_id[w]].start <= covered_clip.start
                                and clips[index_by_id[w]].end >= covered_clip.end
                                for w in contest.winner_clip_ids)):
                        # Spoken instructions recorded later cannot contain
                        # pixels from a distinct earlier visual operation.
                        # The single whole-video selector already chose this
                        # action; a verbal equivalence claim cannot erase it.
                        reason = 'distinct_focused_visual_action_preserved'
                        continue
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
                    cta_destination_already_selected = actions[i] != "select" and any(
                        j != i and actions[j] == "select" and
                        decisions[clips[j].clip_id].action == "select" and
                        not decisions[clips[j].clip_id].trailing_recording_word_count and
                        clips[j].source_asset_id == clips[i].source_asset_id and
                        _has_purchase_action(clips[j].text) and
                        _purchase_destination(clips[j].text) == _purchase_destination(clips[i].text)
                        for j in range(len(clips))
                    )
                    if (_has_purchase_action(clips[i].text) and not winner_has_cta and
                            not cta_destination_already_selected and
                            not selected_cta_elsewhere):
                        if (decisions[clip_id].reason_code not in {"failed_delivery", "recording_process_bts"}
                                and decisions[clip_id].relation not in {"failed", "bts"}):
                            actions[i] = "select"
                            overrides[i] = "purchase_action_not_covered_by_winner"
                        reason = "purchase_action_coverage_conflict"
                        continue
                    # Broad AV audience regions prove that someone addresses
                    # the camera, not that this is the successful take. If
                    # that evidence alone rescued a failed delivery and the
                    # provider independently confirmed a complete winning
                    # take, let the comparison remove the failed alternate.
                    # Material amounts, conditions and destinations above
                    # still require actual coverage by the winner.
                    if (decisions[clip_id].reason_code == "failed_delivery" or
                            decisions[clip_id].relation == "failed") and not (
                                actions[i] == "select" and
                                overrides[i] == "av_audience_unique_content_overrides_failed_label"
                            ):
                        reason = "failed_delivery_preserved_against_equivalence"
                        continue
                    if (tuple(sorted(contest.winner_clip_ids)), clip_id) in protected:
                        reason = "conflicting_material_unique_preserved"
                        continue
                    if actions[i] == "select":
                        actions[i] = "discard"
                        overrides[i] = "whole_take_equivalent_covered"
                if reason not in {"conflicting_material_unique_preserved", "purchase_action_coverage_conflict",
                                  "distinct_focused_visual_action_preserved", "material_claim_not_covered_by_winner",
                                  "failed_delivery_preserved_against_equivalence",
                                  "post_reset_complete_take_preserved",
                                  "focused_av_clean_action_preserved"}:
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
            r"\b(?:puedes?|pueden|podras?)\s+(?:encontrar|comprar|conseguir|adquirir|ordenar|pedir)(?:lo|la)?\b"
            r"|\b(?:compra|compralo|pidelo|ordenalo|encuentralo|adquierelo)\b"
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
) -> list[dict]:
    """Guard a concrete omitted claim, never promote a retry by vocabulary alone.

    Best-take editing can intentionally omit adjective and style variations.
    Keep a disputed alternate in the audit, but promote it to the played
    timeline only for an unrepresented quantity, audience condition, or CTA.
    A source-verified spoken continuation is protected by a separate authority.
    """
    def plain(text):
        value = unicodedata.normalize('NFKD', str(text or '').casefold())
        return ''.join(ch for ch in value if not unicodedata.combining(ch))

    spoken_numbers = {'uno': '1', 'una': '1', 'one': '1', 'dos': '2', 'two': '2',
                      'tres': '3', 'three': '3', 'cinco': '5', 'five': '5',
                      'diez': '10', 'ten': '10', 'quince': '15', 'fifteen': '15',
                      'veinte': '20', 'twenty': '20', 'treinta': '30', 'thirty': '30',
                      'sesenta': '60', 'sixty': '60', 'noventa': '90', 'ninety': '90'}
    def quantities(value):
        normalized = plain(value)
        # 'uno ve esos cuerpos' is an impersonal Spanish pronoun, not a
        # quantitative promise. Resolve one/uno/una as a number only when the
        # following token is an explicit measure/count unit. Other written
        # quantities (for example 'treinta dias') remain protected.
        one_units = (r'dias?|days?|semanas?|weeks?|meses?|months?|horas?|hours?|'
                     r'gramos?|grams?|miligramos?|milligrams?|mg|kg|kilos?|'
                     r'scoops?|cucharaditas?|cucharadas?|bottles?|botellas?|'
                     r'litros?|liters?|mililitros?|milliliters?|ml|'
                     r'capsulas?|capsules?|tabletas?|tablets?|pastillas?|pills?|'
                     r'dosis|dose|doses|porciones?|servings?|gotas?|drops?|'
                     r'veces?|times?|repeticiones?|reps?|'
                     r'por\s+ciento|percent|porcentaje|personas?|people')
        def replace_number(match):
            if match.group() in {'uno', 'una', 'one'} and not re.match(
                    r'\s+(?:' + one_units + r')\b', normalized[match.end():]):
                return match.group()
            return spoken_numbers[match.group()]
        numeric = re.sub(r'\b(?:' + '|'.join(spoken_numbers) + r')\b',
                         replace_number, normalized)
        return set(re.findall(r'\b(\d+(?:[.,]\d+)?)\s*([a-z]+)?', numeric))

    conflicts = []
    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        if actions[index] not in {"swap", "discard"}:
            continue
        # Some provider responses call a discarded fragment retry_winner while
        # simultaneously labeling it redundant_retry. The explicit discarded
        # action still needs the same material-claim protection; the relation
        # label cannot silently exempt an uncovered amount or condition.
        if decision.relation not in {"retry_alternate", "retry_winner"} or decision.reason_code not in {
            "usable_alternate", "redundant_retry",
        }:
            continue
        # A claim repeated by another source cannot cover this creator's take.
        selected_same_source = [other for j, other in enumerate(clips)
                                if j != index and actions[j] == 'select' and
                                other.source_asset_id == clip.source_asset_id]
        selected_text = ' '.join(other.text for other in selected_same_source)
        normalized = plain(clip.text)
        selected_norm = plain(selected_text)
        amounts = quantities(clip.text)
        selected_amounts = quantities(selected_text)
        conditional = re.search(r'\b(?:si|if)\s+(?:(?:tu\s+)?(?:estas\s+)?|you\s+(?:are\s+)?)'
                                r'(?:usando|utilizando|using|taking|use|usas?)\s+([\w-]+)', normalized)
        missing_condition = bool(conditional and
                                 conditional.group(1) not in _content_tokens(selected_text))
        unique_cta = (_has_purchase_action(clip.text) and
                      not any(actions[j] == 'select' and
                                  other.source_asset_id == clip.source_asset_id and
                                  _has_purchase_action(other.text) and
                                  _purchase_destination(other.text) == _purchase_destination(clip.text)
                                  for j, other in enumerate(clips)))
        negative_claims = re.findall(
            r'\b(?:sin|without|free of)\s+(?!ningun\b|any\b)(?:\w+\s+){0,2}'
            r'(?:azucar|sugar|gluten|lactosa|lactose|alergen\w*|allergen\w*|'
            r'calorias|calories|alcohol|cafeina|caffeine)\b|'
            r'\b(?:no\s+contiene|does\s+not\s+contain|doesn.t\s+contain)\s+'
            r'(?:\w+\s+){0,3}\w+', normalized)
        negative_claim = bool(negative_claims and
                              any(claim not in selected_norm for claim in negative_claims))
        protected_topic = bool(re.search(
            r'\b(?:allerg\w*|alerg\w*|diabet\w*|diagnos\w*|diagnost\w*|'
            r'ingredient\w*|ingrediente\w*)\b', normalized)
            and normalized not in selected_norm)
        protected = bool(amounts - selected_amounts or missing_condition or unique_cta or
                         negative_claim or protected_topic)
        unique = sorted(_content_tokens(clip.text) - _content_tokens(selected_text))
        if protected:
            actions[index] = "select"
            overrides[index] = "material_retry_claim_preserved"
        if protected or unique:
            conflicts.append({"clip_id": clip.clip_id, "source_asset_id": clip.source_asset_id,
                              "status": "material_claim_preserved" if protected else "model_discard_pending_review",
                              "missing_quantity": sorted(amounts - selected_amounts),
                              "missing_condition": missing_condition, "unique_cta": unique_cta,
                              "missing_negation": negative_claim, "protected_topic": protected_topic,
                              "lexical_difference": unique[:16]})
    return conflicts


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
        # Focused local delivery probes refine broad mixed/uncertain whole-source
        # observations. They remain advisory; only high-confidence audience spans
        # participate in the same coverage and unique-content gates below.
        regions = tuple(evidence.get("regions") or ()) + tuple(
            evidence.get("focused_delivery_regions") or ())
        for region in regions:
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
    """Protect an uncovered concrete claim when broad AV disputes failure.

    A broad audience label establishes address to camera, not successful
    delivery. Ordinary lexical novelty cannot override an explicit failed
    label; focused AV can still rescue word-aligned clean content separately.
    """
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
        def amounts(value):
            normalized = unicodedata.normalize('NFKD', str(value or '').casefold())
            normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
            for word, number in (('thirty', '30'), ('treinta', '30'),
                                 ('fifteen', '15'), ('quince', '15')):
                normalized = re.sub(r'\b' + word + r'\b', number, normalized)
            return set(re.findall(r'\b(\d+(?:[.,]\d+)?)\s*([a-z]+)?', normalized))
        selected_text = ' '.join(other.text for j, other in enumerate(clips)
                                 if actions[j] == 'select' and
                                 other.source_asset_id == clip.source_asset_id)
        normalized = unicodedata.normalize('NFKD', clip.text.casefold())
        normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
        condition = re.search(r'\b(?:si|if)\s+(?:(?:tu\s+)?(?:estas\s+)?|you\s+(?:are\s+)?)'
                              r'(?:usando|utilizando|using|taking|use|usas?)\s+([\w-]+)', normalized)
        missing_condition = bool(condition and condition.group(1) not in
                                 _content_tokens(selected_text))
        if len(unique) < 3 or not (amounts(clip.text) - amounts(selected_text)
                                   or missing_condition):
            continue
        actions[index] = "select"
        overrides[index] = "av_audience_unique_content_overrides_failed_label"
        selected_tokens.update(tokens)


def _preserve_short_audience_continuations(draft, clips, decisions, actions, overrides):
    """Keep a short, useful bridge inside the same verified clean delivery.

    A candidate-level redundancy label can sever a spoken transition between
    two selected beats. Require independent high-confidence AV coverage,
    close source adjacency on both sides, and information absent from the
    already selected speech. Never promote recording talk or failed delivery.
    """
    audience = _high_confidence_audience_spans(draft)
    for i, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        if (actions[i] != "discard" or decision.reason_code != "redundant_retry"
                or decision.relation not in {"retry_alternate", "retry_winner"}
                or not 0 < clip.end - clip.start <= 4):
            continue
        earlier = next((clips[j] for j in range(i - 1, -1, -1)
                        if actions[j] == "select" and clips[j].source_asset_id == clip.source_asset_id), None)
        later = next((clips[j] for j in range(i + 1, len(clips))
                      if actions[j] == "select" and clips[j].source_asset_id == clip.source_asset_id), None)
        if (earlier is None or later is None or not 0 <= clip.start - earlier.end <= 2
                or not 0 <= later.start - clip.end <= 12):
            continue
        if not any(start <= earlier.end and clip.start >= start and end >= clip.end
                   for start, end in audience.get(clip.source_asset_id, ())):
            continue
        if any(other.source_asset_id == clip.source_asset_id and
               decisions[other.clip_id].reason_code == "recording_process_bts" and
               earlier.end <= other.start < clip.end for other in clips):
            continue
        selected_text = ' '.join(other.text for j, other in enumerate(clips)
                                 if actions[j] == "select" and
                                 other.source_asset_id == clip.source_asset_id)
        if len(_content_tokens(clip.text) - _content_tokens(selected_text)) < 2:
            continue
        actions[i] = "select"
        overrides[i] = "av_continuous_audience_bridge_preserved"


def _refine_failed_delivery_from_focused_av(draft, clips, decisions, actions, overrides):
    """Keep only word-aligned clean speech confirmed by a focused AV probe.

    A localized probe may identify a valid delivery inside a broad candidate
    that also contains laughter or a reset. When ASR word timings exist, trim
    to words fully inside the focused audience span; never promote the entire
    mixed candidate or invent new timestamps.
    """
    whole = (draft.diagnostics or {}).get("whole_video_context") or {}
    focused = {}
    for source in whole.get("sources") or ():
        try:
            evidence = json.loads(str(source.get("audiovisual_evidence") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        rows = []
        for region in evidence.get("focused_delivery_regions") or ():
            try:
                start, end = float(region["start"]), float(region["end"])
                confidence = float(region.get("confidence", 0))
                if (region.get("role") == "audience" and .90 <= confidence <= 1
                        and math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    rows.append((start, end))
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        if rows:
            focused[str(source.get("source_asset_id") or "")] = tuple(rows)

    refined = {}
    covered_tokens = set().union(*(
        _content_tokens(clip.text) for index, clip in enumerate(clips)
        if actions[index] == "select"
    )) if any(action == "select" for action in actions) else set()
    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        if decision.reason_code != "failed_delivery" or not clip.words:
            continue
        if actions[index] not in {"discard", "select"}:
            continue
        for start, end in focused.get(clip.source_asset_id, ()):
            words = tuple(word for word in clip.words
                          if float(word.start) >= start and float(word.end) <= end
                          and float(word.start) < float(word.end))
            if len(words) < 3:
                continue
            text = " ".join(str(word.text).strip() for word in words if str(word.text).strip())
            if len(_content_tokens(text) - covered_tokens) < 3:
                continue
            refined[clip.clip_id] = replace(
                clip, start=min(float(word.start) for word in words),
                end=max(float(word.end) for word in words), text=text,
                caption_text=text, words=words,
            )
            actions[index] = "select"
            overrides[index] = "focused_av_clean_delivery_preserved"
            covered_tokens.update(_content_tokens(text))
            break
    return refined


def _refine_repeated_opening_alternate(draft, clips, decisions, actions, overrides):
    """Preserve a clean second start with a distinct physical instruction.

    Require repeated source words, localized AV audience evidence and an
    instruction absent from selected speech. The earlier false start remains
    discarded; the clean words retain their original source timing.
    """
    focused = {}
    for source in ((draft.diagnostics or {}).get("whole_video_context") or {}).get("sources") or ():
        try:
            evidence = json.loads(str(source.get("audiovisual_evidence") or ""))
        except (TypeError, ValueError):
            continue
        spans = []
        for region in evidence.get("focused_delivery_regions") or ():
            try:
                start, end = float(region["start"]), float(region["end"])
                confidence = float(region.get("confidence", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if (region.get("role") == "audience" and confidence >= .95
                    and math.isfinite(start) and math.isfinite(end) and start < end):
                spans.append((start, end))
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        focused[str(source.get("source_asset_id") or "")] = merged
    refined = {}
    selected_text = ' '.join(clip.text for i, clip in enumerate(clips) if actions[i] == "select")
    selected_tokens = _content_tokens(selected_text)
    for i, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        if (actions[i] != "swap" or decision.reason_code != "usable_alternate"
                or decision.relation != "retry_alternate" or len(clip.words) < 14
                or clip.end - clip.start < 9):
            continue
        plain = [re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', w.text.casefold()))
                 for w in clip.words]
        start_index = None
        for second in range(8, len(plain) - 7):
            phrase = plain[second:second + 4]
            if not all(phrase) or not any(plain[first:first + 4] == phrase
                                          for first in range(max(0, second - 24), second - 3)):
                continue
            start_index = second - 1 if second and plain[second - 1] in {'if', 'si'} else second
            break
        if start_index is None:
            continue
        tail = clip.words[start_index:]
        if len(tail) < 8 or not clip.start + 4 <= tail[0].start < clip.end - 4:
            continue
        spans = focused.get(clip.source_asset_id, ())
        if not any(abs(start - tail[0].start) <= 2 for start, end in spans):
            continue
        covered = sum(max(0., min(clip.end, end) - max(tail[0].start, start))
                      for start, end in spans)
        if covered < .7 * (clip.end - tail[0].start):
            continue
        text = ' '.join(w.text for w in tail)
        unique = _content_tokens(text) - selected_tokens
        action = re.search(r'\b(?:wash|clean|apply|pour|mix|spray|lavar|lava|aplica|mezcla|'
                           r'echa|agrega|pones|poner)\b', text.casefold())
        if not action or len(unique) < 2:
            continue
        refined[clip.clip_id] = replace(clip, start=float(tail[0].start),
                                        end=float(tail[-1].end), text=text,
                                        caption_text=text, words=tuple(tail))
        actions[i] = "select"
        overrides[i] = "focused_av_clean_second_start_preserved"
    return refined


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
        # A unique amount may already have promoted the left instruction.
        # Keep checking its independently verified AV demonstration so
        # Boundary can join the two selected pieces without a dead gap.
        if ((actions[i] not in {"swap", "discard"} and not (
                    actions[i] == "select" and overrides[i] == "material_retry_claim_preserved"))
                or d.reason_code != "redundant_retry"
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


def _av_recording_only_tail(draft, clip):
    """Propose a CTA-adjacent aside only when AV observes its exact words as recording-only.

    The recording-tail executor still demands a matching transcript and an
    independent measured pause. This proposal happens before Selection Freeze.
    """
    if not clip.words or len(clip.words) < 4:
        return 0, 0.0
    whole = (draft.diagnostics or {}).get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if str(source.get("source_asset_id")) != clip.source_asset_id:
            continue
        try:
            evidence = json.loads(str(source.get("audiovisual_evidence") or ""))
        except (TypeError, ValueError):
            continue
        for region in (*tuple(evidence.get("regions") or ()),
                       *tuple(evidence.get("focused_delivery_regions") or ())):
            try:
                start, end = float(region["start"]), float(region["end"])
                confidence = float(region.get("confidence", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if (region.get("role") != "recording_only" or not .95 <= confidence <= 1
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not clip.start < start < clip.end <= end + .001):
                continue
            count = 0
            for word in reversed(clip.words):
                if float(word.start) < start or float(word.end) > end + .001:
                    break
                count += 1
            if not 1 <= count <= 8 or len(clip.words) - count < 3:
                continue
            if _has_purchase_action(" ".join(w.text for w in clip.words[:-count])):
                return count, confidence
    return 0, 0.0


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
    refined_clips = {}
    for clip in clips:
        decision = decisions[clip.clip_id]
        action, safety_override = _effective_action(decision, current.get(clip.clip_id, "swap"))
        actions.append(action)
        overrides.append(safety_override)

    _enforce_single_retry_family_winner(clips, decisions, actions, overrides)
    if v2_request:
        retry_conflicts = _preserve_retry_alternates_with_unique_information(
            clips, decisions, actions, overrides)
        _preserve_unique_content_when_av_contradicts_failed(
            draft, clips, decisions, actions, overrides,
        )
        _preserve_short_audience_continuations(
            draft, clips, decisions, actions, overrides,
        )
        refined_clips = _refine_failed_delivery_from_focused_av(
            draft, clips, decisions, actions, overrides,
        )
        refined_clips.update(_refine_repeated_opening_alternate(
            draft, clips, decisions, actions, overrides,
        ))
        _preserve_continuous_demonstration(draft, clips, decisions, actions, overrides)
        competition_audit = _apply_v2_take_competitions(
            clips, decisions, actions, overrides, plan.take_competitions,
        )
        index_by_id = {clip.clip_id: i for i, clip in enumerate(clips)}
        for left_id, right_id in plan.continuation_links:
            left_i, right_i = index_by_id[left_id], index_by_id[right_id]
            if actions[right_i] == "select" and actions[left_i] != "select":
                actions[left_i] = "select"
                overrides[left_i] = "source_verified_necessary_continuation"

    selected: list[DraftClip] = []
    alternates: list[DraftClip] = []
    discarded: list[DraftClip] = []
    audit: list[dict] = []
    tail_audit: list[dict] = []

    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        action = actions[index]
        normalized_clip = replace(refined_clips.get(clip.clip_id, clip),
                                  selected=(action == "select"))
        av_recording_only = False
        tail_decision = decision
        if v2_request and action == "select" and not decision.trailing_recording_word_count:
            count, confidence = _av_recording_only_tail(draft, normalized_clip)
            if count and not any(actions[j] == "select" and
                                 other.source_asset_id == clip.source_asset_id and
                                 other.start >= clip.end for j, other in enumerate(clips)):
                tail_decision = replace(decision, trailing_recording_word_count=count,
                                        trailing_recording_confidence=confidence)
                av_recording_only = True
        if v2_request and action == "select" and tail_decision.trailing_recording_word_count:
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
                normalized_clip, tail_decision, diagnostics,
                selected_clips=coverage_clips, av_recording_only=av_recording_only)
            tail_audit.append(tail_row)
            if tail_row["action"] == "trim" and av_recording_only:
                overrides[index] = "av_recording_only_tail_removed"
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
    if plan.competition_review is not None:
        diagnostics["competition_review"] = dict(plan.competition_review)
    if plan.candidate_intervals:
        diagnostics["candidate_intervals"] = [dict(row) for row in plan.candidate_intervals]
    if tail_audit:
        diagnostics["v2_recording_tail"] = tail_audit
    if v2_request:
        diagnostics["v2_take_competitions"] = competition_audit
        diagnostics["v2_retry_claim_conflicts"] = retry_conflicts
        diagnostics["v2_verified_continuation_links"] = [list(pair) for pair in plan.continuation_links]
        diagnostics["v2_continuation_observations"] = list(plan.continuation_evidence)
    return replace(
        draft,
        selected=tuple(selected),
        alternates=tuple(alternates),
        discarded=tuple(discarded),
        diagnostics=diagnostics,
    )
