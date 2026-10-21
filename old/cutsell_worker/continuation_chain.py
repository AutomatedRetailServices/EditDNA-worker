"""D-289.1: a SENTENCE-CONTINUATION CHAIN as one realization in the family
competition.

`take_grouping.sentence_continuation` (deterministic recording-process
evidence: an incomplete head that stops on a dangling function word, and a
lower-case tail that finishes the sentence, adjacent, same source) tells
IdeaClusterer that two candidates are ONE delivery split at a mid-sentence
pause. The cohesion pass records every such chain under
`distinct_idea_grouping_safety.continuation_chains`. This module is the one
place the DOWNSTREAM authorities learn what that means, so none of them can
judge the tail (a predicate without its quantifier, whose meaning read alone
may be the OPPOSITE of the sentence) or the head (a number without its
predicate) as a standalone competitor:

* the pipeline folds each chain onto its HEAD for ranking, labels and the
  semantic BestTake: the head is presented with the chain's joined text and
  span (`fold_family_members`), the tails leave the ranked candidate list,
  and the judge row records `continuation_member_ids` (the tails) and
  `realization_text` (the sentence);
* whichever bucket the head lands in, its tails land in the same one --
  `compose_selected` is followed by `bind_continuation_tails`, and the two
  authorities that move clips between buckets after that (the deterministic
  BestTake authority and ClaimCoverageBestTake) move a row's tails together
  with its head (`continuation_member_ids`);
* every claim-level judgment over a family member (ClaimCoverageBestTake,
  StoryValidator's lost-critical-claims ledger) evaluates the row's
  `realization_text` -- the complete sentence -- never the truncated head
  (`evaluation_clip`). The DraftClip's own `text`/`caption_text` are never
  rewritten: the tail's words are rendered from the tail's own clip.

A chain therefore either WINS as a unit (head and tail both kept, in
source order, the pause between them handled by the ordinary interior-gap
authority) or LOSES as a unit. Nothing here reads a Video00 phrase, id or
timestamp.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from .contracts import CandidateTake

CONTINUATION_MEMBER_IDS_KEY = "continuation_member_ids"
REALIZATION_TEXT_KEY = "realization_text"


def chain_tails_by_head(
    chains: Iterable[Sequence[str]], take_by_id: Mapping[str, CandidateTake],
) -> dict[str, tuple[str, ...]]:
    """`continuation_chains` rows -> {head_clip_id: (tail_clip_ids...)} in
    start order. A chain with a member the pipeline no longer holds is
    dropped whole (fail open: nothing is folded on partial evidence)."""
    out: dict[str, tuple[str, ...]] = {}
    for chain in chains or ():
        ids = [str(cid) for cid in chain if str(cid) in take_by_id]
        if len(ids) != len(list(chain)) or len(ids) < 2:
            continue
        ordered = sorted(ids, key=lambda cid: (take_by_id[cid].start, take_by_id[cid].end, cid))
        out[ordered[0]] = tuple(ordered[1:])
    return out


def chain_text(head: CandidateTake, tails: Sequence[CandidateTake]) -> str:
    return " ".join(str(t.text or "").strip() for t in (head, *tails)).strip()


def fold_family_members(
    members: Sequence[CandidateTake], tails_by_head: Mapping[str, Sequence[str]],
    take_by_id: Mapping[str, CandidateTake],
) -> tuple[tuple[CandidateTake, ...], dict[str, tuple[str, ...]]]:
    """Fold every chain whose head AND tails are all in `members` onto its
    head: the head is replaced by an evaluation candidate carrying the
    chain's joined text, span, words and (re-graded) completeness; the
    tails are removed from the member list. Returns (folded_members,
    {head_id: tail_ids}) -- the second value is exactly what the judge row
    must record. A chain not wholly inside `members` is left untouched."""
    from .take_segmentation import _looks_complete_idea

    member_ids = {m.clip_id for m in members}
    folded_tails: dict[str, tuple[str, ...]] = {}
    hidden: set[str] = set()
    for head_id, tail_ids in tails_by_head.items():
        if head_id in member_ids and all(t in member_ids for t in tail_ids):
            folded_tails[head_id] = tuple(tail_ids)
            hidden.update(tail_ids)
    if not folded_tails:
        return tuple(members), {}
    out: list[CandidateTake] = []
    for member in members:
        if member.clip_id in hidden:
            continue
        tail_ids = folded_tails.get(member.clip_id)
        if not tail_ids:
            out.append(member)
            continue
        tails = [take_by_id[t] for t in tail_ids]
        text = chain_text(member, tails)
        end = max(member.end, *(t.end for t in tails))
        words = tuple(member.words) + tuple(w for t in tails for w in t.words)
        out.append(replace(
            member, text=text, end=end, words=words,
            complete_idea=_looks_complete_idea(text, max(0.0, end - member.start)),
        ))
    return tuple(out), folded_tails


def unify_chain_realizations(
    takes: Sequence[CandidateTake], tails_by_head: Mapping[str, Sequence[str]],
) -> tuple[tuple[CandidateTake, ...], dict[str, str]]:
    """D-291.2 (RAW #125): a chain is ONE realization for the CANONICAL
    model too, not only for the family competition. Every tail's
    `realization_id` becomes its head's (D-050A's own "a physical split
    preserves realization identity" invariant, applied at the one place
    the pipeline learns two takes are one delivery split at a pause), so
    the Semantic Ledger registers one `RealizationRecord` holding both
    clips, the Realization Resolver evaluates/keeps/discards the sentence
    as a unit, and the authoritative application moves both clips
    together. Returns (takes with tails restamped, {tail_clip_id:
    head_realization_id}) -- the second value is what `DraftClip.parent_
    realization_id` records on each tail. A head without a
    `realization_id` (never minted) leaves its chain untouched: nothing is
    unified on partial identity."""
    by_id = {take.clip_id: take for take in takes}
    parent_by_tail: dict[str, str] = {}
    for head_id, tail_ids in tails_by_head.items():
        head = by_id.get(head_id)
        head_rid = getattr(head, "realization_id", None) if head is not None else None
        if not head_rid:
            continue
        for tail_id in tail_ids:
            if tail_id in by_id:
                parent_by_tail[tail_id] = str(head_rid)
    if not parent_by_tail:
        return tuple(takes), {}
    out = tuple(
        replace(take, realization_id=parent_by_tail[take.clip_id]) if take.clip_id in parent_by_tail else take
        for take in takes
    )
    return out, parent_by_tail


def continuation_member_ids(row: Mapping | None) -> tuple[str, ...]:
    if not row:
        return ()
    return tuple(str(cid) for cid in (row.get(CONTINUATION_MEMBER_IDS_KEY) or ()))


def rows_by_clip_id(groups: Iterable[Mapping]) -> dict[str, Mapping]:
    """{clip_id: its ranked row} over `take_judge_groups` -- the row carries
    the chain fields for its head."""
    out: dict[str, Mapping] = {}
    for group in groups or ():
        for row in (group.get("ranked") or ()):
            cid = str(row.get("clip_id") or "")
            if cid:
                out[cid] = row
    return out


def evaluation_clip(row: Mapping | None, clip):
    """The clip as the family competition must JUDGE it: a chain head is
    judged on the complete sentence (`realization_text`); any other clip on
    its own text. Never written back to the draft."""
    text = (row or {}).get(REALIZATION_TEXT_KEY)
    if not text or str(text) == str(getattr(clip, "text", "") or ""):
        return clip
    return replace(clip, text=str(text))


def bind_continuation_tails(
    selected: Sequence[CandidateTake], kept: Sequence[CandidateTake],
    tails_by_head: Mapping[str, Sequence[str]],
) -> tuple[CandidateTake, ...]:
    """After `compose_selected`: the tails of every selected head join the
    selection (source order), so a chain is kept or dropped as a unit."""
    if not tails_by_head:
        return tuple(selected)
    kept_by_id = {t.clip_id: t for t in kept}
    selected_ids = {t.clip_id for t in selected}
    out = list(selected)
    for head_id, tail_ids in tails_by_head.items():
        if head_id not in selected_ids:
            continue
        for tail_id in tail_ids:
            if tail_id in selected_ids or tail_id not in kept_by_id:
                continue
            out.append(kept_by_id[tail_id])
            selected_ids.add(tail_id)
    return tuple(sorted(out, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
