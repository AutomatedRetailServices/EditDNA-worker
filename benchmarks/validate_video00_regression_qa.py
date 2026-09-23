from __future__ import annotations

import json
import re
import sys
import unicodedata

try:
    from benchmarks.video00_semantic_alignment import (
        TINY_FRAGMENT_TOKEN_COUNT, _content_tokens, _coverage, find_coverage_span,
    )
except ModuleNotFoundError:
    from video00_semantic_alignment import (
        TINY_FRAGMENT_TOKEN_COUNT, _content_tokens, _coverage, find_coverage_span,
    )

# D-106: the contradiction/polarity/numeric-conflict safety gate reuses the
# SAME general, already-proven production detector BestTake safety itself
# relies on (D-063 dominance safety, D-101's single-winner veto) rather than
# a second, divergent heuristic invented here -- `any_pair_contradicts` is a
# pure, deterministic text function (no provider call, no engine state) that
# flags a negation-conflict (one side negates a shared proposition the other
# doesn't) or a number-conflict. Importing it here is READING a general
# production primitive for QA judgment, not feeding QA/benchmark data INTO
# production -- the import direction the D-098/D-099 "QA-only" boundary
# actually cares about (`video00_semantic_alignment.py` stays zero-import in
# the other direction) is unaffected.
try:
    from cutsell_worker.contradiction_signal import any_pair_contradicts, detect_text_contradiction
except ModuleNotFoundError:  # pragma: no cover - import-path fallback only
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from cutsell_worker.contradiction_signal import any_pair_contradicts, detect_text_contradiction


# D-106 (QA semantics correction): MEANING PRESERVATION ("did CutSell
# preserve the required meaning?") and PREFERRED-REALIZATION PARITY ("did
# CutSell select the same realization the QA reference did?") are
# deliberately separate axes -- see D-105/D-106 decision entries. A
# semantically-equivalent paraphrase may PASS meaning while FAILING parity;
# neither check is hidden.
#
# Meaning-preservation equivalence credit reuses the SAME
# `semantic_idea_equivalence` merge evidence the ENGINE ITSELF already
# computed during grouping (`diagnostics.semantic_idea_equivalence.merges`
# in the result JSON) and that `final_story_coherence_validation._same_
# idea_paraphrase_credit` already reuses downstream in production -- this
# harness only reads that already-computed number, it never re-derives
# equivalence itself and never feeds Gold/Cut.ai text into it. The
# confidence floor matches the SAME already-approved standard
# (`_SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD` / D-061 Phase 1), not a new bar
# invented for this check.
_EQUIVALENCE_CONFIDENCE_FLOOR = 0.85


def _load(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    raw = unicodedata.normalize("NFKC", str(text or ""))
    return " ".join(raw.split()).casefold()


def _selected_texts(result: dict) -> list[str]:
    return [_norm((row or {}).get("text")) for row in (result.get("selected") or [])]


def _selected_rows(result: dict) -> list[tuple[str, str]]:
    """(clip_id, normalized text) for every selected row -- clip_id is
    optional in older result fixtures (defaults to "")."""
    return [
        (str((row or {}).get("clip_id") or ""), _norm((row or {}).get("text")))
        for row in (result.get("selected") or [])
    ]


def _discarded_rows(result: dict) -> list[tuple[str, str]]:
    return [
        (str((row or {}).get("clip_id") or ""), _norm((row or {}).get("text")))
        for row in (result.get("discarded") or [])
    ]


def _equivalence_merges(result: dict) -> tuple:
    diagnostics = result.get("diagnostics") or {}
    sie = diagnostics.get("semantic_idea_equivalence") or {}
    return tuple(sie.get("merges") or ())


def _equivalence_confidence(merges: tuple, left_id: str, right_id: str) -> float:
    """Highest confidence any EXISTING engine-produced `semantic_idea_
    equivalence` merge record already established for this exact pair --
    reused verbatim, never recomputed here and never derived from Gold/
    Cut.ai text (the QA reference is used ONLY to name the canonical
    proposition being evaluated, per D-106 -- this function never sees
    it)."""
    if not left_id or not right_id:
        return 0.0
    pair = {left_id, right_id}
    best = 0.0
    for merge in merges:
        if {str(merge.get("left_clip_id") or ""), str(merge.get("right_clip_id") or "")} == pair:
            try:
                best = max(best, float(merge.get("confidence") or 0.0))
            except (TypeError, ValueError):
                continue
    return best


def _find_realization(rows: list[tuple[str, str]], target_tokens: frozenset[str]) -> tuple[str, str] | None:
    """First (clip_id, text) row among `rows` whose own content covers
    `target_tokens` at the same coverage bar `find_coverage_span` already
    uses -- a single-window search (each row judged independently, not a
    multi-segment rechunk search: discarded clips are not one ordered
    sequence)."""
    for clip_id, text in rows:
        if find_coverage_span([_content_tokens(text)], target_tokens) is not None:
            return clip_id, text
    return None


def _equivalence_credited_order_match(
    result: dict, selected_rows: list[tuple[str, str]], target_tokens: frozenset[str], *, start: int,
) -> tuple[int, str] | None:
    """D-148 tier 2: `align()` (D-032) already resolves light ASR wording
    variance and clean rechunking/composite splits. A genuinely different
    PARAPHRASE -- not just reworded ASR noise -- of a required fact can
    still fail `align()`'s bidirectional/per-segment coverage floors even
    though the fact is fully, correctly present, at the correct position.
    This reuses the SAME engine-confirmed equivalence-credit evidence
    `_evaluate_meaning_preservation` already relies on (D-106), including
    that function's own DISCARDED-only realization search: the reference
    realization must come from a clip the engine actually had and rejected
    (real evidence of "the engine saw this exact content and chose a
    different, credited-equivalent delivery"), never from among the
    already-selected rows themselves -- searching selected rows here would
    let a clip `align()` already rejected for insufficient bidirectional
    coverage "find itself" as its own realization with no real equivalence
    evidence at all. Cross-checked against the engine's OWN
    `semantic_idea_equivalence` diagnostics for a SELECTED clip at or after
    `start` it already judged equivalent, above the same confidence floor,
    with the same anti-contradiction safety gate. Returns
    `(index_in_selected_rows, candidate_text)` or None -- `start`-bounded,
    so this can never place a match earlier than where `align()` left off,
    preserving real order enforcement."""
    realization = _find_realization(_discarded_rows(result), target_tokens)
    if realization is None:
        return None
    realization_id, realization_text = realization
    merges = _equivalence_merges(result)
    for index in range(start, len(selected_rows)):
        selected_id, selected_text = selected_rows[index]
        confidence = _equivalence_confidence(merges, realization_id, selected_id)
        if confidence < _EQUIVALENCE_CONFIDENCE_FLOOR:
            continue
        if any_pair_contradicts([realization_text, selected_text]):
            continue
        return index, selected_text
    return None


def _evaluate_meaning_preservation(check: dict, result: dict, texts: list[str]) -> dict:
    """Return the meaning row: `status` is one of PASS / FAIL / UNCERTAIN.
    Never reads Gold/Cut.ai text -- `check["text"]` names the canonical
    proposition (supplied by the manifest, itself built from the QA
    reference offline) but every PASS/FAIL/UNCERTAIN decision is made from
    the engine's OWN selected/discarded/diagnostics output only. This is
    deliberately a SEPARATE question from `preferred_realization_parity`
    below -- see D-106."""
    check_id = str(check.get("id") or "unnamed")
    target = str(check.get("text") or "")
    protected = bool(check.get("protected", False))
    target_tokens = _content_tokens(target)

    direct_span = find_coverage_span([_content_tokens(t) for t in texts], target_tokens)
    if direct_span is not None:
        start, end = direct_span
        matched_text = " ".join(texts[start:end])
        # D-106 SAFETY: content-token coverage alone (the same primitive
        # `required_exact` has always used) is negation-blind -- "not"/
        # "never" are stopwords, so a directly-negated candidate can score
        # full coverage of the target's content tokens. Never trust a
        # coverage match without also confirming it does not itself
        # contradict the canonical proposition.
        if not any_pair_contradicts([target, matched_text]):
            return {"id": check_id, "kind": "meaning_preservation", "status": "PASS", "reason": "exact_realization_selected"}

    if protected:
        # Protected propositions (diagnosis identity, numbers, negation,
        # correction) never receive equivalence credit -- the canonical
        # realization must itself be present.
        return {"id": check_id, "kind": "meaning_preservation", "status": "FAIL", "reason": "missing_required_segment_protected"}

    discarded = _discarded_rows(result)
    realization = _find_realization(discarded, target_tokens)
    if realization is None:
        # The canonical proposition is not selected AND no discarded clip
        # even carries it -- this is a genuine content loss, not merely a
        # different realization choice.
        return {"id": check_id, "kind": "meaning_preservation", "status": "FAIL", "reason": "missing_required_segment_no_realization_found"}

    discarded_id, discarded_text = realization
    merges = _equivalence_merges(result)
    selected = _selected_rows(result)
    best_confidence = 0.0
    best_text = ""
    for selected_id, selected_text in selected:
        confidence = _equivalence_confidence(merges, discarded_id, selected_id)
        if confidence > best_confidence:
            best_confidence = confidence
            best_text = selected_text

    if best_confidence < _EQUIVALENCE_CONFIDENCE_FLOOR:
        # The engine's own equivalence evidence for this pair is too weak
        # (or absent) to prove the realizations are interchangeable --
        # never silently PASS on weak/uncertain similarity.
        return {
            "id": check_id, "kind": "meaning_preservation", "status": "UNCERTAIN",
            "reason": "insufficient_equivalence_evidence", "best_equivalence_confidence": round(best_confidence, 4),
        }

    if any_pair_contradicts([target, best_text]):
        # D-106 SAFETY: contradiction, polarity flip, or a materially
        # different protected proposition (number/diagnosis identity) --
        # equivalence credit is REFUSED regardless of confidence. Reused
        # verbatim from the same production safety gate, not a local
        # reimplementation.
        return {
            "id": check_id, "kind": "meaning_preservation", "status": "FAIL",
            "reason": "protected_contradiction_detected", "best_equivalence_confidence": round(best_confidence, 4),
        }

    return {
        "id": check_id, "kind": "meaning_preservation", "status": "PASS",
        "reason": "equivalent_realization_credited", "equivalence_confidence": round(best_confidence, 4),
    }


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# D-289.10: presence of THE realization, not coverage of its content. A
# different take of the same idea shares most of the vocabulary (RAW #124:
# the discarded later take's tokens were 9/14 = 0.64 covered by the
# selected monolith, and the monolith's 0.60 covered by it -- both clear
# `_PRECISE_SEARCH_MIN_COVERAGE`), so a 0.6 bar in either direction still
# reads shared content as presence. The realization's OWN wording must be
# (nearly) all there: 0.9 of the target's content tokens in ONE selected
# row (ASR punctuation/casing variance survives NFKC + casefold; a take
# that drops one distinctive word in seven does not), and that row must
# itself be mostly the target (0.6 reverse -- a longer take that merely
# contains the words is not the realization either).
_REALIZATION_PRESENCE_MIN_TARGET_COVERAGE = 0.9


def realization_present(selected_text: str, target_text: str, *, min_coverage: float | None = None) -> bool:
    """D-289.10: True iff `selected_text` IS a realization of `target_text`
    -- the target's content tokens are covered by the row at >=
    `_REALIZATION_PRESENCE_MIN_TARGET_COVERAGE` (0.9) AND the row's own
    content tokens are covered by the target at >= `min_coverage` (default
    `_PRECISE_SEARCH_MIN_COVERAGE`, 0.6). One-directional 0.6 coverage
    (what `required_exact` measures) is satisfied by any other take that
    shares most of the words; the 0.9 target bar plus the reverse
    direction is what rules that out."""
    floor = _PRECISE_SEARCH_MIN_COVERAGE if min_coverage is None else float(min_coverage)
    row_tokens = _content_tokens(selected_text)
    target_tokens = _content_tokens(target_text)
    if not row_tokens or not target_tokens:
        return False
    return (
        _coverage(target_tokens, row_tokens) >= _REALIZATION_PRESENCE_MIN_TARGET_COVERAGE
        and _coverage(row_tokens, target_tokens) >= floor
    )


def _find_present_realization(rows: list[tuple[str, str]], target: str) -> tuple[str, str] | None:
    """The first selected `(clip_id, text)` row that `realization_present`
    accepts for `target`, or None. Single rows only -- never a multi-row
    window, which is `required_exact`'s coverage question, not presence."""
    for clip_id, text in rows:
        if realization_present(text, target):
            return clip_id, text
    return None


def _split_sentences(text: str) -> list[str]:
    """D-148: general, language-agnostic sentence split on terminal
    punctuation -- used ONLY to expand a `required_order` manifest anchor
    that itself bundles multiple facts into one reference string (e.g. two
    Human-Gold sentences joined as one anchor) into its own atomic gold
    segments, each independently searched IN ORDER by `_find_semantic`
    below. Feeding a single compound gold segment straight to a coverage
    search defeats per-fact credit: a candidate that only covers HALF the
    compound text (a real, common split -- one clip per fact) then has no
    partial credit to receive and the whole anchor reads as missing. A
    one-sentence anchor (the common case) round-trips to a single-element
    list, unchanged."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(str(text or "").strip())]
    return [p for p in parts if p]


_PRECISE_SEARCH_MAX_WINDOW = 3
_PRECISE_SEARCH_MIN_COVERAGE = 0.6


def _find_semantic(texts: list[str], target: str, *, start: int = 0) -> tuple[int, int] | None:
    """D-032: content-coverage span search, not byte-equal text matching --
    a required fact rechunked into a bigger/smaller candidate segment, or
    restated with light ASR wording variance, is still recognized as
    present. See `video00_semantic_alignment.py`'s own docstring for
    exactly why exact-text matching false-positived on genuine re-chunking
    (RAW 33402023395).

    D-152 (Gate 6 correction, real RAW #118 audit): deliberately does NOT
    call that module's own `find_coverage_span` here anymore.
    `find_coverage_span` prefers the EARLIEST starting window, THEN the
    smallest -- sensible for its own "does this content appear starting
    around here" question, but unsound for an order check: a real audit
    found it can let an EARLIER, semantically unrelated candidate segment
    get silently absorbed into a wider window that also happens to include
    the real (later, correctly-matching) segment, because the wider window
    still clears coverage once the real segment's own content is folded
    in -- which can hide genuine reordering instead of catching it. This
    local search instead prefers the SMALLEST sufficient window first (most
    precise, least likely to accidentally absorb unrelated content), and
    only the earliest position within that smallest size -- reusing the
    exact same coverage primitive and threshold, just a safer preference
    order for this file's own order/content checks."""
    candidate_tokens = [_content_tokens(t) for t in texts]
    target_tokens = _content_tokens(target)
    for window in range(1, _PRECISE_SEARCH_MAX_WINDOW + 1):
        for c_start in range(start, len(candidate_tokens) - window + 1):
            available = frozenset().union(*candidate_tokens[c_start:c_start + window])
            if _coverage(target_tokens, available) >= _PRECISE_SEARCH_MIN_COVERAGE:
                return c_start, c_start + window
    return None


# D-289.11 (RAW #124 MP4 review, user-reported CTA repetition): CTA
# PRESENCE is not CTA UNIQUENESS. `required_exact`/`required_realization`
# answer "is this segment there?"; neither notices that the preserved
# conclusion already CLOSED on the words the standalone CTA re-opens with
# ("... Así que cuídate." then "Por eso cuídate, aliméntate ..."). This
# detector is QA's own (independent of the production trim in
# `final_boundary_authority._trim_reopened_closings`): an earlier selected
# row within a short lookback that ends a sentence with the 1..6 tokens the
# later row opens with, after at most two leading connectives. It reports
# the repetition; it never decides whether production was right to keep it.
_ORDERED_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+")
# Discourse connectives only -- never articles/determiners: "... a biopsia."
# then "La biopsia confirmó ..." is a noun re-mention, not a repeated closing.
_REPEATED_CLOSING_CONNECTIVES = frozenset({
    "por", "eso", "así", "asi", "que", "entonces", "pues", "bueno", "y", "e", "o", "sea",
    "ahora", "también", "tambien", "además", "ademas",
    "so", "and", "then", "therefore", "hence", "well", "okay", "ok", "now", "also",
})
# A lone function word is never a closing phrase (width-1 guard).
_REPEATED_CLOSING_FUNCTION_TOKENS = _REPEATED_CLOSING_CONNECTIVES | frozenset({
    "a", "de", "en", "el", "la", "los", "las", "un", "una", "lo", "se", "bien",
    "the", "to", "of", "in", "on", "at", "it", "is",
})
_REPEATED_CLOSING_TERMINAL = (".", "!", "?", "…")
_REPEATED_CLOSING_MAX_LOOKBACK_ROWS = 3
_REPEATED_CLOSING_MAX_LEADING_CONNECTIVES = 2
_REPEATED_CLOSING_MAX_PHRASE_TOKENS = 6


def _ordered_tokens(text: str) -> list[str]:
    return _ORDERED_TOKEN_RE.findall(_norm(text))


def _asr_anchor_token(token: str) -> str:
    """Compare a short QA anchor without treating accent drift as content."""
    return "".join(
        letter for letter in unicodedata.normalize("NFKD", token)
        if not unicodedata.combining(letter)
    )


def _asr_anchor_matches(token: str, anchor: str) -> bool:
    token, anchor = _asr_anchor_token(token), _asr_anchor_token(anchor)
    if token == anchor:
        return True
    # Allow one ASR insertion/deletion/substitution in a LONG content word.
    # A one-letter change in "no" or a number must never get this credit.
    if min(len(token), len(anchor)) < 8 or abs(len(token) - len(anchor)) > 1:
        return False
    if len(token) == len(anchor):
        return sum(a != b for a, b in zip(token, anchor)) == 1
    shorter, longer = sorted((token, anchor), key=len)
    return any(longer[:pos] + longer[pos + 1:] == shorter for pos in range(len(longer)))


def _has_ordered_anchors(text: str, anchors: list[str], *, max_span_tokens: int) -> bool:
    words = _ordered_tokens(text)
    for start, word in enumerate(words):
        if not _asr_anchor_matches(word, anchors[0]):
            continue
        at = start
        for anchor in anchors[1:]:
            at = next(
                (pos for pos in range(at + 1, min(len(words), start + max_span_tokens))
                 if _asr_anchor_matches(words[pos], anchor)),
                len(words),
            )
            if at == len(words):
                break
        else:
            return True
    return False


def _repeated_closing_match(left_text: str, right_text: str) -> dict | None:
    """The phrase `right_text` re-opens with that `left_text` closed on, or
    None: `{"skip": n, "repeated_tokens": [...]}`."""
    if not str(left_text or "").rstrip().endswith(_REPEATED_CLOSING_TERMINAL):
        return None
    lt = _ordered_tokens(left_text)
    rt = _ordered_tokens(right_text)
    if not lt or not rt:
        return None
    for skip in range(0, _REPEATED_CLOSING_MAX_LEADING_CONNECTIVES + 1):
        if skip and any(token not in _REPEATED_CLOSING_CONNECTIVES for token in rt[:skip]):
            break
        for width in range(_REPEATED_CLOSING_MAX_PHRASE_TOKENS, 0, -1):
            if len(lt) < width or len(rt) < skip + width:
                continue
            if lt[-width:] != rt[skip:skip + width]:
                continue
            if width == 1 and lt[-1] in _REPEATED_CLOSING_FUNCTION_TOKENS:
                continue
            return {"skip": skip, "repeated_tokens": list(rt[skip:skip + width])}
    return None


def find_repeated_closings(
    rows: list[tuple[str, str]], *, only_index: int | None = None,
    all_prior: bool = False,
) -> list[dict]:
    """Every selected row (or only `only_index`) that re-opens with the
    closing phrase of one of its previous `_REPEATED_CLOSING_MAX_LOOKBACK_
    ROWS` rows -- nearest earlier row first, one finding per later row.
    A targeted QA check can inspect all prior rows; production's short,
    speech-safe trim window is deliberately unaffected."""
    findings: list[dict] = []
    for index, (clip_id, text) in enumerate(rows):
        if only_index is not None and index != only_index:
            continue
        limit = index if all_prior else _REPEATED_CLOSING_MAX_LOOKBACK_ROWS
        for back in range(1, limit + 1):
            earlier = index - back
            if earlier < 0:
                break
            match = _repeated_closing_match(rows[earlier][1], text)
            if match is None:
                continue
            findings.append({
                "earlier_clip_id": rows[earlier][0],
                "earlier_index": earlier,
                "later_clip_id": clip_id,
                "later_index": index,
                "intervening_rows": back - 1,
                "repeated_tokens": match["repeated_tokens"],
                "leading_connective_count": match["skip"],
            })
            break
    return findings


def _locate_target_row(rows: list[tuple[str, str]], target: str) -> int | None:
    """Index of the selected row realizing `target` (presence first, then
    the first row of the smallest coverage window)."""
    for index, (_, text) in enumerate(rows):
        if realization_present(text, target):
            return index
    span = _find_semantic([text for _, text in rows], target)
    return None if span is None else span[0]


def validate(result_path: str, manifest_path: str) -> tuple[bool, dict]:
    result = _load(result_path)
    manifest = _load(manifest_path)
    texts = _selected_texts(result)
    joined = "\n".join(texts)
    failures: list[dict] = []
    passes: list[str] = []
    warnings: list[dict] = []
    # D-106: meaning-preservation and preferred-realization-parity are
    # tracked in their OWN buckets, separate from `failures`/`passes`.
    # Parity mismatches are recorded but never gate `qa_pass` -- an
    # editorial take-choice disagreement is not a meaning-safety defect.
    # UNCERTAIN meaning verdicts are conservative (never silently PASS) and
    # DO gate `qa_pass`, listed in `meaning_uncertain`, distinct from a
    # proven `meaning_failures` FAIL.
    meaning_passes: list[str] = []
    meaning_failures: list[dict] = []
    meaning_uncertain: list[dict] = []
    parity_passes: list[str] = []
    parity_failures: list[dict] = []

    # D-032: a changed segment COUNT is not by itself evidence of content
    # loss -- benign re-chunking (ASR/attempt-reconstruction merging or
    # splitting segments differently between runs) changes the count while
    # every idea stays fully present. Recorded as a warning, not a failure;
    # the required_exact/required_order/forbidden_contains checks below are
    # what actually decide whether real content is missing.
    expected_count = int(manifest.get("expected_selected_count") or 0)
    if expected_count:
        actual_count = len(texts)
        if actual_count != expected_count:
            warnings.append({
                "id": "selection_count_23",
                "kind": "count",
                "expected": expected_count,
                "actual": actual_count,
                "reason": "count_differs_not_treated_as_failure_see_D-032",
            })
        else:
            passes.append("selection_count_23")

    # D-289.11: observability scan -- every re-opened closing in the selection
    # is reported as a warning regardless of the manifest, so a RAW's QA log
    # shows the repetition even where no `repeated_closing_absent` check
    # gates it (adding that check to a baseline manifest is a Product Owner
    # decision; this scan never changes `qa_pass`).
    for finding in find_repeated_closings(_selected_rows(result)):
        warnings.append({
            "id": "repeated_closing_scan",
            "kind": "repeated_closing_detected",
            "reason": "observability_only_gated_by_repeated_closing_absent_check",
            **finding,
        })

    for check in manifest.get("checks") or []:
        check_id = str(check.get("id") or "unnamed")
        kind = str(check.get("kind") or "")

        if kind == "required_exact":
            span = _find_semantic(texts, check.get("text"))
            if span is None:
                failures.append({"id": check_id, "kind": kind, "reason": "missing_required_segment"})
            else:
                passes.append(check_id)
            continue

        if kind == "required_realization":
            # D-289.10 (RAW #124 QA note): `required_exact` asks "is this
            # content COVERED by the selection?" -- a token-coverage search
            # that a DIFFERENT realization sharing most of the words can
            # satisfy (RAW #124: the discarded later take's text was 9/13
            # covered by the selected monolith, so the check passed while
            # the take it names was gone). This kind asks the stricter
            # question "is THIS realization present?": one selected row
            # must cover the target AND be covered by it (both >=
            # `_PRECISE_SEARCH_MIN_COVERAGE`) -- shared vocabulary in a
            # longer or shorter different take never counts. Reported in
            # its own bucket next to the existing kinds; the manifest
            # decides where it is used (none of the baselines here does).
            row = _find_present_realization(_selected_rows(result), check.get("text"))
            if row is None:
                failures.append({"id": check_id, "kind": kind, "reason": "realization_not_present_only_shared_content"})
            else:
                passes.append(check_id)
            continue

        if kind == "repeated_closing_absent":
            # D-289.11: the row realizing `text` must NOT re-open with the
            # closing phrase of a nearby earlier selected row. Presence of
            # the segment is a precondition, never the verdict.
            rows = _selected_rows(result)
            if check.get("target_final_selected"):
                target_index = (
                    len(rows) - 1 if rows and _locate_target_row(rows[-1:], check.get("text")) is not None
                    else None
                )
            else:
                target_index = _locate_target_row(rows, check.get("text"))
            if target_index is None:
                failures.append({"id": check_id, "kind": kind, "reason": "missing_required_segment"})
                continue
            found = find_repeated_closings(
                rows, only_index=target_index,
                all_prior=bool(check.get("search_all_prior")),
            )
            if found:
                failures.append({
                    "id": check_id, "kind": kind,
                    "reason": "repeated_closing_reopens_required_segment",
                    "detail": found[0],
                })
            else:
                passes.append(check_id)
            continue

        if kind == "forbidden_contains":
            needle = _norm(check.get("text"))
            if needle and needle in joined:
                failures.append({"id": check_id, "kind": kind, "reason": "historical_bad_take_returned"})
            else:
                passes.append(check_id)
            continue

        if kind == "forbidden_realization":
            # A literal forbidden substring misses punctuation-only ASR
            # changes ("aquí detrás" vs "aquí, detrás"). The same
            # symmetric take-identity check as required_realization asks
            # whether the rejected realization itself was selected.
            row = _find_present_realization(_selected_rows(result), check.get("text"))
            if row is not None:
                failures.append({
                    "id": check_id, "kind": kind,
                    "reason": "forbidden_realization_selected",
                    "clip_id": row[0],
                })
            else:
                passes.append(check_id)
            continue

        if kind == "required_contiguous_phrase":
            # Preserve short, meaning-critical openings (including "no")
            # within ONE source delivery. A token search over joined clips
            # would accept a negation spliced from a different take.
            required = _ordered_tokens(check.get("text"))
            present = bool(required) and any(
                any(_ordered_tokens(text)[pos:pos + len(required)] == required
                    for pos in range(len(_ordered_tokens(text)) - len(required) + 1))
                for _, text in _selected_rows(result)
            )
            if present:
                passes.append(check_id)
            else:
                failures.append({
                    "id": check_id, "kind": kind,
                    "reason": "phrase_not_contiguous_within_one_delivery",
                })
            continue

        if kind == "required_ordered_anchors":
            # Use only for a compact, single delivery's key actions where
            # harmless ASR spelling/punctuation drift is known to occur.
            raw_anchors = check.get("tokens")
            if not isinstance(raw_anchors, list) or not raw_anchors or not all(
                isinstance(value, str) and len(_ordered_tokens(value)) == 1
                for value in raw_anchors
            ):
                raise ValueError(f"invalid ordered-anchors check: {check_id}")
            anchors = [_ordered_tokens(value)[0] for value in raw_anchors]
            max_span = check.get("max_span_tokens")
            if type(max_span) is not int or max_span < len(anchors):
                raise ValueError(f"invalid ordered-anchors span: {check_id}")
            reference = check.get("positive_text")
            if reference is not None and (not isinstance(reference, str) or not reference.strip()):
                raise ValueError(f"invalid ordered-anchors positive reference: {check_id}")
            candidates = _selected_rows(result)
            if check.get("must_be_final_selected"):
                candidates = candidates[-1:]
            matches = [
                text for _, text in candidates
                if _has_ordered_anchors(text, anchors, max_span_tokens=max_span)
            ]
            if matches and any(
                reference is None or not (
                    (verdict := detect_text_contradiction(reference, text)).negation_conflict
                    or verdict.number_conflict
                ) for text in matches
            ):
                passes.append(check_id)
            else:
                failures.append({
                    "id": check_id, "kind": kind,
                    "reason": (
                        "ordered_actions_polarity_conflict" if matches
                        else "ordered_actions_not_present_in_one_delivery"
                    ),
                })
            continue

        if kind == "phrase_count":
            # A check for a single closing exhortation must establish both
            # presence and uniqueness, even when the prior delivery and CTA
            # are separated by several other selected clips.
            phrase = [_asr_anchor_token(token) for token in _ordered_tokens(check.get("text"))]
            expected = check.get("expected_count")
            if not phrase or type(expected) is not int or expected < 0:
                raise ValueError(f"invalid phrase-count check: {check_id}")
            count = 0
            for _, selected_text in _selected_rows(result):
                tokens = [_asr_anchor_token(token) for token in _ordered_tokens(selected_text)]
                count += sum(
                    tokens[pos:pos + len(phrase)] == phrase
                    for pos in range(len(tokens) - len(phrase) + 1)
                )
            if count == expected:
                passes.append(check_id)
            else:
                failures.append({
                    "id": check_id, "kind": kind,
                    "reason": "phrase_occurrence_count_mismatch",
                    "expected_count": expected, "observed_count": count,
                })
            continue

        if kind in {"required_source_overlap", "forbidden_source_overlap"}:
            # Only a benchmark-specific manifest supplies these source
            # coordinates. They never feed the production editor. They
            # distinguish two takes with similar wording and catch a
            # truncated failed attempt that no text-only search can name.
            start = float(check["source_start_sec"])
            end = float(check["source_end_sec"])
            minimum = float(check["min_overlap_sec"])
            if not (0 <= start < end and 0 < minimum <= end - start):
                raise ValueError(f"invalid source-overlap check: {check_id}")
            selected = result.get("selected") or []
            if any("start" not in row or "end" not in row for row in selected):
                failures.append({
                    "id": check_id, "kind": kind,
                    "reason": "source_interval_unavailable",
                })
                continue
            overlapping = sorted(
                (
                    max(start, float(row["start"])),
                    min(end, float(row["end"])),
                    str(row.get("clip_id") or ""),
                ) for row in selected
                if min(end, float(row["end"])) > max(start, float(row["start"]))
            )
            # Legitimate re-chunking can split one preferred take into
            # several clips. Sum the UNION of covered source time, never
            # require that a single row carry the entire delivery or double
            # count overlapping clips of the same source.
            coverage_end = start
            covered = 0.0
            for row_start, row_end, _ in overlapping:
                covered += max(0.0, row_end - max(row_start, coverage_end))
                coverage_end = max(coverage_end, row_end)
            matching_ids = [row_id for _, _, row_id in overlapping] if covered >= minimum else []
            if (kind == "required_source_overlap") == bool(matching_ids):
                passes.append(check_id)
            else:
                failures.append({
                    "id": check_id, "kind": kind,
                    "reason": "source_overlap_missing" if kind == "required_source_overlap" else "forbidden_source_overlap_selected",
                    "clip_ids": matching_ids,
                    "covered_source_sec": round(covered, 3),
                })
            continue

        if kind == "required_order":
            # D-148 (Gate 6 correction, real RAW #118 audit): a required-
            # order manifest anchor can itself bundle MULTIPLE facts as one
            # reference string (e.g. two Human-Gold sentences joined into
            # one anchor). The previous single-span `_find_semantic` search
            # required that WHOLE bundle to be covered by ONE candidate
            # window -- so a candidate that keeps every fact, in the correct
            # order, but realizes them as a DIFFERENT split (its own clip
            # for each fact, differently worded) registered as
            # "required_sequence_missing_or_reordered": a real causal-order
            # violation and a preferred-realization/rechunking difference
            # were being reported as the SAME failure.
            #
            # D-148 first tried reusing `align()` (D-032) wholesale for this
            # check, since it already has general COMPOSITE/RECHUNKED
            # reasoning -- but a real audit against the actual RAW #118
            # artifact found `align()`'s bounded `MAX_SKIP` (built for a
            # near-parallel gold/candidate walk, e.g.
            # `validate_video00_selection_lock.py`'s full-timeline
            # comparison) is the wrong tool for a required_order check's
            # SPARSE anchors: a handful of required facts scattered across
            # a full-length selection with long stretches of legitimately
            # unrelated content between them, which `align()` cannot skip
            # over. Reverted to this check's own original, proven,
            # UNBOUNDED cursor + `_find_semantic` search (`find_coverage_
            # span` scans every remaining candidate index, not a bounded
            # skip) -- this still finds a COMPOSITE match natively
            # (`find_coverage_span` already tries multi-segment windows),
            # and `start=cursor` (not `cursor+1`) still lets two adjacent
            # gold facts share the same already-matched candidate segment,
            # exactly like the pre-D-148 code.
            #
            # What's NEW here, kept from D-148: (1) each anchor is flattened
            # into its own sentences FIRST (see `_split_sentences`'s
            # docstring), so a compound anchor gets independent, in-order
            # credit per fact instead of needing ONE window to cover all of
            # it; (2) a sentence `_find_semantic` still can't place falls
            # back to the SAME engine-confirmed equivalence-credit evidence
            # `meaning_preservation` checks already use (D-106) before
            # declaring a real failure -- a genuine paraphrase, not just
            # light ASR variance. A sentence found by NEITHER path is still
            # a hard, blocking failure; order itself is always enforced --
            # the cursor only ever moves forward.
            wanted = [sentence for item in (check.get("texts") or []) for sentence in _split_sentences(item)]
            selected_rows = _selected_rows(result)
            cursor = 0
            missing_text: str | None = None
            realization_notes: list[dict] = []
            for sentence in wanted:
                # D-152: a TINY fragment produced by sentence-splitting a
                # compound anchor (<= `TINY_FRAGMENT_TOKEN_COUNT` content
                # tokens -- e.g. a trailing "perfectamente." or an
                # intentionally truncated quote like "Me salía por") carries
                # too little content to independently prove or disprove
                # anything on its own (the SAME doctrine `video00_semantic_
                # alignment.py`'s own module comment already established).
                # The ORIGINAL, pre-split compound-anchor search tolerated
                # exactly this kind of thin trailing content as noise within
                # the whole anchor's own coverage check; requiring it to
                # independently pass its own search is a real regression a
                # RAW #118 audit caught (`pimples_micro_order`) -- skip it
                # entirely rather than searching or advancing the cursor.
                if len(_content_tokens(sentence)) <= TINY_FRAGMENT_TOKEN_COUNT:
                    continue
                span = _find_semantic(texts, sentence, start=cursor)
                if span is not None:
                    start_index, end_index = span
                    if end_index - start_index > 1:
                        realization_notes.append({
                            "id": check_id, "kind": "required_order_realization_variance",
                            "reason": "rechunked_or_composite_not_exact_reference_wording",
                            "gold_text": sentence, "candidate_text": " ".join(texts[start_index:end_index]),
                        })
                    cursor = start_index
                    continue
                credited = _equivalence_credited_order_match(
                    result, selected_rows, _content_tokens(sentence), start=cursor,
                )
                if credited is None:
                    missing_text = sentence
                    break
                matched_index, matched_text = credited
                realization_notes.append({
                    "id": check_id, "kind": "required_order_realization_variance",
                    "reason": "equivalent_realization_credited_not_reference_wording",
                    "gold_text": sentence, "candidate_text": matched_text,
                })
                cursor = matched_index
            if missing_text is not None:
                failures.append({
                    "id": check_id, "kind": kind, "reason": "required_sequence_missing_or_reordered",
                    "missing_text": missing_text,
                })
            else:
                passes.append(check_id)
                # D-148: a rechunked/composite/equivalence-credited note
                # means the causal order IS intact -- every required fact is
                # present, in the required order -- just realized with a
                # different segment split/wording than the QA reference.
                # That is a preferred-realization-parity signal (recorded,
                # visible, never blocking), not an order defect.
                parity_failures.extend(realization_notes)
            continue

        if kind == "meaning_preservation":
            # D-106: "did CutSell preserve the required meaning?" -- a
            # distinct question from whether the QA-preferred wording was
            # itself selected (see `preferred_realization_parity` below).
            row = _evaluate_meaning_preservation(check, result, texts)
            if row["status"] == "PASS":
                meaning_passes.append(check_id)
            elif row["status"] == "UNCERTAIN":
                meaning_uncertain.append(row)
            else:
                meaning_failures.append(row)
            continue

        if kind == "preferred_realization_parity":
            # D-106: "did CutSell select the same/preferred realization as
            # the QA reference?" -- recorded, NEVER hidden, but never gates
            # `qa_pass`: a semantically-equivalent paraphrase legitimately
            # fails this while passing meaning preservation above.
            span = _find_semantic(texts, check.get("text"))
            if span is None:
                parity_failures.append({"id": check_id, "kind": kind, "reason": "preferred_realization_not_selected"})
            else:
                parity_passes.append(check_id)
            continue

        failures.append({"id": check_id, "kind": kind, "reason": "unknown_check_kind"})

    qa_pass = not failures and not meaning_failures and not meaning_uncertain
    report = {
        "schema_version": manifest.get("schema_version"),
        "baseline_run_id": manifest.get("baseline_run_id"),
        "qa_pass": qa_pass,
        "selected_count": len(texts),
        "passed_check_count": len(passes),
        "failed_check_count": len(failures),
        "passed_checks": passes,
        "failed_checks": failures,
        "warnings": warnings,
        # D-106: meaning-safety and editorial-parity are reported as
        # distinct, equally-visible axes -- neither hidden inside the other.
        "meaning_preservation": {
            "passed_check_count": len(meaning_passes),
            "failed_check_count": len(meaning_failures),
            "uncertain_check_count": len(meaning_uncertain),
            "passed_checks": meaning_passes,
            "failed_checks": meaning_failures,
            "uncertain_checks": meaning_uncertain,
        },
        "preferred_realization_parity": {
            "passed_check_count": len(parity_passes),
            "failed_check_count": len(parity_failures),
            "passed_checks": parity_passes,
            "failed_checks": parity_failures,
            "note": "editorial/take-selection mismatch only -- never gates qa_pass",
        },
    }
    return qa_pass, report


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_video00_regression_qa.py RESULT_JSON MANIFEST_JSON", file=sys.stderr)
        return 2
    ok, report = validate(sys.argv[1], sys.argv[2])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
