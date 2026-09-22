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
    from cutsell_worker.contradiction_signal import any_pair_contradicts
except ModuleNotFoundError:  # pragma: no cover - import-path fallback only
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from cutsell_worker.contradiction_signal import any_pair_contradicts


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

        if kind == "forbidden_contains":
            needle = _norm(check.get("text"))
            if needle and needle in joined:
                failures.append({"id": check_id, "kind": kind, "reason": "historical_bad_take_returned"})
            else:
                passes.append(check_id)
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
