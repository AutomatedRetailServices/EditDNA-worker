"""General ordered semantic alignment for Human Gold regression QA -- D-032.

`validate_video00_selection_lock.py` used to compare the candidate's
`selected` list against the Human Gold baseline INDEX BY INDEX: gold[0] vs
candidate[0], gold[1] vs candidate[1], and so on. RAW 33402023395 showed
exactly why that is wrong: one benign re-chunking difference (the ASR/
attempt-reconstruction stage merging what the baseline counted as two
segments into one, or vice versa) shifts every later index by one position
and cascades into a wall of "missing_segment"/"text_changed" errors for
content that is, in fact, fully present -- just chunked differently. The
same brittleness affects `validate_video00_regression_qa.py`'s
`required_exact`/`required_order` checks: they search the WHOLE candidate
list (already position-independent) but require byte-for-byte normalized
text equality, so the identical rechunking (or a harmless ASR wording
variance) makes a genuinely-present fact register as "missing".

This module replaces exact/positional comparison with ORDERED SEMANTIC
ALIGNMENT: it walks gold and candidate segments together, in order, using
CONTENT-TOKEN OVERLAP (not exact text equality) to recognize a match, and
allows the window on either side to grow past 1 segment so it can
recognize:

    Gold[i]                  <-> Candidate[j]                  (EXACT)
    Gold[i], Gold[i+1]        <-> Candidate[j]                  (RECHUNKED -- gold split, candidate merged)
    Gold[i]                  <-> Candidate[j], Candidate[j+1]   (COMPOSITE -- gold merged, candidate split)

A gold segment with no explaining candidate window anywhere ahead is
MISSING. A candidate segment nothing in gold ever needed is EXTRA, unless
its own content closely overlaps a gold segment ALREADY matched elsewhere
-- then it is a DUPLICATE (the same idea rendered twice). Because the walk
is strictly left-to-right on both sides (never moving backward), a
candidate realization for an idea that should come later appearing before
an earlier gold idea's realization cannot be "found" out of order -- the
earlier gold segment simply reports MISSING, which is the correct signal
for a real ordering break, not a silently-accepted match.

Human Gold stays QA/oracle only: this module is read-only benchmark
tooling with zero import from `cutsell_worker`, and nothing here feeds
Human Gold's answers into runtime Selection/Boundary decisions -- it only
ever runs after the fact, in CI, to compare an already-produced candidate
against the oracle.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

EXACT = "EXACT"
RECHUNKED = "RECHUNKED"
COMPOSITE = "COMPOSITE"
MISSING = "MISSING"
EXTRA = "EXTRA"
DUPLICATE = "DUPLICATE"

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+")
# General bilingual stopwords -- the same class of function words
# `final_sibling_grouping._content` already excludes; kept as an
# independent copy here since benchmark tooling must not import
# cutsell_worker (see module docstring).
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "so",
    "that", "the", "this", "to", "was", "we", "were", "what", "with", "you", "your",
    "al", "como", "con", "cuando", "de", "del", "el", "en", "ella", "ellas", "ellos",
    "es", "esta", "este", "la", "las", "le", "les", "lo", "los", "mi", "mis", "o",
    "para", "pero", "por", "porque", "que", "se", "si", "su", "sus", "un", "una",
})

# Bounded search: how many segments a window may span on either side, and
# how far ahead an unrelated EXTRA candidate segment may be skipped before
# giving up and searching a smaller/no window. Small on purpose -- this is
# tolerance for genuine re-chunking/ASR noise, not a license to explain
# away real content loss with a coincidental distant token overlap.
MAX_WINDOW = 3
MAX_SKIP = 2
MIN_COVERAGE = 0.6
# When a window spans 2+ segments on either side, the AGGREGATE coverage
# alone is not enough proof: a strongly-covered segment can mathematically
# "average out" a completely uncovered neighbor and still clear
# MIN_COVERAGE overall. Every individual segment folded into a multi-
# segment window must clear this lower floor on its own, so a genuinely
# absent gold segment can never hide behind a well-matched one it happens
# to be windowed together with.
PER_SEGMENT_MIN_COVERAGE = 0.45
# A segment with this few content tokens or fewer ("perfectamente.", "y
# eso.") is a connector/trailing fragment, not an independent idea -- it is
# EXEMPT from the per-segment floor above. Such a fragment's own content is
# too thin to independently prove or disprove strong coverage against any
# window; it exists only to be windowed together with its real neighbor,
# which is exactly the legitimate RECHUNKED case this floor must not block.
TINY_FRAGMENT_TOKEN_COUNT = 2


def _normalize(text: str) -> str:
    raw = unicodedata.normalize("NFKC", str(text or ""))
    return " ".join(raw.split())


def _content_tokens(text: str) -> frozenset[str]:
    lowered = _normalize(text).casefold()
    return frozenset(
        token for token in _TOKEN_RE.findall(lowered)
        if len(token) >= 3 and token not in _STOP
    )


def _coverage(needed: frozenset[str], available: frozenset[str]) -> float:
    """Fraction of `needed`'s own tokens found in `available` -- not
    symmetric (this asks "is gold's content explained by candidate", not
    the other way around)."""
    if not needed:
        return 1.0
    return len(needed & available) / len(needed)


@dataclass(frozen=True)
class AlignmentRow:
    gold_span: tuple[int, int]        # [start, end) gold segment indices covered
    candidate_span: tuple[int, int]   # [start, end) candidate segment indices covered
    relation: str                     # EXACT | RECHUNKED | COMPOSITE | MISSING
    gold_text: str
    candidate_text: str
    content_coverage: float           # gold content found in the matched candidate window


@dataclass(frozen=True)
class AlignmentResult:
    rows: tuple[AlignmentRow, ...]
    extra_candidate_indices: tuple[int, ...]
    duplicate_candidate_indices: tuple[int, ...]
    missing_count: int

    @property
    def aligned(self) -> bool:
        """True only when every gold segment found a realization AND no
        candidate segment duplicates one already covered. EXTRA candidate
        content (present, not asked for by any gold segment) does not by
        itself fail alignment -- it is recorded for review, not treated as
        loss; a caller wanting a stricter "no extra content at all" gate
        can additionally check `extra_candidate_indices`."""
        return self.missing_count == 0 and not self.duplicate_candidate_indices


def _best_match(
    gold_tokens: list[frozenset[str]],
    candidate_tokens: list[frozenset[str]],
    g: int,
    c: int,
) -> tuple[int, int, int, float] | None:
    """Search for the smallest (skip, window) match starting at gold index
    `g` and candidate index >= `c`. Returns (skip, wg, wc, coverage) or
    None. Prefers, in order: least skip (don't jump ahead when a fine
    match exists nearby), smallest combined window (closest to a clean
    1:1 EXACT match), then highest coverage.

    Two guards keep this from over-matching:

    - BIDIRECTIONAL coverage: a window pair is only accepted when gold's
      content is covered by the candidate window AND the candidate
      window's content is explained by gold, both >= MIN_COVERAGE. Without
      the second direction, a single wg=1/wc=1 "EXACT" match would greedily
      consume a candidate segment that actually contains gold[g] merged
      with gold[g+1]'s content too (a genuine RECHUNKED case) -- accepting
      the narrow match first would then strand gold[g+1] as falsely
      MISSING, having already consumed the only segment that explains it.
    - Skip guard: a `skip` this large would jump PAST some candidate
      content on the way to a match -- but only when that skipped content
      does not itself look like it belongs to gold[g:] (checked both
      directions, same threshold). If it does, skipping over it would mask
      real reordering/duplication as a clean match; refusing the skip
      leaves gold[g] correctly reporting MISSING instead.
    """
    best: tuple[tuple[int, int, float], int, int, int, float] | None = None
    max_g = len(gold_tokens)
    max_c = len(candidate_tokens)
    remaining_gold = frozenset().union(*gold_tokens[g:]) if g < max_g else frozenset()
    for skip in range(0, MAX_SKIP + 1):
        c_start = c + skip
        if c_start >= max_c:
            break
        if skip > 0:
            skipped = frozenset().union(*candidate_tokens[c:c_start])
            looks_like_future_gold = (
                _coverage(skipped, remaining_gold) >= MIN_COVERAGE
                or _coverage(remaining_gold, skipped) >= MIN_COVERAGE
            )
            if looks_like_future_gold:
                continue
        for wg in range(1, min(MAX_WINDOW, max_g - g) + 1):
            needed = frozenset().union(*gold_tokens[g:g + wg])
            for wc in range(1, min(MAX_WINDOW, max_c - c_start) + 1):
                if wg > 1 and wc > 1:
                    # A genuine granularity difference (RECHUNKED/COMPOSITE)
                    # always has exactly one side still at window size 1 --
                    # gold split but candidate merged, or gold whole but
                    # candidate split. Growing BOTH sides at once is bag-of-
                    # words matching two independent multi-segment spans
                    # against each other, which is order-blind (it cannot
                    # tell gold=[A,B] from candidate=[B,A] apart) and risks
                    # silently absorbing real reordering or duplication.
                    continue
                available = frozenset().union(*candidate_tokens[c_start:c_start + wc])
                forward_coverage = _coverage(needed, available)
                backward_coverage = _coverage(available, needed)
                if forward_coverage < MIN_COVERAGE or backward_coverage < MIN_COVERAGE:
                    continue
                if wg > 1 and any(
                    len(gold_tokens[g + i]) > TINY_FRAGMENT_TOKEN_COUNT
                    and _coverage(gold_tokens[g + i], available) < PER_SEGMENT_MIN_COVERAGE
                    for i in range(wg)
                ):
                    continue
                if wc > 1 and any(
                    len(candidate_tokens[c_start + i]) > TINY_FRAGMENT_TOKEN_COUNT
                    and _coverage(candidate_tokens[c_start + i], needed) < PER_SEGMENT_MIN_COVERAGE
                    for i in range(wc)
                ):
                    continue
                key = (skip, wg + wc, -forward_coverage)
                if best is None or key < best[0]:
                    best = (key, skip, wg, wc, forward_coverage)
    if best is None:
        return None
    _, skip, wg, wc, coverage = best
    return skip, wg, wc, coverage


def find_coverage_span(
    candidate_tokens: list[frozenset[str]], target_tokens: frozenset[str], start: int = 0,
) -> tuple[int, int] | None:
    """Find the candidate span (starting at or after `start`) whose content
    tokens cover `target_tokens` at or above `MIN_COVERAGE` -- the single-
    fact building block `validate_video00_regression_qa.py`'s
    `required_exact`/`required_order` checks use instead of byte-equal text
    matching, so a rechunked or lightly-reworded required fact is still
    recognized as present. Prefers the earliest, then smallest, matching
    window. Returns None if no window anywhere from `start` onward covers
    it."""
    best: tuple[tuple[int, int, float], int, int] | None = None
    for c_start in range(start, len(candidate_tokens)):
        for wc in range(1, min(MAX_WINDOW, len(candidate_tokens) - c_start) + 1):
            available = frozenset().union(*candidate_tokens[c_start:c_start + wc])
            coverage = _coverage(target_tokens, available)
            if coverage < MIN_COVERAGE:
                continue
            key = (c_start - start, wc, -coverage)
            if best is None or key < best[0]:
                best = (key, c_start, wc)
    if best is None:
        return None
    _, c_start, wc = best
    return c_start, c_start + wc


def align(gold_segments: list[str], candidate_segments: list[str]) -> AlignmentResult:
    """Ordered semantic alignment of `gold_segments` against
    `candidate_segments`, both already in their own intended reading order.
    See module docstring for the exact algorithm and relation vocabulary.
    """
    gold_tokens = [_content_tokens(t) for t in gold_segments]
    candidate_tokens = [_content_tokens(t) for t in candidate_segments]

    rows: list[AlignmentRow] = []
    consumed: set[int] = set()
    g = 0
    c = 0
    while g < len(gold_segments):
        match = _best_match(gold_tokens, candidate_tokens, g, c)
        if match is None:
            rows.append(AlignmentRow(
                gold_span=(g, g + 1), candidate_span=(c, c),
                relation=MISSING, gold_text=gold_segments[g], candidate_text="",
                content_coverage=0.0,
            ))
            g += 1
            continue

        skip, wg, wc, coverage = match
        c_start = c + skip
        relation = EXACT if (wg == 1 and wc == 1) else (COMPOSITE if wc > 1 and wg == 1 else RECHUNKED)
        rows.append(AlignmentRow(
            gold_span=(g, g + wg), candidate_span=(c_start, c_start + wc),
            relation=relation,
            gold_text=" ".join(gold_segments[g:g + wg]),
            candidate_text=" ".join(candidate_segments[c_start:c_start + wc]),
            content_coverage=round(coverage, 4),
        ))
        consumed.update(range(c_start, c_start + wc))
        g += wg
        c = c_start + wc

    unconsumed = [i for i in range(len(candidate_segments)) if i not in consumed]
    matched_rows = [row for row in rows if row.relation != MISSING]
    duplicates: list[int] = []
    extras: list[int] = []
    for i in unconsumed:
        own = candidate_tokens[i]
        is_duplicate = any(
            _coverage(own, frozenset().union(*gold_tokens[row.gold_span[0]:row.gold_span[1]])) >= MIN_COVERAGE
            for row in matched_rows
        )
        (duplicates if is_duplicate else extras).append(i)

    missing_count = sum(1 for row in rows if row.relation == MISSING)
    return AlignmentResult(
        rows=tuple(rows),
        extra_candidate_indices=tuple(extras),
        duplicate_candidate_indices=tuple(duplicates),
        missing_count=missing_count,
    )
