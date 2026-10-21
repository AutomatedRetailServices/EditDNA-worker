"""D-050A: canonical identity/provenance minting helpers.

See ``docs/CUTSELL_DECISIONS.md`` D-050 (architecture audit) and D-050A
(this migration) for full context. This module is the SINGLE place every
canonical id in the identity chain is computed, so no two stages can
independently reinvent (and silently disagree about) the same id.

Additive-only, shadow-metadata scope (D-050A): every function here is a
pure, deterministic string-in/string-out helper with no side effects and
no dependency on any editorial decision. Nothing in the current pipeline
reads these ids to make a KEEP/DISCARD, winner, coverage, or freeze
decision yet -- that is deliberately deferred to D-050B/C. Wiring these
ids onto ``CandidateTake``/``DraftClip``/``Claim`` only adds observability;
it must never change ``selected``/``discarded``/``alternates`` membership,
ordering, claim-coverage overrides, or Freeze outcomes. See
``tests/test_cutsell_d050a_canonical_identity.py``'s behavioral-parity
tests, which assert exactly this against the real CleanCutBench chain.

PHYSICAL vs SEMANTIC IDENTITY (D-050A design note, per the D-050A
directive's explicit anti-jitter requirement)
=============================================
Two very different kinds of "identity" are minted below, and mixing them
up is exactly the mistake this module exists to prevent:

- ``source_span_id`` is a PHYSICAL OBSERVATION identity. It answers "which
  exact recorded span is this." It is allowed -- expected -- to be
  sensitive to the ASR engine's exact timestamps, mirroring
  ``source_identity.stable_clip_id``'s own existing, already-proven shape
  (``source_asset_id`` + start/end + text). A few milliseconds of ASR
  jitter between two transcriptions of the identical audio legitimately
  produces a different ``source_span_id`` -- that is correct, not a bug:
  it is a different physical observation of (very nearly) the same event.

- ``attempt_id``, ``realization_id``, and ``semantic_idea_id`` are
  CANONICAL SEMANTIC identities. They answer "which idea/delivery is this
  regardless of exactly how the ASR engine happened to slice the audio
  this run." The D-050A directive is explicit: do NOT derive these from
  ``hash(start_ms, end_ms)`` the way ``source_span_id``/``stable_clip_id``
  legitimately are. Every semantic-id minting function below is a pure
  function of CONTENT (normalized spoken text, casefolded/whitespace-
  collapsed) and STRUCTURAL membership (which spans/candidates/groups
  compose it) -- start/end are never part of the hash input. This is what
  makes "small ASR timing differences do not become semantic identity
  solely because timestamps differ" true by construction rather than by
  convention: the same spoken words fused from the same underlying
  members always mint the same semantic id, even when their exact
  boundary timestamps differ run-to-run.

This module does not yet attempt cross-run identity stability (matching
"the same idea" across two independent runs of the same source) -- that is
D-050's own Phase 9 (Semantic Selection Stability) and is explicitly
out of scope here. D-050A's contribution is narrower and load-bearing for
it: making sure a semantic id, WITHIN one run, never accidentally depends
on a timestamp, so a future cross-run aligner has a real chance of two
runs' ids agreeing whenever their content and structure genuinely match.

ID OWNERSHIP (D-050A directive Section 3 -- one minting owner each)
=====================================================================
- ``source_span_id``   -- take_segmentation.py (ASR/evidence layer), at the
                          point each raw ``CandidateTake`` is first built
                          (including its own internal boundary-fragment
                          repair join, which is itself a small fusion).
- ``attempt_id``        -- attempt_reconstruction.py's ``_merge_attempt``,
                          the one place fused/passthrough attempts are
                          built (also reused, unmodified, by
                          ``_preserve_borderline_subspans``/
                          ``preserved_subspan_candidates``).
- ``realization_id``    -- pipeline.py's ``build_flow_b_draft``, at the
                          point surviving candidates ("kept") are finalized
                          immediately before entering take-grouping --
                          the "candidate/realization normalization" layer
                          the D-050 audit named without its own module.
- ``semantic_idea_id`` /
  ``retry_family_id``   -- pipeline.py's ``build_flow_b_draft``, from the
                          FINAL post-semantic-equivalence group id (the
                          only place ``TakeGroup.group_id``/membership key
                          is minted). D-050A deliberately mints both from
                          the same group key -- see the D-050 audit's own
                          Phase 3 finding that these two concepts are
                          currently conflated in the real architecture;
                          separating them is D-050B/C work, not attempted
                          here. Giving them distinct fields now (rather
                          than one shared field) avoids a second migration
                          later.
- ``canonical_claim_id`` -- semantic_claims.py's ``extract_claims``, at
                          ``Claim`` construction, from claim_type +
                          content_tokens only (never source_clip_id/exact
                          text) -- so two near-duplicate restatements of
                          the same fact from different sibling
                          realizations of one idea share a
                          ``canonical_claim_id`` today, in metadata only;
                          nothing reads this field for coverage decisions
                          yet (that dedup is D-050C's own scoped fix for
                          the D-049 Case B gap this audit already found).
- ``render_fragment_id`` /
  ``parent_realization_id`` -- the two physical-split sites,
                          post_selection_interior_gap_trim.py and
                          human_boundary_polish_v5.py, exactly mirroring
                          the existing D-036 ``render_fragment_id``/
                          ``parent_semantic_clip_id`` pattern. No
                          downstream stage recomputes an upstream id;
                          every consumer of these fields only ever reads
                          what one of the six owners above already wrote.

No function in this module is ever expected to be called from more than
one owning module -- if a second call site starts minting the same kind of
id, that is exactly the duplicated-authority problem D-050 was written to
name, reintroduced.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Mapping, Sequence

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Whitespace-collapsed, casefolded text -- the ONLY text
    representation any semantic-id hash below is ever computed from.
    Deliberately does not strip punctuation/accents: normalizing further
    is a semantic-equivalence-arbiter-tier judgment (take_grouping_
    provider.py already owns that), not an identity-minting one."""
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip().casefold()


def _content_fingerprint(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def mint_source_span_id(source_asset_id: str, start: float, end: float, text: str) -> str:
    """Physical observation identity for one raw ASR-produced span.

    Timestamp-sensitive BY DESIGN (see module docstring) -- this is a
    physical-evidence id, not a semantic one. Mirrors
    ``source_identity.stable_clip_id``'s own existing shape exactly, under
    a distinct prefix so the two ids are never confusable in diagnostics.
    """
    raw = f"{source_asset_id}|{float(start):.3f}|{float(end):.3f}|{_normalize_text(text)}".encode("utf-8")
    return "span_" + hashlib.sha256(raw).hexdigest()[:20]


def mint_attempt_id(member_source_span_ids: Sequence[str]) -> str:
    """AttemptReconstructor's identity for one delivery attempt (fused from
    one or more member spans, or a lone passthrough span).

    Deliberately membership-anchored (the sorted set of member ids), never
    timestamp-anchored: re-fusing the identical set of underlying spans
    always mints the same ``attempt_id`` regardless of the exact boundary
    timing computed for the fused span this run.
    """
    raw = "|".join(sorted(str(value) for value in member_source_span_ids if value)).encode("utf-8")
    return "att_" + hashlib.sha256(raw).hexdigest()[:20]


def mint_realization_id(source_asset_id: str, attempt_id: str | None, text: str) -> str:
    """One specific recorded delivery of an idea -- minted once, in
    pipeline.py, on the COMPLETE candidate pool, before ANY editorial
    stage (clean_cut, provider judgements, hybrid/composite resolution,
    grouping) can keep, discard, or transform a candidate (D-050D1 --
    relocated from its original post-composite-resolution point after an
    audit found every candidate those stages removed never received an
    identity at all).

    Content-anchored (source lineage + attempt lineage + normalized
    spoken text), never raw timestamps: a candidate whose ASR timing
    shifted slightly between runs but whose fused-attempt lineage and
    spoken content did not still mints the same ``realization_id``.
    """
    raw = f"{source_asset_id}|{attempt_id or ''}|{_content_fingerprint(text)}".encode("utf-8")
    return "real_" + hashlib.sha256(raw).hexdigest()[:20]


def mint_semantic_idea_id(group_key: str) -> str:
    """The canonical idea identity for a fully-resolved (post-semantic-
    equivalence) take group. See the module docstring's ID OWNERSHIP
    section for why this is also used, unmodified, as ``retry_family_id``
    in D-050A."""
    return "idea_" + hashlib.sha256(str(group_key).encode("utf-8")).hexdigest()[:20]


def mint_retry_family_id(group_key: str) -> str:
    """D-050A intentionally mints this identically to
    ``mint_semantic_idea_id`` -- see the module docstring. Kept as its own
    function (rather than a bare alias) so a future D-050B/C separation of
    the two concepts only has to change this one function's body, not
    every call site."""
    return mint_semantic_idea_id(group_key)


def mint_canonical_claim_id(claim_type: str, content_tokens: Iterable[str]) -> str:
    """Groups claims by (claim_type, content-token set) only -- explicitly
    NOT by source_clip_id or exact text -- so two independently-sourced,
    differently-worded restatements of the same fact from sibling
    realizations of one merged idea share a ``canonical_claim_id`` in
    metadata today. This is observability only in D-050A: nothing in
    ``claim_coverage_best_take.py`` reads this field yet. Wiring it into
    an actual coverage-dedup decision is D-050C's scoped fix for the D-049
    Case B finding.
    """
    tokens = "|".join(sorted(str(token) for token in content_tokens))
    raw = f"{claim_type}|{tokens}".encode("utf-8")
    return "cclaim_" + hashlib.sha256(raw).hexdigest()[:20]


def build_identity_chain_diagnostics(draft) -> list[dict]:
    """Provider-neutral observability view (D-050A Section 7): for every
    clip in ``draft.selected``, the full identity chain in one place --
    ``source_span_id`` is intentionally omitted here (attempt fusion may
    combine several; the per-attempt member list already lives in
    ``diagnostics.attempt_reconstruction.attempts[*].member_clip_ids`` and
    is not duplicated here) -- without reconstructing it from several
    unrelated diagnostics dicts. Read-only: this never feeds back into any
    decision. Duck-typed on plain attribute access so it accepts a
    ``DraftClip`` or anything shaped like one; a clip minted before D-050A
    (or a clip type predating these fields entirely) simply reports
    ``None`` for fields it never had -- never a crash.
    """
    chain = []
    for clip in getattr(draft, "selected", ()) or ():
        chain.append({
            "clip_id": getattr(clip, "clip_id", None),
            "realization_id": getattr(clip, "realization_id", None),
            "semantic_idea_id": getattr(clip, "semantic_idea_id", None),
            "retry_family_id": getattr(clip, "retry_family_id", None),
            "take_group_id": getattr(clip, "take_group_id", None),
            "render_fragment_id": getattr(clip, "render_fragment_id", None),
            "parent_realization_id": getattr(clip, "parent_realization_id", None),
            "parent_semantic_clip_id": getattr(clip, "parent_semantic_clip_id", None),
        })
    return chain
