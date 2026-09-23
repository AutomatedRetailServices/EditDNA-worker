"""D-153/D-154 (Gate 6 corrections, real RAW #118 audit, explicit Product
Owner acceptance criterion): System Watch+Listen's own self-reported
`gate_mode`/`blocking` fields said "advisory_v1"/`False` unconditionally,
even though `benchmarks/clean_raw_gate.py` already independently blocked on
a real `perceptual_status == "FAIL"` -- a genuine canon-vs-code discrepancy
the audit's README named explicitly.

D-153 first replaced the hardcoded advisory fields with a single boolean
`blocks_delivery`. That was corrected again (D-154, this file) into an
EXPLICIT 4-state status -- a bare boolean cannot distinguish "a confirmed
defect to fix" from "evidence a human still needs to look at", and D-153
also wrongly treated ERROR the same as UNCERTAIN/NOT_IMPLEMENTED (an error
means the measurement never ran at all, which is not merely "uncertain").

The corrected acceptance criterion `PerceptualReview.watch_listen_status`
implements:
- EVALUATED_FAIL on any capability -> BLOCKED
- ERROR on any capability -> BLOCKED too
- UNCERTAIN or NOT_IMPLEMENTED (nothing BLOCKED) -> HUMAN_REVIEW_REQUIRED
  (the render is kept for a human to review; SYSTEM_PASS/Ready/automatic
  delivery are withheld)
- SYSTEM_PASS only when every capability in the v1 acceptance set is
  implemented, evaluated, AND EVALUATED_PASS
- HUMAN_APPROVED is NEVER computed automatically -- only
  `apply_human_watch_listen_approval()`, given an explicit human decision,
  can reach it, and only by promoting HUMAN_REVIEW_REQUIRED or SYSTEM_PASS
  (never BLOCKED).
"""
from cutsell_worker import perceptual_watch_listen as pwl


def _report(status, findings=()):
    return pwl.CapabilityReport("cap", status, "mp4_measured", tuple(findings))


def _review(*capabilities):
    return pwl.PerceptualReview(
        status=pwl.overall_status(capabilities),
        gate_mode=pwl.GATE_MODE_STATE_MACHINE_V1,
        capabilities=capabilities,
    )


# ---------------------------------------------------------------------------
# 1. EVALUATED_FAIL
# ---------------------------------------------------------------------------

def test_evaluated_fail_is_blocked():
    review = _review(_report(pwl.EVALUATED_FAIL))
    assert review.watch_listen_status == pwl.WATCH_LISTEN_BLOCKED
    assert review.has_confirmed_blocking_defect is True
    assert review.allows_automatic_delivery is False
    payload = review.as_dict()
    assert payload["watch_listen_status"] == pwl.WATCH_LISTEN_BLOCKED
    assert payload["has_confirmed_blocking_defect"] is True
    assert payload["allows_automatic_delivery"] is False


def test_evaluated_fail_still_blocks_alongside_not_implemented_capabilities():
    """Mixed-evidence guard: NOT_IMPLEMENTED capabilities never mask a real
    EVALUATED_FAIL from BLOCKED."""
    review = _review(
        _report(pwl.EVALUATED_FAIL),
        _report(pwl.NOT_IMPLEMENTED), _report(pwl.NOT_IMPLEMENTED),
        _report(pwl.NOT_IMPLEMENTED), _report(pwl.NOT_IMPLEMENTED),
    )
    assert review.watch_listen_status == pwl.WATCH_LISTEN_BLOCKED


# ---------------------------------------------------------------------------
# 2. ERROR
# ---------------------------------------------------------------------------

def test_error_is_blocked_not_merely_uncertain():
    """D-154's own correction of D-153: ERROR means the measurement itself
    never ran -- there is no reliable evidence at all, which is at least
    as unsafe as a confirmed FAIL, never merely "uncertain"."""
    review = _review(_report(pwl.ERROR))
    assert review.watch_listen_status == pwl.WATCH_LISTEN_BLOCKED
    assert review.has_confirmed_blocking_defect is True
    assert review.allows_automatic_delivery is False


def test_error_alongside_evaluated_pass_still_blocks():
    review = _review(_report(pwl.EVALUATED_PASS), _report(pwl.ERROR))
    assert review.watch_listen_status == pwl.WATCH_LISTEN_BLOCKED


# ---------------------------------------------------------------------------
# 3. UNCERTAIN
# ---------------------------------------------------------------------------

def test_uncertain_alone_requires_human_review_not_blocked():
    review = _review(_report(pwl.UNCERTAIN))
    assert review.watch_listen_status == pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert review.has_confirmed_blocking_defect is False
    assert review.allows_automatic_delivery is False
    payload = review.as_dict()
    assert payload["watch_listen_status"] == pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert payload["has_confirmed_blocking_defect"] is False
    assert payload["allows_automatic_delivery"] is False
    assert payload["human_watch_listen_required"] is True


# ---------------------------------------------------------------------------
# 4. NOT_IMPLEMENTED
# ---------------------------------------------------------------------------

def test_not_implemented_alone_requires_human_review_not_blocked():
    """The core acceptance criterion: with real capabilities still
    NOT_IMPLEMENTED (4 of 8, today), a review carrying ONLY that status
    (and otherwise clean) must require human review, never BLOCKED and
    never SYSTEM_PASS."""
    review = _review(
        _report(pwl.EVALUATED_PASS),
        _report(pwl.NOT_IMPLEMENTED), _report(pwl.NOT_IMPLEMENTED),
        _report(pwl.NOT_IMPLEMENTED), _report(pwl.NOT_IMPLEMENTED),
    )
    assert review.watch_listen_status == pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert review.has_confirmed_blocking_defect is False
    assert review.allows_automatic_delivery is False
    # `status` is still honestly non-PASS -- this never changes what
    # counts as a clean review, only what blocks delivery vs. what merely
    # requires a human to look.
    assert review.status != pwl.REVIEW_PASS


# ---------------------------------------------------------------------------
# 5. All capabilities approved (EVALUATED_PASS) -> SYSTEM_PASS, and the
#    HUMAN_APPROVED promotion path
# ---------------------------------------------------------------------------

def test_all_capabilities_evaluated_pass_reaches_system_pass():
    review = _review(_report(pwl.EVALUATED_PASS), _report(pwl.EVALUATED_PASS))
    assert review.watch_listen_status == pwl.WATCH_LISTEN_SYSTEM_PASS
    assert review.has_confirmed_blocking_defect is False
    assert review.allows_automatic_delivery is True
    assert review.status == pwl.REVIEW_PASS


def test_human_approval_promotes_system_pass_to_human_approved():
    review = _review(_report(pwl.EVALUATED_PASS))
    assert pwl.apply_human_watch_listen_approval(review, approved=True) == pwl.WATCH_LISTEN_HUMAN_APPROVED


def test_human_approval_promotes_human_review_required_to_human_approved():
    review = _review(_report(pwl.NOT_IMPLEMENTED))
    assert pwl.apply_human_watch_listen_approval(review, approved=True) == pwl.WATCH_LISTEN_HUMAN_APPROVED


def test_human_approval_never_promotes_a_blocked_review():
    """A confirmed defect is a root-authority fix or a re-render, never
    something a human sign-off launders through this gate."""
    review = _review(_report(pwl.EVALUATED_FAIL))
    assert pwl.apply_human_watch_listen_approval(review, approved=True) == pwl.WATCH_LISTEN_BLOCKED


def test_approval_false_never_changes_the_automated_status():
    for capability_status in (pwl.EVALUATED_PASS, pwl.NOT_IMPLEMENTED, pwl.EVALUATED_FAIL):
        review = _review(_report(capability_status))
        assert pwl.apply_human_watch_listen_approval(review, approved=False) == review.watch_listen_status


def test_gate_mode_constant_is_the_state_machine_identifier():
    # `review_rendered_candidate` itself (which stamps every real review
    # with this constant) is exercised end-to-end in
    # test_cutsell_d097_perceptual_watch_listen_and_clean_raw_gate.py.
    assert pwl.GATE_MODE_STATE_MACHINE_V1 == "state_machine_v1_blocked_human_review_system_pass"


def test_watch_listen_statuses_tuple_names_all_four_states():
    assert set(pwl.WATCH_LISTEN_STATUSES) == {
        pwl.WATCH_LISTEN_BLOCKED, pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        pwl.WATCH_LISTEN_SYSTEM_PASS, pwl.WATCH_LISTEN_HUMAN_APPROVED,
    }


# ---------------------------------------------------------------------------
# D-155 (independent audit): a review with zero capabilities must never
# read as SYSTEM_PASS -- "a review without capabilities cannot become
# SYSTEM_PASS" -- there is no evidence to grant automatic delivery on.
# ---------------------------------------------------------------------------

def test_empty_capabilities_requires_human_review_never_system_pass():
    review = _review()
    assert review.watch_listen_status == pwl.WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert review.has_confirmed_blocking_defect is False
    assert review.allows_automatic_delivery is False
