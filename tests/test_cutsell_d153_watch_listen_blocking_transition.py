"""D-153 (Gate 6 correction, real RAW #118 audit, explicit Product Owner
acceptance criterion): System Watch+Listen's own self-reported `gate_mode`/
`blocking` fields said "advisory_v1"/`False` unconditionally, even though
`benchmarks/clean_raw_gate.py` already independently blocked on a real
`perceptual_status == "FAIL"` -- a genuine canon-vs-code discrepancy the
audit's README named explicitly.

The explicit acceptance criterion this correction implements: `blocks_
delivery` is True exactly when at least one capability that was ACTUALLY
EVALUATED reports EVALUATED_FAIL. A capability that is UNCERTAIN, ERROR, or
still NOT_IMPLEMENTED never blocks delivery on its own -- with 4 of 8
capabilities still NOT_IMPLEMENTED today, requiring the full review to
reach REVIEW_PASS before anything could deliver would mean NOTHING could
ever deliver until all four ship, which is not what "block on a real
defect" means. `status` itself is unchanged -- the review still never
silently PASSes while anything is NOT_IMPLEMENTED/UNCERTAIN/ERROR; only
what BLOCKS delivery changes.
"""
from cutsell_worker import perceptual_watch_listen as pwl


def _report(status, findings=()):
    return pwl.CapabilityReport("cap", status, "mp4_measured", tuple(findings))


def test_evaluated_fail_blocks_delivery():
    review = pwl.PerceptualReview(
        status=pwl.REVIEW_FAIL, gate_mode=pwl.GATE_MODE_BLOCKING_V1_EVALUATED_FAIL_ONLY,
        capabilities=(_report(pwl.EVALUATED_FAIL),),
    )
    assert review.blocks_delivery is True
    assert review.as_dict()["blocking"] is True


def test_not_implemented_alone_never_blocks_delivery():
    """The core acceptance criterion: with real capabilities still
    NOT_IMPLEMENTED, a review carrying ONLY that status (and otherwise
    clean) must never block -- only a real EVALUATED_FAIL does."""
    review = pwl.PerceptualReview(
        status=pwl.REVIEW_UNCERTAIN, gate_mode=pwl.GATE_MODE_BLOCKING_V1_EVALUATED_FAIL_ONLY,
        capabilities=(
            _report(pwl.EVALUATED_PASS),
            _report(pwl.NOT_IMPLEMENTED),
            _report(pwl.NOT_IMPLEMENTED),
            _report(pwl.NOT_IMPLEMENTED),
            _report(pwl.NOT_IMPLEMENTED),
        ),
    )
    assert review.blocks_delivery is False
    assert review.as_dict()["blocking"] is False
    # `status` is still honestly non-PASS -- this correction never changes
    # what counts as a clean review, only what blocks delivery.
    assert review.status != pwl.REVIEW_PASS


def test_uncertain_alone_never_blocks_delivery():
    review = pwl.PerceptualReview(
        status=pwl.REVIEW_UNCERTAIN, gate_mode=pwl.GATE_MODE_BLOCKING_V1_EVALUATED_FAIL_ONLY,
        capabilities=(_report(pwl.UNCERTAIN),),
    )
    assert review.blocks_delivery is False


def test_error_alone_never_blocks_delivery():
    review = pwl.PerceptualReview(
        status=pwl.REVIEW_UNCERTAIN, gate_mode=pwl.GATE_MODE_BLOCKING_V1_EVALUATED_FAIL_ONLY,
        capabilities=(_report(pwl.ERROR),),
    )
    assert review.blocks_delivery is False


def test_a_real_fail_still_blocks_even_alongside_not_implemented_capabilities():
    """Mixed-evidence guard: NOT_IMPLEMENTED capabilities never mask a real
    EVALUATED_FAIL from blocking delivery."""
    review = pwl.PerceptualReview(
        status=pwl.REVIEW_FAIL, gate_mode=pwl.GATE_MODE_BLOCKING_V1_EVALUATED_FAIL_ONLY,
        capabilities=(
            _report(pwl.EVALUATED_FAIL),
            _report(pwl.NOT_IMPLEMENTED),
            _report(pwl.NOT_IMPLEMENTED),
            _report(pwl.NOT_IMPLEMENTED),
            _report(pwl.NOT_IMPLEMENTED),
        ),
    )
    assert review.blocks_delivery is True


def test_all_evaluated_pass_never_blocks():
    review = pwl.PerceptualReview(
        status=pwl.REVIEW_PASS, gate_mode=pwl.GATE_MODE_BLOCKING_V1_EVALUATED_FAIL_ONLY,
        capabilities=(_report(pwl.EVALUATED_PASS), _report(pwl.EVALUATED_PASS)),
    )
    assert review.blocks_delivery is False
    assert review.status == pwl.REVIEW_PASS


def test_gate_mode_constant_is_the_new_blocking_identifier():
    # `review_rendered_candidate` itself (which stamps every real review
    # with this constant) is exercised end-to-end in
    # test_cutsell_d097_perceptual_watch_listen_and_clean_raw_gate.py.
    assert pwl.GATE_MODE_BLOCKING_V1_EVALUATED_FAIL_ONLY == "blocking_v1_evaluated_fail_only"
