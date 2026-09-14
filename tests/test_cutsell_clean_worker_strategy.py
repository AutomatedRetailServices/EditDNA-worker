from cutsell_worker.contracts import CandidateTake, EditStrategy, MediaSignals, SemanticLabel, SemanticRole
from cutsell_worker.strategy import choose_strategy


def _take(clip_id: str, *, face: float, product: float) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=0.0,
        end=2.0,
        text="product demo",
        signals=MediaSignals(
            source_asset_id="src-1",
            start=0.0,
            end=2.0,
            face_visibility=face,
            product_visibility=product,
        ),
    )


def test_faceless_strategy_can_be_detected_from_visual_evidence_without_labels():
    strategy = choose_strategy((), (_take("a", face=0.05, product=0.80),))
    assert strategy == EditStrategy.FACELESS


def test_product_visible_feature_content_is_demo_not_generic_direct_sales():
    takes = (_take("a", face=0.75, product=0.90), _take("b", face=0.70, product=0.85))
    labels = (
        SemanticLabel("a", SemanticRole.FEATURES, 0.90),
        SemanticLabel("b", SemanticRole.BENEFITS, 0.90),
    )
    assert choose_strategy(labels, takes) == EditStrategy.DEMO


def test_storytelling_still_wins_when_story_dominates():
    labels = (
        SemanticLabel("a", SemanticRole.STORY, 0.90),
        SemanticLabel("b", SemanticRole.STORY, 0.90),
        SemanticLabel("c", SemanticRole.BENEFITS, 0.90),
    )
    assert choose_strategy(labels, ()) == EditStrategy.STORYTELLING
