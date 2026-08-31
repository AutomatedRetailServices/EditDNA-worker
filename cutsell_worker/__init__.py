"""CutSell clean worker package.

Mobile-first Flow B engine. Modules are intentionally small and communicate
through versioned typed contracts defined in ``cutsell_worker.contracts``.
"""

from .commercial_observability import initialize_observability
from .clean_cut_retry import install_clean_cut_contract_recovery
from .recording_process_context import install_recording_process_context_cleanup
from .recording_suffix_trim import install_visual_self_critique_suffix_trim
from .script_consult_trim import install_script_consult_pause_trim
from .delivery_edge_trim import install_delivery_edge_trim
from .interior_performance_break import install_interior_performance_break_split
from .dangling_delivery import install_dangling_delivery_cleanup
from .internal_self_correction import install_internal_self_correction_trim
from .internal_retake_winner import install_internal_retake_winner
from .internal_repeat_trim import install_internal_repeat_trim
from .recording_breaks import install_recording_break_cleanup
from .restart_questions import install_short_restart_question_cleanup
from .frustrated_restart import install_soft_frustration_restart_cleanup
from .micro_restart_cleanup import install_micro_restart_cleanup
from .product_handling_failure import install_product_handling_failure_cleanup
from .micro_self_talk import install_micro_self_talk_cleanup
from .orphan_retry_cleanup import install_orphan_retry_cleanup
from .incomplete_retry_suffix import install_incomplete_retry_suffix_cleanup
from .interstitial_retry_debris import install_interstitial_retry_debris_cleanup
from .trailing_retry_restart import install_trailing_retry_restart_trim
from .merged_self_review import install_merged_self_review_cleanup
from .word_search_attempts import install_word_search_attempt_cleanup
from .recording_meta_continuation import install_recording_meta_continuation_cleanup
from .story_coverage_guard import install_story_coverage_guard
from .superseded_attempt_cleanup import install_superseded_attempt_cleanup
from .lexical_self_correction import install_explicit_lexical_self_correction_cut
from .semantic_fragment_guard import install_semantic_fragment_guard
from .editorial_guardrails_v2 import install_editorial_guardrails_v2
from .complete_retry_identity_guard import install_complete_retry_identity_guard
from .speech_safe_dead_air_guard import install_speech_safe_dead_air_guard
from .incomplete_bridge_retry_authority import install_incomplete_bridge_retry_authority
from .terminal_sentence_boundary_guard import install_terminal_sentence_boundary_guard
from .boundary_retry_tail_guard import install_boundary_retry_tail_guard
# hybrid_retry_completion_integrity, hybrid_story_guard, hybrid_alternate_integrity,
# hybrid_cross_group_retry_integrity, hybrid_failed_continuation_integrity,
# hybrid_retry_winner_authority, hybrid_gold_reconciliation, hybrid_failed_soft_restore,
# hybrid_unavailable_retry_fallback, hybrid_complementary_delivery_guard,
# hybrid_semantic_complementary_rescue, hybrid_semantic_composite_bridge,
# hybrid_composite_best_take, hybrid_semantic_conflict_arbitration, and
# post_selection_complementary_family_stabilizer are no longer installed as a
# monkeypatch chain here (D-023): composite_resolver.py calls their own pure
# functions directly, in the same order, from pipeline.py. Each module's
# install_*() function still exists unchanged for its own monkeypatch-based
# tests, it is just no longer invoked from this file.
from .failed_prefix_completion_rescue import install_failed_prefix_completion_rescue
from .local_retry_grouping import install_local_retry_grouping
from .retry_group_integrity import install_retry_group_integrity
from .attempt_boundary_integrity import install_attempt_boundary_integrity
from .final_sibling_grouping import install_final_sibling_grouping
from .session_grouping_bridge import install_session_grouping_bridge
from .global_session_sibling_bridge import install_global_session_sibling_bridge
from .selection_integrity import install_selection_integrity
from .semantic_best_take_integrity import install_semantic_best_take_integrity
from .final_delivery_integrity import install_final_delivery_integrity
from .terminal_delivery_reconciliation import install_terminal_delivery_reconciliation
from .temporal_word_boundary_integrity import install_temporal_word_boundary_integrity
from .final_draft_retry_integrity import install_final_draft_retry_integrity
from .selected_failed_bridge_integrity import install_selected_failed_bridge_integrity
from .round8_retry_reconciliation import install_round8_retry_reconciliation
from .round9_orphan_prefix_integrity import install_round9_orphan_prefix_integrity
from .round11_semantic_retry_cleanup import install_round11_semantic_retry_cleanup
from .short_bts_process_cleanup import install_short_bts_process_cleanup
from .post_selection_incomplete_bridge_authority import install_post_selection_incomplete_bridge_authority
from .post_selection_internal_retake_trim import install_post_selection_internal_retake_trim
from .final_selection_retry_arbiter import install_final_selection_retry_arbiter
from .selection_boundary_contract import install_selection_freeze, install_boundary_selection_invariant
from .post_selection_edge_only_boundary import install_post_selection_edge_only_boundary
from .post_selection_interior_gap_trim import install_post_selection_interior_gap_trim
from .post_selection_continuity_coalescer import install_post_selection_continuity_coalescer
from .audio_boundary_completion_install import install_audio_boundary_completion

install_clean_cut_contract_recovery()
install_recording_process_context_cleanup()
install_visual_self_critique_suffix_trim()
install_script_consult_pause_trim()
install_delivery_edge_trim()
install_interior_performance_break_split()
install_dangling_delivery_cleanup()
install_internal_self_correction_trim()
install_internal_retake_winner()
install_internal_repeat_trim()
install_recording_break_cleanup()
install_short_restart_question_cleanup()
install_soft_frustration_restart_cleanup()
install_micro_restart_cleanup()
install_product_handling_failure_cleanup()
install_micro_self_talk_cleanup()
install_orphan_retry_cleanup()
install_incomplete_retry_suffix_cleanup()
install_interstitial_retry_debris_cleanup()
install_trailing_retry_restart_trim()
install_merged_self_review_cleanup()
install_word_search_attempt_cleanup()
install_recording_meta_continuation_cleanup()
install_story_coverage_guard()
install_superseded_attempt_cleanup()
install_explicit_lexical_self_correction_cut()
install_semantic_fragment_guard()
install_editorial_guardrails_v2()
install_complete_retry_identity_guard()
install_speech_safe_dead_air_guard()
install_terminal_sentence_boundary_guard()
install_boundary_retry_tail_guard()
install_incomplete_bridge_retry_authority()
install_failed_prefix_completion_rescue()
install_local_retry_grouping()
install_retry_group_integrity()
install_attempt_boundary_integrity()
install_final_sibling_grouping()
install_session_grouping_bridge()
install_global_session_sibling_bridge()
install_selection_integrity()
install_semantic_best_take_integrity()
install_final_delivery_integrity()
install_terminal_delivery_reconciliation()
install_temporal_word_boundary_integrity()
install_final_draft_retry_integrity()
install_selected_failed_bridge_integrity()
install_round8_retry_reconciliation()
install_round9_orphan_prefix_integrity()
install_round11_semantic_retry_cleanup()
install_short_bts_process_cleanup()
install_post_selection_incomplete_bridge_authority()

# ------------------------------ SELECTION PHASE ------------------------------
# Every operation that may change spoken content or membership must execute here.
install_post_selection_internal_retake_trim()
install_final_selection_retry_arbiter()

# Hard semantic phase barrier. Everything after this point is Boundary-only.
install_selection_freeze()

# ------------------------------- BOUNDARY PHASE ------------------------------
# Boundary may adjust timestamps and fragment structure, never semantic membership.
install_post_selection_edge_only_boundary()
install_post_selection_interior_gap_trim()
install_post_selection_continuity_coalescer()
# post_selection_composite_handoff_trim is intentionally not installed: it removes a
# selected spoken fragment after Boundary and therefore violates phase ownership.
install_audio_boundary_completion()

# Last wrapper: refuse any final timeline whose ordered spoken token stream differs from
# the frozen Selection stream. A bad edit fails closed instead of shipping corrupted.
install_boundary_selection_invariant()

# Raw benchmark trigger marker: validate RAW105 near-tied sonography bridge arbitration.
__version__ = "0.1.0"
OBSERVABILITY_STATUS = initialize_observability(service="cutsell-worker")
