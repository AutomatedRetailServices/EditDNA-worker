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
from .hybrid_retry_completion_integrity import install_hybrid_retry_completion_integrity
from .hybrid_story_guard import install_hybrid_story_coverage_guard
from .hybrid_alternate_integrity import install_hybrid_alternate_integrity
from .hybrid_cross_group_retry_integrity import install_hybrid_cross_group_retry_integrity
from .hybrid_failed_continuation_integrity import install_hybrid_failed_continuation_integrity
from .hybrid_retry_winner_authority import install_hybrid_retry_winner_authority
from .hybrid_gold_reconciliation import install_hybrid_gold_reconciliation
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
from .hybrid_failed_soft_restore import install_hybrid_failed_soft_restore
from .hybrid_unavailable_retry_fallback import install_hybrid_unavailable_retry_fallback
from .hybrid_complementary_delivery_guard import install_hybrid_complementary_delivery_guard
from .hybrid_semantic_complementary_rescue import install_hybrid_semantic_complementary_rescue
from .hybrid_semantic_composite_bridge import install_hybrid_semantic_composite_bridge
from .hybrid_composite_best_take import install_hybrid_composite_best_take
from .hybrid_performance_retry_restore_guard import install_hybrid_performance_retry_restore_guard
from .post_selection_incomplete_bridge_authority import install_post_selection_incomplete_bridge_authority
from .post_selection_interior_gap_trim import install_post_selection_interior_gap_trim
from .post_selection_internal_retake_trim import install_post_selection_internal_retake_trim
from .post_selection_continuity_coalescer import install_post_selection_continuity_coalescer
from .post_selection_composite_handoff_trim import install_post_selection_composite_handoff_trim
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
install_hybrid_retry_completion_integrity()
install_hybrid_story_coverage_guard()
install_hybrid_alternate_integrity()
install_hybrid_cross_group_retry_integrity()
# Cross-group may otherwise delete the later clean take as lexical coverage of an
# earlier take. Reinstall the physical incomplete-bridge authority after it so the
# complete -> incomplete reset -> complete retake pattern owns final retry direction.
install_incomplete_bridge_retry_authority()
install_hybrid_failed_continuation_integrity()
install_hybrid_retry_winner_authority()
install_hybrid_gold_reconciliation()
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
install_hybrid_failed_soft_restore()
# Last Hybrid fallback for later clean deliveries after a window failure.
install_hybrid_unavailable_retry_fallback()
# Final Hybrid authority: restore complete complementary sub-deliveries with unique
# audience-facing tails, and when Hybrid is unavailable suppress only an immediate
# undecided incomplete restart already delivered completely just before it.
install_hybrid_complementary_delivery_guard()
# A reset-backed full alternate is not redundant merely because it overlaps a winner.
# Preserve and split it when it carries material unique audience-facing information.
install_hybrid_semantic_complementary_rescue()
# Bridge semantic rescues into Composite Best Take, while revoking strong same-opening
# retries that are alternate deliveries rather than complementary information.
install_hybrid_semantic_composite_bridge()
# Composite authority runs after complementary recovery. It can rescue a complete
# performance-only deletion with unique information, combine complementary sub-deliveries
# instead of one monolithic retry, and split those deliveries before Best Take.
install_hybrid_composite_best_take()
# Composite may still restore an earlier semantically failed take when reset evidence is
# treated as performance-only. Revoke that restore when it shares the same strong opening
# as a later authoritative winner in the same source.
install_hybrid_performance_retry_restore_guard()
# Final physical authority: after every semantic/grouping pass, a proven consecutive
# complete -> rejected incomplete bridge -> complete retry may promote the discarded
# clean retry without relying on the LLM's inconsistent failed/winner label.
install_post_selection_incomplete_bridge_authority()
# Best Take is now stable, so selected logical clips may safely split at speech-free
# multimodal performance resets without invalidating selection identity.
install_post_selection_interior_gap_trim()
# Spoken internal attempts may only yield when a later clean retake repeats the opening,
# covers the earlier audience-facing content, and preserves all numbers/negations.
install_post_selection_internal_retake_trim()
# Over-segmented fragments from the same source should not manufacture jump cuts when
# only a tiny natural gap separates them and no retry/reset evidence exists there.
install_post_selection_continuity_coalescer()
# After interior splits/coalescing, trim only a redundant final sibling when a later
# selected delivery takes over while earlier siblings preserve unique information.
install_post_selection_composite_handoff_trim()
install_audio_boundary_completion()

# Keep this bootstrap path in the raw Video00 benchmark trigger set; touching this file
# intentionally retriggers the exact-head raw benchmark when editorial guards change.
# Raw benchmark trigger marker: continuity coalescer installed.
# Raw benchmark trigger marker: validate covered incomplete retry suppression.
# Raw benchmark trigger marker: validate repeated-opening unique-tail preservation.
# Raw benchmark trigger marker: validate complementary Hybrid deliveries.
# Raw benchmark trigger marker: tuned unavailable-prior restart fallback.
# Raw benchmark trigger marker: final Hybrid wrapper bound into pipeline.
# Raw benchmark trigger marker: composite Best Take authority installed.
# Raw benchmark trigger marker: semantic complementary full-alternate rescue installed.
# Raw benchmark trigger marker: post-selection composite handoff trim installed.
# Raw benchmark trigger marker: semantic rescue Composite bridge installed.
# Raw benchmark trigger marker: performance retry restore guard installed.
__version__ = "0.1.0"
OBSERVABILITY_STATUS = initialize_observability(service="cutsell-worker")
