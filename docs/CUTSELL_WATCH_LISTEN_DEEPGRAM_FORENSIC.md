# Deepgram full-engine Watch & Listen forensic

Evidence: run 36067890052, test 01a7acde8243c8410bb6f3d6823515c99f084c14.
Read-only investigation; no new paid request, no engine or authority modification.

## Confirmed active path

- brain_runtime.py selects RunPodLocalWholeVideoProvider and clean_cut_provider=None.
- whole_video_local.py builds a bounded ASR transcript summary with source metadata;
  analyze accepts frame samples but does not inspect their image content.
- Local MediaPipe/OpenCV and audio measurements do run; whole_video_context contains
  motion/reset events. It is incorrect to say no visual/audio processing happened.
- hybrid_google.py builds the semantic cleanup request with a text part only.
  Its compact prompt contains transcripts and computed evidence, not raw AV.
- clean_cut_judge_status: requested=false, available=false, provider=none.
- whole_video_editorial_reasoning, watch_listen_besttake_v2, and
  watch_listen_besttake_guard_authority report disabled. The first is explicitly
  diagnostics-only in its integration module: enabling it cannot fix selection.
- perceptual_watch_listen runs after selection/render, returns UNCERTAIN and
  HUMAN_REVIEW_REQUIRED. Several perceptual capabilities are NOT_IMPLEMENTED.

## Trace of retained preparation speech

Hybrid decisions identify 'How am I supposed to say?' as bts at 0.80 and
'Pull it. Yeah. There you go. I just need a pep top. That’s it?' as bts at 0.75.
Both are kept_fail_open, semantic_delete_recommended=false, and appear in the
final selected list. 'Too many people. Ready? Set.' is bts 0.80 in one window
and failed 0.75 in another, also retained. Thus this is not only a recognition
problem: some preparation speech was classified yet retained.

hybrid_session_cleanup.py requires 0.84 for corroborated/clustered bts deletion
recommendations. It intentionally defers non-mechanical deletion to the
existing authoritative resolver (D-081).

pipeline._semantic_best_take uses a 0.85 floor plus deterministic_unusable
for bts singletons. Otherwise a singleton returns single_member_no_contest.
Offline invocation of the actual function confirmed:

| Label/confidence | Deterministic unusable | Outcome |
| --- | --- | --- |
| bts / 0.80 | true | keep, single_member_no_contest |
| bts / 0.90 | false | keep, single_member_no_contest |
| bts / 0.90 | true | discard, single_bts_unusable |

This establishes the retention mechanism for below-floor singleton labels;
not a complete counterfactual replay of every selected fragment. Some selected
mixed fragments also received conflicting winner/failed labels in overlapping
windows. Do not treat lowering a threshold as a validated general solution.

## Correction boundary

The missing capability is contextual AV confirmation of ambiguous preparation,
mixed takes, and delivery before the existing selection authority freezes the
plan. Supply time-bounded AV evidence to that authority, preserve valid content
inside mixed fragments, and cover legitimate audience-directed questions,
product instructions, and authentic reactions in regression cases. A phrase
blacklist or unconditional deletion of uncertain speech would violate the
preserve-valid-content doctrine. Activating diagnostic flags alone does not
supply this capability.

No fix or quality improvement is claimed. Production unchanged. No new paid run.
