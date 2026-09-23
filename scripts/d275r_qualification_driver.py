"""D-275R Stages 3-22: live qualification of the real, downloaded iPhone
source samples against the actual current D-271/D-272/D-274/D-274E/D-274F
chain, plus a bounded final-render sanity check (Stage 20). Docs-only
gate -- this script imports and calls production code UNCHANGED, never
edits it. Every printed field comes from a real function call against a
real downloaded file; nothing here is fabricated.
"""
import dataclasses
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_normalization_executor as sne
from cutsell_worker import worker_runtime_capability as wrc
from cutsell_worker import worker_job
from cutsell_worker import output_format_qc as ofq
from cutsell_worker import render as render_mod
from cutsell_worker.render_plan import RenderSegment

SAMPLES_DIR = "samples"
OUT_DIR = "artifact"
os.makedirs(OUT_DIR, exist_ok=True)

# Stage 5: the representative qualification set, with the reason each was
# selected (real filenames as returned by the Stage 1 S3 inventory).
SELECTED = [
    {
        "key": "Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4",
        "local": "video00_raw.mp4",
        "reason": "Video00 itself -- the one file this repo already has "
                   "the most independent corroborating context for (named "
                   "throughout CLAUDE.md/D-095 as the canonical RAW). MP4 "
                   "container, smallest of the 'VIDEO-timestamp' family.",
    },
    {
        "key": "Editdna longform validation/copy_9E4975E5-79EF-43EF-9440-5F06AC0A5581.MP4",
        "local": "copy_9E4975E5.MP4",
        "reason": "Explicitly named by the Product Owner in the D-275R "
                   "directive itself as a known corpus member. MP4 "
                   "container, 'copy_<UUID>' iOS Photos/Files export "
                   "filename pattern.",
    },
    {
        "key": "Editdna longform validation/copy_0C08368A-1FA1-4217-B9AB-A05BE613B6A4.MOV",
        "local": "copy_0C08368A.MOV",
        "reason": "Largest .MOV in the corpus (103 MB) -- best real "
                   "candidate for higher-bitrate/resolution/HEVC-style "
                   "content; QuickTime .MOV container diversity vs the "
                   ".MP4 samples above.",
    },
    {
        "key": "Editdna longform validation/copy_767E1C78-4791-4AFD-B6C1-49CEEC5F73EE.MOV",
        "local": "copy_767E1C78.MOV",
        "reason": "Smallest .MOV in the corpus (26.6 MB) -- a second, "
                   "distinct .MOV sample for orientation/rotation "
                   "diversity without re-running the largest file twice.",
    },
]

# Explicitly excluded from selection, with reasons (Stage 5 documentation
# requirement) -- never silently dropped.
EXCLUDED = [
    {
        "key": "Editdna longform validation/5E01F214-A364-4F4B-8F25-D39B1E2B21D2.MP4",
        "reason": "CLAUDE.md-designated Human Gold QA-ONLY oracle file -- "
                  "reserved for editorial benchmarking, never blended into "
                  "engineering format-qualification sampling.",
    },
    {
        "key": "Editdna longform validation/D40F1D43-7391-44D5-8D83-09CB62FBF397.MP4",
        "reason": "CLAUDE.md-designated Cut.ai QA-ONLY oracle file -- same "
                  "reason as above.",
    },
    {
        "key": "Editdna longform validation/v12044gd0000d46k2m7og65re0trr1rg.MP4",
        "reason": "Filename matches a TikTok CDN-style internal video-id "
                  "pattern ('v<digits>g<digits>...'), not the iOS "
                  "'copy_<UUID>' or 'VIDEO-<timestamp>' patterns the rest "
                  "of the corpus shares -- excluded from the iPhone-only "
                  "qualification set pending separate provenance "
                  "confirmation, per this gate's own no-fabricated-"
                  "provenance rule. Not asserted to be non-iPhone, only "
                  "not asserted to BE iPhone without further evidence.",
    },
    {
        "key": "Editdna longform validation/VIDEO-2026-07-30-09-21-35.mp4 and VIDEO-2026-07-30-09-24-13.mp4 and VIDEO-2026-07-30-10-22-46.mp4",
        "reason": "Same 'VIDEO-<timestamp>' family as the one already "
                  "selected; skipped per Stage 5's own "
                  "'do not run every duplicate file unnecessarily' "
                  "instruction -- one representative of this family is "
                  "sufficient, remaining .MOV 'copy_' files are similarly "
                  "skipped for the same reason (7 total copy_*.MOV/MP4 "
                  "files exist; 2 were selected for container+size "
                  "diversity).",
    },
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_to_dict(profile):
    d = dataclasses.asdict(profile)
    return d


def decision_to_dict(decision):
    return {
        "decision": decision.decision,
        "policy_version": decision.policy_version,
        "reason_codes": list(decision.reason_codes),
        "blocking_reasons": list(decision.blocking_reasons),
        "normalization_reasons": list(decision.normalization_reasons),
        "warnings": list(decision.warnings),
        "source_profile_status": decision.source_profile_status,
        "source_format_class": decision.source_format_class,
        "user_facing_error_code": decision.user_facing_error_code,
    }


def main():
    results = []
    runtime_capability = wrc.get_worker_runtime_capability_input()
    capability_diag = wrc.describe_worker_capability_diagnostics()
    print("=== D-275R runtime capability (this runner, real, not fabricated) ===")
    print(json.dumps(capability_diag, indent=2, default=str))

    for entry in SELECTED:
        local_path = os.path.join(SAMPLES_DIR, entry["local"])
        print(f"\n=== D-275R sample: {entry['key']} ===")
        print(f"selection reason: {entry['reason']}")

        if not os.path.exists(local_path):
            print(f"SAMPLE_NOT_DOWNLOADED: {local_path}")
            results.append({"key": entry["key"], "error": "not_downloaded"})
            continue

        original_sha_before = sha256_of(local_path)
        original_size = os.path.getsize(local_path)

        # Stage 3: raw D-271 profile.
        profile = smp.probe_source_media_profile(local_path)
        profile_dict = profile_to_dict(profile)

        # Stage 6: initial D-272 policy.
        initial_decision = sfp.evaluate_source_format_policy(
            profile, runtime_capability=runtime_capability,
        )

        # Stage 8/9: the REAL live worker_job seam -- profile -> policy ->
        # plan -> executor (if NORMALIZE_REQUIRED) -> reprobe -> reevaluate
        # -> format QC, exactly as run_flow_b_job calls it in production.
        out_dir = os.path.join(OUT_DIR, "normalized", entry["local"])
        os.makedirs(out_dir, exist_ok=True)
        source_id = f"d275r-{entry['local']}"
        t0 = time.monotonic()
        resolved_paths, source_format_diagnostics, normalization_diagnostics, blocked_sources = (
            worker_job.resolve_sources_for_editorial_entry(
                {source_id: local_path}, output_directory=out_dir,
            )
        )
        wall_time_sec = time.monotonic() - t0

        original_sha_after = sha256_of(local_path)
        source_preserved = (original_sha_before == original_sha_after)

        resolved_path = resolved_paths.get(source_id)
        resolved_kind = "ACCEPT_ORIGINAL" if resolved_path == local_path else (
            "NORMALIZED" if resolved_path else "BLOCKED"
        )
        normalized_sha = None
        if resolved_kind == "NORMALIZED" and resolved_path and os.path.exists(resolved_path):
            normalized_sha = sha256_of(resolved_path)

        timeout_headroom_sec = None
        if resolved_kind == "NORMALIZED":
            timeout_headroom_sec = round(sne.NORMALIZATION_FFMPEG_TIMEOUT_SEC - wall_time_sec, 2)

        result = {
            "key": entry["key"],
            "selection_reason": entry["reason"],
            "original_size_bytes": original_size,
            "original_sha256_before": original_sha_before,
            "original_sha256_after": original_sha_after,
            "source_preserved": source_preserved,
            "stage3_profile": profile_dict,
            "stage6_initial_policy": decision_to_dict(initial_decision),
            "live_resolution": {
                "resolved_kind": resolved_kind,
                "resolved_path": resolved_path,
                "resolved_path_exists": bool(resolved_path and os.path.exists(resolved_path)),
                "blocked": bool(blocked_sources),
                "source_format_diagnostics": source_format_diagnostics,
                "normalization_diagnostics": normalization_diagnostics,
                "wall_time_sec": round(wall_time_sec, 2),
                "timeout_headroom_sec": timeout_headroom_sec,
                "normalized_sha256": normalized_sha,
            },
        }

        if resolved_kind == "NORMALIZED" and resolved_path and os.path.exists(resolved_path):
            final_profile = smp.probe_source_media_profile(resolved_path)
            final_decision = sfp.evaluate_source_format_policy(
                final_profile, runtime_capability=runtime_capability,
            )
            format_qc = ofq.verify_output_format(final_profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
            result["stage_reprobe_reevaluate"] = {
                "final_profile": profile_to_dict(final_profile),
                "final_decision": decision_to_dict(final_decision),
                "normalized_source_qc_status": format_qc.status,
                "normalized_source_qc_failed_checks": list(format_qc.failed_checks),
            }
            # Stage 22: A/V relation before/after.
            result["av_relation"] = {
                "before_video_duration_sec": profile.duration_sec,
                "after_video_duration_sec": final_profile.duration_sec,
                "before_audio_sample_rate_hz": profile.audio_sample_rate_hz,
                "after_audio_sample_rate_hz": final_profile.audio_sample_rate_hz,
            }

        print(json.dumps(result, indent=2, default=str))
        results.append(result)

    with open(os.path.join(OUT_DIR, "d275r-per-sample-results.json"), "w") as f:
        json.dump({"selected": results, "excluded": EXCLUDED}, f, indent=2, default=str)

    # Stage 20 -- bounded final-render sanity subset: the first TWO samples
    # that reached a resolved canonical path (ACCEPT or NORMALIZED), never
    # a blocked one. Minimal 3-second single-segment timeline, no editorial
    # judgment.
    render_results = []
    bounded_subset = [r for r in results if r.get("live_resolution", {}).get("resolved_path_exists")][:2]
    for r in bounded_subset:
        key = r["key"]
        entry = next(e for e in SELECTED if e["key"] == key)
        resolved_path = r["live_resolution"]["resolved_path"]
        actual_resolved = resolved_path if resolved_path and os.path.exists(resolved_path) else None
        if actual_resolved is None:
            render_results.append({"key": key, "error": "resolved_path_not_locatable_for_render_step"})
            continue

        duration = r["stage3_profile"].get("duration_sec") or 3.0
        clip_end = min(3.0, float(duration))
        segment = RenderSegment(
            clip_id="d275r-sanity-1",
            source_asset_id="d275r-sanity",
            source_path=actual_resolved,
            start=0.0,
            end=clip_end,
        )
        render_out = os.path.join(OUT_DIR, f"render_sanity_{entry['local']}.mp4")
        try:
            render_mod.render_preview([segment], render_out, width=1080, height=1920)
            render_profile = smp.probe_source_media_profile(render_out)
            final_qc = ofq.verify_output_format(render_profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
            render_results.append({
                "key": key,
                "clip_end_sec": clip_end,
                "final_render_qc_status": final_qc.status,
                "final_render_qc_failed_checks": list(final_qc.failed_checks),
            })
        except Exception as exc:  # noqa: BLE001 -- report, never hide
            render_results.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if os.path.exists(render_out):
                os.remove(render_out)  # never uploaded/kept -- QC verdict only

    print("\n=== D-275R Stage 20 -- bounded final-render sanity subset ===")
    print(json.dumps(render_results, indent=2, default=str))
    with open(os.path.join(OUT_DIR, "d275r-render-sanity-results.json"), "w") as f:
        json.dump(render_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
