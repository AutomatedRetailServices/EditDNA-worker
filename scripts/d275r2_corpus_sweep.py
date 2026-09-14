"""D-275R2: profile the 14 remaining, previously-unsampled real-iPhone
source objects (D-271 profile only -- no normalization), search the
findings for genuinely new format-diversity evidence (VFR, rotation
metadata, HDR, 10-bit, highest resolution, landscape, multi-action),
and run the SAME live D-274F qualification chain D-275R already proved
(resolve_sources_for_editorial_entry) ONLY on the specific new samples
that add evidence -- never a duplicate re-run of a property this corpus
already proved. Docs-only gate: imports and calls production code
UNCHANGED, edits nothing. Every printed field is a real measurement
against a real downloaded file.
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

SAMPLES_DIR = "samples2"
OUT_DIR = "artifact"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Stage 1 -- the complete 18-object manifest (from D-275R's own real S3
# inventory, run 34830943403). The 4 already qualified in D-275R carry
# their own already-proven results inline; the 14 below are profiled
# fresh in this gate.
ALREADY_SAMPLED = {
    "Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4": {
        "size": 38700219, "d275r_result": "ACCEPT (H264 SDR), render QC PASS",
    },
    "Editdna longform validation/copy_9E4975E5-79EF-43EF-9440-5F06AC0A5581.MP4": {
        "size": 68730266, "d275r_result": "NORMALIZE_REQUIRED->NORMALIZED (HEVC SDR), format QC PASS, render QC PASS",
    },
    "Editdna longform validation/copy_0C08368A-1FA1-4217-B9AB-A05BE613B6A4.MOV": {
        "size": 103385722, "d275r_result": "NORMALIZE_REQUIRED->NORMALIZED (HEVC SDR), format QC PASS",
    },
    "Editdna longform validation/copy_767E1C78-4791-4AFD-B6C1-49CEEC5F73EE.MOV": {
        "size": 26614759, "d275r_result": "NORMALIZE_REQUIRED->NORMALIZED (HEVC SDR), format QC PASS",
    },
}

# The 14 remaining, previously-unsampled objects. QA-oracle files and
# the one unconfirmed-provenance filename are included here for PURE
# read-only D-271 format profiling only (a bounded ffprobe call is not
# an editorial exposure) -- never for normalization/qualification
# unless Stage 4 genuinely selects them for a new property, and even
# then the QA-oracle files are excluded from selection by policy (see
# EXCLUDE_FROM_QUALIFICATION below), consistent with D-275R's own
# established boundary and CLAUDE.md's QA-ONLY designation.
REMAINING_14 = {
    "Editdna longform validation/5E01F214-A364-4F4B-8F25-D39B1E2B21D2.MP4": "5E01F214.MP4",
    "Editdna longform validation/D40F1D43-7391-44D5-8D83-09CB62FBF397.MP4": "D40F1D43.MP4",
    "Editdna longform validation/VIDEO-2026-07-30-09-21-35.mp4": "VIDEO_092135.mp4",
    "Editdna longform validation/VIDEO-2026-07-30-09-24-13.mp4": "VIDEO_092413.mp4",
    "Editdna longform validation/VIDEO-2026-07-30-10-22-46.mp4": "VIDEO_102246.mp4",
    "Editdna longform validation/copy_0A524783-376E-4FD4-AE19-F2C722177CDC.MOV": "copy_0A524783.MOV",
    "Editdna longform validation/copy_2E76C007-26DD-48F9-9AF7-41C8849A602D.MP4": "copy_2E76C007.MP4",
    "Editdna longform validation/copy_3EC2186B-4AA9-4449-BACD-52CF82719B74.MP4": "copy_3EC2186B.MP4",
    "Editdna longform validation/copy_69D72849-50AA-45F7-9B13-BA21505530E5.MOV": "copy_69D72849.MOV",
    "Editdna longform validation/copy_7053BB2F-D130-4DC9-81CA-F431B7FF4038.MOV": "copy_7053BB2F.MOV",
    "Editdna longform validation/copy_7A0721D7-BDA7-4EE6-8A72-F7B46F19E716.MOV": "copy_7A0721D7.MOV",
    "Editdna longform validation/copy_82A297CA-95F0-4546-A161-EA0FCD8804E1.MOV": "copy_82A297CA.MOV",
    "Editdna longform validation/copy_CFB5F467-42AC-442A-94BF-ACA8514070E6.MP4": "copy_CFB5F467.MP4",
    "Editdna longform validation/v12044gd0000d46k2m7og65re0trr1rg.MP4": "v12044gd0000.MP4",
}

# Never selected for NEW normalization/qualification runs even if they
# happen to show an interesting property -- QA-ONLY oracle files
# (CLAUDE.md) and the one unconfirmed-device-provenance filename.
EXCLUDE_FROM_QUALIFICATION = {
    "Editdna longform validation/5E01F214-A364-4F4B-8F25-D39B1E2B21D2.MP4":
        "CLAUDE.md-designated Human Gold QA-ONLY oracle file.",
    "Editdna longform validation/D40F1D43-7391-44D5-8D83-09CB62FBF397.MP4":
        "CLAUDE.md-designated Cut.ai QA-ONLY oracle file.",
    "Editdna longform validation/v12044gd0000d46k2m7og65re0trr1rg.MP4":
        "TikTok-CDN-pattern filename, provenance not yet confirmed as iPhone (D-275R Stage 5 exclusion carried forward).",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_to_dict(profile):
    return dataclasses.asdict(profile)


def decision_to_dict(decision):
    return {
        "decision": decision.decision,
        "reason_codes": list(decision.reason_codes),
        "source_format_class": decision.source_format_class,
        "user_facing_error_code": decision.user_facing_error_code,
    }


def diversity_flags(profile):
    """Stage 3 -- the exact real-evidence signals this gate searches for."""
    coded_w, coded_h = profile.coded_width, profile.coded_height
    max_dim = max([d for d in (coded_w, coded_h) if d], default=0)
    is_landscape = bool(coded_w and coded_h and coded_w > coded_h)
    return {
        "is_vfr": profile.vfr_status not in ("CFR", "UNKNOWN", None),
        "has_rotation_metadata": bool(profile.rotation_degrees) and profile.rotation_source != "NONE",
        "is_hdr": profile.hdr_status not in ("SDR", "UNKNOWN", None),
        "is_10bit": bool(profile.bit_depth and profile.bit_depth > 8),
        "is_landscape": is_landscape,
        "max_dimension_px": max_dim,
        "codec": profile.video_codec,
    }


def qualify_one(key, local_path, out_subdir):
    """The SAME live D-274F seam D-275R already proved -- profile ->
    policy -> (if NORMALIZE_REQUIRED) plan -> executor -> reprobe ->
    reevaluate -> format QC, via the real, unchanged
    worker_job.resolve_sources_for_editorial_entry."""
    runtime_capability = wrc.get_worker_runtime_capability_input()
    original_sha_before = sha256_of(local_path)

    profile = smp.probe_source_media_profile(local_path)
    initial_decision = sfp.evaluate_source_format_policy(profile, runtime_capability=runtime_capability)

    out_dir = os.path.join(OUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    source_id = f"d275r2-{out_subdir}"
    t0 = time.monotonic()
    resolved_paths, source_format_diagnostics, normalization_diagnostics, blocked_sources = (
        worker_job.resolve_sources_for_editorial_entry({source_id: local_path}, output_directory=out_dir)
    )
    wall_time_sec = time.monotonic() - t0
    original_sha_after = sha256_of(local_path)

    resolved_path = resolved_paths.get(source_id)
    resolved_kind = "ACCEPT_ORIGINAL" if resolved_path == local_path else ("NORMALIZED" if resolved_path else "BLOCKED")
    normalized_sha = sha256_of(resolved_path) if resolved_kind == "NORMALIZED" and resolved_path and os.path.exists(resolved_path) else None
    timeout_headroom_sec = round(sne.NORMALIZATION_FFMPEG_TIMEOUT_SEC - wall_time_sec, 2) if resolved_kind == "NORMALIZED" else None

    result = {
        "key": key,
        "original_sha256_before": original_sha_before,
        "original_sha256_after": original_sha_after,
        "source_preserved": original_sha_before == original_sha_after,
        "stage3_profile": profile_to_dict(profile),
        "initial_policy": decision_to_dict(initial_decision),
        "resolved_kind": resolved_kind,
        "resolved_path": resolved_path,
        "blocked": bool(blocked_sources),
        "normalization_diagnostics": normalization_diagnostics,
        "wall_time_sec": round(wall_time_sec, 2),
        "timeout_headroom_sec": timeout_headroom_sec,
        "normalized_sha256": normalized_sha,
    }
    if resolved_kind == "NORMALIZED" and resolved_path and os.path.exists(resolved_path):
        final_profile = smp.probe_source_media_profile(resolved_path)
        final_decision = sfp.evaluate_source_format_policy(final_profile, runtime_capability=runtime_capability)
        format_qc = ofq.verify_output_format(final_profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
        result["final_profile"] = profile_to_dict(final_profile)
        result["final_decision"] = decision_to_dict(final_decision)
        result["normalized_source_qc_status"] = format_qc.status
        result["normalized_source_qc_failed_checks"] = list(format_qc.failed_checks)
    return result


def render_sanity(key, resolved_path, duration_hint):
    clip_end = min(3.0, float(duration_hint or 3.0))
    segment = RenderSegment(clip_id="d275r2-sanity", source_asset_id="d275r2-sanity",
                             source_path=resolved_path, start=0.0, end=clip_end)
    render_out = os.path.join(OUT_DIR, "render_sanity_d275r2.mp4")
    try:
        render_mod.render_preview([segment], render_out, width=1080, height=1920)
        render_profile = smp.probe_source_media_profile(render_out)
        final_qc = ofq.verify_output_format(render_profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
        return {"key": key, "clip_end_sec": clip_end, "final_render_qc_status": final_qc.status,
                "final_render_qc_failed_checks": list(final_qc.failed_checks)}
    except Exception as exc:  # noqa: BLE001
        return {"key": key, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if os.path.exists(render_out):
            os.remove(render_out)


def main():
    profiles = {}
    print("=== D-275R2 Stage 2 -- profiling the 14 remaining corpus objects (D-271 only, no normalization) ===")
    for key, local_name in REMAINING_14.items():
        local_path = os.path.join(SAMPLES_DIR, local_name)
        if not os.path.exists(local_path):
            print(f"NOT_DOWNLOADED: {key}")
            profiles[key] = {"error": "not_downloaded"}
            continue
        profile = smp.probe_source_media_profile(local_path)
        pdict = profile_to_dict(profile)
        flags = diversity_flags(profile)
        profiles[key] = {"profile": pdict, "diversity_flags": flags, "size_bytes": os.path.getsize(local_path)}
        print(f"\n--- {key} ---")
        print(json.dumps({"profile_summary": {
            "container": pdict["container_name"], "codec": pdict["video_codec"],
            "coded_wh": [pdict["coded_width"], pdict["coded_height"]],
            "bit_depth": pdict["bit_depth"], "vfr_status": pdict["vfr_status"],
            "hdr_status": pdict["hdr_status"], "rotation_degrees": pdict["rotation_degrees"],
            "rotation_source": pdict["rotation_source"], "duration_sec": pdict["duration_sec"],
        }, "diversity_flags": flags}, indent=2, default=str))
        # NOTE: deliberately NOT deleted here. An earlier version of this
        # script deleted each file immediately after profiling and relied
        # on a "re-download" for any Stage 4 candidate -- but this script
        # has no AWS credentials of its own (they exist only in the shell
        # scope of the workflow's download step, a separate GH Actions
        # step whose plain `export`s do not persist to this step), so
        # that "re-download" was never actually reachable and silently
        # produced zero qualification results. Root-caused via the
        # first real run (34834797093) printing an empty Stage 13 render
        # list. Fix: keep all 14 files on disk (under 1 GB total, well
        # within runner disk headroom) until Stage 4 selection is known,
        # then clean up in one pass below.
        pass

    with open(os.path.join(OUT_DIR, "d275r2-profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2, default=str)

    # Stage 3/4 -- format diversity search + candidate selection. Never
    # select a QA-oracle or unconfirmed-provenance file, never select a
    # file that adds no NEW property this corpus hasn't already proven
    # (H264 SDR portrait and HEVC SDR portrait are already CLOSED by
    # D-275R -- a plain duplicate of either is not reselected here).
    candidates = {}  # category -> key
    valid = {k: v for k, v in profiles.items() if "profile" in v and k not in EXCLUDE_FROM_QUALIFICATION}
    for key, v in valid.items():
        flags = v["diversity_flags"]
        if flags["is_vfr"] and "vfr" not in candidates:
            candidates["vfr"] = key
        if flags["has_rotation_metadata"] and "rotation" not in candidates:
            candidates["rotation"] = key
        if flags["is_hdr"] and "hdr" not in candidates:
            candidates["hdr"] = key
        if flags["is_10bit"] and "10bit" not in candidates:
            candidates["10bit"] = key
        if flags["is_landscape"] and "landscape" not in candidates:
            candidates["landscape"] = key

    # Highest resolution across the WHOLE 18-file corpus (already-sampled
    # max was copy_0C08368A at 896x1920 = 1920px max dim) -- select a new
    # highest only if this sweep's own valid set exceeds that.
    prior_max_dim = 1920
    best_res_key, best_res_val = None, prior_max_dim
    for key, v in valid.items():
        md = v["diversity_flags"]["max_dimension_px"]
        if md and md > best_res_val:
            best_res_key, best_res_val = key, md
    if best_res_key:
        candidates["higher_resolution"] = best_res_key

    print("\n=== D-275R2 Stage 3/4 -- format-diversity search result ===")
    print(json.dumps({"candidates_selected_for_qualification": candidates,
                       "excluded_from_selection": EXCLUDE_FROM_QUALIFICATION,
                       "prior_corpus_max_dimension_px": prior_max_dim}, indent=2, default=str))

    if not candidates:
        print("\nNO NEW FORMAT-DIVERSITY CANDIDATES FOUND in the 14 profiled objects "
              "(excluding QA-oracle + unconfirmed-provenance files). No additional "
              "normalization/qualification run performed -- nothing new to qualify.")
        with open(os.path.join(OUT_DIR, "d275r2-qualification-results.json"), "w") as f:
            json.dump({"candidates": {}, "qualification_results": [], "render_sanity_results": []}, f, indent=2, default=str)
        return

    # Stage 5-10 -- fully qualify ONLY the selected candidate(s) (the
    # file is still present on disk from the download step above -- see
    # the note in the Stage 2 loop for why this script never attempts a
    # standalone "re-download"), never a duplicate re-run of an
    # already-proven property.
    qual_results = []
    render_results = []
    unique_keys = sorted(set(candidates.values()))
    print(f"\n=== D-275R2 fully qualifying {len(unique_keys)} selected candidate(s) ===")
    for key in unique_keys:
        local_name = REMAINING_14[key]
        local_path = os.path.join(SAMPLES_DIR, local_name)
        if not os.path.exists(local_path):
            qual_results.append({"key": key, "error": "candidate_file_unexpectedly_missing_from_disk"})
            print(f"CANDIDATE_FILE_MISSING: {key}")
            continue
        categories = [c for c, k in candidates.items() if k == key]
        print(f"\n--- qualifying {key} (categories: {categories}) ---")
        result = qualify_one(key, local_path, out_subdir=local_name.replace(".", "_"))
        result["diversity_categories"] = categories
        print(json.dumps(result, indent=2, default=str))
        qual_results.append(result)

        if result.get("resolved_path") and os.path.exists(result["resolved_path"]):
            duration_hint = result["stage3_profile"].get("duration_sec")
            rr = render_sanity(key, result["resolved_path"], duration_hint)
            render_results.append(rr)
        os.remove(local_path)

    print("\n=== D-275R2 Stage 13 -- final-render sanity on newly-qualified samples ===")
    print(json.dumps(render_results, indent=2, default=str))

    # Cleanup: remove any un-selected downloaded files still on disk
    # (disk hygiene only, not a correctness requirement on an ephemeral
    # runner).
    for local_name in REMAINING_14.values():
        p = os.path.join(SAMPLES_DIR, local_name)
        if os.path.exists(p):
            os.remove(p)

    with open(os.path.join(OUT_DIR, "d275r2-qualification-results.json"), "w") as f:
        json.dump({"candidates": candidates, "qualification_results": qual_results,
                    "render_sanity_results": render_results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
