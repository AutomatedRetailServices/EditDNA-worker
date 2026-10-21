"""Production check after deploying a0ab25c (Mix + Plan my week). Writes ONLY under a test user (never the creator's
library, never deletes): (1) one of her real recordings uploaded and cut by the deployed worker -> Ready, source, posting
caption, splits for the Gold score, compared with her existing cut; (2) her products / brands / videos (read-only) for the
duplicate review; (3) a real Mix (3 hooks + 2 bodies + 2 benefits + 2 CTAs) started through the same route code the
server runs, processed by the deployed worker; joins, loudness and caption timing measured on the delivered videos;
Haiku vs Sonnet on the same clips. Report encrypted (no keys, no media)."""
import json
import os
import time

import modal

app = modal.App("cutsell-prod-verify")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .pip_install("fastapi", "pydantic", "requests", "numpy")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir("cutsell_app", remote_path="/root/cutsell_app", ignore=["**/__pycache__/**", "**/*.pyc"])
)
REAL_USER = "1625a6e9"
TEST_USER = "cutsell-prod-verify-20261021"
GOLD_N01_SEC = 103.333333
MODAL_RATE = 2 * 0.0000131 + 4 * 0.00000222        # the worker: 2 cores, 4 GiB


@app.function(image=image, secrets=[secret], timeout=1750, cpu=2.0, memory=4096)
def probe():
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path

    import requests
    from redis import Redis
    from rq.job import Job

    t_all = time.time()
    report = {"errors": []}
    r = Redis.from_url(os.environ["REDIS_URL"])
    from cutsell_worker import post_plan, product_catalog as catalog, project_store, mix_store
    from cutsell_worker import mix_engine as M
    from cutsell_worker.draft_store import get_draft_snapshot
    from cutsell_worker.storage import download_source
    from cutsell_worker.uploads import create_presigned_upload
    from cutsell_worker.simple_engine import asr, engine as E
    from cutsell_worker.post_captions import spoken_text
    work = Path(tempfile.mkdtemp())

    def drain():
        try:
            modal.Function.from_name("cutsell-worker", "drain_queue").spawn()
            return "spawned"
        except Exception as exc:
            return f"spawn_failed:{exc.__class__.__name__}"

    def job_seconds(job_id):
        try:
            j = Job.fetch(job_id, connection=r)
            if j.started_at and j.ended_at:
                return round((j.ended_at - j.started_at).total_seconds(), 1), j.get_status()
            return None, j.get_status()
        except Exception as exc:
            return None, f"err:{exc.__class__.__name__}"

    def upload(local, project_id, name, ctype):
        size = Path(local).stat().st_size
        signed = create_presigned_upload(project_id=project_id, user_id=TEST_USER, original_name=name, content_type=ctype,
                                         size_bytes=size)
        with open(local, "rb") as f:
            if signed["method"] == "PUT":
                resp = requests.put(signed["upload_url"], data=f, headers=signed["headers"], timeout=300)
            else:
                resp = requests.post(signed["upload_url"], data=signed["fields"], files={"file": (name, f, ctype)},
                                     timeout=300)
        resp.raise_for_status()
        return signed["source_uri"]

    # ---------------------------------------------------------------- the creator (read-only)
    user = None
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        snap = json.loads(r.get(key))
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
            user = snap["user_id"]
            break
    data = catalog.load(user, client=r)
    projects = project_store.list_projects(user_id=user, limit=500, client=r)
    ready = post_plan.candidates(user, projects, client=r)
    ready_ids = {v["project_id"] for v in ready}
    drafts = {}
    for p in projects:
        try:
            drafts[p["project_id"]] = get_draft_snapshot(user_id=user, project_id=p["project_id"], client=r)
        except Exception:
            pass

    # ---- (2) products, brands, videos
    brands = {bid: b.get("name") for bid, b in data["brands"].items()}
    rows = {}
    for pid, prod in data["products"].items():
        rows[pid] = {"product_id": pid, "name": prod.get("name"), "brand": brands.get(prod.get("brand_id")),
                     "category": prod.get("category"), "videos": [], "ready": 0}
    for p in projects:
        entry = data["videos"].get(p["project_id"]) or {}
        prod = entry.get("product_id")
        snap = drafts.get(p["project_id"]) or {}
        src = (snap.get("sources") or p.get("sources") or [{}])
        draft = snap.get("draft") or {}
        item = {"project_id": p["project_id"][-6:], "title": p.get("title"), "state": p.get("state"),
                "created_at": p.get("created_at"), "ready": p["project_id"] in ready_ids,
                "mix": bool(entry.get("mix_id")), "sources": len(src),
                "source_sec": [round(float(s.get("duration_sec") or 0), 1) for s in src],
                "recording": [hashlib.sha256(str(s.get("uri") or "").encode()).hexdigest()[:6] for s in src],
                "spoken": (spoken_text(draft) or "")[:160], "tag_raw": {k: entry.get(k) for k in ("detected", "label",
                                                                                                  "tag", "source")
                                                                       if entry.get(k)}}
        if prod in rows:
            rows[prod]["videos"].append(item)
            rows[prod]["ready"] += item["ready"]
        else:
            rows.setdefault("_none", {"name": None, "brand": None, "videos": [], "ready": 0})["videos"].append(item)
            rows["_none"]["ready"] += item["ready"]
    report["products"] = list(rows.values())
    report["catalog_counts"] = {"brands": len(data["brands"]), "products": len(data["products"]),
                                "videos_tagged": len(data["videos"]), "projects": len(projects), "ready": len(ready)}

    # every recording of hers once (by source duration), with its words density
    every, seen = [], set()
    product_of = {pid: v.get("product_id") for pid, v in data["videos"].items() if v.get("product_id")}
    for pid, snap in drafts.items():
        srcs = snap.get("sources") or []
        if len(srcs) != 1 or not srcs[0].get("uri"):
            continue
        dur = round(float(srcs[0].get("duration_sec") or 0), 1)
        if not dur or dur in seen:
            continue
        seen.add(dur)
        draft = snap.get("draft") or {}
        words = sum(len(c.get("words") or []) for c in (draft.get("selected") or []) + (draft.get("alternates") or []))
        splits = (((draft.get("diagnostics") or {}).get("simple_engine") or {}).get("sources") or [{}])[0].get("splits")
        every.append({"uri": srcs[0]["uri"], "dur": dur, "name": srcs[0].get("original_name") or "video.mov",
                      "product": product_of.get(pid), "density": words / dur, "project_id": pid, "splits": splits,
                      "engine": ((draft.get("diagnostics") or {}).get("engine_version"))})

    # ---------------------------------------------------------------- (1) a real cut in production
    try:
        target = min(every, key=lambda v: abs(v["dur"] - GOLD_N01_SEC))
        report["cut_source"] = {"sec": target["dur"], "is_gold_N01": abs(target["dur"] - GOLD_N01_SEC) < 0.6,
                                "her_cut_engine": target["engine"], "her_splits": target["splits"]}
        local = download_source(target["uri"], str(work / "cut_src.mov"))
        proj = project_store.create_project(user_id=TEST_USER, title="Prod check · cut", client=r)
        pid = proj["project_id"]
        uri = upload(local, pid, "cut_src.mov", "video/quicktime")
        from cutsell_worker.queueing import enqueue_flow_b
        from cutsell_worker.source_identity import stable_source_id
        from cutsell_worker.worker_wake import wake_worker
        t0 = time.time()
        sub = enqueue_flow_b({"project_id": pid, "user_id": TEST_USER, "language_hint": None, "audio_overlap": False,
                              "auto_zoom_enabled": True,
                              "sources": [{"source_asset_id": stable_source_id(pid, 0, "cut_src.mov"),
                                           "original_name": "cut_src.mov", "source_order": 0,
                                           "duration_sec": target["dur"], "uri": uri, "metadata": {}}]})
        report["cut_wake"] = {"api_wake": wake_worker().get("status"), "drain": drain()}
        state = None
        while time.time() - t0 < 900:
            state = project_store.get_project(user_id=TEST_USER, project_id=pid, client=r).get("state")
            if state in ("draft_ready", "failed", "error"):
                break
            time.sleep(5)
        snap = get_draft_snapshot(user_id=TEST_USER, project_id=pid, client=r) if state == "draft_ready" else {}
        draft = snap.get("draft") or {}
        se = (draft.get("diagnostics") or {}).get("simple_engine") or {}
        src0 = (se.get("sources") or [{}])[0]
        entry = post_plan.video(TEST_USER, pid, client=r)
        secs, status = job_seconds(sub.job_id)
        report["cut"] = {"state": state, "seconds_to_ready": round(time.time() - t0, 1), "job_seconds": secs,
                         "job_status": str(status), "engine_version": (draft.get("diagnostics") or {}).get("engine_version"),
                         "model": se.get("model"), "cut_sec": se.get("cut_duration_sec"), "splits": src0.get("splits"),
                         "usage": src0.get("usage"), "plan_entry": entry,
                         "selected_text": " ".join(c.get("text") or "" for c in draft.get("selected") or [])[:1500]}
    except Exception as exc:
        report["errors"].append(f"cut: {exc.__class__.__name__}: {str(exc)[:300]}")

    # ---------------------------------------------------------------- (3) a real Mix in production
    try:
        by_product = {}
        for v in every:
            if v["product"] and 15 <= v["dur"] <= 180:
                by_product.setdefault(v["product"], []).append(v)
        best = max((sorted(vs, key=lambda v: -v["density"])[:2] for vs in by_product.values() if len(vs) >= 2),
                   key=lambda pair: min(v["density"] for v in pair))
        vids = sorted(best, key=lambda v: -v["dur"])
        prod = data["products"][vids[0]["product"]]
        report["mix_product"] = {"name": prod.get("name"), "brand": brands.get(prod.get("brand_id")),
                                 "recordings_sec": [v["dur"] for v in vids]}
        for i, v in enumerate(vids):
            v["local"] = download_source(v["uri"], str(work / f"m{i}.mov"))
            heard = asr.transcribe_words(asr.extract_audio(v["local"], str(work / f"m{i}.mp3")))
            v["words"] = [w for w in heard["words"] if w["e"] > w["s"]]
        report["deepgram_setup_sec"] = round(sum(v["dur"] for v in vids), 1)

        def chunks(v, sizes):
            units = E.lines_of(v["words"])
            out, i = [], 0
            for lo, hi in sizes:
                a = b = None
                while i < len(units):
                    s0, s1 = v["words"][units[i][0]]["s"], v["words"][units[i][-1]]["e"]
                    if a is not None and s1 - a > hi:
                        break
                    a = s0 if a is None else a
                    b = s1
                    i += 1
                    if b - a >= lo:
                        break
                if a is None:
                    break
                out.append((max(0.0, a - 0.4), min(v["dur"], b + 0.4)))
            return out

        def last(v):
            units = E.lines_of(v["words"])
            a = v["words"][units[-1][0]]["s"]
            for u in reversed(units[:-1]):
                if v["words"][units[-1][-1]]["e"] - a >= 3:
                    break
                a = v["words"][u[0]]["s"]
            return max(0.0, a - 0.4), min(v["dur"], v["words"][-1]["e"] + 0.4)

        A, B = vids[0], vids[-1]
        ca = chunks(A, [(3, 8), (3, 8), (6, 14), (5, 12)])
        cb = chunks(B, [(3, 8), (6, 14), (5, 12)])
        plan_src = {"Hook": [(A, ca[0]), (B, cb[0]), (A, ca[1])], "Body": [(A, ca[2]), (B, cb[1])],
                    "Benefit": [(A, ca[3]), (B, cb[2])], "CTA": [(A, last(A)), (B, last(B))]}
        mix = project_store.create_project(user_id=TEST_USER, title="Mix", client=r)
        mix_id = mix["project_id"]
        sections, voices = [], []
        for name, spans in plan_src.items():
            items = []
            for ci, (v, (a, b)) in enumerate(spans):
                cid = f"{name[:2].lower()}{ci}"
                clip, voice = work / f"{cid}.mov", work / f"{cid}_voice.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", v["local"],
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "160k",
                                str(clip)], check=True)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vn", "-ac", "1", "-c:a", "aac", "-b:a",
                                "64k", str(voice)], check=True)
                items.append({"uri": upload(clip, mix_id, f"{cid}.mov", "video/quicktime"),
                              "audio_uri": upload(voice, mix_id, f"{cid}_voice.mp4", "video/mp4"),
                              "original_name": f"{cid}.mov", "duration_sec": round(b - a, 2)})
                voices.append({"section": name, "file": str(voice), "sec": round(b - a, 2)})
            sections.append({"name": name, "clips": items})
        report["mix_clips_sec"] = {s["name"]: [c["duration_sec"] for c in s["clips"]] for s in sections}
        made = catalog.create(TEST_USER, brand=report["mix_product"]["brand"] or "Brand",
                              product=report["mix_product"]["name"] or "Product", client=r)
        from cutsell_app import mix_routes
        t0 = time.time()
        started = mix_routes.start(mix_routes.MixJobRequest(user_id=TEST_USER, project_id=mix_id,
                                                            product_id=made["product_id"], sections=sections))
        report["mix_wake"] = drain()
        rec = {}
        while time.time() - t0 < 1000:
            rec = mix_store.load(TEST_USER, mix_id, client=r)
            if rec.get("state") in ("ready", "needs_retry", "failed"):
                break
            time.sleep(5)
        secs, status = job_seconds(started.get("job_id"))
        report["mix"] = {"state": rec.get("state"), "seconds_to_ready": round(time.time() - t0, 1), "job_seconds": secs,
                         "job_status": str(status), "usage": rec.get("usage"), "timing": rec.get("timing"),
                         "videos": len(rec.get("combos") or []),
                         "ready": sum(c.get("state") in ("ready", "draft_ready") for c in rec.get("combos") or []),
                         "clips": {cid: {k: c.get(k) for k in ("label", "state", "duration", "kept_sec", "error")}
                                   for cid, c in (rec.get("clips") or {}).items()},
                         "kept_text": {cid: " ".join(w["w"] for w in c.get("captions") or [])
                                       for cid, c in (rec.get("clips") or {}).items()}}

        # QA on every delivered video: joins (clicks / volume steps), loudness per part, posting caption
        qa = []
        for k, combo in enumerate(rec.get("combos") or []):
            row = {"code": combo.get("code"), "state": combo.get("state")}
            try:
                snap = get_draft_snapshot(user_id=TEST_USER, project_id=combo["project_id"], client=r)
                d = snap.get("draft") or {}
                se = (d.get("diagnostics") or {}).get("simple_engine") or {}
                parts = [(c["start"], c["end"]) for c in d.get("selected") or []]
                path = download_source(snap["sources"][0]["uri"], str(work / f"combo{k}.mp4"))
                joins = [b for a, b in parts[:-1]]
                row["duration"] = se.get("cut_duration_sec")
                row["joins"] = M.join_report(path, joins)
                lufs = []
                for a, b in parts:
                    seg = work / "seg.wav"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", path, "-vn",
                                    str(seg)], check=True)
                    l = M.loudness(str(seg))
                    lufs.append(round(l["input_i"], 1) if l else None)
                row["parts_lufs"] = lufs
                row["lufs"] = (M.loudness(path) or {}).get("input_i")
                entry = post_plan.video(TEST_USER, combo["project_id"], client=r)
                row["post_caption"] = entry.get("caption")
                row["post_source"] = entry.get("source")
                row["caption_cost"] = (entry.get("caption_log") or {}).get("cost_usd")
                if k < 3:
                    heard = asr.transcribe_words(asr.extract_audio(path, str(work / f"qa{k}.mp3")))["words"]
                    deltas, missing = [], 0
                    for c in se.get("captions") or []:
                        cand = [h for h in heard if E._norm(h["w"]) == E._norm(c["w"]) and abs(h["s"] - c["s"]) < 1.0]
                        if cand:
                            deltas.append(min(abs(h["s"] - c["s"]) for h in cand))
                        else:
                            missing += 1
                    deltas.sort()
                    row["captions_vs_fresh_asr"] = {
                        "words": len(se.get("captions") or []), "matched": len(deltas),
                        "median_abs_s": round(deltas[len(deltas) // 2], 3) if deltas else None,
                        "p90_abs_s": round(deltas[int(len(deltas) * 0.9)], 3) if deltas else None,
                        "max_abs_s": round(deltas[-1], 3) if deltas else None, "duration": row["duration"]}
            except Exception as exc:
                row["error"] = f"{exc.__class__.__name__}: {str(exc)[:200]}"
            qa.append(row)
        report["mix_qa"] = qa

        # Haiku vs Sonnet on the same clip transcripts (mix_clip prompt, nothing stored)
        clips_words = []
        for v in voices:
            heard = asr.transcribe_words(asr.extract_audio(v["file"], v["file"] + ".mp3"))
            clips_words.append({"section": v["section"], "words": [w for w in heard["words"] if w["e"] > w["s"]]})
        report["deepgram_compare_sec"] = round(sum(v["sec"] for v in voices), 1)
        compare = {}
        for label, model in (("haiku", "claude-haiku-5-5"), ("sonnet", "claude-sonnet-5-5")):
            t1 = time.time()
            answer, units, usage = M.ask(clips_words, model=model)
            kept = []
            for i, c in enumerate(clips_words):
                lab = M.clip_labels(answer, i, c["words"], units[i])
                kept.append(" ".join(w["w"] for w, l in zip(c["words"], lab) if l[0] == "keep"))
            compare[label] = {"usage": usage, "seconds": round(time.time() - t1, 1), "kept": kept}
        report["model_compare"] = compare
    except Exception as exc:
        import traceback
        report["errors"].append(f"mix: {exc.__class__.__name__}: {str(exc)[:300]} {traceback.format_exc()[-600:]}")

    report["modal_rate_per_sec"] = MODAL_RATE
    report["probe_seconds"] = round(time.time() - t_all, 1)
    return report


@app.local_entrypoint()
def main():
    out = probe.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, default=str)
    print("::notice title=prod verify::done in", out.get("probe_seconds"), "s; errors:", len(out.get("errors") or []))
