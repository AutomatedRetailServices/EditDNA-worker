"""Mix real test (read-only on production): 9 clips cut from the creator's own Rhino videos (3 hooks, 2 bodies,
2 benefits, 2 CTAs), the real Mix job (Deepgram + one mix_clip call + ffmpeg) on a fake Redis and local storage,
Haiku vs Sonnet on the same clips, all 24 and "Pick 12", join / caption checks, and a normal cut of a ~35 s video for
the cost comparison. Nothing is written to production Redis or storage. Report encrypted (numbers only)."""
import json
import os
import time

import modal

app = modal.App("cutsell-mix-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .pip_install("fakeredis")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
REAL_USER = "1625a6e9"


@app.function(image=image, secrets=[secret], timeout=1750, cpu=2.0, memory=4096)
def probe():
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path
    from redis import Redis
    import fakeredis

    report = {"steps": {}}
    t_all = time.time()
    prod = Redis.from_url(os.environ["REDIS_URL"])
    from cutsell_worker import product_catalog as catalog
    from cutsell_worker.simple_engine import engine as E

    # ---- find the creator and her Rhino videos (read-only)
    user = None
    for key in prod.scan_iter("cutsell:v1:draft:*", count=500):
        snap = json.loads(prod.get(key))
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
            user = snap["user_id"]
            break
    data = catalog.load(user, client=prod)
    product_of = {pid: v.get("product_id") for pid, v in data["videos"].items() if v.get("product_id")}
    from cutsell_worker.draft_store import draft_key
    from cutsell_worker.storage import download_source
    from cutsell_worker.simple_engine import asr
    work = Path(tempfile.mkdtemp())
    # distinct recordings (the same video was cut more than once): by source duration; speech density from the draft
    seen, every = set(), []
    for key in prod.scan_iter("cutsell:v1:draft:*", count=500):
        snap = json.loads(prod.get(key))
        if snap.get("user_id") != user:
            continue
        src = (snap.get("sources") or [{}])[0]
        dur = round(float(src.get("duration_sec") or 0), 1)
        draft = snap.get("draft") or {}
        words = sum(len(c.get("words") or []) for c in (draft.get("selected") or []) + (draft.get("alternates") or []))
        if src.get("uri") and dur and dur not in seen:
            seen.add(dur)
            every.append({"uri": src["uri"], "dur": dur, "product": product_of.get(snap.get("project_id")),
                          "density": words / dur})
    # the product with two clearly spoken recordings (a real creator records the same product again)
    by_product = {}
    for v in every:
        if v["product"] and 15 <= v["dur"] <= 180:
            by_product.setdefault(v["product"], []).append(v)
    best = max((sorted(vs, key=lambda v: -v["density"])[:2] for vs in by_product.values() if len(vs) >= 2),
               key=lambda pair: min(v["density"] for v in pair))
    vids = sorted(best, key=lambda v: -v["dur"])
    report["test_product"] = data["products"][vids[0]["product"]]["name"]
    report["rhino_videos"] = [v["dur"] for v in vids]
    report["density_words_per_sec"] = [round(v["density"], 2) for v in vids]
    for i, v in enumerate(vids):
        v["local"] = download_source(v["uri"], str(work / f"src{i}.mov"))
        heard = asr.transcribe_words(asr.extract_audio(v["local"], str(work / f"src{i}.mp3")))
        v["words"] = [w for w in heard["words"] if w["e"] > w["s"]]
    report["deepgram_source_sec"] = round(sum(v["dur"] for v in vids), 1)

    def chunks(v, sizes):
        """Consecutive pieces of the recording, each made of whole sentences (min, max seconds)."""
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
            out.append((max(0.0, a - 0.6), min(v["dur"], b + 0.6)))
        return out

    A, B = vids[0], vids[-1]
    ca = chunks(A, [(4, 9), (4, 9), (6, 14), (5, 12)])
    cb = chunks(B, [(4, 9), (6, 14), (5, 12)])
    def last(v):
        units = E.lines_of(v["words"])
        a = v["words"][units[-1][0]]["s"]
        for u in reversed(units[:-1]):
            if v["words"][units[-1][-1]]["e"] - a >= 3:
                break
            a = v["words"][u[0]]["s"]
        return max(0.0, a - 0.6), min(v["dur"], v["words"][-1]["e"] + 0.6)
    plan_src = {"Hook": [(A, ca[0]), (B, cb[0]), (A, ca[1])], "Body": [(A, ca[2]), (B, cb[1])],
                "Benefit": [(A, ca[3]), (B, cb[2])], "CTA": [(A, last(A)), (B, last(B))]}
    files, sections = {}, []
    for name, spans in plan_src.items():
        items = []
        for ci, (v, (a, b)) in enumerate(spans):
            cid = f"{name[:2].lower()}{ci}"
            clip = work / f"{cid}.mov"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", v["local"],
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "160k", str(clip)],
                           check=True)
            voice = work / f"{cid}.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "64k",
                            str(voice)], check=True)
            files[f"s3://local/{cid}.mov"], files[f"s3://local/{cid}.m4a"] = str(clip), str(voice)
            items.append({"clip_id": cid, "uri": f"s3://local/{cid}.mov", "audio_uri": f"s3://local/{cid}.m4a",
                          "original_name": f"{cid}.mov", "duration_sec": round(b - a, 2)})
        sections.append({"name": name, "clips": items})
    report["clips_raw_sec"] = {s["name"]: [c["duration_sec"] for c in s["clips"]] for s in sections}

    # ---- the real job on a fake Redis / local storage
    from cutsell_worker import mix_engine as M, mix_job as J, mix_store as S
    from cutsell_worker import draft_store, project_store
    fake = fakeredis.FakeRedis()
    for module in (project_store, draft_store, S, catalog):
        module._redis_client = lambda c=None, _r=fake: _r
    import cutsell_worker.timeline_asset_storage as tas
    import cutsell_worker.timeline_asset_s3 as ta3
    tas.store_timeline_assets = lambda **k: {"status": "probe_skipped"}
    ta3.register_primary_video = lambda **k: {"status": "probe_skipped"}
    import cutsell_worker.notifications as notes
    notes.publish_notification = lambda **k: {"notification_id": "probe"}
    import cutsell_worker.usage_limits as ul
    ul.record_processing_minutes = lambda **k: {}

    def fetch(uri, dest):
        Path(dest).write_bytes(Path(files[uri]).read_bytes())
        return dest

    def put(local, key, ctype):
        target = work / ("put_" + key.replace("/", "_"))
        target.write_bytes(Path(local).read_bytes())
        files["s3://local/" + key] = str(target)
        return "s3://local/" + key

    made = []

    def after(**kw):
        t = time.time()
        J._after_combo(**kw)
        made.append({"n": kw["combo"]["n"], "code": kw["combo"]["code"], "duration": round(kw["duration"], 2),
                     "after_sec": round(time.time() - t, 2), "path": kw["local"], "timeline": kw["timeline"],
                     "at": time.time()})
        keep = work / f"keep_{kw['combo']['n']:03d}_{len(made)}.mp4"
        keep.write_bytes(Path(kw["local"]).read_bytes())
        made[-1]["path"] = str(keep)

    def new_mix(how_many, model):
        os.environ["CUTSELL_MIX_MODEL"] = model
        mix = project_store.create_project(user_id="probe", title="Mix")
        prod_entry = catalog.create("probe", brand="Rhino", product="Retractable Straps")
        codes = M.section_codes([s["name"] for s in sections])
        secs = [{"name": s["name"], "code": codes[i], "clips": s["clips"]} for i, s in enumerate(sections)]
        clips = {c["clip_id"]: {"state": "waiting", "section": s["name"], "label": f"{codes[i]}{k + 1}"}
                 for i, s in enumerate(sections) for k, c in enumerate(s["clips"])}
        combos = []
        for k, picks in enumerate(M.plan([len(s["clips"]) for s in sections], how_many), start=1):
            p = project_store.create_project(user_id="probe", title=f"c{k}")
            combos.append({"n": k, "picks": picks, "code": M.combo_code(picks, codes), "project_id": p["project_id"],
                           "state": "processing"})
        S.save_new({"mix_id": mix["project_id"], "user_id": "probe", "product_id": prod_entry["product_id"],
                    "sections": secs, "combos": combos, "clips": clips, "state": "processing", "usage": {}, "timing": {}})
        return mix["project_id"]

    def run(how_many, model, label):
        made.clear()
        mix_id = new_mix(how_many, model)
        deps = J.Deps(fetch=fetch, put=put, client=fake, after_combo=after)
        t0 = time.time()
        out = J.run_mix_job({"user_id": "probe", "mix_id": mix_id}, deps)
        rec = S.load("probe", mix_id)
        t = rec.get("timing") or {}
        report["steps"][label] = {
            "state": out["state"], "videos": out["ready"], "total": out["total"], "seconds_total": round(time.time() - t0, 1),
            "seconds_to_clean": round(t.get("cleaned_at", t0) - t0, 1),
            "seconds_to_clean_clips": round(t.get("rendered_at", t0) - t0, 1),
            "seconds_to_first_video": round(t.get("first_ready_at", t0) - t0, 1),
            "seconds_to_last_video": round(t.get("last_ready_at", t0) - t0, 1),
            "usage": rec.get("usage"), "failed": out["failed_clips"],
            "clips": {cid: {k: c.get(k) for k in ("label", "state", "duration", "kept_sec", "words", "parts", "fixes", "language",
                                                  "moments", "lufs_in", "clean_duration", "error")}
                      for cid, c in rec["clips"].items()},
            "kept_text": {cid: " ".join(w["w"] for w in c.get("captions") or []) for cid, c in rec["clips"].items()},
            "durations": [m["duration"] for m in made], "after_combo_sec": [m["after_sec"] for m in made],
            "codes": [m["code"] for m in made],
        }
        return rec, list(made)

    rec24, made24 = run(None, "claude-haiku-5-5", "all24_haiku")

    # ---- QA on the 24: joins, loudness per part, captions vs a fresh transcription of 3 videos
    qa = []
    from cutsell_worker.simple_engine import asr
    for m in made24:
        joins = [b for a, b in m["timeline"]["parts"][:-1]]
        jr = M.join_report(m["path"], joins)
        parts_lufs = []
        for a, b in m["timeline"]["parts"]:
            seg = work / "seg.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", m["path"], "-vn", str(seg)],
                           check=True)
            l = M.loudness(str(seg))
            parts_lufs.append(round(l["input_i"], 1) if l else None)
        shape = M.probe(m["path"])
        qa.append({"code": m["code"], "duration": m["duration"], "size": [shape["width"], shape["height"]],
                   "lufs": (M.loudness(m["path"]) or {}).get("input_i"), "parts_lufs": parts_lufs,
                   "joins": jr})
    report["qa"] = qa
    cap_checks = []
    for m in made24[:3]:
        audio = asr.extract_audio(m["path"], str(work / f"qa{m['n']}.mp3"))
        heard = asr.transcribe_words(audio)["words"]
        deltas, missing = [], 0
        for c in m["timeline"]["captions"]:
            cand = [h for h in heard if E._norm(h["w"]) == E._norm(c["w"]) and abs(h["s"] - c["s"]) < 1.0]
            if cand:
                deltas.append(min(abs(h["s"] - c["s"]) for h in cand))
            else:
                missing += 1
        deltas.sort()
        cap_checks.append({"code": m["code"], "words": len(m["timeline"]["captions"]), "matched": len(deltas),
                           "median_abs_s": round(deltas[len(deltas) // 2], 3) if deltas else None,
                           "p90_abs_s": round(deltas[int(len(deltas) * 0.9)], 3) if deltas else None,
                           "max_abs_s": round(deltas[-1], 3) if deltas else None})
    report["captions_vs_fresh_asr"] = cap_checks
    report["deepgram_qa_sec"] = round(sum(m["duration"] for m in made24[:3]), 1)

    # ---- Sonnet vs Haiku on the same transcripts (labels only), plus a second Haiku answer for stability
    clips_words = []
    for s in sections:
        for c in s["clips"]:
            heard = asr.transcribe_words(asr.extract_audio(files[c["audio_uri"]], str(work / f"w_{c['clip_id']}.mp3")))
            clips_words.append({"section": s["name"], "clip_id": c["clip_id"],
                                "words": [w for w in heard["words"] if w["e"] > w["s"]]})
    report["deepgram_compare_sec"] = round(sum(c["duration_sec"] for s in sections for c in s["clips"]), 1)
    compare = {}
    for label, model in (("haiku_a", "claude-haiku-5-5"), ("haiku_b", "claude-haiku-5-5"), ("sonnet_a", "claude-sonnet-5-5"),
                         ("sonnet_b", "claude-sonnet-5-5")):
        t0 = time.time()
        answer, units, usage = M.ask([{"section": c["section"], "words": c["words"]} for c in clips_words], model=model)
        kept = {}
        for i, c in enumerate(clips_words):
            lab = M.clip_labels(answer, i, c["words"], units[i])
            kept[c["clip_id"]] = {"kept": " ".join(w["w"] for w, l in zip(c["words"], lab) if l[0] == "keep"),
                                  "hidden": " ".join(w["w"] for w, l in zip(c["words"], lab) if l[0] != "keep")}
        compare[label] = {"usage": usage, "seconds": round(time.time() - t0, 1), "clips": kept,
                          "fix": answer.get("fix"), "zoom": answer.get("zoom")}
    report["model_compare"] = compare
    report["clip_transcripts"] = {c["clip_id"]: " ".join(w["w"] for w in c["words"]) for c in clips_words}

    # ---- Pick 12 (fresh, Haiku)
    rec12, made12 = run(12, "claude-haiku-5-5", "pick12_haiku")

    # ---- a normal cut of a ~35 s video, for the cost comparison (the live engine, untouched)
    target = min(every, key=lambda v: abs(v["dur"] - 35))
    target["local"] = download_source(target["uri"], str(work / "normal_src.mov"))
    audio = asr.extract_audio(target["local"], str(work / "normal.mp3"))
    heard = asr.transcribe_words(audio)
    t0 = time.time()
    out = E.process([w for w in heard["words"] if w["e"] > w["s"]], target["dur"], target["local"], video_path=target["local"])
    usage = out["usage"]

    def sonnet_cost(u):
        return (int(u.get("input_tokens") or 0) * 2.0 + int(u.get("output_tokens") or 0) * 10.0
                + int(u.get("cache_creation_input_tokens") or 0) * 2.5 + int(u.get("cache_read_input_tokens") or 0) * 0.2) / 1e6
    report["normal_cut"] = {"source_sec": round(target["dur"], 1), "cut_sec": out["cut_duration"], "calls": len(usage),
                            "input_tokens": sum(int(u.get("input_tokens") or 0) for u in usage),
                            "output_tokens": sum(int(u.get("output_tokens") or 0) for u in usage),
                            "claude_usd": round(sum(sonnet_cost(u) for u in usage), 5),
                            "deepgram_usd": round(target["dur"] / 60 * 0.0043, 5), "seconds": round(time.time() - t0, 1)}
    report["probe_seconds"] = round(time.time() - t_all, 1)
    return report


@app.local_entrypoint()
def main():
    out = probe.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("::notice title=mix probe::done in", out.get("probe_seconds"), "s")
