"""Read-only on production. With the code of feat/mix-sonnet-tight-joins (Sonnet mix_clip, ~0.1 s joins, language fix):
(a) the SAME 9 clips as the production Mix: pause and clicks at every join, loudness, captions vs a fresh transcription,
    and the same measures on the 24 videos production made (before); (b) a Spanish Mix from her Spanish recording;
(c) Plan one-cut-per-recording on her real library; (d) the full text of her untagged videos (for product assignment).
Nothing written to production. Report encrypted."""
import json
import os
import time

import modal

app = modal.App("cutsell-mix-cues-check")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .pip_install("fakeredis", "numpy")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
REAL_USER = "1625a6e9"
TEST_USER = "cutsell-prod-verify-20261021"


def pauses(path, joins):
    """Per join: the silence around it (s), from the sound itself (10 ms frames below the clip's voice level - 30 dB)."""
    import subprocess
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    n = 160
    fr = x[: len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-6)
    thr = float(np.percentile(db, 95)) - 30
    out = []
    for t in joins:
        i = min(len(db) - 1, int(t / 0.01))
        a = b = i
        while a > 0 and db[a - 1] <= thr:
            a -= 1
        while b < len(db) and db[b] <= thr:
            b += 1
        out.append(round((b - a) * 0.01, 2) if db[i] <= thr else 0.0)
    return out


@app.function(image=image, secrets=[secret], timeout=1700, cpu=2.0, memory=4096)
def probe():
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path
    from redis import Redis
    import fakeredis

    t_all = time.time()
    report = {"errors": []}
    prod = Redis.from_url(os.environ["REDIS_URL"])
    from cutsell_worker import post_plan, project_store, mix_store, product_catalog as catalog, draft_store
    from cutsell_worker import mix_engine as M, mix_job as J
    from cutsell_worker.draft_store import get_draft_snapshot
    from cutsell_worker.storage import download_source
    from cutsell_worker.simple_engine import asr, engine as E
    work = Path(tempfile.mkdtemp())
    report["model"] = M.model_name()

    # ---- production state, read before anything is patched
    user = None
    for key in prod.scan_iter("cutsell:v1:draft:*", count=500):
        snap = json.loads(prod.get(key))
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
            user = snap["user_id"]
            break
    projects = project_store.list_projects(user_id=user, limit=500, client=prod)
    plan = post_plan.load(user, client=prod)
    cands = post_plan.candidates(user, projects, client=prod, plan=plan)
    data = catalog.load(user, client=prod)
    report["plan_dedupe"] = {"ready_videos": sum(p.get("state") == "draft_ready" for p in projects),
                             "planned_videos": len(cands),
                             "kept": [{"title": v["title"], "product": v.get("product"), "other_cuts": v.get("other_cuts", 0),
                                       "created_at": v["created_at"], "sec": v["duration_sec"]} for v in cands]}
    untagged = []
    spanish_src = None
    for p in projects:
        link = data["videos"].get(p["project_id"]) or {}
        try:
            snap = get_draft_snapshot(user_id=user, project_id=p["project_id"], client=prod)
        except Exception:
            continue
        d = snap.get("draft") or {}
        clips = sorted((c for part in ("selected", "alternates", "discarded") for c in d.get(part) or []),
                       key=lambda c: (c.get("source_order") or 0, c.get("start") or 0))
        text = " ".join(" ".join(w.get("text") or "" for w in c.get("words") or []) for c in clips)
        srcs = snap.get("sources") or []
        if not link.get("product_id"):
            untagged.append({"id": p["project_id"][-6:], "state": p.get("state"), "sec": [s.get("duration_sec") for s in srcs],
                             "text": text[:700]})
        if len(srcs) == 1 and abs(float(srcs[0].get("duration_sec") or 0) - 106.3) < 0.5 and srcs[0].get("uri"):
            spanish_src = srcs[0]["uri"]
    report["untagged"] = untagged
    mix_rec = None
    for p in project_store.list_projects(user_id=TEST_USER, limit=100, client=prod):
        if p.get("state") == catalog.MIX_CONTAINER_STATE:
            mix_rec = mix_store.load(TEST_USER, p["project_id"], client=prod)
    before = []
    for k, combo in enumerate(mix_rec["combos"][:12]):
        try:
            snap = get_draft_snapshot(user_id=TEST_USER, project_id=combo["project_id"], client=prod)
            parts = [(c["start"], c["end"]) for c in snap["draft"]["selected"]]
            path = download_source(snap["sources"][0]["uri"], str(work / f"before{k}.mp4"))
            before.append({"code": combo["code"], "duration": parts[-1][1], "pauses": pauses(path, [b for a, b in parts[:-1]])})
        except Exception as exc:
            before.append({"error": exc.__class__.__name__})
    report["before"] = before

    def edges(rec):
        out = {}
        for cid, c in (rec.get("clips") or {}).items():
            caps = c.get("captions") or []
            dur = sum(b - a for a, b in c.get("parts") or []) or c.get("kept_sec") or 0
            if caps:
                out[cid] = {"label": c.get("label"), "lead": round(caps[0]["s"], 3), "tail": round(dur - caps[-1]["e"], 3),
                            "first": caps[0]["w"], "last": caps[-1]["w"]}
        return out
    report["edges_before"] = edges(mix_rec)

    # ---- the new code on fake Redis
    fake = fakeredis.FakeRedis()
    for module in (project_store, draft_store, mix_store, catalog):
        module._redis_client = lambda c=None, _r=fake: _r
    import cutsell_worker.timeline_asset_storage as tas
    import cutsell_worker.timeline_asset_s3 as ta3
    tas.store_timeline_assets = lambda **k: {"status": "probe_skipped"}
    ta3.register_primary_video = lambda **k: {"status": "probe_skipped"}
    import cutsell_worker.notifications as notes
    notes.publish_notification = lambda **k: {"notification_id": "probe"}
    import cutsell_worker.usage_limits as ul
    ul.record_processing_minutes = lambda **k: {}
    import cutsell_worker.post_captions as pc
    pc.caption_once = lambda **k: {"status": "probe_skipped"}
    files = {}

    def fetch(uri, dest):
        if uri in files:
            Path(dest).write_bytes(Path(files[uri]).read_bytes())
            return dest
        return download_source(uri, dest)

    def put(local, key, ctype):
        target = work / ("put_" + key.replace("/", "_"))
        target.write_bytes(Path(local).read_bytes())
        files["s3://local/" + key] = str(target)
        return "s3://local/" + key

    seen_labels = []
    _orig_labels = M.clip_labels
    _orig_drop = M.drop_hidden_words

    def logged_labels(answer, index, words, units):
        lab = _orig_labels(answer, index, words, units)
        seen_labels.append({"index": index, "lines": [" ".join(words[k]["w"] for k in u) for u in units],
                            "hidden": [words[k]["w"] for k, (st, _) in enumerate(lab) if st != "keep"],
                            "answer": next((c for c in answer.get("clips") or [] if c.get("clip") == index + 1), None)})
        return lab
    M.clip_labels = logged_labels

    def logged_drop(parts, words, lab):
        out = _orig_drop(parts, words, lab)
        seen_labels.append({"drop_in": parts, "drop_out": out})
        return out
    M.drop_hidden_words = logged_drop
    report["seen_labels"] = seen_labels

    def run(sections, label, lang):
        made = []

        def after(**kw):
            J._after_combo(**kw)
            keep = work / f"{label}_{kw['combo']['n']:03d}.mp4"
            keep.write_bytes(Path(kw["local"]).read_bytes())
            made.append({"code": kw["combo"]["code"], "duration": round(kw["duration"], 2), "path": str(keep),
                         "parts": kw["timeline"]["parts"], "captions": kw["timeline"]["captions"]})
        mix = project_store.create_project(user_id="probe", title="Mix")
        prod_entry = catalog.create("probe", brand="B", product="P")
        codes = M.section_codes([s["name"] for s in sections])
        secs = [{"name": s["name"], "code": codes[i], "clips": s["clips"]} for i, s in enumerate(sections)]
        clips = {c["clip_id"]: {"state": "waiting", "section": s["name"], "label": f"{codes[i]}{k + 1}"}
                 for i, s in enumerate(sections) for k, c in enumerate(s["clips"])}
        combos = []
        for k, picks in enumerate(M.plan([len(s["clips"]) for s in sections], None), start=1):
            p = project_store.create_project(user_id="probe", title=f"c{k}")
            combos.append({"n": k, "picks": picks, "code": M.combo_code(picks, codes), "project_id": p["project_id"],
                           "state": "processing"})
        mix_store.save_new({"mix_id": mix["project_id"], "user_id": "probe", "product_id": prod_entry["product_id"],
                            "sections": secs, "combos": combos, "clips": clips, "state": "processing", "usage": {},
                            "timing": {}})
        t0 = time.time()
        out = J.run_mix_job({"user_id": "probe", "mix_id": mix["project_id"]},
                            J.Deps(fetch=fetch, put=put, client=fake, after_combo=after))
        rec = mix_store.load("probe", mix["project_id"])
        qa = []
        for m in made:
            joins = [b for a, b in m["parts"][:-1]]
            jr = M.join_report(m["path"], joins)
            lufs = []
            for a, b in m["parts"]:
                seg = work / "seg.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", m["path"], "-vn",
                                str(seg)], check=True)
                l = M.loudness(str(seg))
                lufs.append(round(l["input_i"], 1) if l else None)
            qa.append({"code": m["code"], "duration": m["duration"], "pauses": pauses(m["path"], joins),
                       "clicks": sum(j["click"] for j in jr), "max_jump": max([j["max_jump"] for j in jr] or [0]),
                       "parts_lufs": lufs})
        cap = []
        for m in made[:6]:
            heard = asr.transcribe_words(asr.extract_audio(m["path"], m["path"] + ".mp3"), language_hint=lang)["words"]
            deltas = []
            for c in m["captions"]:
                cand = [h for h in heard if E._norm(h["w"]) == E._norm(c["w"]) and abs(h["s"] - c["s"]) < 1.0]
                if cand:
                    deltas.append(min(abs(h["s"] - c["s"]) for h in cand))
            deltas.sort()
            cap.append({"code": m["code"], "words": len(m["captions"]), "heard_words": len(heard), "matched": len(deltas),
                        "median_abs_s": round(deltas[len(deltas) // 2], 3) if deltas else None,
                        "max_abs_s": round(deltas[-1], 3) if deltas else None,
                        "captions": " ".join(c["w"] for c in m["captions"]),
                        "heard": " ".join(h["w"] for h in heard)})
        report["_rec_" + label] = rec
        report[label] = {"state": out["state"], "videos": out["ready"], "seconds": round(time.time() - t0, 1),
                         "usage": rec.get("usage"), "timing": rec.get("timing"),
                         "clips": {cid: {k: c.get(k) for k in ("label", "language", "heard_twice", "duration", "kept_sec",
                                                                "parts")}
                                   for cid, c in rec["clips"].items()},
                         "kept_text": {cid: " ".join(w["w"] for w in c.get("captions") or [])
                                       for cid, c in rec["clips"].items()},
                         "qa": qa, "captions_vs_fresh_asr": cap}

    try:
        same = [{"name": s["name"], "clips": [{k: c.get(k) for k in ("clip_id", "uri", "audio_uri", "original_name",
                                                                       "duration_sec")} for c in s["clips"]]}
                for s in mix_rec["sections"]]
        run(same, "same_clips_v2", "en")
        report["edges_after"] = edges(report.pop("_rec_same_clips_v2"))
        dbg = {}
        for sec in []:
            for c in sec["clips"]:
                loc = download_source(c["audio_uri"] or c["uri"], str(work / ("dbg_" + c["clip_id"] + ".mp4")))
                ws = [w for w in asr.transcribe_words(asr.extract_audio(loc, str(work / ("dbg_" + c["clip_id"] + ".mp3"))))["words"]
                      if w["e"] > w["s"]]
                dbg[c["clip_id"]] = [" ".join(ws[k]["w"] for k in u) for u in E.lines_of(ws)]
        report["job_mode_lines"] = dbg
    except Exception:
        import traceback
        report["errors"].append("same: " + traceback.format_exc()[-800:])

    # ---- (b) Spanish Mix: done in the previous run
    try:
        raise StopIteration
        local = download_source(spanish_src, str(work / "es.mov"))
        heard = asr.transcribe_words(asr.extract_audio(local, str(work / "es.mp3")))
        words = [w for w in heard["words"] if w["e"] > w["s"]]
        report["spanish_source_language"] = heard.get("language")
        sents = [(words[u[0]]["s"], words[u[-1]]["e"]) for u in E.lines_of(words)]
        # 9 consecutive pieces of whole sentences, 2.5 - 9 s each
        spans, i = [], 0
        while len(spans) < 9 and i < len(sents):
            a = sents[i][0]
            j = i
            while j + 1 < len(sents) and sents[j][1] - a < 2.5:
                j += 1
            if sents[j][1] - a <= 9:
                spans.append((max(0, a - 0.3), sents[j][1] + 0.3))
            i = j + 1
        names = ["Hook"] * 3 + ["Body"] * 2 + ["Benefit"] * 2 + ["CTA"] * 2
        sections = {}
        for k, (name, (a, b)) in enumerate(zip(names, spans)):
            cid = f"e{k}"
            clip, voice = work / f"{cid}.mov", work / f"{cid}_voice.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", local, "-c:v",
                            "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "160k", str(clip)],
                           check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "64k",
                            str(voice)], check=True)
            files[f"s3://local/{cid}.mov"], files[f"s3://local/{cid}.mp4"] = str(clip), str(voice)
            sections.setdefault(name, []).append({"clip_id": cid, "uri": f"s3://local/{cid}.mov",
                                                  "audio_uri": f"s3://local/{cid}.mp4", "original_name": f"{cid}.mov",
                                                  "duration_sec": round(b - a, 2)})
        report["spanish_clips_sec"] = {n: [c["duration_sec"] for c in cs] for n, cs in sections.items()}
        run([{"name": n, "clips": cs} for n, cs in sections.items()], "spanish_v2", "es")
    except Exception:
        import traceback
        pass
    report["probe_seconds"] = round(time.time() - t_all, 1)
    return report


@app.local_entrypoint()
def main():
    out = probe.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, default=str)
    print("::notice title=mix v2 check::done in", out.get("probe_seconds"), "s")
