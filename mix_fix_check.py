"""Read-only on production: the test user's posting captions (cut + 24 Mix videos), then the fixed Mix job (fix/mix-language)
run on fake Redis / local storage on (a) the SAME 9 clips the production Mix used and (b) a Mix of two DIFFERENT Rhino
recordings. Nothing written to production. Report encrypted."""
import json
import os
import time

import modal

app = modal.App("cutsell-mix-fix-check")
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


@app.function(image=image, secrets=[secret], timeout=1500, cpu=2.0, memory=4096)
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

    # ---- posting captions written in production for the test user (read-only)
    plan = post_plan.load(TEST_USER, client=prod)
    projects = project_store.list_projects(user_id=TEST_USER, limit=100, client=prod)
    report["test_projects"] = [{"title": p.get("title"), "state": p.get("state"), "id": p["project_id"][-6:]} for p in projects]
    report["test_plan_videos"] = {pid[-6:]: {k: v.get(k) for k in ("source", "caption", "caption_source")}
                                  | {"cost": (v.get("caption_log") or {}).get("cost_usd")}
                                  for pid, v in plan["videos"].items()}
    mix_rec = None
    for p in projects:
        if p.get("state") == catalog.MIX_CONTAINER_STATE:
            mix_rec = mix_store.load(TEST_USER, p["project_id"], client=prod)
    report["prod_clip_language"] = {cid: c.get("language") for cid, c in (mix_rec or {}).get("clips", {}).items()}

    # ---- the fixed job on fake Redis
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

    def run(sections, label):
        made = []

        def after(**kw):
            J._after_combo(**kw)
            keep = work / f"{label}_{kw['combo']['n']:03d}.mp4"
            keep.write_bytes(Path(kw["local"]).read_bytes())
            made.append({"code": kw["combo"]["code"], "duration": round(kw["duration"], 2), "path": str(keep),
                         "parts": kw["timeline"]["parts"], "captions": kw["timeline"]["captions"]})
        mix = project_store.create_project(user_id="probe", title="Mix")
        prod_entry = catalog.create("probe", brand="Rhino", product="Retractable Straps")
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
            jr = M.join_report(m["path"], [b for a, b in m["parts"][:-1]])
            qa.append({"code": m["code"], "duration": m["duration"], "joins": jr,
                       "lufs": (M.loudness(m["path"]) or {}).get("input_i")})
        cap = []
        for m in made[:3]:
            heard = asr.transcribe_words(asr.extract_audio(m["path"], m["path"] + ".mp3"), language_hint="en")["words"]
            deltas = []
            for c in m["captions"]:
                cand = [h for h in heard if E._norm(h["w"]) == E._norm(c["w"]) and abs(h["s"] - c["s"]) < 1.0]
                if cand:
                    deltas.append(min(abs(h["s"] - c["s"]) for h in cand))
            deltas.sort()
            cap.append({"code": m["code"], "words": len(m["captions"]), "matched": len(deltas),
                        "median_abs_s": round(deltas[len(deltas) // 2], 3) if deltas else None,
                        "max_abs_s": round(deltas[-1], 3) if deltas else None})
        report[label] = {"state": out["state"], "videos": out["ready"], "seconds": round(time.time() - t0, 1),
                         "usage": rec.get("usage"), "timing": rec.get("timing"),
                         "clips": {cid: {k: c.get(k) for k in ("label", "language", "heard_twice", "duration", "kept_sec")}
                                   for cid, c in rec["clips"].items()},
                         "kept_text": {cid: " ".join(w["w"] for w in c.get("captions") or [])
                                       for cid, c in rec["clips"].items()},
                         "durations": [m["duration"] for m in made], "qa": qa, "captions_vs_fresh_asr": cap}

    try:
        same = [{"name": s["name"], "clips": [{k: c.get(k) for k in ("clip_id", "uri", "audio_uri", "original_name",
                                                                       "duration_sec")} for c in s["clips"]]}
                for s in mix_rec["sections"]]
        run(same, "same_clips_fixed")
    except Exception as exc:
        import traceback
        report["errors"].append("same: " + traceback.format_exc()[-800:])

    # ---- (b) two DIFFERENT Rhino recordings of hers (distinct spoken text / length)
    try:
        user = None
        for key in prod.scan_iter("cutsell:v1:draft:*", count=500):
            snap = json.loads(prod.get(key))
            if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
                user = snap["user_id"]
                break
        data = catalog.load(user, client=prod)
        rhino = next(pid for pid, p in data["products"].items() if "Retractable" in (p.get("name") or ""))
        recs = {}
        for pid, v in data["videos"].items():
            if v.get("product_id") != rhino:
                continue
            snap = get_draft_snapshot(user_id=user, project_id=pid, client=prod)
            src = (snap.get("sources") or [{}])[0]
            dur = round(float(src.get("duration_sec") or 0))
            if src.get("uri") and dur not in recs:
                recs[dur] = src["uri"]
        report["rhino_recordings_sec"] = sorted(recs)
        vids = []
        for i, (dur, uri) in enumerate(sorted(recs.items())[:2]):
            local = download_source(uri, str(work / f"r{i}.mov"))
            heard = asr.transcribe_words(asr.extract_audio(local, str(work / f"r{i}.mp3")))
            vids.append({"local": local, "dur": dur, "words": [w for w in heard["words"] if w["e"] > w["s"]],
                         "language": heard.get("language")})
        report["rhino_transcripts"] = [" ".join(w["w"] for w in v["words"])[:900] for v in vids]

        def sentences(v):
            out = []
            for u in E.lines_of(v["words"]):
                out.append((v["words"][u[0]]["s"], v["words"][u[-1]]["e"]))
            return out

        def pick(v, lo, hi, start):
            sents = sentences(v)
            for i in range(start, len(sents)):
                a = sents[i][0]
                for j in range(i, len(sents)):
                    if sents[j][1] - a >= lo:
                        if sents[j][1] - a <= hi:
                            return (max(0, a - 0.3), min(v["dur"], sents[j][1] + 0.3)), j + 1
                        break
            return None, start
        A, B = vids[0], vids[1]
        spans, ia, ib = {}, 0, 0
        plan_ab = [("Hook", A), ("Hook", B), ("Hook", A), ("Body", A), ("Body", B), ("Benefit", A), ("Benefit", B),
                   ("CTA", A), ("CTA", B)]
        sizes = {"Hook": (2.5, 7), "Body": (5, 14), "Benefit": (4, 12), "CTA": (3, 10)}
        pos = {id(A): 0, id(B): 0}
        sections = {}
        for k, (name, v) in enumerate(plan_ab):
            if name == "CTA":
                s = sentences(v)
                span = (max(0, s[-1][0] - 0.3), min(v["dur"], s[-1][1] + 0.3))
                if span[1] - span[0] < 3 and len(s) > 1:
                    span = (max(0, s[-2][0] - 0.3), span[1])
            else:
                span, pos[id(v)] = pick(v, *sizes[name], pos[id(v)])
            if not span:
                continue
            cid = f"d{k}"
            clip, voice = work / f"{cid}.mov", work / f"{cid}_voice.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{span[0]:.3f}", "-to", f"{span[1]:.3f}", "-i",
                            v["local"], "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a",
                            "160k", str(clip)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "64k",
                            str(voice)], check=True)
            files[f"s3://local/{cid}.mov"], files[f"s3://local/{cid}.mp4"] = str(clip), str(voice)
            sections.setdefault(name, []).append({"clip_id": cid, "uri": f"s3://local/{cid}.mov",
                                                  "audio_uri": f"s3://local/{cid}.mp4", "original_name": f"{cid}.mov",
                                                  "duration_sec": round(span[1] - span[0], 2)})
        report["distinct_clips_sec"] = {n: [c["duration_sec"] for c in cs] for n, cs in sections.items()}
        run([{"name": n, "clips": cs} for n, cs in sections.items()], "distinct_fixed")
        # Sonnet on the same clips (labels only, nothing stored)
        words = []
        for n, cs in sections.items():
            for c in cs:
                h = asr.transcribe_words(asr.extract_audio(files[c["audio_uri"]], files[c["audio_uri"]] + ".mp3"),
                                         language_hint="en")
                words.append({"section": n, "words": [w for w in h["words"] if w["e"] > w["s"]]})
        cmp = {}
        for label, model in (("haiku", "claude-haiku-5-5"), ("sonnet", "claude-sonnet-5-5")):
            answer, units, usage = M.ask(words, model=model)
            kept = []
            for i, c in enumerate(words):
                lab = M.clip_labels(answer, i, c["words"], units[i])
                kept.append(" ".join(w["w"] for w, l in zip(c["words"], lab) if l[0] == "keep"))
            cmp[label] = {"usage": usage, "kept": kept}
        report["distinct_model_compare"] = cmp
        report["distinct_clip_words"] = [" ".join(w["w"] for w in c["words"]) for c in words]
    except Exception:
        import traceback
        report["errors"].append("distinct: " + traceback.format_exc()[-800:])
    report["probe_seconds"] = round(time.time() - t_all, 1)
    return report


@app.local_entrypoint()
def main():
    out = probe.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, default=str)
    print("::notice title=mix fix check::done in", out.get("probe_seconds"), "s")
