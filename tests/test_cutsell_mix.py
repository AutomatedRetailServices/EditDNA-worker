"""Mix: combinations, the one-call clean of every clip, clean clips, joins and the job end to end (fakeredis,
synthetic media, no network)."""
import json
import subprocess
from collections import Counter
from pathlib import Path

import fakeredis
import pytest

from cutsell_worker import mix_engine as engine
from cutsell_worker import mix_job, mix_store
from cutsell_worker import draft_store, product_catalog, project_store


@pytest.fixture()
def r(monkeypatch):
    client = fakeredis.FakeRedis()
    for module in (project_store, draft_store, mix_store, product_catalog):
        monkeypatch.setattr(module, "_redis_client", lambda c=None, _r=client: _r)
    return client


# ---------------------------------------------------------------- combinations

def test_codes_follow_the_section_names():
    assert engine.section_codes(["Hook", "Body", "Benefit", "CTA"]) == ["H", "B", "Be", "C"]
    assert engine.section_codes(["Hook", "Hook", "Demo"]) == ["H", "Ho", "D"]
    assert engine.combo_code([0, 1, 0, 1], ["H", "B", "Be", "C"]) == "H1 · B2 · Be1 · C2"


def test_all_24_and_every_clip_used_evenly():
    combos = engine.plan([3, 2, 2, 2])
    assert len(combos) == 24 and len({tuple(c) for c in combos}) == 24
    for n in (6, 12, 18):
        chosen = engine.plan([3, 2, 2, 2], n)
        assert len({tuple(c) for c in chosen}) == n
        for s, size in enumerate([3, 2, 2, 2]):
            use = Counter(c[s] for c in chosen)
            assert max(use.values()) - min(use.values()) <= 1 and len(use) == size
        # never the same start or the same end twice in a row
        assert all(a[0] != b[0] and a[3] != b[3] for a, b in zip(chosen, chosen[1:]))


def test_limits():
    with pytest.raises(ValueError):
        engine.plan([11, 1])
    with pytest.raises(ValueError):
        engine.plan([1] * 7)
    with pytest.raises(ValueError):
        engine.plan([10, 10, 2])          # 200 combinations
    assert len(engine.plan([10, 10, 2], 100)) == 100


# ---------------------------------------------------------------- the one AI call

def _words(text, start=0.5, step=0.4):
    out, t = [], start
    for w in text.split():
        out.append({"w": w, "s": round(t, 2), "e": round(t + step - 0.1, 2)})
        t += step
    return out


def test_one_call_for_every_clip_and_labels_per_clip():
    clips = [{"section": "Hook", "words": _words("Wait. Okay. This cleanser changed my skin.")},
             {"section": "CTA", "words": _words("Grab it grab it now at the link.")}]
    sent = []

    def fake(body):
        sent.append(body)
        return json.dumps({"clips": [{"clip": 1, "hide": [[1, 2, "n"]], "trim": []},
                                     {"clip": 2, "hide": [], "trim": [[1, "grab it"]]}],
                           "fix": [[1, 3, "cleanser", "Cleanser"]], "zoom": [[1, 3, "This", "hook"]]}), \
            {"input_tokens": 900, "output_tokens": 300}

    answer, units, usage = engine.ask(clips, call=fake, model="claude-haiku-5-5")
    assert len(sent) == 1 and "CLIP 2 · part: CTA" in sent[0]["messages"][0]["content"][1]["text"]
    assert usage["calls"] == 1 and usage["cost_usd"] == pytest.approx((900 * 0.10 + 300 * 0.50) / 1e6)
    lab1 = engine.clip_labels(answer, 0, clips[0]["words"], units[0])
    assert [w["w"] for w, l in zip(clips[0]["words"], lab1) if l[0] == "keep"] == "This cleanser changed my skin.".split()
    lab2 = engine.clip_labels(answer, 1, clips[1]["words"], units[1])
    assert [w["w"] for w, l in zip(clips[1]["words"], lab2) if l[0] == "keep"] == "grab it now at the link.".split()
    kept = {k for k, l in enumerate(lab1) if l[0] == "keep"}
    fixes, moments = engine.fixes_and_moments(answer, 0, clips[0]["words"], units[0], kept)
    assert fixes == [[3, "cleanser", "Cleanser"]] and moments == [{"i": 2, "kind": "hook"}]


def test_an_answer_that_would_empty_a_clip_keeps_it():
    words = _words("Use code SAVE10 today.")
    units = engine._engine.lines_of(words)
    lab = engine.clip_labels({"clips": [{"clip": 1, "hide": [[1, 1, "n"]], "trim": []}]}, 0, words, units)
    assert all(l[0] == "keep" for l in lab)


def test_captions_and_moments_move_onto_the_combination():
    t = engine.combo_timeline([
        {"duration": 2.0, "captions": [{"w": "Hi", "s": 0.1, "e": 0.4}], "moments": [{"t": 0.1, "kind": "hook"}]},
        {"duration": 3.0, "captions": [{"w": "Buy", "s": 0.2, "e": 0.5}], "moments": []}])
    assert t["parts"] == [(0.0, 2.0), (2.0, 5.0)] and t["duration"] == 5.0
    assert t["captions"][1] == {"w": "Buy", "s": 2.2, "e": 2.5}
    assert t["moments"] == [{"s": 0.1, "kind": "hook"}]
    assert engine.clip_time(3.2, [(1.0, 2.0), (3.0, 4.0)]) == 1.2


# ---------------------------------------------------------------- media + the job

def _clip(path: Path, seconds: float, freq: int, volume: float, size="360x640"):
    """A synthetic clip: speech-like tone bursts (on 0.3 s / off 0.1 s) after 0.4 s of silence."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=f={freq}:d={seconds}", "-filter_complex",
                    f"[1:a]volume='if(lt(t,0.4),0,if(lt(mod(t,0.4),0.3),{volume},0))':eval=frame[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                    "-shortest", str(path)], check=True)
    return str(path)


def test_clean_clips_are_even_in_loudness_and_joins_do_not_click(tmp_path):
    loud = _clip(tmp_path / "loud.mp4", 3, 300, 0.6)
    quiet = _clip(tmp_path / "quiet.mp4", 3, 500, 0.05, size="640x360")
    a = engine.render_clean_clip(loud, [(0.4, 1.5), (1.9, 2.8)], (360, 640), str(tmp_path / "a.mov"))
    b = engine.render_clean_clip(quiet, [(0.4, 2.8)], (360, 640), str(tmp_path / "b.mov"))
    assert abs(engine.loudness(a["path"])["input_i"] - (-16)) < 1.0
    assert abs(engine.loudness(b["path"])["input_i"] - (-16)) < 1.0
    combo = engine.join_clips([a["path"], b["path"], a["path"]], str(tmp_path / "combo.mp4"))
    shape = engine.probe(combo["path"])
    assert (shape["width"], shape["height"]) == (360, 640)
    assert combo["duration"] == pytest.approx(a["duration"] * 2 + b["duration"], abs=0.1)
    report = engine.join_report(combo["path"], [a["duration"], a["duration"] + b["duration"]])
    assert not any(j["click"] for j in report)


def _mix(r, tmp_path, counts=(2, 1, 1, 2), names=("Hook", "Body", "Benefit", "CTA"), how_many=None):
    user = "u1"
    mix = project_store.create_project(user_id=user, title="Mix")
    product = product_catalog.create(user, brand="CeraVe", product="Hydrating Cleanser")
    sections, clips, files = [], {}, {}
    for si, (name, count) in enumerate(zip(names, counts)):
        items = []
        for ci in range(count):
            cid = f"s{si}c{ci}"
            files[f"s3://b/{cid}.mp4"] = _clip(tmp_path / f"{cid}.mp4", 2.0, 300 + 100 * si + 30 * ci, 0.3 + 0.1 * ci)
            files[f"s3://b/{cid}.m4a"] = files[f"s3://b/{cid}.mp4"]          # the voice the app sends first
            items.append({"clip_id": cid, "uri": f"s3://b/{cid}.mp4", "audio_uri": f"s3://b/{cid}.m4a",
                          "original_name": f"{cid}.mp4"})
            clips[cid] = {"state": "waiting", "section": name, "label": f"{name[0]}{ci + 1}"}
        sections.append({"name": name, "code": engine.section_codes(list(names))[si], "clips": items})
    combos = []
    for n, picks in enumerate(engine.plan(list(counts), how_many), start=1):
        p = project_store.create_project(user_id=user, title=f"combo {n}")
        combos.append({"n": n, "picks": picks, "code": engine.combo_code(picks, [s["code"] for s in sections]),
                       "project_id": p["project_id"], "state": "processing"})
    product_catalog.place_mix(user, mix["project_id"], {c["project_id"]: c["code"] for c in combos},
                              product_id=product["product_id"])
    mix_store.save_new({"mix_id": mix["project_id"], "user_id": user, "product_id": product["product_id"],
                        "sections": sections, "combos": combos, "clips": clips, "state": "processing",
                        "usage": {}, "timing": {}}, client=r)
    return user, mix["project_id"], files


def _deps(r, files, tmp_path, missing=(), llm_calls=None, fail_transcribe=()):
    calls = llm_calls if llm_calls is not None else []
    made = []

    def fetch(uri, dest):
        if uri in missing:
            raise FileNotFoundError(uri)
        src = files.get(uri) or files[uri]
        Path(dest).write_bytes(Path(src).read_bytes())
        return dest

    def put(local, key, content_type):
        target = tmp_path / "store" / key.replace("/", "_")
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(Path(local).read_bytes())
        files["s3://b/" + key] = str(target)
        return "s3://b/" + key

    def transcribe(audio, **k):
        if any(f in audio for f in fail_transcribe):
            raise RuntimeError("deepgram down")
        return {"language": "en", "words": [{"w": "um", "s": 0.4, "e": 0.65}] +
                [{"w": w, "s": 0.8 + 0.4 * i, "e": 1.0 + 0.4 * i} for i, w in enumerate("this works great".split())]}

    def llm(body):
        calls.append(body)
        n = body["messages"][0]["content"][1]["text"].count("=== CLIP")
        return json.dumps({"clips": [{"clip": i + 1, "hide": [], "trim": [[1, "um"]]} for i in range(n)],
                           "fix": [], "zoom": [[1, 1, "this", "hook"]]}), {"input_tokens": 1000, "output_tokens": 200}

    def after(**kw):
        made.append(kw)
        mix_job._after_combo(**kw)

    deps = mix_job.Deps(transcribe=transcribe, llm=llm, fetch=fetch, put=put, client=r, after_combo=after,
                        extract_audio=lambda src, dst: src)
    return deps, calls, made


def test_the_job_makes_every_video_with_one_ai_call(r, tmp_path):
    user, mix_id, files = _mix(r, tmp_path)
    deps, calls, made = _deps(r, files, tmp_path)
    out = mix_job.run_mix_job({"user_id": user, "mix_id": mix_id}, deps)
    assert out["state"] == "ready" and out["ready"] == out["total"] == 4
    assert len(calls) == 1                                   # ONE mix_clip call for all 6 clips
    record = mix_store.load(user, mix_id, client=r)
    assert record["usage"]["llm"]["calls"] == 1 and record["usage"]["units"] == 4
    # each video: a normal draft (one clip per part), captions on its timeline, its product with MIX + code
    first = record["combos"][0]
    snap = draft_store.get_draft_snapshot(user_id=user, project_id=first["project_id"])
    selected = snap["draft"]["selected"]
    assert len(selected) == 4 and selected[0]["start"] == 0.0
    caps = snap["draft"]["diagnostics"]["simple_engine"]["captions"]
    assert [c["w"] for c in caps][:3] == ["this", "works", "great"] and "um" not in [c["w"] for c in caps]
    assert caps[3]["s"] >= selected[1]["start"] - 0.01      # the Body's words start after the Hook
    assert project_store.get_project(user_id=user, project_id=first["project_id"])["state"] == "draft_ready"
    entry = product_catalog.video_entry(user, first["project_id"])
    assert entry["mix_code"] == first["code"] and entry["source"] == "user"
    # the Auto Zoom moment found once per clip, ready to be placed on every video that uses the clip
    assert record["clips"]["s0c0"]["moments"][0]["kind"] == "hook"
    assert all(k["kind"] == "hook" for k in snap["draft"].get("zoom_keyframes") or [])


def test_videos_still_uploading_then_video_ready_finishes(r, tmp_path):
    user, mix_id, files = _mix(r, tmp_path, counts=(2, 1, 1, 1))
    deps, calls, _ = _deps(r, files, tmp_path, missing={"s3://b/s3c0.mp4"})
    out = mix_job.run_mix_job({"user_id": user, "mix_id": mix_id}, deps)
    assert out["state"] == "waiting_for_videos"
    record = mix_store.load(user, mix_id, client=r)
    assert all(c["state"] in ("clean", "rendered") for c in record["clips"].values())
    deps2, calls2, _ = _deps(r, files, tmp_path, llm_calls=calls)
    out = mix_job.run_mix_job({"user_id": user, "mix_id": mix_id}, deps2)
    assert out["state"] == "ready" and len(calls) == 1      # nothing cleaned twice


def test_a_failed_clip_is_named_and_retried_alone(r, tmp_path):
    user, mix_id, files = _mix(r, tmp_path, counts=(2, 1, 1, 1))
    deps, calls, _ = _deps(r, files, tmp_path, fail_transcribe=("s0c1",))
    out = mix_job.run_mix_job({"user_id": user, "mix_id": mix_id}, deps)
    assert out["state"] == "needs_retry" and out["ready"] == 1
    assert [f["clip_id"] for f in out["failed_clips"]] == ["s0c1"] and out["failed_clips"][0]["label"] == "H2"
    record = mix_store.load(user, mix_id, client=r)
    rendered_before = {cid: c.get("clean_uri") for cid, c in record["clips"].items() if c.get("clean_uri")}

    def reset(d):
        d["clips"]["s0c1"] = {"state": "waiting", "section": "Hook", "label": "H2"}
    mix_store.update(user, mix_id, reset, client=r)
    deps2, _, made = _deps(r, files, tmp_path, llm_calls=calls)
    out = mix_job.run_mix_job({"user_id": user, "mix_id": mix_id}, deps2)
    assert out["state"] == "ready" and out["ready"] == 2 and len(made) == 1      # only the missing video made
    assert "CLIP 2" not in calls[-1]["messages"][0]["content"][1]["text"]        # only that clip was cleaned again
    after = mix_store.load(user, mix_id, client=r)
    assert all(after["clips"][cid]["clean_uri"] == uri for cid, uri in rendered_before.items())


def test_the_api_plans_starts_and_reports(monkeypatch, r):
    from fastapi.testclient import TestClient
    from cutsell_app import mix_routes
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(mix_routes.router)
    jobs = []
    monkeypatch.setattr(mix_routes, "_enqueue", lambda u, m: jobs.append(m) or type("J", (), {"job_id": "j1"})())
    monkeypatch.setattr(mix_routes, "validate_product_source_uri", lambda uri, **k: None)
    client = TestClient(app)
    preview = client.post("/v1/mix/plan", json={"user_id": "u1", "sections": ["Hook", "Body", "Benefit", "CTA"],
                                                "counts": [3, 2, 2, 2], "how_many": 12}).json()
    assert preview["total"] == 24 and len(preview["videos"]) == 12 and preview["videos"][0]["code"] == "H1 · B1 · Be1 · C1"
    mix = project_store.create_project(user_id="u1", title="Mix")
    product = product_catalog.create("u1", brand="CeraVe", product="Hydrating Cleanser")
    body = {"user_id": "u1", "project_id": mix["project_id"], "product_id": product["product_id"], "how_many": 6,
            "sections": [{"name": n, "clips": [{"uri": f"s3://b/{n}{i}.mov", "duration_sec": 8} for i in range(c)]}
                         for n, c in zip(["Hook", "Body", "Benefit", "CTA"], [3, 2, 2, 2])]}
    started = client.post("/v1/mix/jobs", json=body)
    assert started.status_code == 202, started.text
    data = started.json()
    assert data["total"] == 6 and data["clip_count"] == 9 and jobs == [mix["project_id"]]
    assert client.post("/v1/mix/jobs", json=body).status_code == 409
    assert project_store.get_project(user_id="u1", project_id=mix["project_id"])["state"] == "mix"
    overview = product_catalog.overview("u1", project_store.list_projects(user_id="u1", limit=100))
    assert overview["brands"][0]["products"][0]["processing"] == 6 and overview["uncategorized"] == []
    listed = client.get("/v1/mix", params={"user_id": "u1"}).json()["mixes"]
    assert listed[0]["videos"][0]["code"] == "H1 · B1 · Be1 · C1"
    assert client.post(f"/v1/mix/{mix['project_id']}/retry", json={"user_id": "u1"}).json()["job_id"] is None


def test_the_cut_engine_is_untouched_by_mix():
    import shutil
    if not shutil.which("git"):
        pytest.skip("no git")
    root = Path(__file__).resolve().parents[1]
    diff = subprocess.run(["git", "diff", "--stat", "f7f5772", "--", "cutsell_worker/simple_engine",
                           "cutsell_worker/simple_engine_adapter.py"], cwd=root, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("no history")
    assert diff.stdout.strip() == ""


def test_a_short_clip_heard_empty_is_heard_again_in_the_mix_language(r, tmp_path):
    user, mix_id, files = _mix(r, tmp_path, counts=(2, 1, 1, 1))
    deps, calls, _ = _deps(r, files, tmp_path)
    plain = deps.transcribe
    asked = []

    def transcribe(audio, **k):
        asked.append(k.get("language_hint"))
        if "s0c1" in audio and not k.get("language_hint"):
            return {"language": None, "words": []}           # Deepgram guessed the wrong language on 2 s alone
        return plain(audio, **k)

    deps.transcribe = transcribe
    out = mix_job.run_mix_job({"user_id": user, "mix_id": mix_id}, deps)
    record = mix_store.load(user, mix_id, client=r)
    assert out["state"] == "ready" and "en" in asked
    assert record["clips"]["s0c1"]["words"] == 4 and record["clips"]["s0c1"]["captions"]


def test_a_clip_heard_in_the_wrong_language_is_heard_again_in_the_mix_language(r, tmp_path):
    """Production, 21 Oct: a 6 s English clip heard alone came back as Dutch ("Deze is er.") and the clip lost its words."""
    user, mix_id, files = _mix(r, tmp_path, counts=(2, 1, 1, 1))
    deps, calls, _ = _deps(r, files, tmp_path)
    plain = deps.transcribe
    asked = []

    def transcribe(audio, **k):
        asked.append((Path(audio).stem.replace("-voice", ""), k.get("language_hint")))
        if "s1c0" in audio and not k.get("language_hint"):
            return {"language": "nl", "words": [{"w": "Deze", "s": 0.2, "e": 0.5}, {"w": "is", "s": 0.5, "e": 0.7},
                                                 {"w": "er.", "s": 0.7, "e": 0.9}]}
        return plain(audio, **k)

    deps.transcribe = transcribe
    out = mix_job.run_mix_job({"user_id": user, "mix_id": mix_id}, deps)
    record = mix_store.load(user, mix_id, client=r)
    assert out["state"] == "ready"
    assert ("s1c0", "en") in asked and record["clips"]["s1c0"]["language"] == "en"
    assert "Deze" not in " ".join(w["w"] for w in record["clips"]["s1c0"]["captions"])
    assert [a for a in asked if a[1]] == [("s1c0", "en")]              # the clips heard right are not paid twice


def test_a_spanish_mix_is_heard_in_spanish_never_forced_to_english(r, tmp_path):
    user, mix_id, files = _mix(r, tmp_path, counts=(2, 1, 1, 1))
    deps, calls, _ = _deps(r, files, tmp_path)
    asked = []

    def transcribe(audio, **k):
        name = Path(audio).stem.replace("-voice", "")
        asked.append((name, k.get("language_hint")))
        if name == "s0c1" and not k.get("language_hint"):        # a 2 s Spanish clip guessed as English alone
            return {"language": "en", "words": [{"w": "Mirror", "s": 0.4, "e": 0.8}]}
        return {"language": k.get("language_hint") or "es",
                "words": [{"w": w, "s": 0.4 + 0.4 * i, "e": 0.7 + 0.4 * i} for i, w in enumerate("mira qué bonito queda".split())]}

    deps.transcribe = transcribe
    out = mix_job.run_mix_job({"user_id": user, "mix_id": mix_id}, deps)
    record = mix_store.load(user, mix_id, client=r)
    assert out["state"] == "ready"
    assert {c["language"] for c in record["clips"].values()} == {"es"}
    assert [a for a in asked if a[1]] == [("s0c1", "es")]                 # heard again in Spanish, nothing in English
    assert "Mirror" not in " ".join(w["w"] for w in record["clips"]["s0c1"]["captions"])


def _tone_wav(path, pieces):
    """pieces: [(seconds, amplitude)] of a 220 Hz voice-like tone over a faint noise floor."""
    import numpy as np
    import wave
    rate, out = 16000, []
    rng = np.random.default_rng(1)
    for sec, amp in pieces:
        t = np.arange(int(sec * rate)) / rate
        out.append(amp * np.sin(2 * np.pi * 220 * t) + 0.002 * rng.standard_normal(len(t)))
    x = (np.clip(np.concatenate(out), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(x.tobytes())
    return str(path)


def test_tighten_edges_leaves_a_tenth_of_a_second_and_never_cuts_the_voice(tmp_path):
    from cutsell_worker import mix_engine as engine
    audio = _tone_wav(tmp_path / "a.wav", [(1.0, 0.0), (1.5, 0.3), (1.2, 0.0)])     # voice from 1.0 to 2.5 s
    # Deepgram's word times start late and end early (1.15 .. 2.35): the voice decides the edge, not the stamps
    words = [{"w": "hello", "s": 1.15, "e": 1.6}, {"w": "there", "s": 1.7, "e": 2.35}]
    parts = engine.tighten_edges([(0.0, 3.7)], words, audio)
    (a, b), = parts
    assert 0.92 <= a <= 0.97 and 2.53 <= b <= 2.58                  # 0.05 s of silence each side, voice whole
    # inner parts untouched; a clip whose voice fills its edges is not shortened
    assert engine.tighten_edges([(0.0, 1.2), (1.3, 3.7)], words, audio)[0][1] == 1.2
    full = _tone_wav(tmp_path / "b.wav", [(2.0, 0.3)])
    assert engine.tighten_edges([(0.0, 2.0)], [{"w": "hi", "s": 0.3, "e": 1.7}], full) == [(0.0, 2.0)]
    assert engine.tighten_edges([(0.0, 3.7)], [], audio) == [(0.0, 3.7)]            # no words: nothing to anchor


def test_mix_clip_uses_sonnet_by_default(monkeypatch):
    from cutsell_worker import mix_engine as engine
    monkeypatch.delenv("CUTSELL_MIX_MODEL", raising=False)
    assert engine.model_name() == "claude-sonnet-5-5"


def test_tighten_edges_leaves_an_edge_alone_when_the_voice_goes_on_past_the_words(tmp_path):
    from cutsell_worker import mix_engine as engine
    audio = _tone_wav(tmp_path / "c.wav", [(0.5, 0.0), (2.5, 0.3), (0.5, 0.0)])    # voice 0.5 .. 3.0
    words = [{"w": "late", "s": 1.4, "e": 1.8}]                    # stamp 0.9 s after the voice starts
    (a, b), = engine.tighten_edges([(0.0, 3.5)], words, audio)
    assert (a, b) == (0.0, 3.5)                                      # voice past both searches: both edges kept


def test_camera_cues_are_hidden_in_a_mix_clip_and_her_lines_are_not():
    from cutsell_worker import mix_engine as engine
    from cutsell_worker.simple_engine import engine as E
    cue = lambda text: engine.is_camera_cue([E._norm(t) for t in text.split()])
    for text in ("Ready? Set.", "Ready.", "Set.", "Action!", "OK, go.", "Okay go", "Rolling.", "One more time.",
                 "Again.", "Take two.", "Three, two, one.", "Uno, dos, tres.", "Are you recording?", "Did you get that?",
                 "¿Ya?", "¿Estás grabando?", "Otra vez.", "Una más.", "Dale.", "Listo, ya.", "And action."):
        assert cue(text), text
    for text in ("Let's go shopping.", "Set it and forget it.", "Ready to glow?", "Go grab yours.", "Two bags.",
                 "Okay.", "One, two, three steps and you're done.", "I'm ready for summer.", "Ya llegó el verano.",
                 "Dale un toque de color."):
        assert not cue(text), text
    words = [{"w": w, "s": i * 0.3, "e": i * 0.3 + 0.25} for i, w in
             enumerate("Ready? Set. These are the Rhino straps. One more time.".split())]
    units = E.lines_of(words)
    lab = engine.clip_labels({"clips": [{"clip": 1, "hide": [], "trim": []}]}, 0, words, units)
    kept = " ".join(w["w"] for w, (st, _) in zip(words, lab) if st == "keep")
    assert kept == "These are the Rhino straps."
    only = [{"w": "Action!", "s": 0.0, "e": 0.4}]                               # never empties a clip
    assert engine.clip_labels({"clips": []}, 0, only, E.lines_of(only))[0][0] == "keep"


def test_the_normal_cut_does_not_hide_camera_cues():
    from cutsell_worker.simple_engine import engine as E
    words = [{"w": w, "s": i * 0.3, "e": i * 0.3 + 0.25} for i, w in enumerate("Ready? Set. These are great.".split())]
    assert all(st == "keep" for st, _ in E.labels({"hide": [], "trim": []}, words, E.lines_of(words)))


def test_a_hidden_cue_said_back_to_back_with_her_words_is_cut_out(tmp_path):
    """Production clip Be2 (21 Oct): Deepgram gave "people. Ready? Set. These" with no gap; the finishing read the
    hidden run as silence and glued it back. A hidden word never plays."""
    import numpy as np
    import wave
    from cutsell_worker import mix_engine as engine
    from cutsell_worker.simple_engine import engine as E
    spec = [("That's okay.", 0.4, 1.28), ("Ugh.", 1.44, 2.0), ("Too many people.", 2.0, 2.64), ("Ready?", 2.64, 3.04),
            ("Set.", 3.04, 3.52), ("These are the Okay.", 3.52, 5.12)]
    words = []
    for text, a, b in spec:
        ws = text.split()
        d = (b - a) / len(ws)
        words += [{"w": w, "s": round(a + i * d, 3), "e": round(a + (i + 1) * d, 3)} for i, w in enumerate(ws)]
    rate, dur = 16000, 6.0
    x = 0.002 * np.random.default_rng(0).standard_normal(int(dur * rate))
    for w in words:
        a, b = int(w["s"] * rate), int(w["e"] * rate)
        x[a:b] += 0.3 * np.sin(2 * np.pi * 200 * np.arange(b - a) / rate)
    path = str(tmp_path / "be2.wav")
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    lab = engine.clip_labels({"clips": [{"clip": 1, "hide": [], "trim": []}]}, 0, words, E.lines_of(words))
    parts = engine.drop_hidden_words(engine.kept_parts(engine.finish_clip(words, lab, dur, path)), words, lab)
    kept = " ".join(w["w"] for w in words if any(a <= (w["s"] + w["e"]) / 2 <= b for a, b in parts))
    assert kept == "That's okay. Ugh. Too many people. These are the Okay."
