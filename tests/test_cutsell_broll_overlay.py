"""B-roll over the voice (broll_overlay.py, PO 9-10 Nov 2026).

No network: Deepgram, Claude and the B-roll matcher are always faked. Media is synthetic (ffmpeg lavfi), made in tmp.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cutsell_worker import broll_overlay as B
from cutsell_worker import overlay_edits, simple_engine_adapter
from cutsell_worker.contracts import ProcessingRequest, SourceAsset, Word, DraftClip
from cutsell_worker.serde import draft_from_dict, result_to_dict
from cutsell_worker.simple_engine import engine as se

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_MODE", "rules")
    monkeypatch.delenv("CUTSELL_BROLL_OVERLAY", raising=False)
    monkeypatch.delenv("CUTSELL_BROLL_AI", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("CUTSELL_BROLL_MATCH_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)          # never a real call from a test
    monkeypatch.delenv("CUTSELL_BROLL_UPLOAD_WAIT_SEC", raising=False)
    clock = FakeClock()
    monkeypatch.setattr(B, "_now", clock.now)                       # no real waiting in any test
    monkeypatch.setattr(B, "_pause", clock.sleep)


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


# 12 spoken lines over 40 s; line 6 talks about the texture, line 12 is the call to action
LINES = [
    (0.4, 2.6, ["Hola", "chicas,", "miren", "esto."]),
    (3.4, 5.6, ["Esta", "crema", "es", "nueva."]),
    (6.4, 8.6, ["La", "compré", "ayer", "mismo."]),
    (9.4, 11.6, ["Huele", "muy", "rico", "también."]),
    (12.4, 14.6, ["Me", "encanta", "cómo", "queda."]),
    (15.4, 17.6, ["Miren", "la", "textura", "suave."]),
    (18.4, 20.6, ["Se", "absorbe", "muy", "rápido."]),
    (21.4, 23.6, ["No", "deja", "nada", "grasoso."]),
    (24.4, 26.6, ["La", "uso", "cada", "mañana."]),
    (27.4, 29.6, ["Mi", "piel", "está", "feliz."]),
    (30.4, 32.6, ["De", "verdad", "la", "recomiendo."]),
    (34.0, 37.5, ["Cómprala", "en", "el", "link", "ya."]),
]
DURATION = 40.0


def _words():
    out = []
    for start, end, tokens in LINES:
        step = (end - start) / len(tokens)
        for i, token in enumerate(tokens):
            out.append({"w": token, "s": round(start + i * step, 3), "e": round(start + (i + 1) * step - 0.03, 3)})
    return out


def _make_voice(path: Path) -> str:
    gate = "+".join(f"between(t,{a},{b})" for a, b, _ in LINES)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s=180x320:d={DURATION}:r=30",
        "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*220*t)*({gate})':d={DURATION}:s=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return str(path)


def _make_broll(path: Path, duration: float, colour: str) -> str:
    # a still colour, then a moving test pattern: the steadiest window is the still part
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={colour}:s=180x320:d={duration / 2}:r=30",
        "-f", "lavfi", "-i", f"testsrc=s=180x320:d={duration / 2}:r=30",
        "-f", "lavfi", "-i", f"sine=f=880:d={duration}",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return str(path)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("broll")
    return {
        "voice": _make_voice(folder / "voice.mp4"),
        "b1": _make_broll(folder / "b1.mov", 6.0, "red"),
        "b2": _make_broll(folder / "b2.mov", 5.0, "green"),
        "b3": _make_broll(folder / "b3.mov", 4.0, "white"),
    }


class FakeLLM:
    """The cut engine's calls (pass, decision, review, word fix + moments): keep everything, mark the CTA."""

    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt: str):
        self.prompts.append(str(prompt))
        if prompt.startswith(se.PASS_PROMPT):
            return '{"passes": [[1, 12, true]], "backbone": [1, 12]}', {"input_tokens": 10, "output_tokens": 5}
        if prompt.startswith(se.FIXWORDS_PROMPT):
            indices = [int(item.split(":", 1)[0]) for item in prompt.split("\n")[-1].split()]
            return json.dumps({"fix": [], "zoom": [[indices[-5], "cta"]]}), {"input_tokens": 5, "output_tokens": 3}
        return '{"hide": [], "trim": []}', {"input_tokens": 20, "output_tokens": 8}


class Hearing:
    def __init__(self):
        self.heard: list[str] = []

    def __call__(self, audio_path, *, language_hint=None):
        self.heard.append(audio_path)
        return {"language": "es", "duration": DURATION, "words": _words()}


def _source(asset_id, order, uri, duration, clip_type=None, **meta):
    metadata = dict(meta)
    if clip_type:
        metadata["clip_type"] = clip_type
    return SourceAsset(source_asset_id=asset_id, project_id="p1", user_id="u1", original_name=Path(uri).name,
                       source_order=order, duration_sec=duration, uri=uri, metadata=metadata)


def _request(media, *, broll=("b1", "b2", "b3"), voice_meta=None):
    sources = [_source("voice", 0, media["voice"], DURATION, "primary", **(voice_meta or {}))]
    for order, key in enumerate(broll, start=1):
        sources.append(_source(key, order, media[key], 5.0, "broll"))
    return ProcessingRequest(project_id="p1", user_id="u1", sources=tuple(sources))


def _paths(media, keys=("voice", "b1", "b2", "b3")):
    return {k: media[k] for k in keys}


def _run(media, request=None, *, ask=None, llm=None, hearing=None, keys=("voice", "b1", "b2", "b3"), **kw):
    return simple_engine_adapter.process_with_simple_engine(
        request or _request(media), _paths(media, keys), transcribe=hearing or Hearing(), llm=llm or FakeLLM(),
        broll_ask=ask, **kw)


def _stable(payload: dict) -> dict:
    """The serialized draft without the per-run prompt-cache counters (they count videos across runs)."""
    out = json.loads(json.dumps(payload))
    for item in out["draft"]["diagnostics"]["simple_engine"]["sources"]:
        item.pop("prompt_cache", None)
    return out


def _texture_ask(content):
    """The matcher: B2 shows the cream's texture -> the phrase "Miren la textura suave."; others: no phrase."""
    text = content[0]["text"]
    number = next(int(line.split()[0][1:]) for line in text.splitlines() if "textura" in line and line.startswith("P"))
    clips = [{"clip": "B1", "shows": "jar on a table", "uses": []},
             {"clip": "B2", "shows": "cream spread on a hand", "uses": [{"phrase": number, "at": 0.83}]},
             {"clip": "B3", "shows": "bathroom shelf", "uses": []}]
    return json.dumps({"clips": clips, "cta_phrase": None}), {"input_tokens": 3000, "output_tokens": 300}


# ------------------------------------------------------------------ d) B-roll is never cut

def test_broll_sources_are_never_heard_cut_or_put_on_the_voice_timeline(media):
    hearing, llm = Hearing(), FakeLLM()
    result = _run(media, ask=_texture_ask, hearing=hearing, llm=llm)
    draft = result.draft
    assert len(hearing.heard) == 1                                   # only the voice went to speech-to-text
    assert {c.source_asset_id for c in (*draft.selected, *draft.alternates, *draft.discarded)} == {"voice"}
    per_source = draft.diagnostics["simple_engine"]["sources"]
    assert [s["source_asset_id"] for s in per_source] == ["voice"]  # no silence refine / decision for B-roll
    plain = FakeLLM()
    _run(media, _request(media, broll=()), llm=plain, keys=("voice",))
    assert len(llm.prompts) == len(plain.prompts)                     # the cut engine made exactly the same calls
    assert all("b1.mov" not in p and "b2.mov" not in p for p in llm.prompts)


# ------------------------------------------------------------------ e) B-roll on top, voice complete

def test_voice_is_identical_with_and_without_broll(media):
    with_broll = _stable(result_to_dict(_run(media, ask=_texture_ask)))
    without = _stable(result_to_dict(_run(media, _request(media, broll=()), keys=("voice",))))
    for key in ("selected", "alternates", "discarded", "zoom_keyframes"):
        assert with_broll["draft"][key] == without["draft"][key], key
    assert with_broll["draft"]["diagnostics"]["simple_engine"] == without["draft"]["diagnostics"]["simple_engine"]
    assert with_broll["draft"]["media_overlays"] and not without["draft"]["media_overlays"]
    assert "broll_overlay" not in without["draft"]["diagnostics"]


def _assert_rules(draft, clip_durations):
    total = B.timeline_duration(draft.selected)
    phrases = B.phrases_of(B.timeline_words(draft.selected))
    overlays = draft.media_overlays
    assert overlays
    starts = {round(p.start, 3) for p in phrases}
    ends = {round(e, 3) for p in phrases for e in p.word_ends}
    diag = draft.diagnostics["broll_overlay"]
    for o in overlays:
        length = o.end - o.start
        assert o.kind == "video" and o.mute_audio is True and o.layout == "full" and o.fill is True
        assert o.start >= B.HOOK_SEC - 1e-6                                   # face for the hook
        assert o.end <= diag["cta_start"] + 1e-6 <= total - B.CTA_MIN_TAIL_SEC + 1e-6   # face for the CTA
        assert B.MIN_SEC - 1e-6 <= length <= B.MAX_SEC + 1e-6
        assert round(o.start, 3) in starts and round(o.end, 3) in ends        # on phrase / word boundaries
        assert o.source_start >= 0 and o.source_end <= clip_durations[o.uri] + 1e-6
        assert abs((o.source_end - o.source_start) - length) < 1e-3
    assert sum(o.end - o.start for o in overlays) <= B.MAX_COVERAGE * total + 1e-6
    assert len({o.uri for o in overlays}) == len(overlays)                   # each clip at most once
    ordered = sorted(overlays, key=lambda o: o.start)
    assert all(a.end <= b.start + 1e-6 for a, b in zip(ordered, ordered[1:]))  # never on top of each other


def test_broll_lands_on_top_obeying_every_rule_and_the_match(media):
    result = _run(media, ask=_texture_ask)
    draft = result.draft
    _assert_rules(draft, {media["b1"]: 6.1, media["b2"]: 5.1, media["b3"]: 4.1})
    assert len(draft.media_overlays) == 3
    texture = next(p for p in B.phrases_of(B.timeline_words(draft.selected)) if "textura" in p.text)
    b2 = next(o for o in draft.media_overlays if o.uri == media["b2"])
    assert b2.start == pytest.approx(texture.start, abs=1e-3)               # "look at the texture" -> B2
    assert b2.source_start <= 0.83 <= b2.source_end                          # around the second the model named
    diag = draft.diagnostics["broll_overlay"]
    assert diag["ai"]["used"] is True and diag["ai"]["frames_sent"] == 9
    assert {s["reason"] for s in diag["sources"] if s["placed"]} <= {"matched", "product", "spread"}
    # the draft survives the stored/served serialization and the API's draft validation
    restored = draft_from_dict(json.loads(json.dumps(result_to_dict(result)))["draft"])
    assert [o.overlay_id for o in restored.media_overlays] == [o.overlay_id for o in draft.media_overlays]


def test_steadiest_window_is_used_when_the_model_names_no_frame(media):
    result = _run(media, ask=lambda content: ('{"clips": [], "cta_phrase": null}', {}))
    b1 = next(o for o in result.draft.media_overlays if o.uri == media["b1"])
    assert b1.source_end <= 3.0 + 0.3                                         # the still half (0-3 s) of B1


@pytest.mark.parametrize("ask", [
    None,                                                                     # no frames asked -> real call is not made
    lambda content: (_ for _ in ()).throw(RuntimeError("Anthropic HTTP 529")),
    lambda content: ("sorry, no json", {}),
    lambda content: ('{"clips": "nope"}', {}),
])
def test_fallback_when_the_matcher_fails_is_an_even_spread_under_the_same_rules(media, monkeypatch, ask):
    if ask is None:
        monkeypatch.setenv("CUTSELL_BROLL_AI", "0")
    result = _run(media, ask=ask or (lambda content: pytest.fail("the matcher must not be called")))
    draft = result.draft
    _assert_rules(draft, {media["b1"]: 6.1, media["b2"]: 5.1, media["b3"]: 4.1})
    diag = draft.diagnostics["broll_overlay"]
    assert diag["ai"]["used"] is False
    assert {s["reason"] for s in diag["sources"] if s["placed"]} == {"spread"}
    assert len(draft.media_overlays) == 3


def test_broll_never_fails_the_cut(media, monkeypatch):
    monkeypatch.setattr(B, "plan_placements", lambda **kw: (_ for _ in ()).throw(ZeroDivisionError()))
    result = _run(media, ask=_texture_ask)
    assert result.draft.selected and result.draft.media_overlays == ()
    assert result.draft.diagnostics["broll_overlay"]["error"] == "ZeroDivisionError"


def test_product_mention_places_broll_when_no_phrase_matches(media):
    request = _request(media, broll=("b1",), voice_meta={"brand": "Lumi", "product": "Piel Feliz"})
    result = _run(media, request, ask=lambda content: ('{"clips": [{"clip": "B1", "shows": "x", "uses": []}], '
                                                       '"cta_phrase": null}', {}),
                  keys=("voice", "b1"))
    (overlay,) = result.draft.media_overlays
    named = next(p for p in B.phrases_of(B.timeline_words(result.draft.selected)) if "piel" in p.text)
    assert overlay.start == pytest.approx(named.start, abs=1e-3)
    assert result.draft.diagnostics["broll_overlay"]["sources"][0]["reason"] == "product"


def test_the_editor_can_move_trim_and_delete_the_placed_blocks(media):
    draft = json.loads(json.dumps(result_to_dict(_run(media, ask=_texture_ask))))["draft"]
    first = draft["media_overlays"][0]
    moved = overlay_edits.update_media_overlay(draft, first["overlay_id"], start=first["start"] + 0.5,
                                               end=first["end"] + 0.5)
    assert moved["media_overlays"][0]["start"] == round(first["start"] + 0.5, 3)
    trimmed = overlay_edits.update_media_overlay(draft, first["overlay_id"], end=first["start"] + 1.0)
    assert trimmed["media_overlays"][0]["end"] == round(first["start"] + 1.0, 3)
    removed = overlay_edits.remove_media_overlay(draft, first["overlay_id"])
    assert len(removed["media_overlays"]) == len(draft["media_overlays"]) - 1
    assert removed["selected"] == draft["selected"]                          # the voice is untouched by the edits


# ------------------------------------------------------------------ regression: no B-roll = as before

def _adapter_before():
    """The adapter exactly as it was before this change (git), loaded next to the current package."""
    root = Path(__file__).resolve().parents[1]
    shown = subprocess.run(["git", "show", "f4798ed6:cutsell_worker/simple_engine_adapter.py"], cwd=root,
                           capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip("no git history")
    name = "cutsell_worker._adapter_before_broll"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "cutsell_worker"
    sys.modules[name] = module
    exec(compile(shown.stdout, "adapter_before.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("meta", [None, {"clip_type": "primary"}, {"album_hint": "Lumi"}])
def test_no_broll_request_gives_the_same_output_as_before(media, meta):
    if not shutil.which("git"):
        pytest.skip("no git")
    before = _adapter_before()
    request = _request(media, broll=(), voice_meta=None)
    if meta is not None:
        request = replace(request, sources=(replace(request.sources[0], metadata=dict(meta)),))
    old = before.process_with_simple_engine(request, _paths(media, ("voice",)), transcribe=Hearing(), llm=FakeLLM())
    new = simple_engine_adapter.process_with_simple_engine(request, _paths(media, ("voice",)), transcribe=Hearing(),
                                                           llm=FakeLLM())
    assert json.dumps(_stable(result_to_dict(new)), sort_keys=True) == json.dumps(_stable(result_to_dict(old)), sort_keys=True)


def test_switch_off_treats_every_source_as_voice_exactly_as_before(media, monkeypatch):
    monkeypatch.setenv("CUTSELL_BROLL_OVERLAY", "0")
    voice, broll = B.split_sources(_request(media).sources)
    assert len(voice) == 4 and broll == []


def test_a_request_of_only_broll_is_cut_as_voice():
    sources = [_source("a", 0, "a.mov", 5, "broll"), _source("b", 1, "b.mov", 5, "broll")]
    voice, broll = B.split_sources(sources)
    assert [s.source_asset_id for s in voice] == ["a", "b"] and broll == []
    assert B.is_broll(_source("c", 0, "c.mov", 5, " BRoll ")) and not B.is_broll(_source("d", 0, "d.mov", 5, "primary"))


# ------------------------------------------------------------------ rules (pure)

def _clip(i, duration, order=0):
    return B.BrollClip(source=_source(f"b{i}", i, f"s3://bk/b{i}.mov", duration, "broll"), duration=duration,
                       order=order)


def _photo_clip(i, order=0):
    return B.BrollClip(source=_source(f"p{i}", i, f"s3://bk/p{i}.jpg", 0.0, "broll", broll_kind="photo"),
                       duration=0.0, kind="photo", order=order, frames=[{"t": 0.0, "jpeg_b64": "AAAA"}])


def _phrases(n=20, step=2.0):
    words = [{"w": f"w{i}.", "s": round(i * step + 0.1, 3), "e": round(i * step + 1.6, 3), "clip": "c"} for i in range(n)]
    return words, B.phrases_of(words)


def test_coverage_never_passes_half_of_the_video():
    words, phrases = _phrases(10, 2.0)                  # a 20 s video
    clips = [_clip(i, 6.0) for i in range(8)]
    placed, unplaced = B.plan_placements(total=20.0, phrases=phrases, words=words, clips=clips,
                                         cta_start=B.protected_tail_start(20.0, phrases), product_times=())
    assert sum(p.end - p.start for p in placed) <= 10.0 + 1e-6
    assert len({p.clip_index for p in placed}) == len(placed) and unplaced


def test_a_clip_shorter_than_two_seconds_is_not_used():
    words, phrases = _phrases()
    placed, unplaced = B.plan_placements(total=40.0, phrases=phrases, words=words, clips=[_clip(0, 1.5)],
                                         cta_start=30.0, product_times=())
    assert placed == [] and unplaced == {0: "clip_shorter_than_min"}


def test_closing_call_to_action_keeps_the_face():
    words, phrases = _phrases()
    assert B.protected_tail_start(40.0, phrases) == pytest.approx(min(37.0, phrases[-1].start))
    assert B.protected_tail_start(40.0, phrases, cta_times=[33.0]) == pytest.approx(32.1)
    assert B.protected_tail_start(40.0, phrases, cta_times=[5.0]) == pytest.approx(37.0)  # not closing
    assert B.protected_tail_start(40.0, phrases, cta_phrase=15) == pytest.approx(phrases[14].start)


def test_parse_answer_keeps_only_valid_suggestions():
    clips = [_clip(1, 5.0), _clip(2, 20.0), _photo_clip(3)]
    answer = B.parse_answer('note {"clips": [{"clip": "B1", "shows": "hand", "uses": [{"phrase": 3, "at": 2.0}, '
                            '{"phrase": 4, "at": 4.0}]}, {"clip": "B9", "uses": [{"phrase": 1, "at": 0}]}, '
                            '{"clip": "B2", "uses": [{"phrase": 99, "at": 1}, {"phrase": 5, "at": 30}, '
                            '{"phrase": 6, "at": 3.5}, {"phrase": 6, "at": 9}, {"phrase": 7, "at": 12}, '
                            '{"phrase": 8, "at": 16}, {"phrase": 9, "at": 18}]}, '
                            '{"clip": "B3", "uses": [{"phrase": 2, "at": 1.5}]}], "cta_phrase": 4}', clips, 10)
    assert answer == {"by_clip": {
        0: {"uses": [{"phrase": 3, "at": 2.0}], "shows": "hand"},                 # a short clip: one use
        1: {"uses": [{"phrase": 5, "at": None}, {"phrase": 6, "at": 3.5}, {"phrase": 7, "at": 12.0}], "shows": ""},
        2: {"uses": [{"phrase": 2, "at": None}], "shows": ""}},                   # a photo: no "at"
        "cta_phrase": 4}
    with pytest.raises(ValueError):
        B.parse_answer("no", clips, 1)


def test_best_window_prefers_the_steady_part_and_the_chosen_frame():
    motion = [9.0] * 8 + [0.5] * 8 + [9.0] * 8       # 6 s at 4 fps, steady between 2 and 4 s
    assert B.best_window(6.0, 2.0, motion=motion) == pytest.approx(2.0)
    assert B.best_window(6.0, 2.0, motion=motion, near=5.5) >= 3.5
    assert B.best_window(3.0, 3.0) == 0.0


def test_the_matcher_call_sends_three_frames_per_clip_and_asks_for_json(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    sent = {}

    class Response:
        status_code = 200

        def json(self):
            return {"content": [{"type": "text", "text": '{"clips": [], "cta_phrase": null}'}],
                    "usage": {"input_tokens": 1, "output_tokens": 1}}

    def post(url, json=None, **kw):
        sent.update(json)
        return Response()

    _, phrases = _phrases(3)
    clip = _clip(1, 5.0)
    clip.frames = [{"t": t, "jpeg_b64": "AAAA"} for t in (0.8, 2.5, 4.2)]
    text, usage = B.match_with_claude(B.build_content(phrases, [clip], product_hint="Lumi"), post=post)
    content = sent["messages"][0]["content"]
    assert sum(1 for c in content if c["type"] == "image") == 3
    assert sent["model"] == "claude-haiku-5-5" and usage["model"] == "claude-haiku-5-5"
    assert sent["output_config"]["format"]["type"] == "json_schema" and usage["step"] == "broll_match"
    assert "PRODUCT: Lumi" in content[0]["text"] and "P1 [" in content[0]["text"]


def test_overlay_uri_may_name_this_projects_own_source_only(monkeypatch):
    from types import SimpleNamespace
    from cutsell_worker import overlay_uploads, uploads
    monkeypatch.setattr(overlay_uploads, "load_runtime_config", lambda: SimpleNamespace(s3_bucket="bk"))
    monkeypatch.setattr(overlay_uploads, "is_readable_bucket", lambda bucket, current: bucket == current)
    mine = uploads.scoped_upload_prefix(user_id="u1", project_id="p1")
    assert overlay_uploads.validate_overlay_uri(f"s3://bk/{mine}x-b.mov", user_id="u1", project_id="p1")[2] == "video"
    for uri in (f"s3://bk/{uploads.scoped_upload_prefix(user_id='u2', project_id='p1')}x-b.mov",
                f"s3://bk/{uploads.scoped_upload_prefix(user_id='u1', project_id='p2')}x-b.mov",
                f"s3://bk/{mine}x-b.png", f"s3://other/{mine}x-b.mov"):
        with pytest.raises(ValueError):
            overlay_uploads.validate_overlay_uri(uri, user_id="u1", project_id="p1")


def test_primary_video_is_the_first_voice_source():
    from cutsell_worker import worker_job
    request = ProcessingRequest(project_id="p1", user_id="u1", sources=(
        _source("b", 0, "b.mov", 5, "broll"), _source("v", 1, "v.mov", 30, "primary")))
    assert worker_job._first_voice_source(request).source_asset_id == "v"
    plain = ProcessingRequest(project_id="p1", user_id="u1", sources=(_source("a", 0, "a.mov", 5), _source("c", 1, "c.mov", 5)))
    assert worker_job._first_voice_source(plain).source_asset_id == "a"


def test_timeline_words_follow_the_cut_order_and_speed():
    clips = (
        DraftClip(clip_id="c1", source_asset_id="v", source_order=0, start=10.0, end=12.0, text="", caption_text="",
                  words=(Word("uno", 10.2, 10.6),)),
        DraftClip(clip_id="c2", source_asset_id="v", source_order=0, start=1.0, end=5.0, text="", caption_text="",
                  words=(Word("dos", 2.0, 2.4),), speed=2.0),
    )
    words = B.timeline_words(clips)
    assert [(w["w"], w["s"]) for w in words] == [("uno", 0.2), ("dos", 2.5)]
    assert B.timeline_duration(clips) == pytest.approx(4.0)
    assert B.source_time_to_timeline(clips, "v", 3.0) == pytest.approx(3.0)


def test_fallback_spread_is_even_over_the_middle_of_the_video():
    words, phrases = _phrases(20, 2.0)                  # a 40 s video, a phrase every 2 s
    clips = [_clip(i, 5.0) for i in range(3)]
    cta = B.protected_tail_start(40.0, phrases)
    placed, _ = B.plan_placements(total=40.0, phrases=phrases, words=words, clips=clips, cta_start=cta,
                                  product_times=())
    starts = [p.start for p in placed]
    targets = [B.HOOK_SEC + (cta - B.HOOK_SEC) * (k + 0.5) / 3 for k in range(3)]
    assert len(starts) == 3 and all(abs(s - t) <= 2.0 for s, t in zip(starts, targets))


# ================================================================== B-roll v2 (PO 11 Nov 2026)

def _make_photo(path: Path, size="1200x800") -> str:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=s={size}:d=1", "-frames:v", "1",
                    str(path)], check=True, capture_output=True)
    return str(path)


@pytest.fixture(scope="module")
def media2(tmp_path_factory, media):
    folder = tmp_path_factory.mktemp("broll2")
    return {**media, "long": _make_broll(folder / "long.mov", 20.0, "yellow"),
            "photo": _make_photo(folder / "photo.jpg"), "voice2": _make_voice(folder / "voice2.mp4")}


def _broll_source(key, order, uri, duration, *, n, keep="0", kind="video"):
    return _source(key, order, uri, duration, "broll", broll_kind=kind, broll_order=str(n), broll_keep_order=keep)


def _phrase_numbers(content, *needles):
    text = content[0]["text"]
    lines = [line for line in text.splitlines() if line.startswith("P") and "[" in line]
    return [next(int(line.split()[0][1:]) for line in lines if needle in line) for needle in needles]


def _picks(*uses):
    """uses[k] = list of (needle, at) for clip B(k+1)."""
    def ask(content):
        clips = []
        for k, items in enumerate(uses, start=1):
            numbers = _phrase_numbers(content, *[n for n, _ in items]) if items else []
            clips.append({"clip": f"B{k}", "shows": "x",
                          "uses": [{"phrase": number, "at": at} for number, (_, at) in zip(numbers, items)]})
        ask.content = content
        return json.dumps({"clips": clips, "cta_phrase": None}), {"input_tokens": 100, "output_tokens": 10,
                                                                    "model": "claude-haiku-5-5"}
    return ask


def _ordered_request(media, keep):
    # broll_order differs from source_order on purpose: the creator's order is b3, b1, b2
    sources = [_source("voice", 0, media["voice"], DURATION, "primary"),
               _broll_source("b1", 1, media["b1"], 6.0, n=2, keep=keep),
               _broll_source("b2", 2, media["b2"], 5.0, n=3, keep=keep),
               _broll_source("b3", 3, media["b3"], 4.0, n=1, keep=keep)]
    return ProcessingRequest(project_id="p1", user_id="u1", sources=tuple(sources))


# ------------------------------------------------------------------ e) keep_order on -> 1-2-3

def test_keep_order_on_shows_the_shots_in_the_creators_order_even_when_the_model_reverses_them(media):
    # in the prompt the clips are B1=b3 (order 1), B2=b1 (order 2), B3=b2 (order 3); the model picks them backwards
    ask = _picks([("cada mañana", 1.0)], [("absorbe", 1.0)], [("Esta crema", 1.0)])
    result = _run(media, _ordered_request(media, "1"), ask=ask)
    overlays = sorted(result.draft.media_overlays, key=lambda o: o.start)
    assert [o.uri for o in overlays] == [media["b3"], media["b1"], media["b2"]]        # 1 before 2 before 3
    assert "ORDER:" in ask.content[0]["text"]
    diag = result.draft.diagnostics["broll_overlay"]
    assert diag["keep_order"] is True
    assert {s["reason"] for s in diag["sources"]} <= {"matched", "reordered"}
    _assert_rules(result.draft, {media["b1"]: 6.1, media["b2"]: 5.1, media["b3"]: 4.1})


def test_keep_order_on_keeps_every_pick_of_the_model_that_respects_the_order(media):
    ask = _picks([("compré", 1.0)], [("textura", 1.0)], [("cada mañana", 1.0)])
    result = _run(media, _ordered_request(media, "1"), ask=ask)
    phrases = B.phrases_of(B.timeline_words(result.draft.selected))
    starts = {o.uri: o.start for o in result.draft.media_overlays}
    for uri, needle in ((media["b3"], "compré"), (media["b1"], "textura"), (media["b2"], "cada")):
        assert starts[uri] == pytest.approx(next(p.start for p in phrases if needle in p.text), abs=1e-3)
    assert {s["reason"] for s in result.draft.diagnostics["broll_overlay"]["sources"]} == {"matched"}


@pytest.mark.parametrize("ask", [None, lambda content: ("not json", {})])
def test_keep_order_fallback_is_an_even_spread_in_the_creators_order(media, monkeypatch, ask):
    if ask is None:
        monkeypatch.setenv("CUTSELL_BROLL_AI", "0")
    result = _run(media, _ordered_request(media, "1"), ask=ask or (lambda c: pytest.fail("no call")))
    overlays = sorted(result.draft.media_overlays, key=lambda o: o.start)
    assert [o.uri for o in overlays] == [media["b3"], media["b1"], media["b2"]]
    assert {s["reason"] for s in result.draft.diagnostics["broll_overlay"]["sources"]} == {"spread"}


def test_best_ordered_assignment_is_monotonic_and_keeps_the_most_picks():
    words, phrases = _phrases(20, 2.0)
    starts = [p for p in phrases if p.start >= 3.0]
    requests = [{"phrase": 15}, {"phrase": 5}, {"phrase": 10}, {"phrase": None}]
    chosen = B.best_ordered_assignment(requests, starts, lo=3.0, hi=36.0, total=40.0, phrases=phrases)
    picked = [starts[j].index for _, j in chosen]
    assert [i for i, _ in chosen] == [0, 1, 2, 3] and picked == sorted(picked)
    assert sum(1 for (i, j) in chosen if starts[j].index == requests[i]["phrase"]) == 2   # 2 of 3 picks kept


# ------------------------------------------------------------------ f) keep_order off -> the model's order

def test_keep_order_off_follows_the_models_placement_freely(media):
    ask = _picks([("cada mañana", 1.0)], [("absorbe", 1.0)], [("compré", 1.0)])
    result = _run(media, _ordered_request(media, "0"), ask=ask)
    overlays = sorted(result.draft.media_overlays, key=lambda o: o.start)
    assert [o.uri for o in overlays] == [media["b2"], media["b1"], media["b3"]]        # as the model said
    assert result.draft.diagnostics["broll_overlay"]["keep_order"] is False
    assert {s["reason"] for s in result.draft.diagnostics["broll_overlay"]["sources"]} == {"matched"}


# ------------------------------------------------------------------ g) long clips

def test_frame_count_rule():
    assert [B.frame_count(d) for d in (0, 4, 9.9, 10, 12.4, 20, 25, 60)] == [0, 3, 3, 4, 5, 8, 10, 10]
    assert B.frame_times(20.0) == [1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75]   # one every 2.5 s
    assert B.frame_times(6.0) == [1.0, 3.0, 5.0]
    assert _clip(0, 9.0).max_uses == 1 and _clip(0, 10.0).max_uses == 2 and _clip(0, 20.0).max_uses == 3


def _long_request(media):
    return ProcessingRequest(project_id="p1", user_id="u1", sources=(
        _source("voice", 0, media["voice"], DURATION, "primary"),
        _broll_source("long", 1, media["long"], 20.0, n=1)))


def test_long_clip_gets_more_frames_and_the_piece_around_the_named_second(media2):
    ask = _picks([("textura", 15.0), ("cada mañana", 4.0)])
    result = _run(media2, _long_request(media2), ask=ask, keys=("voice", "long"))
    assert sum(1 for c in ask.content if c["type"] == "image") == 8
    assert "may be used 3 times" in "".join(c.get("text", "") for c in ask.content)
    pieces = sorted(result.draft.media_overlays, key=lambda o: o.start)
    assert len(pieces) == 2 and {o.overlay_id for o in pieces} == {"ai-broll-001", "ai-broll-001-2"}
    phrases = B.phrases_of(B.timeline_words(result.draft.selected))
    texture = next(p for p in phrases if "textura" in p.text)
    first, second = pieces
    assert first.start == pytest.approx(texture.start, abs=1e-3) and first.source_start <= 15.0 <= first.source_end
    assert second.source_start <= 4.0 <= second.source_end
    assert first.source_end <= second.source_start or second.source_end <= first.source_start   # different parts


def test_the_same_part_of_a_long_clip_is_never_used_twice(media2):
    ask = _picks([("textura", 15.0), ("cada mañana", 14.5)])                         # the same part twice
    result = _run(media2, _long_request(media2), ask=ask, keys=("voice", "long"))
    assert len(result.draft.media_overlays) == 1
    words, phrases = _phrases()
    clip = _clip(0, 20.0)
    suggestions = {0: {"uses": [{"phrase": 5, "at": 10.0}, {"phrase": 12, "at": 11.0}]}}
    placed, _ = B.plan_placements(total=40.0, phrases=phrases, words=words, clips=[clip], suggestions=suggestions,
                                  cta_start=35.0)
    assert len(placed) == 1


# ------------------------------------------------------------------ h) photos

def _photo_request(media, keep="0"):
    return ProcessingRequest(project_id="p1", user_id="u1", sources=(
        _source("voice", 0, media["voice"], DURATION, "primary"),
        _broll_source("b1", 1, media["b1"], 6.0, n=1, keep=keep),
        _broll_source("photo", 2, media["photo"], 0.0, n=2, keep=keep, kind="photo")))


def test_photo_is_one_image_and_becomes_a_ken_burns_photo_overlay(media2):
    hearing = Hearing()
    ask = _picks([], [("textura", None)])
    result = _run(media2, _photo_request(media2), ask=ask, hearing=hearing, keys=("voice", "b1", "photo"))
    assert len(hearing.heard) == 1                                       # a photo is never sent to speech-to-text
    assert "B2: PHOTO" in "".join(c.get("text", "") for c in ask.content)
    assert sum(1 for c in ask.content if c["type"] == "image") == 3 + 1  # 3 frames of b1 + the photo
    photo = next(o for o in result.draft.media_overlays if o.uri == media2["photo"])
    assert photo.kind == "photo" and photo.motion == "ken_burns"
    assert (photo.motion_zoom_from, photo.motion_zoom_to) == (1.0, 1.08)
    assert photo.layout == "full" and photo.fill is True and photo.mute_audio is True
    assert photo.source_start == 0.0 and photo.source_end is None
    assert B.MIN_SEC - 1e-6 <= photo.end - photo.start <= B.MAX_SEC + 1e-6
    item = next(o for o in result_to_dict(result)["draft"]["media_overlays"] if o["kind"] == "photo")
    assert {k: item[k] for k in ("motion", "motion_zoom_from", "motion_zoom_to", "layout", "fill", "mute_audio")} == {
        "motion": "ken_burns", "motion_zoom_from": 1.0, "motion_zoom_to": 1.08, "layout": "full", "fill": True,
        "mute_audio": True}
    restored = draft_from_dict(json.loads(json.dumps(result_to_dict(result)))["draft"])
    assert next(o for o in restored.media_overlays if o.kind == "photo").motion == "ken_burns"
    assert result.draft.diagnostics["simple_engine"]["sources"][0]["source_asset_id"] == "voice"


def test_a_video_overlay_never_carries_motion_and_still_overlays_serialize_as_before():
    from cutsell_worker.contracts import MediaOverlay
    from cutsell_worker.serde import _json_safe
    still = MediaOverlay(overlay_id="o", kind="video", uri="s3://b/v.mov", start=1.0, end=3.0)
    assert set(_json_safe(still)) == {"overlay_id", "kind", "uri", "start", "end", "x", "y", "width", "source_start",
                                     "source_end", "mute_audio"}
    base = {"overlay_id": "o", "uri": "s3://b/v.mov", "start": 0.0, "end": 1.0}
    with pytest.raises(ValueError):
        draft_from_dict({"project_id": "p", "selected": [], "media_overlays": [{**base, "kind": "video",
                                                                                "motion": "ken_burns"}]})


def _old_renderer():
    root = Path(__file__).resolve().parents[1]
    shown = subprocess.run(["git", "show", "623cfd8f:cutsell_worker/media_overlay_render.py"], cwd=root,
                           capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip("no git history")
    name = "cutsell_worker._overlay_render_before_v2"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "cutsell_worker"
    sys.modules[name] = module
    exec(compile(shown.stdout, "render_before.py", "exec"), module.__dict__)
    return module


def test_renderer_zooms_a_ken_burns_photo_and_leaves_every_other_overlay_byte_identical(media2, tmp_path):
    from cutsell_worker import media_overlay_render as R
    from cutsell_worker.contracts import MediaOverlay
    photo = MediaOverlay(overlay_id="p", kind="photo", uri="s3://b/p.jpg", start=4.0, end=7.0, width=1.0,
                         layout="full", fill=True, motion="ken_burns", motion_zoom_from=1.0, motion_zoom_to=1.08)
    command = R.build_final_overlay_command(media2["voice"], str(tmp_path / "out.mp4"),
                                            media_overlays=[R.LocalMediaOverlay(photo, media2["photo"])],
                                            width=180, height=320)
    graph = command[command.index("-filter_complex") + 1]
    assert "zoompan=z='1.00000+0.08000*min(on/89,1)'" in graph and "s=180x320" in graph and "d=90" in graph
    assert "x='iw/2-(iw/zoom/2)'" in graph and "crop=360:640" in graph            # centred, fills 9:16
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    frames = []
    for t in (4.1, 6.9):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(tmp_path / "out.mp4"), "-frames:v", "1",
                              "-vf", "scale=45:80,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
        frames.append(raw)
    assert frames[0] and frames[0] != frames[1]                                    # the picture moves (zoom)
    # every overlay without motion: exactly the command of the renderer before this change
    old = _old_renderer()
    others = [
        MediaOverlay(overlay_id="v", kind="video", uri="s3://b/v.mov", start=1.0, end=3.0, width=1.0, layout="full",
                     fill=True, source_start=0.5, source_end=2.5),
        MediaOverlay(overlay_id="s", kind="photo", uri="s3://b/s.jpg", start=2.0, end=5.0, width=1.0, layout="full",
                     fill=True),
        MediaOverlay(overlay_id="f", kind="photo", uri="s3://b/f.jpg", start=0.5, end=2.0, x=0.3, y=0.4, width=0.5),
    ]
    paths = {"v": media2["b1"], "s": media2["photo"], "f": media2["photo"]}
    new_cmd = R.build_final_overlay_command("in.mp4", "out.mp4", width=180, height=320, media_overlays=[
        R.LocalMediaOverlay(o, paths[o.overlay_id]) for o in others])
    old_cmd = old.build_final_overlay_command("in.mp4", "out.mp4", width=180, height=320, media_overlays=[
        old.LocalMediaOverlay(o, paths[o.overlay_id]) for o in others])
    assert new_cmd == old_cmd


def test_photo_source_is_accepted_end_to_end_by_the_upload_and_uri_checks(monkeypatch):
    from types import SimpleNamespace
    from cutsell_worker import overlay_uploads, uploads

    class FakeS3:
        def generate_presigned_post(self, **kw):
            return {"url": "https://up", "fields": {}}

    config = lambda: SimpleNamespace(s3_bucket="bk", aws_region="us-east-1")
    monkeypatch.setattr(uploads, "load_runtime_config", config)
    monkeypatch.setattr(overlay_uploads, "load_runtime_config", config)
    monkeypatch.setattr(overlay_uploads, "is_readable_bucket", lambda bucket, current: bucket == current)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    signed = uploads.create_presigned_upload(project_id="p1", user_id="u1", original_name="IMG_1.jpg",
                                             content_type="image/jpeg", size_bytes=2_000_000, client=FakeS3())
    assert signed["source_uri"].endswith("-IMG_1.jpg") and signed["content_type"] == "image/jpeg"
    for name, kind, size in (("IMG_1.jpg", "video/quicktime", 100), ("IMG_1.mov", "image/jpeg", 100),
                             ("IMG_1.png", "image/png", 100), ("IMG_1.jpg", "image/jpeg", 41 * 1024 * 1024)):
        with pytest.raises(ValueError):
            uploads.prepare_upload_target(project_id="p1", user_id="u1", original_name=name, content_type=kind,
                                          size_bytes=size)
    uri = signed["source_uri"]
    with pytest.raises(ValueError):
        uploads.validate_product_source_uri(uri, project_id="p1", user_id="u1")       # not where a photo is unexpected
    assert uploads.validate_product_source_uri(uri, project_id="p1", user_id="u1", allow_photo=True)
    assert overlay_uploads.validate_overlay_uri(uri, user_id="u1", project_id="p1")[2] == "photo"
    other = uri.replace(uploads.scoped_upload_prefix(user_id="u1", project_id="p1"),
                        uploads.scoped_upload_prefix(user_id="u2", project_id="p1"))
    with pytest.raises(ValueError):
        overlay_uploads.validate_overlay_uri(other, user_id="u1", project_id="p1")
    # the API: a JPEG only as a B-roll photo; 0 minutes for the usage meter
    from cutsell_app.auth_middleware import _is_broll_photo
    assert _is_broll_photo({"uri": uri, "metadata": {"clip_type": "broll", "broll_kind": "photo"}})
    assert not _is_broll_photo({"uri": uri, "metadata": {"clip_type": "primary"}})
    assert not _is_broll_photo({"uri": "s3://bk/x.mov", "metadata": {"clip_type": "broll"}})


def test_submit_route_takes_a_jpeg_only_as_a_broll_photo(monkeypatch):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    import cutsell_app.main as api
    seen = []
    monkeypatch.setattr(api, "validate_product_source_uri", lambda uri, **kw: seen.append((uri, kw)))
    monkeypatch.setattr(api, "enqueue_flow_b", lambda payload: SimpleNamespace(job_id="j", queue_name="q"))
    from cutsell_app import auth_middleware
    monkeypatch.setattr(auth_middleware, "check_processing_allowance",
                        lambda **kw: SimpleNamespace(allowed=True, reason=""), raising=False)
    sources = [{"original_name": "v.mov", "uri": "s3://b/v.mov", "source_order": 0, "metadata": {"clip_type": "primary"}},
               {"original_name": "p.jpg", "uri": "s3://b/p.jpg", "source_order": 1,
                "metadata": {"clip_type": "broll", "broll_kind": "photo", "broll_order": "1", "broll_keep_order": "1"}}]
    request = api.FlowBSubmitRequest(project_id="p1", user_id="u1", sources=sources)
    api.submit_flow_b(request)
    assert [kw["allow_photo"] for _, kw in seen] == [False, True]


def test_photo_counts_zero_minutes_in_the_cut(media2, monkeypatch):
    from cutsell_worker import simple_engine_adapter as adapter
    asked = []
    monkeypatch.setattr(adapter, "check_processing_allowance",
                        lambda **kw: asked.append(kw["durations_sec"]) or SimpleNamespaceAllowed())
    _run(media2, _photo_request(media2), ask=_picks([], []), keys=("voice", "b1", "photo"))
    assert asked[0][2] == 0.0 and asked[0][0] == pytest.approx(DURATION, abs=0.2)


class SimpleNamespaceAllowed:
    allowed = True
    reason = ""


def test_only_photos_and_no_voice_is_refused_clearly():
    sources = [_source("p", 0, "p.jpg", 0, "broll", broll_kind="photo")]
    assert B.split_sources(sources) == ([], [])


# ------------------------------------------------------------------ i) Haiku for the images, the cut model unchanged

def test_the_image_call_uses_the_broll_model_and_the_cut_model_is_unchanged(media, monkeypatch):
    from cutsell_worker.simple_engine import llm as se_llm
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CUTSELL_SIMPLE_ENGINE_MODEL", raising=False)
    bodies = []

    class Response:
        status_code = 200

        def json(self):
            return {"content": [{"type": "text", "text": '{"clips": [], "cta_phrase": null}'}],
                    "usage": {"input_tokens": 2600, "output_tokens": 300}}

    post = lambda url, json=None, **kw: bodies.append(json) or Response()
    ask = lambda content: B.match_with_claude(content, post=post)
    result = _run(media, ask=ask)
    assert bodies[0]["model"] == "claude-haiku-5-5"                              # default: Haiku
    assert result.draft.diagnostics["simple_engine"]["model"] == "claude-sonnet-5-5" == se_llm.model_name()
    ai = result.draft.diagnostics["broll_overlay"]["ai"]
    assert ai["model"] == "claude-haiku-5-5" and ai["cost_usd"] == pytest.approx((2600 * 0.10 + 300 * 0.50) / 1e6)
    monkeypatch.setenv("CUTSELL_BROLL_MATCH_MODEL", "claude-haiku-9-9")          # read at call time, no redeploy
    _run(media, ask=ask)
    assert bodies[1]["model"] == "claude-haiku-9-9" and se_llm.model_name() == "claude-sonnet-5-5"
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_MODEL", "claude-sonnet-x")
    assert B.match_model() == "claude-haiku-9-9"                                 # the two never mix


# ------------------------------------------------------------------ b) several talking videos joined in order

def test_several_primary_sources_are_joined_in_order_and_overlays_live_on_the_joined_timeline(media2):
    hearing = Hearing()
    request = ProcessingRequest(project_id="p1", user_id="u1", sources=(
        _source("voice", 0, media2["voice"], DURATION, "primary"),
        _source("voice2", 1, media2["voice2"], DURATION, "primary"),
        _broll_source("b1", 2, media2["b1"], 6.0, n=1)))

    def ask(content):
        lines = [l for l in content[0]["text"].splitlines() if l.startswith("P") and "textura" in l]
        second = int(lines[1].split()[0][1:])                                    # "textura" of the SECOND video
        return json.dumps({"clips": [{"clip": "B1", "shows": "x", "uses": [{"phrase": second, "at": 1.0}]}],
                           "cta_phrase": None}), {}

    result = _run(media2, request, ask=ask, hearing=hearing, keys=("voice", "voice2", "b1"))
    draft = result.draft
    assert len(hearing.heard) == 2                                               # both talks heard, B-roll never
    assert [c.source_asset_id for c in draft.selected] == sorted((c.source_asset_id for c in draft.selected),
                                                                  key=lambda s: s != "voice")   # voice, then voice2
    first_len = sum(c.end - c.start for c in draft.selected if c.source_asset_id == "voice")
    (overlay,) = draft.media_overlays
    assert first_len < overlay.start < overlay.end <= B.timeline_duration(draft.selected)
    in_second = next(c for c in draft.selected if c.source_asset_id == "voice2" and c.text and "textura" in c.text)
    assert overlay.start == pytest.approx(B.source_time_to_timeline(draft.selected, "voice2", in_second.words[0].start),
                                          abs=1e-3)


# ------------------------------------------------------------------ audio first: B-roll without audio_uri

def test_broll_still_uploading_is_placed_blind_and_never_fails_the_cut(media2):
    hearing = Hearing()
    request = _photo_request(media2)
    result = simple_engine_adapter.process_with_simple_engine(
        request, {"voice": media2["voice"]}, transcribe=hearing, llm=FakeLLM(), broll_ask=_picks([], []),
        pending_broll={"b1", "photo"})
    assert len(hearing.heard) == 1 and result.draft.selected
    kinds = sorted(o.kind for o in result.draft.media_overlays)
    assert kinds == ["photo", "video"]                                           # both placed from the declared data
    assert result.draft.diagnostics["broll_overlay"]["ai"] == {"used": False, "reason": "no_frames"}


def test_worker_accepts_audio_first_primary_with_broll_that_has_no_audio_uri(media2, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from cutsell_worker import worker_job
    from cutsell_worker.simple_engine import asr as se_asr, llm as se_llm
    voice_audio = str(tmp_path / "voice.m4a")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", media2["voice"], "-vn", "-c:a", "aac", voice_audio], check=True)
    monkeypatch.setenv("CUTSELL_ENGINE", "simple")
    monkeypatch.setenv("CUTSELL_BROLL_AI", "0")
    hearing = Hearing()
    monkeypatch.setattr(se_asr, "transcribe_words", hearing)
    monkeypatch.setattr(se_llm, "call_anthropic", FakeLLM())
    downloads, validated, drafts = [], [], []
    present = {voice_audio: voice_audio, "s3://bk/up/b1.mov": media2["b1"], "s3://bk/up/photo.jpg": media2["photo"]}

    def download(uri, destination, *, client=None):
        downloads.append(uri)
        if uri not in present:
            raise FileNotFoundError(uri)                                          # still uploading
        shutil.copyfile(present[uri], destination)
        return destination

    monkeypatch.setattr(worker_job, "validate_product_source_uri", lambda uri, **kw: validated.append((uri, kw)))
    monkeypatch.setattr(worker_job, "download_source", download)
    monkeypatch.setattr(worker_job, "resolve_sources_for_editorial_entry",
                        lambda paths, **kw: (dict(paths), [], [], []))
    monkeypatch.setattr(worker_job, "load_runtime_config", lambda: SimpleNamespace(asr_model="tiny"))
    monkeypatch.setattr(worker_job, "create_initial_draft", lambda **kw: drafts.append(kw))
    monkeypatch.setattr(worker_job, "safe_update_project", lambda **kw: {"status": "saved"})
    monkeypatch.setattr(worker_job, "publish_notification", lambda **kw: {"notification_id": "n1"})
    monkeypatch.setattr(worker_job, "generate_filmstrip", lambda *a, **kw: [])
    monkeypatch.setattr(worker_job, "waveform_peaks", lambda *a, **kw: [0.0])
    monkeypatch.setattr(worker_job, "store_timeline_assets", lambda **kw: {"status": "ok"})
    minutes = []
    monkeypatch.setattr(worker_job, "record_processing_minutes", lambda **kw: minutes.append(kw["minutes"]))
    monkeypatch.setattr(worker_job, "release_processing_slot", lambda **kw: None)
    broll_meta = lambda n, kind: {"clip_type": "broll", "broll_kind": kind, "broll_order": n, "broll_keep_order": "1"}
    payload = {"project_id": "p1", "user_id": "u1", "sources": [
        {"source_asset_id": "voice", "uri": "s3://bk/up/voice-not-there-yet.mov", "original_name": "v.mov",
         "source_order": 0, "duration_sec": DURATION, "metadata": {"clip_type": "primary", "audio_uri": voice_audio}},
        {"source_asset_id": "b1", "uri": "s3://bk/up/b1.mov", "original_name": "b1.mov", "source_order": 1,
         "duration_sec": 6.0, "metadata": broll_meta("1", "video")},
        {"source_asset_id": "b2", "uri": "s3://bk/up/b2-not-there-yet.mov", "original_name": "b2.mov",
         "source_order": 2, "duration_sec": 5.0, "metadata": broll_meta("2", "video")},
        {"source_asset_id": "photo", "uri": "s3://bk/up/photo.jpg", "original_name": "photo.jpg", "source_order": 3,
         "duration_sec": 0.0, "metadata": broll_meta("3", "photo")}]}

    result = worker_job.run_flow_b_job(payload)

    assert len(hearing.heard) == 1                                               # only the primary's voice
    assert "s3://bk/up/voice-not-there-yet.mov" not in downloads                 # the primary cut from its audio
    assert {uri: kw.get("allow_photo") for uri, kw in validated if "up/" in uri} == {
        "s3://bk/up/voice-not-there-yet.mov": False, "s3://bk/up/b1.mov": False,
        "s3://bk/up/b2-not-there-yet.mov": False, "s3://bk/up/photo.jpg": True}
    assert result["broll_pending_upload"] == ["b2"] and result["audio_first_sources"] == ["voice"]
    wait = result["broll_upload_wait"]
    assert wait["still_missing"] == ["b2"] and wait["waited_sec"] == pytest.approx(60.0)   # waited the cap, then blind
    overlays = sorted(result["draft"]["media_overlays"], key=lambda o: o["start"])
    assert [o["uri"] for o in overlays] == ["s3://bk/up/b1.mov", "s3://bk/up/b2-not-there-yet.mov",
                                            "s3://bk/up/photo.jpg"]              # 1-2-3, the pending one too
    assert overlays[2]["kind"] == "photo" and overlays[2]["motion"] == "ken_burns"
    assets = {s["source_asset_id"]: s["timeline_assets"]["status"] for s in drafts[0]["sources"]}
    assert assets["photo"] == "not_applicable" and assets["b2"] == "waiting_for_video"
    assert minutes and minutes[0] == pytest.approx((DURATION + 6.0) / 60.0, abs=0.01)   # photo 0, pending 0



# ================================================================== PO decisions 11 Nov: same-clip gap, slots, upload wait

def test_two_pieces_of_one_clip_are_at_least_four_seconds_apart_or_the_second_is_dropped():
    words, phrases = _phrases(20, 2.0)                  # a phrase every 2 s
    clip = _clip(0, 20.0)
    close = {0: {"uses": [{"phrase": 5, "at": 3.0}, {"phrase": 6, "at": 15.0}]}}       # 2 s apart on the timeline
    placed, _ = B.plan_placements(total=40.0, phrases=phrases, words=words, clips=[clip], suggestions=close,
                                  cta_start=35.0)
    assert len(placed) == 1                             # never violated: the second piece is dropped
    far = {0: {"uses": [{"phrase": 5, "at": 3.0}, {"phrase": 9, "at": 15.0}]}}
    placed, _ = B.plan_placements(total=40.0, phrases=phrases, words=words, clips=[clip], suggestions=far,
                                  cta_start=35.0)
    assert len(placed) == 2 and placed[1].start - placed[0].end >= B.SAME_CLIP_GAP_SEC - 1e-6
    for suggestions in (close, far):                    # whatever the answer, the rule holds
        placed, _ = B.plan_placements(total=40.0, phrases=phrases, words=words, clips=[clip, _clip(1, 5.0)],
                                      suggestions=suggestions, cta_start=35.0)
        same = sorted((p for p in placed if p.clip_index == 0), key=lambda p: p.start)
        assert all(b.start - a.end >= B.SAME_CLIP_GAP_SEC - 1e-6 for a, b in zip(same, same[1:]))


@pytest.mark.parametrize("picks", [
    {0: {"uses": [{"phrase": 4, "at": 3.0}, {"phrase": 8, "at": 15.0}]}, 1: {"uses": [{"phrase": 12, "at": 1.0}]}},
    {0: {"uses": [{"phrase": 4, "at": 3.0}, {"phrase": 5, "at": 15.0}]}, 1: {"uses": [{"phrase": 6, "at": 1.0}]}},
    {1: {"uses": [{"phrase": 4, "at": 1.0}]}, 0: {"uses": [{"phrase": 8, "at": 3.0}, {"phrase": 12, "at": 15.0}]}},
    None,
])
def test_keep_order_long_clip_pieces_stay_in_its_slot_with_face_in_between(picks):
    words, phrases = _phrases(20, 2.0)
    clips = [_clip(0, 20.0, order=1), _clip(1, 5.0, order=2)]
    suggestions = picks or {0: {"uses": [{"phrase": 4, "at": 3.0}, {"phrase": 6, "at": 15.0}]}}
    placed, _ = B.plan_placements(total=40.0, phrases=phrases, words=words, clips=clips, suggestions=suggestions,
                                  cta_start=35.0, keep_order=True)
    timeline = sorted(placed, key=lambda p: p.start)
    assert [p.clip_index for p in timeline] == [0, 0, 1]                              # 1A, 1B, 2
    a, b, two = timeline
    assert b.start - a.end >= B.SAME_CLIP_GAP_SEC - 1e-6                              # face between 1A and 1B
    assert two.start >= b.end - 1e-6                                                  # 2 after 1B
    assert not (a.source_start < b.source_start + (b.end - b.start) and b.source_start < a.source_start + (a.end - a.start))


def test_wait_for_uploads_sees_a_file_that_arrives_after_some_seconds():
    clock = FakeClock()
    arrives_at = 7.0
    fetch = lambda: "/tmp/b.mov" if clock.t >= arrives_at else None
    arrived, info = B.wait_for_uploads({"b": fetch}, cap=60.0, clock=clock.now, sleep=clock.sleep)
    assert arrived == {"b": "/tmp/b.mov"} and info["still_missing"] == []
    assert info["waited_sec"] == pytest.approx(9.0) and info["polls"] == 4              # polls at 0, 3, 6, 9 s


def test_wait_for_uploads_total_cap_is_shared_by_all_files():
    clock = FakeClock()
    calls = []
    never = lambda name: (lambda: calls.append(name) or None)
    arrived, info = B.wait_for_uploads({"b1": never("b1"), "b2": never("b2")}, cap=60.0,
                                       clock=clock.now, sleep=clock.sleep)
    assert arrived == {} and info["still_missing"] == ["b1", "b2"]
    assert info["waited_sec"] == pytest.approx(60.0) and sum(clock.slept) == pytest.approx(60.0)   # not 2 x 60
    assert calls.count("b1") == calls.count("b2") == info["polls"]                   # both polled together
    rejected = lambda: (_ for _ in ()).throw(B.UploadRejected("x"))
    arrived, info = B.wait_for_uploads({"bad": rejected}, cap=60.0, clock=clock.now, sleep=clock.sleep)
    assert info["rejected"] == ["bad"] and info["polls"] == 1                         # an unusable file: no more polls
    assert B.upload_wait_sec({}) == 60.0 and B.upload_wait_sec({"CUTSELL_BROLL_UPLOAD_WAIT_SEC": "0"}) == 0.0
    assert B.upload_wait_sec({"CUTSELL_BROLL_UPLOAD_WAIT_SEC": "9999"}) == B.MAX_UPLOAD_WAIT_SEC
    arrived, info = B.wait_for_uploads({"b": never("z")}, cap=0.0, clock=clock.now, sleep=clock.sleep)
    assert info["polls"] == 1 and info["waited_sec"] == 0.0                           # cap 0: one look, no wait


def test_broll_that_arrives_during_the_wait_is_looked_at_by_the_ai(media2):
    seen = {}

    def ask(content):
        seen["images"] = sum(1 for c in content if c["type"] == "image")
        seen["text"] = "".join(c.get("text", "") for c in content)
        return '{"clips": [], "cta_phrase": null}', {}

    def wait(pending):
        assert pending == {"b1", "photo"}
        return {"b1": media2["b1"]}, {"arrived": ["b1"], "still_missing": ["photo"], "waited_sec": 60.0}

    result = simple_engine_adapter.process_with_simple_engine(
        _photo_request(media2), {"voice": media2["voice"]}, transcribe=Hearing(), llm=FakeLLM(), broll_ask=ask,
        pending_broll={"b1", "photo"}, wait_for_broll=wait)
    assert seen["images"] == 3 and "B1 (6.0 s)" in seen["text"]                       # b1 arrived: its 3 frames seen
    diag = result.draft.diagnostics["broll_overlay"]
    assert diag["upload_wait"]["still_missing"] == ["photo"]
    by_id = {s["source_asset_id"]: s for s in diag["sources"]}
    assert by_id["b1"]["video_available"] is True and by_id["photo"]["video_available"] is False
    assert any(o.kind == "photo" for o in result.draft.media_overlays)               # the missing photo: placed blind


def test_worker_waits_for_a_broll_upload_and_uses_it_when_it_arrives(media2, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from cutsell_worker import worker_job
    from cutsell_worker.simple_engine import asr as se_asr, llm as se_llm
    monkeypatch.setenv("CUTSELL_ENGINE", "simple")
    monkeypatch.setenv("CUTSELL_BROLL_AI", "0")
    clock = FakeClock()
    monkeypatch.setattr(B, "_now", clock.now)
    monkeypatch.setattr(B, "_pause", clock.sleep)
    monkeypatch.setattr(se_asr, "transcribe_words", Hearing())
    monkeypatch.setattr(se_llm, "call_anthropic", FakeLLM())
    attempts = {"b1": 0}

    def download(uri, destination, *, client=None):
        if uri.endswith("b1.mov"):
            attempts["b1"] += 1
            if clock.t < 5.0:                                                  # on storage only after 5 s
                raise FileNotFoundError(uri)
            shutil.copyfile(media2["b1"], destination)
            return destination
        shutil.copyfile(media2["voice"], destination)
        return destination

    gate = []
    monkeypatch.setattr(worker_job, "validate_product_source_uri", lambda uri, **kw: None)
    monkeypatch.setattr(worker_job, "download_source", download)
    monkeypatch.setattr(worker_job, "resolve_sources_for_editorial_entry",
                        lambda paths, **kw: (gate.append(sorted(paths)) or (dict(paths), [], [], [])))
    monkeypatch.setattr(worker_job, "load_runtime_config", lambda: SimpleNamespace(asr_model="tiny"))
    drafts, minutes = [], []
    monkeypatch.setattr(worker_job, "create_initial_draft", lambda **kw: drafts.append(kw))
    monkeypatch.setattr(worker_job, "safe_update_project", lambda **kw: {"status": "saved"})
    monkeypatch.setattr(worker_job, "publish_notification", lambda **kw: {"notification_id": "n1"})
    monkeypatch.setattr(worker_job, "generate_filmstrip", lambda *a, **kw: [])
    monkeypatch.setattr(worker_job, "waveform_peaks", lambda *a, **kw: [0.0])
    monkeypatch.setattr(worker_job, "store_timeline_assets", lambda **kw: {"status": "ok"})
    monkeypatch.setattr(worker_job, "record_processing_minutes", lambda **kw: minutes.append(kw["minutes"]))
    monkeypatch.setattr(worker_job, "release_processing_slot", lambda **kw: None)
    payload = {"project_id": "p1", "user_id": "u1", "sources": [
        {"source_asset_id": "voice", "uri": "s3://bk/up/voice.mov", "original_name": "v.mov", "source_order": 0,
         "duration_sec": DURATION, "metadata": {"clip_type": "primary"}},
        {"source_asset_id": "b1", "uri": "s3://bk/up/b1.mov", "original_name": "b1.mov", "source_order": 1,
         "duration_sec": 6.0, "metadata": {"clip_type": "broll", "broll_kind": "video", "broll_order": "1"}}]}

    result = worker_job.run_flow_b_job(payload)

    assert "broll_pending_upload" not in result
    assert result["broll_upload_wait"]["arrived"] == ["b1"] and result["broll_upload_wait"]["waited_sec"] == 6.0
    assert attempts["b1"] == 1 + 3                                                    # first try, then polls 0, 3, 6
    assert ["b1"] in gate                                                             # the late clip went through the gate
    source = result["draft"]["diagnostics"]["broll_overlay"]["sources"][0]
    assert source["video_available"] is True and source["frames"] == 3              # looked at normally
    assert minutes[0] == pytest.approx((DURATION + 6.0) / 60.0, abs=0.01)
    assets = {s["source_asset_id"]: s["timeline_assets"]["status"] for s in drafts[0]["sources"]}
    assert assets["b1"] == "ok"
