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
    clips = [{"clip": "B1", "shows": "jar on a table", "phrase": None, "frame": None},
             {"clip": "B2", "shows": "cream spread on a hand", "phrase": number, "frame": 1},
             {"clip": "B3", "shows": "bathroom shelf", "phrase": None, "frame": None}]
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
    assert b2.source_start <= 0.83 <= b2.source_end                          # around the frame the model chose
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
    result = _run(media, request, ask=lambda content: ('{"clips": [{"clip": "B1", "shows": "x", "phrase": null, '
                                                       '"frame": null}], "cta_phrase": null}', {}),
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

def _clip(i, duration):
    return B.BrollClip(source=_source(f"b{i}", i, f"s3://bk/b{i}.mov", duration, "broll"), duration=duration)


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
    answer = B.parse_answer('note {"clips": [{"clip": "B1", "shows": "hand", "phrase": 3, "frame": 2}, '
                            '{"clip": "B9", "phrase": 1}, {"clip": "B2", "phrase": 99, "frame": 7}], '
                            '"cta_phrase": 4}', 2, 10)
    assert answer == {"by_clip": {0: {"phrase": 3, "frame": 1, "shows": "hand"},
                                  1: {"phrase": None, "frame": None, "shows": ""}}, "cta_phrase": 4}
    with pytest.raises(ValueError):
        B.parse_answer("no", 1, 1)


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
