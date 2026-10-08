"""Simple engine (CUTSELL_ENGINE=simple): engine logic, the ProcessingResult adapter and the worker switch.

No network: the LLM and the ASR are always faked. Media is synthetic (ffmpeg lavfi): tone bursts where
the fake words are, silence in between, so the real silence-refine code runs on real audio.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import config as cutsell_config
from cutsell_worker import draft_edits, simple_engine_adapter, worker_job
from cutsell_worker.contracts import JobState, ProcessingRequest, SourceAsset
from cutsell_worker.serde import draft_from_dict, result_to_dict
from cutsell_worker.simple_engine import asr as se_asr
from cutsell_worker.simple_engine import engine as se
from cutsell_worker.simple_engine import llm as se_llm

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")

# four spoken lines; line 1 is an abandoned start that line 2 retakes
LINES = [
    (0.5, 2.5, ["Este", "producto", "es"]),
    (3.5, 5.5, ["Este", "producto", "es", "increíble."]),
    (7.0, 9.0, ["Lo", "uso", "todos", "los", "días."]),
    (10.0, 11.5, ["Cómpralo", "ya."]),
]
DURATION = 12.0


def _words():
    out = []
    for start, end, tokens in LINES:
        step = (end - start) / len(tokens)
        for i, token in enumerate(tokens):
            out.append({"w": token, "s": round(start + i * step, 3), "e": round(start + (i + 1) * step - 0.02, 3)})
    return out


def _make_video(path: Path) -> str:
    gate = "+".join(f"between(t,{a},{b})" for a, b, _ in LINES)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={DURATION}:r=30",
        "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*220*t)*({gate})':d={DURATION}:s=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return str(path)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return _make_video(tmp_path_factory.mktemp("simple_engine") / "raw.mp4")


class FakeLLM:
    """Answers the pass step and the decision step like the real model would (JSON after some notes)."""

    def __init__(self, decisions=None):
        self.prompts: list[str] = []
        self.decisions = list(decisions or [{"hide": [[1, 1, "a"]], "trim": []}])

    def __call__(self, prompt: str):
        self.prompts.append(prompt)
        if prompt.startswith(se.PASS_PROMPT):
            return 'One pass.\n{"passes": [[1, 4, true]], "backbone": [1, 4]}', {"input_tokens": 10, "output_tokens": 5}
        decision = self.decisions[0] if len(self.decisions) == 1 else self.decisions.pop(0)
        return "Notes first.\n" + json.dumps(decision), {"input_tokens": 20, "output_tokens": 8}


def _fake_transcribe(audio_path, *, language_hint=None):
    assert Path(audio_path).exists()
    return {"language": "es", "duration": DURATION, "words": _words()}


def _request(paths: dict[str, str]) -> ProcessingRequest:
    return ProcessingRequest(
        project_id="p1", user_id="u1",
        sources=tuple(
            SourceAsset(source_asset_id=asset_id, project_id="p1", user_id="u1", original_name=Path(path).name,
                        source_order=order, duration_sec=DURATION, uri=path)
            for order, (asset_id, path) in enumerate(paths.items())
        ),
    )


# ------------------------------------------------------------------ engine

def test_parse_takes_last_json_and_tolerates_line_prefix():
    assert se.parse('thinking {"x": 1}\n{"hide": [[2, 3, "r"]]}') == {"hide": [[2, 3, "r"]]}
    assert se.parse('{"hide": [[L2, L3, "r"]]}') == {"hide": [[2, 3, "r"]]}
    with pytest.raises(ValueError):
        se.parse("no json here")


def test_process_hides_abandoned_line_and_never_chops_a_kept_word(video):
    words = _words()
    out = se.process(words, DURATION, video, video_path=video, llm=FakeLLM())
    visible = [s for s in out["splits"] if s["visible"]]
    hidden_spoken = [s for s in out["splits"] if not s["visible"] and s["reason"] == "frase abandonada"]
    assert len(visible) == 3 and hidden_spoken
    assert visible[0]["start"] >= 2.5                      # the abandoned first line is not in the cut
    for word in words[3:]:                                 # every kept word sits whole inside one visible part
        assert any(s["start"] <= word["s"] and word["e"] <= s["end"] for s in visible), word
    # full, gap-free, non-destructive timeline
    assert out["splits"][0]["start"] == 0.0 and out["splits"][-1]["end"] == DURATION
    assert all(a["end"] == b["start"] for a, b in zip(out["splits"], out["splits"][1:]))
    # captions are on the cut timeline and grouped in threes at most
    assert [c["w"] for c in out["captions"]] == [w["w"] for w in words[3:]]
    assert out["captions"][0]["s"] < 0.5 and out["captions"][-1]["e"] <= out["cut_duration"] + 1e-6
    assert all(len(g["words"]) <= 3 for g in out["caption_groups"])
    assert out["passes"] == 1 and len(out["usage"]) == 2


def test_malformed_answer_is_asked_again(video):
    class Flaky(FakeLLM):
        def __call__(self, prompt):
            if not prompt.startswith(se.PASS_PROMPT) and "retry" not in prompt:
                self.prompts.append(prompt)
                return "sorry, no json", {}
            return super().__call__(prompt)

    llm = Flaky()
    out = se.process(_words(), DURATION, video, llm=llm)
    assert sum(1 for s in out["splits"] if s["visible"]) == 3
    assert any("-retry1)" in p for p in llm.prompts)


def test_answer_that_hides_almost_everything_is_asked_once_more(video):
    words = _words() * 3                                   # >= 30 words triggers the net
    words = [{"w": w["w"], "s": round(w["s"] + 12 * (i // 14), 3), "e": round(w["e"] + 12 * (i // 14), 3)}
             for i, w in enumerate(words)]
    llm = FakeLLM(decisions=[{"hide": [[1, 99, "n"]]}, {"hide": []}])
    splits, info = se.decide(words, 36.0, llm)
    assert any(s["visible"] for s in splits) and len(info["usage"]) == 3


def test_loose_piece_rule_hides_a_lone_word():
    splits = [{"start": 0.0, "end": 1.0, "visible": True, "reason": "se queda"},
              {"start": 1.0, "end": 9.0, "visible": False, "reason": "retoma"}]
    log = se.orphans(splits, [{"w": "Otro", "s": 0.2, "e": 0.6}])
    assert splits[0]["visible"] is False and splits[0]["reason"] == "pedazo suelto" and log


# ------------------------------------------------------------------ config switch

def test_engine_defaults_to_legacy_and_rejects_unknown_values():
    assert cutsell_config.selected_engine({}) == "legacy"
    assert cutsell_config.selected_engine({"CUTSELL_ENGINE": " Simple "}) == "simple"
    with pytest.raises(ValueError):
        cutsell_config.selected_engine({"CUTSELL_ENGINE": "v3"})


# ------------------------------------------------------------------ adapter

def test_adapter_returns_the_existing_draft_contract(video):
    stages = []
    result = simple_engine_adapter.process_with_simple_engine(
        _request({"s1": video}), {"s1": video},
        progress=lambda stage, percent: stages.append((stage, percent)),
        transcribe=_fake_transcribe, llm=FakeLLM(),
    )
    assert result.state is JobState.DRAFT_READY and result.stage_status["engine"] == "simple"
    draft = result.draft
    assert [c.text for c in draft.selected] == ["Este producto es increíble.", "Lo uso todos los días.", "Cómpralo ya."]
    assert [c.text for c in draft.alternates] == ["Este producto es"]
    assert all(c.selected for c in draft.selected) and not any(c.selected for c in draft.alternates)
    assert len({c.clip_id for c in (*draft.selected, *draft.alternates)}) == 4
    assert all(c.words and c.words[0].start >= c.start - 0.06 and c.words[-1].end <= c.end + 0.06 for c in draft.selected)
    assert [s for s, _ in stages] == ["transcribing", "analyzing", "composing"]

    # survives the same serialization the worker stores and the API serves
    payload = json.loads(json.dumps(result_to_dict(result)))
    restored = draft_from_dict(payload["draft"])
    assert [c.clip_id for c in restored.selected] == [c.clip_id for c in draft.selected]
    engine_diag = restored.diagnostics["simple_engine"]
    assert restored.diagnostics["engine"] == "simple"
    assert [c["w"] for c in engine_diag["captions"]][:4] == ["Este", "producto", "es", "increíble."]
    assert engine_diag["caption_groups"] and engine_diag["sources"][0]["language"] == "es"

    # the editor's existing "restore" edit can bring the hidden spoken part back
    edited = draft_edits.restore_clip(payload["draft"], draft.alternates[0].clip_id, position=0)
    assert edited["selected"][0]["text"] == "Este producto es" and not edited["alternates"]


def test_adapter_multi_source_keeps_ids_unique_and_shifts_captions(video, tmp_path):
    second = str(tmp_path / "raw2.mp4")
    shutil.copyfile(video, second)
    result = simple_engine_adapter.process_with_simple_engine(
        _request({"a": video, "b": second}), {"a": video, "b": second}, transcribe=_fake_transcribe, llm=FakeLLM(),
    )
    selected = result.draft.selected
    assert len(selected) == 6 and len({c.clip_id for c in selected}) == 6
    assert [c.source_asset_id for c in selected] == ["a"] * 3 + ["b"] * 3
    caps = result.draft.diagnostics["simple_engine"]["captions"]
    assert all(x["s"] <= y["s"] + 1e-6 for x, y in zip(caps, caps[1:]))      # one monotonic timeline


def test_adapter_fails_loudly_without_speech(video):
    with pytest.raises(ValueError, match="no speech"):
        simple_engine_adapter.process_with_simple_engine(
            _request({"s1": video}), {"s1": video},
            transcribe=lambda path, *, language_hint=None: {"language": None, "duration": DURATION, "words": []},
            llm=FakeLLM(),
        )


def test_llm_and_asr_need_their_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        se_llm.call_anthropic("hi")
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        se_asr.transcribe_words(str(tmp_path / "x.mp3"))


# ------------------------------------------------------------------ worker switch

@pytest.fixture
def wired_worker(monkeypatch):
    calls = {"legacy": 0, "drafts": []}

    def fake_download_source(uri, destination, *, client=None):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(uri, destination)
        return destination

    def fake_legacy(request, local_paths, **kwargs):
        calls["legacy"] += 1
        return object()

    monkeypatch.setattr(worker_job, "validate_product_source_uri", lambda uri, **kw: None)
    monkeypatch.setattr(worker_job, "download_source", fake_download_source)
    monkeypatch.setattr(worker_job, "process_local_sources", fake_legacy)
    monkeypatch.setattr(worker_job, "load_runtime_config", lambda: SimpleNamespace(asr_model="tiny"))
    monkeypatch.setattr(worker_job, "create_initial_draft", lambda **kw: calls["drafts"].append(kw))
    monkeypatch.setattr(worker_job, "safe_update_project", lambda **kw: {"status": "saved", "project_state": kw.get("state")})
    monkeypatch.setattr(worker_job, "publish_notification", lambda **kw: {"notification_id": "n1"})
    monkeypatch.setattr(worker_job, "generate_filmstrip", lambda *a, **kw: [])
    monkeypatch.setattr(worker_job, "waveform_peaks", lambda *a, **kw: [0.0])
    monkeypatch.setattr(worker_job, "store_timeline_assets", lambda **kw: {"status": "ok"})
    monkeypatch.setattr(worker_job, "record_processing_minutes", lambda **kw: None)
    monkeypatch.setattr(worker_job, "release_processing_slot", lambda **kw: None)
    return calls


def _payload(uri: str) -> dict:
    return {"project_id": "p1", "user_id": "u1",
            "sources": [{"source_asset_id": "s1", "uri": uri, "original_name": Path(uri).name}]}


def test_worker_runs_simple_engine_without_building_the_legacy_stack(wired_worker, monkeypatch, video):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy brain / Whisper must not be built when CUTSELL_ENGINE=simple")

    monkeypatch.setenv("CUTSELL_ENGINE", "simple")
    monkeypatch.setattr(worker_job, "build_brain_runtime", forbidden)
    monkeypatch.setattr(worker_job, "FasterWhisperASR", forbidden)
    monkeypatch.setattr(se_asr, "transcribe_words", _fake_transcribe)
    monkeypatch.setattr(se_llm, "call_anthropic", FakeLLM())

    result = worker_job.run_flow_b_job(_payload(video))

    assert wired_worker["legacy"] == 0
    assert result["engine"] == "simple" and result["brain_backend"] == "simple_engine"
    assert result["hybrid_provider"] == "anthropic" and result["hybrid_primary_model"] == se_llm.DEFAULT_MODEL
    stored = wired_worker["drafts"][0]["draft"]
    assert len(stored["selected"]) == 3 and len(stored["alternates"]) == 1
    assert stored["diagnostics"]["engine"] == "simple"
    draft_from_dict(stored)                                 # what the API will hand to the export job


def test_worker_default_is_still_the_legacy_engine(wired_worker, monkeypatch, video):
    monkeypatch.delenv("CUTSELL_ENGINE", raising=False)
    brain = SimpleNamespace(
        backend="runpod_local", external_calls_enabled=False,
        hybrid_settings=SimpleNamespace(provider="google", primary_model="m"),
        semantic_provider=None, whole_video_provider=None, visual_provider=None, take_grouping_provider=None,
        take_judge_provider=None, clean_cut_provider=None, composer_provider=None, draft_review_provider=None,
        editorial_judge=None,
    )
    monkeypatch.setattr(worker_job, "build_brain_runtime", lambda config: brain)
    monkeypatch.setattr(worker_job, "FasterWhisperASR", lambda **kw: object())
    monkeypatch.setattr(worker_job, "result_to_dict", lambda result: {"draft": {"selected": []}})

    result = worker_job.run_flow_b_job(_payload(video))

    assert wired_worker["legacy"] == 1
    assert "engine" not in result and result["brain_backend"] == "runpod_local"


# ------------------------------------------------------------------ Human Gold scorer

def test_gold_file_and_scorer_reproduce_the_recorded_reference_score():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "simple_engine_eval.py"
    spec = importlib.util.spec_from_file_location("simple_engine_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gold = json.loads(module.GOLD.read_text(encoding="utf-8"))
    assert len(gold["videos"]) == 25 and sum(len(v["decisions"]) for v in gold["videos"]) == 555
    result = module.score()                                # frozen v1.0 run on the 24 videos that have one
    assert (result["agree"], result["total"]) == (478, 514)
    # a cut that keeps everything agrees only with the "keep" marks
    video = gold["videos"][0]
    total, agree, removed, kept = module.score_video(video["decisions"], [[0.0, video["duration"]]])
    assert removed == 0 and agree == sum(d["mark"] == "keep" for d in video["decisions"]) and agree + kept == total


# ------------------------------------------------------------------ export captions

def test_simple_engine_export_uses_short_timed_captions_and_legacy_drafts_do_not(video, tmp_path):
    from dataclasses import replace as dc_replace

    from cutsell_worker.render import _caption_filter
    from cutsell_worker.render_plan import build_render_plan

    result = simple_engine_adapter.process_with_simple_engine(
        _request({"s1": video}), {"s1": video}, transcribe=_fake_transcribe, llm=FakeLLM(),
    )
    draft = result.draft
    plan = build_render_plan(draft, {"s1": video})
    middle = plan[1]                                          # "Lo uso todos los días."
    assert [text for _, _, text in middle.caption_cues] == ["Lo uso todos", "los días."]
    assert all(0 <= a < b <= middle.duration_sec + 1e-6 for a, b, _ in middle.caption_cues)
    assert all(x[1] <= y[0] + 1e-6 for x, y in zip(middle.caption_cues, middle.caption_cues[1:]))

    assert _caption_filter(middle, tmp_path / "part.mp4")
    ass = (tmp_path / "part.ass").read_text(encoding="utf-8")
    assert ass.count("Dialogue:") == 2 and "}Lo uso todos\n" in ass and "}los días.\n" in ass
    # Editor v2 look: lower third of a 1080x1920 frame, centred
    assert "Style: Caption,Montserrat ExtraBold," in ass and "PlayResY: 1920" in ass and "\\an5\\pos(540,1455)" in ass
    # each cue carries its own word timings, used by the Highlight looks
    assert [[w for _s, _e, w in ws] for ws in middle.caption_cue_words] == [["Lo", "uso", "todos"], ["los", "días."]]

    # captions off -> no cues, no caption text
    off = build_render_plan(dc_replace(draft, captions_enabled=False), {"s1": video})
    assert all(not seg.caption_cues and not seg.caption_text for seg in off)
    # a caption the user rewrote by hand is shown whole, exactly as typed
    edited_clip = dc_replace(draft.selected[1], caption_text="¡Mi favorito!")
    edited = build_render_plan(dc_replace(draft, selected=(draft.selected[0], edited_clip, draft.selected[2])), {"s1": video})
    assert edited[1].caption_cues == () and edited[1].caption_text == "¡Mi favorito!"
    # any draft that is not from the simple engine keeps the single whole-clip caption
    legacy = build_render_plan(dc_replace(draft, diagnostics={}), {"s1": video})
    assert all(seg.caption_cues == () for seg in legacy)
    assert _caption_filter(legacy[1], tmp_path / "legacy.mp4")
    assert (tmp_path / "legacy.srt").read_text(encoding="utf-8").count("-->") == 1


def test_timed_caption_text_cannot_inject_a_second_cue(tmp_path):
    from cutsell_worker.render import _caption_filter
    from cutsell_worker.render_plan import RenderSegment

    seg = RenderSegment(clip_id="c", source_asset_id="s", source_path="x.mp4", start=0.0, end=2.0,
                        caption_text="hola", caption_cues=((0.0, 1.0, "hola\n\n9\n00:00:00,000 --> 00:00:09,000\nmalo"),))
    assert _caption_filter(seg, tmp_path / "p.mp4")
    body = (tmp_path / "p.ass").read_text(encoding="utf-8")
    assert body.count("Dialogue:") == 1 and body.rstrip().endswith("hola 9 00:00:00,000 --> 00:00:09,000 malo")


def test_caption_looks_highlight_only_the_spoken_word_and_strip_styling_commands():
    from cutsell_worker.caption_render import CAPTION_PRESETS, build_caption_ass

    cues = ((0.0, 1.2, "me salió este"),)
    words = (((0.0, 0.3, "me"), (0.3, 0.8, "salió"), (0.8, 1.1, "este")),)
    assert {"classic", "yellow", "highlight", "highlight_green", "highlight_red", "highlight_blue",
            "box", "box_light", "clean"} == set(CAPTION_PRESETS)
    green = build_caption_ass(cues, words, preset="highlight_green", duration_sec=5.0)
    lines = [line for line in green.splitlines() if line.startswith("Dialogue:")]
    assert len(lines) == 3                                   # one step per spoken word
    assert "{\\1c&H0078FF39}me{" in lines[0] and "{\\1c&H0078FF39}salió{" in lines[1] and "{\\1c&H0078FF39}este{" in lines[2]
    assert all(line.count("\\1c&H0078FF39") == 1 for line in lines)   # never more than one coloured word
    assert "\\1c&H003A45FF" in build_caption_ass(cues, words, preset="highlight_red", duration_sec=5.0)
    assert "\\1c&H00FFA61A" in build_caption_ass(cues, words, preset="highlight_blue", duration_sec=5.0)
    # the other looks draw the phrase once, with no per-word colour
    for preset in ("classic", "yellow", "box", "box_light", "clean"):
        body = build_caption_ass(cues, words, preset=preset, duration_sec=5.0)
        assert body.count("Dialogue:") == 1 and "\\1c" not in body
    # word timings that do not match the cue text -> phrase drawn plainly, never dropped
    plain = build_caption_ass(cues, (((0.0, 0.3, "otra"),),), preset="highlight", duration_sec=5.0)
    assert plain.count("Dialogue:") == 1 and "\\1c" not in plain
    # text can never carry its own styling commands or open another line
    hostile = build_caption_ass(((0.0, 1.0, "hola {\\pos(0,0)}\\N x"),), (), preset="classic", duration_sec=5.0)
    assert hostile.count("Dialogue:") == 1 and hostile.rstrip().endswith("}hola pos(0,0)N x")
    # cues are clamped to the real duration and an unknown preset falls back to classic
    assert build_caption_ass(((4.99, 6.0, "tarde"),), (), preset="classic", duration_sec=5.0) == ""
    assert "Dialogue:" in build_caption_ass(cues, words, preset="no-existe", duration_sec=5.0)


def test_caption_font_choice_reaches_the_export_and_every_face_ships_with_its_licence(tmp_path):
    from cutsell_worker.caption_render import CAPTION_FONTS, DEFAULT_CAPTION_FONT, FONTS_DIR, build_caption_ass
    from cutsell_worker.caption_settings import patch_caption_settings
    from cutsell_worker.render import _caption_filter
    from cutsell_worker.render_plan import RenderSegment

    assert set(CAPTION_FONTS) == {"montserrat", "poppins", "roboto", "oswald", "anton", "luckiest_guy",
                                  "bebas_neue", "inter", "bangers"}
    assert DEFAULT_CAPTION_FONT == "montserrat"
    shipped = {path.name for path in FONTS_DIR.iterdir()}
    assert sum(name.endswith((".ttf", ".otf")) for name in shipped) == len(CAPTION_FONTS)
    assert sum(name.startswith("LICENSE-") for name in shipped) == len(CAPTION_FONTS)

    cues = ((0.0, 1.0, "hola"),)
    assert "Style: Caption,Montserrat ExtraBold,94," in build_caption_ass(cues, (), preset="classic", duration_sec=2.0)
    assert "Style: Caption,Anton,116," in build_caption_ass(cues, (), preset="classic", duration_sec=2.0, font="anton")
    assert "Style: Caption,Montserrat ExtraBold," in build_caption_ass(cues, (), preset="classic", duration_sec=2.0, font="no-existe")

    seg = RenderSegment(clip_id="c", source_asset_id="s", source_path="x.mp4", start=0.0, end=2.0,
                        caption_text="hola", caption_cues=cues, caption_font="bebas_neue")
    flt = _caption_filter(seg, tmp_path / "p.mp4")
    assert "fontsdir=" in flt and str(FONTS_DIR) in flt
    assert "Style: Caption,Bebas Neue,112," in (tmp_path / "p.ass").read_text(encoding="utf-8")

    draft = {"selected": [], "caption_preset": "classic"}
    assert patch_caption_settings(draft, font="oswald")["caption_font"] == "oswald"
    with pytest.raises(ValueError):
        patch_caption_settings(draft, font="comic-sans")


def test_caption_position_and_size_apply_to_the_whole_video_and_stay_inside_the_frame(tmp_path):
    from cutsell_worker.caption_render import PLAY_RES_X, PLAY_RES_Y, build_caption_ass, caption_layout
    from cutsell_worker.caption_settings import patch_caption_settings
    from cutsell_worker.render import _caption_filter
    from cutsell_worker.render_plan import RenderSegment, _can_coalesce
    from dataclasses import replace as dc_replace
    import re

    cues = ((0.0, 1.0, "me salió este"), (1.0, 2.0, "innecesarias, esta compra"))

    def events(**layout):
        body = build_caption_ass(cues, (), preset="classic", duration_sec=3.0, **layout)
        style = next(line for line in body.splitlines() if line.startswith("Style:")).split(",")
        rows = [re.search(r"pos\((\d+),(\d+)\)(?:\\blur1\.2)?(?:\\fs(\d+))?", line).groups()
                for line in body.splitlines() if line.startswith("Dialogue:")]
        return int(style[2]), int(style[19]), int(style[20]), rows

    size, left, right, rows = events()
    assert (size, left, right) == (94, 36, 36)
    assert [(int(a), int(b)) for a, b, _ in rows] == [(540, 1455)] * 2          # same place for every cue
    # moved up and enlarged: one setting, every cue follows it
    size, _l, _r, rows = events(y=0.2, scale=1.5)
    assert size == 141 and {(a, b) for a, b, _ in rows} == {("540", "384")}
    # dragged into a corner at the largest size: the centre is pulled in, the column stays
    # symmetric and inside the frame, and a word too wide for the column is drawn smaller
    size, left, right, rows = events(x=1.0, y=1.0, scale=2.0)
    cx, cy = int(rows[0][0]), int(rows[0][1])
    column = PLAY_RES_X - left - right
    assert left == right and column >= 500
    assert cx - column // 2 >= 0 and cx + column // 2 <= PLAY_RES_X
    assert cy + size <= PLAY_RES_Y and size == 188
    assert int(rows[1][2]) < int(rows[0][2]) < size                       # longer word -> smaller
    assert events(x=0.0, y=0.0, scale=2.0)[3][0][:2] == (str(PLAY_RES_X - cx), str(PLAY_RES_Y - cy))

    assert caption_layout() == (0.5, 0.758, 1.0)
    for bad in ({"x": 1.2}, {"y": -0.1}, {"scale": 0.2}, {"scale": 3}, {"x": "izquierda"}, {"y": float("nan")}, {"scale": True}):
        with pytest.raises(ValueError):
            caption_layout(**bad)

    draft = {"selected": [], "caption_x": 0.3}
    moved = patch_caption_settings(draft, y=0.25, scale=1.4)
    assert (moved["caption_x"], moved["caption_y"], moved["caption_scale"]) == (0.3, 0.25, 1.4)
    assert draft == {"selected": [], "caption_x": 0.3}
    with pytest.raises(ValueError):
        patch_caption_settings(draft, scale=9)

    seg = RenderSegment(clip_id="c", source_asset_id="s", source_path="x.mp4", start=0.0, end=2.0,
                        caption_text="hola", caption_cues=((0.0, 1.0, "hola"),), caption_y=0.2, caption_scale=1.5)
    assert _caption_filter(seg, tmp_path / "p.mp4")
    assert "pos(540,384)" in (tmp_path / "p.ass").read_text(encoding="utf-8")
    twin = dc_replace(seg, clip_id="d", start=2.0, end=4.0)
    assert _can_coalesce(seg, twin) and not _can_coalesce(seg, dc_replace(twin, caption_scale=1.0))
