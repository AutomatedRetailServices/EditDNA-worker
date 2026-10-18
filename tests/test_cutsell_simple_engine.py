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


@pytest.fixture(autouse=True)
def _rules_path(monkeypatch):
    """These tests count the 1.2 decision calls; the 1.2.1 sale review has its own tests below."""
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_MODE", "rules")
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
        if prompt.startswith(se.FIXWORDS_PROMPT):
            return "[]", {"input_tokens": 5, "output_tokens": 1}
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
    assert out["passes"] == 1 and len(out["usage"]) == 4   # pass + decision + one review call + word fixer


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
    assert any(s["visible"] for s in splits) and len(info["usage"]) == 4


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
    # 26 videos since G01 (Swanny's own Gold, 37 decisions) was added; it has no frozen run yet
    assert len(gold["videos"]) == 26 and sum(len(v["decisions"]) for v in gold["videos"]) == 592
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


def test_deepgram_request_sends_brand_keyterms(monkeypatch, tmp_path):
    from cutsell_worker.simple_engine import asr
    seen = {}

    class R:
        status_code = 200
        def json(self):
            return {"metadata": {"duration": 1.0}, "results": {"channels": [{"detected_language": "es", "alternatives": [
                {"words": [{"word": "shein", "punctuated_word": "Shein", "start": 0.1, "end": 0.4}]}]}]}}

    def fake_post(url, params=None, **kw):
        seen["params"] = params
        return R()

    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    monkeypatch.setattr(asr.requests, "post", fake_post)
    audio = tmp_path / "a.mp3"; audio.write_bytes(b"x")
    out = asr.transcribe_words(str(audio))
    terms = [v for k, v in seen["params"] if k == "keyterm"]
    assert "Shein" in terms and ("smart_format", "false") in seen["params"] and ("detect_language", "true") in seen["params"]
    assert out["words"][0]["w"] == "Shein"


def test_decision_prompt_has_restarted_section_rule():
    from cutsell_worker.simple_engine import engine
    assert "RESTARTED SECTION" in engine.DECISION_PROMPT and engine.VERSION == "1.2.4"


def _review_words():
    text = ["Compre este set.", "Tambien tengo otro vestido, en rojo.", "Tambien tengo este vestido en rojo, trae su cinturon.", "Chao."]
    words, t = [], 0.0
    for line in text:
        for w in line.split():
            words.append({"w": w, "s": round(t, 2), "e": round(t + 0.3, 2)}); t += 0.35
        t += 1.0
    return words


def test_review_hides_a_repeated_announcement_only_when_two_answers_agree():
    from cutsell_worker.simple_engine import engine
    words = _review_words(); units = engine.lines_of(words)
    lab = [("keep", "se queda")] * len(words)
    answers = iter(['{"type":"haul","hide":[[2,3]],"unhide":[]}', '{"type":"haul","hide":[[2,3]],"unhide":[]}', '{"type":"haul","hide":[],"unhide":[]}'])
    calls = []
    def call(prompt):
        calls.append(prompt); return next(answers), {"input_tokens": 1, "output_tokens": 1}
    new, usage, info = engine.review(words, units, lab, call)
    assert info["haul"] and info["hide"] == [2] and len(calls) == 2 and len(usage) == 2   # 2 agreeing answers decide
    assert all(new[k][0] == "hide" for k in units[1]) and all(new[k][0] == "keep" for k in units[2])


def test_review_on_an_ordinary_pitch_costs_one_call_and_changes_nothing():
    from cutsell_worker.simple_engine import engine
    words = _review_words(); units = engine.lines_of(words)
    lab = [("keep", "se queda")] * len(words)
    calls = []
    def call(prompt):
        calls.append(prompt); return '{"type":"other","hide":[[2,3]],"unhide":[]}', {}
    new, usage, info = engine.review(words, units, lab, call)
    assert len(calls) == 1 and new == lab and not info["haul"]


def test_a_silence_split_never_hides_a_spoken_word():
    from cutsell_worker.simple_engine import engine
    words = [{"w": "este", "s": 1.0, "e": 1.3}, {"w": "set", "s": 1.6, "e": 1.9}, {"w": "blusa.", "s": 2.4, "e": 2.8}]
    splits = [{"start": 0.0, "end": 1.4, "visible": True, "reason": "se queda"},
              {"start": 1.4, "end": 2.0, "visible": False, "reason": "silencio"},
              {"start": 2.0, "end": 3.0, "visible": True, "reason": "se queda"}]
    out = engine.keep_spoken_words(splits, words)
    assert out == [{"start": 0.0, "end": 3.0, "visible": True, "reason": "se queda"}]


def test_review_retries_a_failed_call_instead_of_skipping_the_review():
    from cutsell_worker.simple_engine import engine
    words = _review_words(); units = engine.lines_of(words)
    lab = [("keep", "se queda")] * len(words)
    replies = iter([RuntimeError("network"), '{"type":"haul","hide":[[2,3]],"unhide":[]}',
                    '{"type":"haul","hide":[[2,3]],"unhide":[]}', '{"type":"haul","hide":[[2,3]],"unhide":[]}'])
    def call(prompt):
        r = next(replies)
        if isinstance(r, Exception): raise r
        return r, {}
    new, usage, info = engine.review(words, units, lab, call)
    assert info["errors"] == 1 and info["hide"] == [2]


def test_breathe_keeps_a_short_pause_but_never_reaches_a_hidden_word():
    from cutsell_worker.simple_engine import engine
    words = [{"w": "uno.", "s": 1.0, "e": 1.5}, {"w": "dos", "s": 1.9, "e": 2.2}, {"w": "tres.", "s": 4.0, "e": 4.5}]
    splits = [{"start": 0.0, "end": 0.9, "visible": False}, {"start": 0.9, "end": 1.6, "visible": True},
              {"start": 1.6, "end": 3.9, "visible": False}, {"start": 3.9, "end": 4.6, "visible": True},
              {"start": 4.6, "end": 6.0, "visible": False}]
    out = engine.breathe(splits, words, 6.0)
    assert out[1]["end"] <= 1.9 + 1e-9 and out[1]["end"] > 1.6           # pause added, hidden "dos" stays out
    assert abs(out[3]["end"] - 4.9) < 1e-6 and abs(out[3]["start"] - 3.75) < 1e-6
    assert all(a["end"] == b["start"] for a, b in zip(out, out[1:])) and out[0]["start"] == 0.0 and out[-1]["end"] == 6.0


def test_fix_words_changes_caption_text_only_for_kept_words():
    from cutsell_worker.simple_engine import engine
    words = [{"w": "compre", "s": 0.0, "e": 0.4}, {"w": "en", "s": 0.4, "e": 0.6}, {"w": "por", "s": 0.6, "e": 0.9},
             {"w": "malo", "s": 5.0, "e": 5.3}]
    splits = [{"start": 0.0, "end": 1.0, "visible": True}, {"start": 1.0, "end": 6.0, "visible": False}]
    def call(prompt):
        return '[[2, "Shein"], [3, "bueno"]]', {}
    shown, fixes, usage = engine.fix_words(words, splits, call)
    assert shown[2]["w"] == "Shein" and shown[3]["w"] == "malo" and words[2]["w"] == "por"
    assert fixes == [[2, "por", "Shein"]] and len(usage) == 1


# ------------------------------------------------------------------ Overlap audio + Auto Zoom AI

class MomentLLM(FakeLLM):
    """Like FakeLLM, but the word-fix answer (v2) also marks two key moments."""

    def __call__(self, prompt: str):
        if prompt.startswith(se.FIXWORDS_PROMPT):
            self.prompts.append(prompt)
            indices = [int(item.split(":", 1)[0]) for item in prompt.split("\n")[-1].split()]
            marks = [[indices[0], "hook"], [indices[-1], "cta"], [indices[1], "bogus_kind"]]
            return json.dumps({"fix": [], "zoom": marks}), {"input_tokens": 5, "output_tokens": 3}
        return super().__call__(prompt)


def _request_with(paths, **flags):
    base = _request(paths)
    from dataclasses import replace
    return replace(base, **flags)


def test_overlap_flag_sets_the_draft_and_every_kept_clip_gets_bounded_rolls(video):
    on = simple_engine_adapter.process_with_simple_engine(
        _request_with({"s1": video}, dialogue_overlap_enabled=True), {"s1": video},
        transcribe=_fake_transcribe, llm=FakeLLM())
    off = simple_engine_adapter.process_with_simple_engine(
        _request_with({"s1": video}, dialogue_overlap_enabled=False), {"s1": video},
        transcribe=_fake_transcribe, llm=FakeLLM())
    assert on.draft.audio_overlap_enabled is True and off.draft.audio_overlap_enabled is False
    words = _words()
    for clip in on.draft.selected:
        assert 0.0 <= clip.audio_pre_roll <= 0.06 and 0.0 <= clip.audio_post_roll <= 0.06
        # a roll never reaches another spoken word
        assert not any(clip.start - clip.audio_pre_roll < w["e"] <= clip.start for w in words)
        assert not any(clip.end <= w["s"] < clip.end + clip.audio_post_roll for w in words)
    assert any(clip.audio_pre_roll > 0 or clip.audio_post_roll > 0 for clip in on.draft.selected)


def test_auto_zoom_keyframes_come_from_the_existing_word_fix_call(video):
    llm = MomentLLM()
    result = simple_engine_adapter.process_with_simple_engine(
        _request_with({"s1": video}, auto_zoom_enabled=True), {"s1": video}, transcribe=_fake_transcribe, llm=llm)
    calls = len(llm.prompts)
    assert calls == len(FakeLLM().prompts) or calls == 4      # same four calls as before: no extra call
    keys = result.draft.zoom_keyframes
    assert result.draft.auto_zoom_enabled is True and keys
    assert {k.kind for k in keys} <= {"hook", "product", "price", "cta"}       # unknown kinds dropped
    assert all(1.0 < k.scale <= 1.12 and 0.8 <= k.duration <= 1.8 for k in keys)
    assert all(b.source_time - a.source_time >= 3.0 for a, b in zip(keys, keys[1:]))
    for k in keys:   # inside a kept clip and away from its joins
        assert any(c.start + 0.3 - 1e-6 <= k.source_time and k.source_time + k.duration <= c.end - 0.3 + 1e-6
                   for c in result.draft.selected), k
    off = simple_engine_adapter.process_with_simple_engine(
        _request_with({"s1": video}, auto_zoom_enabled=False), {"s1": video}, transcribe=_fake_transcribe, llm=MomentLLM())
    assert off.draft.auto_zoom_enabled is False and off.draft.zoom_keyframes   # kept, so the editor can turn them on


def test_word_fix_still_accepts_the_old_list_answer():
    from cutsell_worker.simple_engine import engine
    words = [{"w": "compre", "s": 0.0, "e": 0.4}, {"w": "por", "s": 0.6, "e": 0.9}]
    splits = [{"start": 0.0, "end": 1.0, "visible": True}]
    shown, fixes, moments, _ = engine.fix_words_and_moments(words, splits, lambda p: ('[[1, "Shein"]]', {}))
    assert fixes == [[1, "por", "Shein"]] and moments == []
    shown, fixes, moments, _ = engine.fix_words_and_moments(
        words, splits, lambda p: ('{"fix": [], "zoom": [[0, "price"], [9, "price"]]}', {}))
    assert fixes == [] and moments == [{"i": 0, "kind": "price"}]      # index 9 is not a kept word


# ------------------------------------------------------------------ 1.2.1 sale review (off by default)

def _sr_words(text):
    return [{"w": w, "s": i * 0.3, "e": i * 0.3 + 0.25} for i, w in enumerate(text.split())]


def _sr(answers):
    from cutsell_worker.simple_engine import engine as E
    words = _sr_words("Huele rico. Mira esto. Huele rico de verdad. Cuesta veinte dólares. Cómpralo ya.")
    units = E.lines_of(words)
    lab = [("keep", "se queda")] * len(words)
    it = iter(answers)
    return E, words, units, E.sale_review(words, units, lab, lambda p: (next(it), {}))


def test_sale_review_applies_a_fix_only_when_two_of_three_readings_agree(monkeypatch):
    from cutsell_worker.simple_engine import engine as E0
    monkeypatch.setattr(E0, "SALE_REVIEW_KINDS", E0.SALE_REVIEW_ALL_KINDS)
    fix = '{"type":"sale","fixes":[{"kind":"repeat","hide":[1],"unhide":[],"why":"dicho dos veces"}]}'
    other = '{"type":"sale","fixes":[{"kind":"repeat","hide":[2],"unhide":[],"why":"x"}]}'
    E, words, units, (lab, usage, info) = _sr([fix, fix, other])
    assert info["applied"] == [["repeat", "hide", 1]] and len(usage) == 2     # two agreeing readings decide
    E, words, units, (lab, usage, info) = _sr([fix, other, fix])
    assert info["applied"] == [["repeat", "hide", 1]] and len(usage) == 3
    assert all(lab[k][0] == "hide" for k in units[0]) and all(lab[k][0] == "keep" for k in units[1])
    E, words, units, (lab, usage, info) = _sr([fix, other, '{"type":"sale","fixes":[]}'])
    assert info["applied"] == [] and all(l[0] == "keep" for l in lab)


def test_sale_review_stops_after_one_call_when_nothing_to_fix_or_not_a_sale():
    for first in ('{"type":"sale","fixes":[]}', '{"type":"other","fixes":[]}'):
        E, words, units, (lab, usage, info) = _sr([first])
        assert len(usage) == 1 and info["applied"] == [] and all(l[0] == "keep" for l in lab)


def test_sale_review_never_loses_the_price(monkeypatch):
    from cutsell_worker.simple_engine import engine as E0
    monkeypatch.setattr(E0, "SALE_REVIEW_KINDS", E0.SALE_REVIEW_ALL_KINDS)
    fix = '{"type":"sale","fixes":[{"kind":"repeat","hide":[4],"unhide":[],"why":"x"}]}'
    E, words, units, (lab, usage, info) = _sr([fix, fix, fix])
    assert info.get("rejected") and all(l[0] == "keep" for l in lab)


def test_review_mode_is_the_default_and_rules_switches_it_off(monkeypatch):
    from cutsell_worker.simple_engine import engine as E
    monkeypatch.delenv("CUTSELL_SIMPLE_ENGINE_MODE", raising=False)
    import inspect
    assert '"review").strip().lower()' in inspect.getsource(E.process) and 'mode == "review"' in inspect.getsource(E.process)


def test_sale_review_knows_the_restarted_pitch():
    from cutsell_worker.simple_engine import engine as E
    assert E.SALE_REVIEW_KINDS == ("restart", "mixed") and "RESTARTED PITCH" in E.SALE_REVIEW_PROMPT


def test_a_reviewed_cut_never_keeps_a_hidden_boundary_word():
    from cutsell_worker.simple_engine import engine as E
    words = [{"w": "Shop.", "s": 79.575, "e": 80.135}, {"w": "Mi", "s": 80.135, "e": 80.695}, {"w": "amor.", "s": 80.695, "e": 81.095}]
    out = E.keep_hidden_words_out([1, 2], [{"start": 71.79, "end": 80.415, "visible": True},
                                           {"start": 80.415, "end": 81.71, "visible": False}], words)
    assert out[0]["end"] == 80.135


# ------------------------------------------------------------------ 1.2.3 savings (same cut, fewer tokens)

def test_third_reading_is_never_needed_when_two_agree_same_result_as_three(monkeypatch):
    """Exhaustive: whenever the first two readings agree, every possible third reading gives the same cut."""
    from cutsell_worker.simple_engine import engine as E0
    monkeypatch.setattr(E0, "SALE_REVIEW_KINDS", E0.SALE_REVIEW_ALL_KINDS)
    opts = ['{"type":"sale","fixes":[]}', '{"type":"other","fixes":[]}',
            '{"type":"sale","fixes":[{"kind":"repeat","hide":[1],"unhide":[],"why":"x"}]}',
            '{"type":"sale","fixes":[{"kind":"repeat","hide":[2],"unhide":[],"why":"x"}]}',
            '{"type":"sale","fixes":[{"kind":"mixed","hide":[1],"unhide":[],"why":"x"}]}']
    for a in opts:
        results = set()
        for c in opts:
            E, words, units, (lab, usage, info) = _sr([a, a, c])
            results.add((tuple(lab), tuple(map(tuple, info["applied"]))))
        assert len(results) == 1


def test_word_fix_reads_the_last_answer_and_does_not_ask_again_for_trailing_text():
    from cutsell_worker.simple_engine import engine
    words = [{"w": "por", "s": 0.0, "e": 0.3}, {"w": "set", "s": 0.4, "e": 0.7}]
    splits = [{"start": 0.0, "end": 1.0, "visible": True, "reason": "se queda"}]
    calls = []
    def call(p):
        calls.append(p)
        return ('{"fix": [[1, "Shein"]], "zoom": [[0, "hook"]]}\n\nWait, I should keep {that} as is.', {})
    shown, fixes, moments, usage = engine.fix_words_and_moments(words, splits, call)
    assert len(calls) == 1 and fixes == [[1, "set", "Shein"]] and moments == [{"i": 0, "kind": "hook"}]
    calls.clear()
    def corrected(p):
        calls.append(p)
        return '{"fix": [[1, "Shein"]], "zoom": []}  Actually: {"fix": [], "zoom": [[0, "hook"]]}', {}
    shown, fixes, moments, usage = engine.fix_words_and_moments(words, splits, corrected)
    assert len(calls) == 1 and fixes == [] and moments == [{"i": 0, "kind": "hook"}]


def test_effort_per_step_default_rollback_and_request(monkeypatch):
    from cutsell_worker.simple_engine import llm
    monkeypatch.delenv("CUTSELL_SIMPLE_ENGINE_EFFORT", raising=False)
    expected = {"pass": "medium", "haul_review": "low", "decision": "", "sale_review": "", "fixwords": ""}
    for name, value in expected.items():
        with llm.step(name):
            assert llm.effort_for() == value, name
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_EFFORT", "high")          # rollback: every step as in 1.2.1
    for name in expected:
        with llm.step(name):
            assert llm.effort_for() == ""
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_EFFORT", "pass=medium,fixwords=low,decision=bogus")
    with llm.step("pass"):
        assert llm.effort_for() == "medium"
    with llm.step("fixwords"):
        assert llm.effort_for() == "low"
    with llm.step("decision"):
        assert llm.effort_for() == ""
    assert llm.effort_for() == ""
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_EFFORT", "medium")
    with llm.step("sale_review"):
        assert llm.effort_for() == "medium"

    sent = {}
    class R:
        status_code = 200
        def json(self):
            return {"content": [{"type": "text", "text": "ok"}], "usage": {}}
    def fake_post(url, json=None, **kw):
        sent.update(json); return R()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_EFFORT", "fixwords=low")
    with llm.step("fixwords"):
        llm.call_anthropic("x")
    assert sent["output_config"] == {"effort": "low"}
    sent.clear()
    with llm.step("decision"):
        llm.call_anthropic("x")
    assert "output_config" not in sent


def test_every_engine_call_is_tagged_with_its_step(video):
    from cutsell_worker.simple_engine import engine, llm
    steps = []
    def call(prompt):
        steps.append(llm._STEP.get())
        if "A PASS is one continuous run" in prompt:
            return '{"passes":[[1,3,true]],"backbone":[1,3]}', {}
        if "FIRST decide the TYPE" in prompt:
            return '{"type":"other","hide":[],"unhide":[]}', {}
        if "checking a FIRST CUT" in prompt:
            return '{"type":"other","fixes":[]}', {}
        if "index:word" in prompt:
            return '{"fix":[],"zoom":[]}', {}
        return '{"hide":[]}', {}
    words = [{"w": w, "s": i * 0.4, "e": i * 0.4 + 0.3} for i, w in enumerate("Hola chicas. Miren esto. Cuesta diez.".split())]
    splits, info = engine.decide_reviewed(words, 4.0, call)
    engine.fix_words_and_moments(words, splits, call)
    assert steps == ["pass", "decision", "haul_review", "sale_review", "fixwords"]


def test_prompt_parts_keep_the_same_text_and_cache_the_fixed_instructions(monkeypatch):
    from cutsell_worker.simple_engine import llm
    p = llm.Prompt("FIXED", "own", "\n(run x)", schema={"type": "object"})
    assert str(p) == "FIXEDown\n(run x)" and p.schema == {"type": "object"}
    blocks = llm._content(p)
    assert [b["text"] for b in blocks] == ["FIXED", "own", "\n(run x)"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in blocks[1]
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_CACHE", "0")
    assert llm._content(p) == "FIXEDown\n(run x)"


def test_every_step_has_a_strict_json_schema():
    from cutsell_worker.simple_engine import engine as E
    assert set(E.SCHEMAS) == {"decision", "haul_review", "fixwords"}   # pass + sale review: free text, as live
    for schema in E.SCHEMAS.values():
        assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    assert '"why"' in E.SALE_REVIEW_PROMPT
