import copy, importlib.util, sys, types
import pytest

def stub(name, **attrs):
    if name in sys.modules:
        return
    if importlib.util.find_spec(name) is None:
        m=types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name]=m
stub("requests"); stub("boto3"); stub("clip"); stub("faster_whisper",WhisperModel=object)
from worker import pipeline

def clip(cid,start,keep=True):
    return {"id":cid,"start":start,"end":start+1,"text":cid,"slot":"STORY","semantic_score":1.0,"meta":{"keep":keep}}

@pytest.fixture
def setup(monkeypatch,tmp_path):
    calls={k:[] for k in ("download","asr","semantic","vision","render")}
    monkeypatch.setattr(pipeline,"S3_BUCKET",None); monkeypatch.setattr(pipeline,"ensure_session_dir",lambda _:str(tmp_path))
    monkeypatch.setattr(pipeline,"download_to_local",lambda u,p:calls["download"].append((u,p)))
    monkeypatch.setattr(pipeline,"probe_duration",lambda p:2.0 if "000" in p or p.endswith("input.mp4") else 3.0)
    monkeypatch.setattr(pipeline,"run_asr",lambda p:calls["asr"].append(p) or [{"path":p}])
    data={"input_000.mp4":[clip("same",4),clip("early",1)],"input_001.mp4":[clip("same",0),clip("drop",2,False)],"input.mp4":[clip("same",4),clip("early",1)]}
    monkeypatch.setattr(pipeline,"sentence_boundary_micro_cuts",lambda x:copy.deepcopy(data[x[0]["path"].rsplit("/",1)[-1]]))
    monkeypatch.setattr(pipeline,"merge_incomplete_phrases",lambda x:x)
    monkeypatch.setattr(pipeline,"enrich_clips_semantic",lambda x:calls["semantic"].append(x[0]["source_index"]) or True)
    monkeypatch.setattr(pipeline,"dedupe_clips",lambda x:x)
    monkeypatch.setattr(pipeline,"run_visual_pass",lambda p,d,c:calls["vision"].append(p) or True)
    monkeypatch.setattr(pipeline,"build_slots_dict",lambda x:{})
    monkeypatch.setattr(pipeline,"build_composer",lambda cs,mode:{"mode":mode,"used_clip_ids":[cs[2]["id"],cs[1]["id"]] if len(cs)>2 else [cs[0]["id"]]})
    monkeypatch.setattr(pipeline,"pretty_print_composer",lambda *x:"diagnostic")
    monkeypatch.setattr(pipeline,"render_funnel_video",lambda src,d,cs,ids:calls["render"].append((src,copy.deepcopy(cs),list(ids))) or str(tmp_path/"final.mp4"))
    monkeypatch.setattr(pipeline,"save_result_json_to_s3",lambda x:None)
    def run(mode="clean",files=None): return pipeline.run_pipeline("s",files=files if files is not None else ["one","two"],mode=mode)
    return run,calls

def test_both_urls_and_each_analysis_stage_once(setup):
    result=setup[0](); calls=setup[1]
    assert [x[0] for x in calls["download"]]==["one","two"]
    assert len(calls["asr"])==len(calls["semantic"])==len(calls["vision"])==2
    assert result["processed_source_indices"]==[0,1]

def test_unique_ids_and_source_metadata_on_every_clip(setup):
    cs=setup[0]()["clips"]; assert len({c["id"] for c in cs})==len(cs)
    assert all(all(k in c for k in ("source_index","source_local","source_start","source_end")) for c in cs)

def test_clean_kept_clips_both_sources_canonical_and_renderer_gets_all(setup):
    result=setup[0]("clean"); expected=["source_000:early","source_000:same","source_001:same"]
    assert result["clean_cut_used_clip_ids"]==expected==setup[1]["render"][0][2]
    assert {c["source_index"] for c in setup[1]["render"][0][1]}=={0,1}

def test_human_composer_membership_both_sources_rendered_canonically(setup):
    result=setup[0]("human"); assert result["composer"]["used_clip_ids"]==["source_000:early","source_001:same"]

def test_durations_diagnostics_and_multi_paths(setup):
    r=setup[0](); assert r["input_file_count"]==2 and r["duration_sec"]==5 and r["input_durations_sec"]==[2,3]
    assert [p.rsplit("/",1)[-1] for p in r["input_files_local"]]==["input_000.mp4","input_001.mp4"]

def test_single_file_historical_path_ids_and_renderer_call(setup):
    r=setup[0](files=["one"]); assert r["input_local"].endswith("/input.mp4") and r["input_files_local"]==[r["input_local"]]
    assert [c["id"] for c in r["clips"]]==["same","early"] and isinstance(setup[1]["render"][0][0],str)

def test_file_two_analysis_failure_no_partial_render(setup,monkeypatch):
    original=pipeline.run_asr
    monkeypatch.setattr(pipeline,"run_asr",lambda p:(_ for _ in ()).throw(ValueError()) if "001" in p else original(p))
    with pytest.raises(RuntimeError,match=r"source_index=1.*URL position 2"): setup[0]()
    assert not setup[1]["render"]

def command(monkeypatch,tmp_path,audio):
    got={}; monkeypatch.setattr(pipeline,"has_audio_stream",lambda p:audio[p])
    def fake(cmd,**kw): got["cmd"]=cmd; return types.SimpleNamespace(returncode=0,stdout="",stderr="")
    monkeypatch.setattr(pipeline.subprocess,"run",fake)
    pipeline.render_funnel_video(["one","two"],str(tmp_path),[{**clip("a",1),"source_index":0},{**clip("b",2),"source_index":1}],["a","b"])
    cmd=got["cmd"]; return cmd,cmd[cmd.index("-filter_complex")+1]

def test_different_resolutions_normalized_without_stretching(monkeypatch,tmp_path):
    _,f=command(monkeypatch,tmp_path,{"one":True,"two":True})
    expected=f"scale={pipeline.OUTPUT_WIDTH}:{pipeline.OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease"
    assert f.count(expected)==2 and f.count(f"pad={pipeline.OUTPUT_WIDTH}:{pipeline.OUTPUT_HEIGHT}")==2

def test_different_frame_rates_normalized_before_concat(monkeypatch,tmp_path):
    cmd,f=command(monkeypatch,tmp_path,{"one":True,"two":True})
    assert f.count(f"fps={pipeline.OUTPUT_FPS}")==2 and cmd[cmd.index("-r")+1]==str(pipeline.OUTPUT_FPS)

def test_video_segments_same_sar_pixel_format_and_dimensions(monkeypatch,tmp_path):
    _,f=command(monkeypatch,tmp_path,{"one":True,"two":True})
    assert f.count("setsar=1")==2 and f.count("format=yuv420p")==2 and f.index("format=yuv420p")<f.index("concat=n=2:v=1")

def test_real_audio_normalized_stereo_48khz(monkeypatch,tmp_path):
    _,f=command(monkeypatch,tmp_path,{"one":True,"two":True})
    norm="aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    assert f.count(norm)==2 and f.count("asetpts=PTS-STARTPTS")==2

@pytest.mark.parametrize("audio,silence,audio_output",[
 ({"one":True,"two":False},1,True),({"one":False,"two":True},1,True),
 ({"one":True,"two":True},0,True),({"one":False,"two":False},0,False)])
def test_audio_matrix_silence_matches_real_format_and_concat(monkeypatch,tmp_path,audio,silence,audio_output):
    cmd,f=command(monkeypatch,tmp_path,audio)
    assert f.count("anullsrc=r=48000:cl=stereo")==silence
    assert ("concat=n=2:v=0:a=1[aout]" in f)==audio_output
    assert ("-c:a" in cmd)==audio_output and ("-an" in cmd)==(not audio_output)
    if silence: assert f.count("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo")==2
    assert "+faststart" in cmd and "libx264" in cmd


def test_composer_does_not_suppress_similar_timestamps_across_sources():
    first = clip("source_000:valid", 2.0)
    second = clip("source_001:valid", 2.1)
    first.update({"source_index": 0, "text": "same valid product statement"})
    second.update({"source_index": 1, "text": "same valid product statement"})

    pipeline.build_composer([first, second], mode="human")

    assert first["meta"]["keep"] is True
    assert second["meta"]["keep"] is True


def test_contiguous_blocks_never_cross_source_timelines():
    first = clip("source_000:hook", 1.0)
    second = clip("source_001:hook", 1.1)
    for source_index, candidate in enumerate((first, second)):
        candidate.update({"source_index": source_index, "slot": "HOOK"})

    blocks = pipeline.group_contiguous_blocks_by_slot([first, second], "HOOK")

    assert [[candidate["id"] for candidate in block] for block in blocks] == [
        ["source_000:hook"],
        ["source_001:hook"],
    ]
