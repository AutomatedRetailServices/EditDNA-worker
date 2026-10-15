"""Editor v5 (Figma 04_Editor "Editor v5 · Paneles"): toolbar, panels, overlay layers / layouts,
draft audio clips (voice-over, music, CutSell sounds), speed, volume, AI notice and App Review items.
Static checks of the iPhone sources; the server side is covered by tests/test_cutsell_editor_v5_*.py."""
import plistlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (IOS / name).read_text()


def test_every_panel_is_a_card_above_the_toolbar_with_close():
    panels = read("EditorV5Panels.swift")
    for name in ("OverlayAddPanelV5", "OverlayPanelV5", "VolumePanelV5", "SpeedPanelV5", "VoiceClipPanelV5",
                 "SoundClipPanelV5", "AudioPanelV5", "VoiceOverSheetV5", "CountdownOverlayV5"):
        assert f"struct {name}" in panels, name
    assert 'Image(closeIsCheck ? "Ed5CloseCheck" : "Ed5CloseX")' in panels
    v4 = read("EditorV4View.swift")
    # The video gets smaller by the panel's height so the timeline stays in view.
    assert "TimelineV4.height(for: model, compact: card != nil) + toolbar + cardHeight" in v4
    assert '"Add overlay", subtitle: "From your Photos"' in panels


def test_overlay_layers_new_row_when_overlapping_and_toast():
    model = read("EditorV5Model.swift")
    assert "static func layer(for start: Double, end: Double, existing:" in model
    assert "return (top + 1, true)" in model
    vm = read("DraftEditorViewModel.swift")
    assert "OverlayLayers.layer(for: begin, end: min(total, begin + 4), existing: existing)" in vm
    v4 = read("EditorV4View.swift")
    assert 'added on a new layer · tap ↶ to undo' in v4
    # Higher layer drawn on top in the preview, as in the export.
    assert "model.overlaysOnScreen.sorted { $0.style.layer < $1.style.layer }" in v4
    timeline = read("EditorV4Timeline.swift")
    assert "case .layer(let layer): overlayTrack(layer: layer" in timeline and "LayerBadge(number: layer + 1)" in timeline


def test_overlay_style_fields_match_the_server():
    model = read("EditorV5Model.swift")
    for case in ("float", "side_by_side", "stacked", "left", "right", "top", "bottom", "full"):
        assert case in model, case
    for key in ('object["layer"]', 'object["layout"]', 'object["edge"]', 'object["fill"]', 'object["rotation"]',
                'object["mute_audio"]', 'object["volume"]'):
        assert key in model, key
    assert "case .soft: return 12" in model and "case .extra_soft: return 28" in model


def test_audio_clips_on_the_draft_with_server_field_names():
    model = read("EditorV5Model.swift")
    for key in ('"audio_id"', '"kind"', '"start"', '"source_in"', '"source_out"', '"volume"', '"asset_id"',
                '"sfx_id"', '"duck_video"', '"clean"', '"enhanced_at_record"'):
        assert key in model, key
    assert "func splitAudioClip(id: String, at time: Double) -> Bool" in model
    assert "func trimAudioClip(id: String, edge: TrimEdge, to time: Double)" in model
    assert 'changed.sfxID = "typing_long"' in model          # Typing stretched past 3 s
    assert "func adoptLegacyVoiceOvers()" in model            # old voice placements join the draft once


def test_speed_is_applied_everywhere_the_timeline_is_measured():
    tracks = read("EditorV2View.swift")
    assert "let length = max(0, end - start) / speed" in tracks
    playback = read("DraftPlaybackView.swift")
    assert "composition.scaleTimeRange(CMTimeRange(start: cursor, duration: durationTime), toDuration: played)" in playback
    assert "item.audioTimePitchAlgorithm = .spectral" in playback
    captions = read("CaptionPreviewRules.swift")
    assert "TimedWord(start: $0.start / rate, end: $0.end / rate" in captions
    vm = read("DraftEditorViewModel.swift")
    assert "/ ClipSpeed.of(clip)" in vm


def test_preview_ducks_the_video_under_a_voice_like_the_export():
    playback = read("DraftPlaybackView.swift")
    assert "let dip: Float = 0.25   // -12 dB" in playback
    assert "model.previewVoices.filter(\\.ducksVideo)" in playback


def test_voice_over_two_steps_with_countdown_and_enhance():
    panels = read("EditorV5Panels.swift")
    assert 'Text("Removes background noise and sounds, so only your voice is heard.")' in panels
    assert "for value in [3, 2, 1]" in panels and "await recorder.start(enhance: enhance)" in panels
    assert '"Preview your recording before saving."' in panels
    recorder = read("VoiceRecorder.swift")
    assert "mode: enhance ? .voiceChat : .spokenAudio" in recorder


def test_audio_panel_has_only_the_five_sounds_and_music_notice():
    model = read("EditorV5Model.swift")
    assert '("tap", "Tap"), ("good", "Good"), ("wrong", "Wrong"), ("glass", "Glass"), ("typing", "Typing")' in model
    assert '"/v1/sfx/manifest"' in model
    v4 = read("EditorV4View.swift")
    assert '"Use only music you own or have rights to."' in v4
    assert "cutsell.musicRightsNoticeShown" in v4


def test_no_audio_files_in_the_repository():
    for path in (ROOT / "mobile").rglob("*"):
        assert path.suffix.lower() not in {".m4a", ".mp3", ".wav", ".aiff", ".caf"}, path


def test_ai_notice_before_the_first_cut():
    consent = read("AIConsent.swift")
    assert "Deepgram" in consent and "Anthropic (Claude)" in consent
    assert 'PrimaryButton(title: "Agree"' in consent
    submit = read("CutSubmission.swift")
    assert "guard await AIConsentCenter.shared.ensure() else { throw AIConsentError.declined }" in submit
    assert submit.index("AIConsentCenter.shared.ensure()") < submit.index("isSubmitting = true")


def test_privacy_manifest_terms_and_permissions():
    manifest = plistlib.loads((IOS / "PrivacyInfo.xcprivacy").read_bytes())
    assert manifest["NSPrivacyTracking"] is False
    types = {item["NSPrivacyAccessedAPIType"] for item in manifest["NSPrivacyAccessedAPITypes"]}
    assert {"NSPrivacyAccessedAPICategoryUserDefaults", "NSPrivacyAccessedAPICategoryFileTimestamp"} <= types
    assert (IOS / "Legal" / "privacy_policy.md").exists() and (IOS / "Legal" / "terms_of_use.md").exists()
    you = read("YouView.swift")
    assert 'Button("Privacy Policy")' in you and 'Button("Terms of Use")' in you
    project = (ROOT / "mobile" / "ios" / "project.yml").read_text()
    for key in ("NSCameraUsageDescription", "NSMicrophoneUsageDescription", "NSPhotoLibraryUsageDescription"):
        assert f"INFOPLIST_KEY_{key}: CutSell" in project


def test_no_technical_text_on_screen():
    for name in ("BrollView.swift", "OverlayView.swift", "VoiceOverView.swift", "TimelineEditorView.swift"):
        source = read(name)
        texts = [line for line in source.splitlines() if 'Text("' in line]
        for line in texts:
            for word in ("backend", "authority", "contract", "gate", "registered", "simulated", "/v1/"):
                assert word not in line, (name, line.strip())


def test_ai_notice_and_policy_name_exactly_the_ai_services_in_use():
    # Audit 15 Oct 2026: the live engine (Modal, CUTSELL_ENGINE=simple) calls Deepgram (audio) and
    # Anthropic (transcript text) only; Gemini / OpenAI code paths are off in production.
    consent = read("AIConsent.swift")
    policy = (IOS / "Legal" / "privacy_policy.md").read_text()
    for text in (consent, policy):
        assert "Deepgram" in text and "Anthropic (Claude)" in text
        for unused in ("OpenAI", "Gemini", "Google", "ChatGPT", "Whisper"):
            assert unused not in text, unused
    assert "Cloudflare" in consent and "Cloudflare (R2)" in policy
    for doc in ("privacy_policy.md", "terms_of_use.md"):
        body = (IOS / "Legal" / doc).read_text()
        assert "arsgroup01@gmail.com" in body and "[contact email" not in body
