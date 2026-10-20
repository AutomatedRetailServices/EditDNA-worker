"""Mix (Figma 1136:2724 / 2804 / 2916): setup, how many, Processing group, Results, voice-first upload."""
from pathlib import Path

IOS = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"


def read(name):
    return (IOS / name).read_text()


def test_mix_screens_follow_figma():
    mix = read("MixView.swift")
    for text in ("Record a few versions of each part. CutSell cleans every clip once and builds every combination for you.",
                 "PRODUCT", "+  Add section", "Pick how many ›", "How many videos?", "combinations are possible",
                 "We mix the clips so each video starts and ends differently.", "PREVIEW",
                 "Tip: post these over several days, between other products. Feeds reward variety.", "MIX"):
        assert text in mix
    assert '["Hook", "Body", "Benefit", "CTA"]' in mix
    assert "maxSections = 6, maxClips = 10, maxVideos = 100" in mix


def test_mix_uses_one_job_and_the_voice_first_upload():
    mix = read("MixView.swift")
    for path in ('"/v1/mix/plan"', '"/v1/mix/jobs"', '"/v1/mix"', '"/v1/mix/\\(mix.mixID)/retry"', '"/v1/exports/jobs"'):
        assert path in mix
    assert "AudioExtractor.extract" in mix and "VoiceUpload.upload" in mix and "BackgroundVideoUploads.shared.start" in mix


def test_mix_in_edits_and_products():
    edits = read("ProjectsView.swift")
    assert "MixGroupRow(mix: mix" in edits and "mixCenter.activeProjectIDs" in edits
    products = read("ProductsView.swift")
    assert "Mix ready: \\(total) videos from" in products and "Export all" in products and 'mixCode = "mix_code"' in products
    hub = read("HomeHubView.swift")
    assert "ForEach([CreateMode.single, .combine]" in hub and "onMix" not in hub
    assert "$0.state != Project.mixContainerState" in read("AppState.swift")


def test_plan_home_has_mix_bulk_in_progress_and_soon():
    plan = read("PlanView.swift")
    for text in ("Make videos in batches and plan what to post.", "New Mix", "Hooks × bodies × CTAs → many videos", "Bulk",
                 "Upload many videos at once", "In progress", "See all in Edits ›", "This week", "SOON", "Plan my week ›"):
        assert text in plan
    assert "MixSetupView(" in plan and "queue.requestedCreateMode = .bulk" in plan and "queue.focus = focus" in plan
    assert "struct PlanView" not in read("MainTabView.swift")
    assert "allowedModes: route.mode == .bulk ? [.bulk] : [.single, .combine]" in read("CreateFlowView.swift")
