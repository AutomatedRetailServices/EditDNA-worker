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
                 "Upload many videos at once", "In progress", "See all in Edits ›", "This week", "Plan my week ›"):
        assert text in plan
    assert "MixSetupView(" in plan and "queue.requestedCreateMode = .bulk" in plan and "queue.focus = focus" in plan
    assert "struct PlanView" not in read("MainTabView.swift")
    assert "allowedModes: route.mode == .bulk ? [.bulk] : [.single, .combine]" in read("CreateFlowView.swift")


def test_plan_my_week_screens():
    week = read("PlanWeekView.swift")
    for text in ("Plan my week", "VIDEOS PER DAY", "DAYS", "REMINDERS", "Build my week", "This week", "Settings", "POSTED", "TODAY",
                 "New Mix ›", "✦ SUGGESTED CAPTION", "Share to TikTok", "Save to Photos", "✓  Mark as posted", "Time to post: "):
        assert text in week
    for path in ('"/v1/plan"', '"/v1/plan/prefs"', '"/v1/plan/week/build"', '"/v1/plan/slots/move"', '"/v1/plan/slots/remove"',
                 '"/v1/plan/posted"', '"/v1/plan/caption"'):
        assert path in week
    assert "requiresExternalPower = true" in week and "requiresNetworkConnectivity = true" in week
    assert "UIActivityViewController" in week and "UIPasteboard.general.string" in week
    plan = read("PlanView.swift")
    assert "SOON" not in plan and "Plan my week ›" in plan and "path = week.hasWeek ? [.week] : [.goal]" in plan
    info = (IOS.parent / "Config" / "Info.plist").read_text()
    assert "ai.cutsell.app.prefetch-posts" in info and "<string>processing</string>" in info


def test_edits_menu_marks_a_video_personal_through_the_plan_route():
    root = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"
    tiles = (root / "ProjectsView.swift").read_text()
    store = (root / "PlanWeekView.swift").read_text()
    assert "Personal · don't plan" in tiles and "Use in Plan" in tiles
    assert '"/v1/plan/personal"' in store and "var personal: [String]?" in store
