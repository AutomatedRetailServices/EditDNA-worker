import SwiftUI

// MARK: - Intro: 4 welcome screens
//
// Shown once, the first time someone reaches the app (after sign-in / "continue without an account"), and again
// from Perfil → Help & Support → Replay intro. When it ends the app is on Create (Figma 945:147 "Tutorial will
// replay from Create"). No Figma frame exists for the screens themselves: they use the app's own pieces
// (brand logo, tab icons, the Mix tile, the CTA gradient, Inter).

@MainActor
final class IntroState: ObservableObject {
    static let shared = IntroState()
    private static let seenKey = "cutsell.intro.seen.v1"
    @Published private(set) var showing = false
    private init() {}

    /// First visit only. UI probes (launched with CUTSELL_* settings) skip it unless CUTSELL_INTRO=1.
    func showIfFirstTime() {
        let env = ProcessInfo.processInfo.environment
        if env["CUTSELL_INTRO"] == "1" { showing = true; return }
        if env.keys.contains(where: { $0.hasPrefix("CUTSELL_") }) { return }
        if !UserDefaults.standard.bool(forKey: Self.seenKey) { showing = true }
    }

    func replay() { showing = true }

    func finish() {
        UserDefaults.standard.set(true, forKey: Self.seenKey)
        showing = false
        CutSellDiagnostics.log("intro_finished", [:])
    }
}

private struct IntroSlide {
    enum Art { case tab(String), mix }
    let art: Art
    let title: String
    let body: String
    let chips: [String]
}

private let introSlides: [IntroSlide] = [
    IntroSlide(art: .tab("TabCreate"), title: "Upload your raw video",
               body: "The AI removes mistakes, pauses and retakes, and leaves a cut ready to sell.",
               chips: ["One", "Many", "Combine"]),
    IntroSlide(art: .mix, title: "Mix: many videos from one recording",
               body: "Record hooks, bodies and CTAs. CutSell combines them into many videos ready to post.",
               chips: ["Hooks", "Bodies", "CTAs"]),
    IntroSlide(art: .tab("TabEdits"), title: "Edit it your way",
               body: "Captions, voice over, overlays, keyframes and Auto Zoom. Your cuts wait for you in Edits.",
               chips: ["Captions", "Voice Over", "Keyframes"]),
    IntroSlide(art: .tab("TabPlan"), title: "Plan your week and post",
               body: "CutSell spreads your videos across the week and reminds you when it's time to post on TikTok.",
               chips: ["Plan my week", "TikTok"]),
]

struct IntroView: View {
    let onFinish: () -> Void
    @State private var index = 0

    private var isLast: Bool { index == introSlides.count - 1 }

    var body: some View {
        ZStack {
            ScreenBackground()
            VStack(spacing: 0) {
                HStack {
                    BrandLogo()
                    Spacer()
                    if !isLast {
                        Button(L10n.tr("Skip"), action: onFinish)
                            .font(CS.inter(15, .semibold))
                            .foregroundStyle(CS.textSecondary)
                            .frame(minWidth: 44, minHeight: 44)
                            .accessibilityIdentifier("intro_skip")
                    }
                }
                .frame(height: 48)
                .padding(.horizontal, 24)
                .padding(.top, 4)

                TabView(selection: $index) {
                    ForEach(introSlides.indices, id: \.self) { i in
                        slideView(introSlides[i])
                            .tag(i)
                            .accessibilityIdentifier("intro_page_\(i + 1)")
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .never))

                HStack(spacing: 8) {
                    ForEach(introSlides.indices, id: \.self) { i in
                        Capsule()
                            .fill(i == index ? AnyShapeStyle(CS.ctaGradient) : AnyShapeStyle(CS.borderDefault))
                            .frame(width: i == index ? 26 : 8, height: 8)
                    }
                }
                .animation(.easeOut(duration: 0.2), value: index)
                .padding(.bottom, 22)

                Button {
                    if isLast { onFinish() } else { withAnimation { index += 1 } }
                } label: {
                    Text(L10n.tr(isLast ? "Start creating" : "Next"))
                        .font(CS.inter(17, .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 56)
                        .background(CS.ctaGradient, in: Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("intro_next")
                .padding(.horizontal, 24)
                .padding(.bottom, 18)
            }
        }
        .preferredColorScheme(.dark)
        .accessibilityIdentifier("introView")
    }

    private func slideView(_ slide: IntroSlide) -> some View {
        VStack(spacing: 0) {
            Spacer(minLength: 12)
            art(slide.art)
            Text(L10n.tr(slide.title))
                .font(CS.inter(28, .bold))
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 40)
            Text(L10n.tr(slide.body))
                .font(CS.inter(16))
                .foregroundStyle(CS.textSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 12)
            HStack(spacing: 8) {
                ForEach(slide.chips, id: \.self) { chip in
                    Text(L10n.tr(chip))
                        .font(CS.inter(13, .semibold))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .padding(.horizontal, 12)
                        .frame(height: 30)
                        .background(Capsule().fill(Color.white.opacity(0.06)))
                        .overlay(Capsule().stroke(CS.chipBorder, lineWidth: 1))
                }
            }
            .padding(.top, 22)
            Spacer(minLength: 12)
        }
        .padding(.horizontal, 28)
    }

    @ViewBuilder
    private func art(_ art: IntroSlide.Art) -> some View {
        ZStack {
            Circle().fill(RadialGradient(colors: [CS.cyan.opacity(0.22), .clear], center: .center, startRadius: 10, endRadius: 120))
                .frame(width: 240, height: 240)
            Circle().stroke(CS.borderDefault, lineWidth: 1).frame(width: 176, height: 176)
            switch art {
            case .tab(let name):
                Image(name).resizable().interpolation(.high).scaledToFit()
                    .frame(width: 120, height: 120)
                    .shadow(color: CS.cyan.opacity(0.55), radius: 18)
            case .mix:
                Text(L10n.tr("⊞"))
                    .font(CS.inter(46, .bold))
                    .foregroundStyle(.white)
                    .frame(width: 112, height: 112)
                    .background(CS.brandFull, in: RoundedRectangle(cornerRadius: 30, style: .continuous))
                    .shadow(color: CS.magenta.opacity(0.45), radius: 18)
            }
        }
        .frame(height: 240)
    }
}

// MARK: - How to use CutSell (Perfil → Help & Support)

private struct GuideTopic: Identifiable {
    let id: String
    let title: String
    let detail: String
    let steps: [String]
}

private struct GuideGroup: Identifiable {
    let id: String
    let label: String
    let topics: [GuideTopic]
}

private let guideGroups: [GuideGroup] = [
    GuideGroup(id: "create", label: "CREATE", topics: [
        GuideTopic(id: "upload", title: "Upload a video", detail: "From your gallery to a cut ready to sell", steps: [
            "In Create, tap Upload videos.",
            "Choose One for a single video, Many to cut several at once, or Combine to join them in order.",
            "Pick your videos. CutSell uploads them and the AI cuts them; you can leave the app meanwhile.",
            "When it's ready you get a notice. The cut is saved in Edits.",
        ]),
        GuideTopic(id: "mix", title: "Mix", detail: "Hooks × bodies × CTAs → many videos", steps: [
            "In Create, tap Mix.",
            "Add your hooks, bodies and CTAs (recorded or from the gallery).",
            "CutSell combines them into many different videos.",
            "Your Mix videos appear in Plan, ready to export.",
        ]),
        GuideTopic(id: "record", title: "Record with CutSell", detail: "Camera with clips · 60 s · 3 min · 10 min", steps: [
            "In Create, tap Record with CutSell.",
            "Choose how long: 60 s, 3 min or 10 min.",
            "Record in clips: tap to start and stop each one. Retakes are fine, the AI keeps the best.",
            "Review the clips and continue; CutSell makes the cut.",
        ]),
    ]),
    GuideGroup(id: "edit", label: "EDIT", topics: [
        GuideTopic(id: "editor", title: "Open the editor", detail: "Cut, split and delete", steps: [
            "In Edits, open a cut.",
            "Slide the timeline under the white line to move through the video.",
            "Tap a track to select it; Split and Delete act on the selected track at the white line.",
            "Undo and Redo are above the timeline.",
        ]),
        GuideTopic(id: "captions", title: "Captions", detail: "Text from your voice, in your style", steps: [
            "In the editor, tap Captions.",
            "CutSell writes them from what you say.",
            "Tap a caption to change its text, style or position.",
        ]),
        GuideTopic(id: "voiceover", title: "Voice Over", detail: "Record your voice over the video", steps: [
            "In the editor, put the white line where the voice should start and tap Voice Over.",
            "Tap the microphone. After the countdown, speak; tap the red square when you're done.",
            "Listen, then use Lower video, Clean up or Retake if you need them.",
            "To close the sheet, swipe it down or tap outside it.",
        ]),
        GuideTopic(id: "keyframes", title: "Keyframes and Auto Zoom", detail: "Move and zoom like in CapCut", steps: [
            "Select the video or an overlay and put the white line where the movement starts.",
            "Tap ◇ to add a keyframe, then move the white line and pinch, drag or turn the picture: a new ◇ appears.",
            "Use ‹ › to jump between keyframes. Tap a ◇ on the timeline to choose how it moves.",
            "Auto Zoom AI adds soft zooms on the key moments; you can delete any of them.",
        ]),
    ]),
    GuideGroup(id: "post", label: "POST", topics: [
        GuideTopic(id: "export", title: "Export and save", detail: "Save the video to Photos", steps: [
            "Open the cut and tap Export.",
            "Choose the quality and export; the video is saved in Photos.",
        ]),
        GuideTopic(id: "plan", title: "Plan my week", detail: "One video per day, with reminders", steps: [
            "In Plan, tap Plan my week.",
            "Choose your goal; CutSell spreads your ready videos across the week.",
            "On the day, you get a reminder to post.",
        ]),
        GuideTopic(id: "tiktok", title: "Post to TikTok", detail: "From the cut to TikTok", steps: [
            "Open a ready cut and tap Post to TikTok.",
            "Post it in TikTok and come back to CutSell.",
            "Answer \"Did you post it?\" so Plan keeps track.",
        ]),
    ]),
    GuideGroup(id: "settings", label: "SETTINGS", topics: [
        GuideTopic(id: "language", title: "Change the language", detail: "English or Español", steps: [
            "Go to Profile → Preferences → App language.",
            "Choose English or Español. The whole app changes at once.",
        ]),
    ]),
]

struct GuidePage: View {
    let back: () -> Void
    @State private var open: Set<String> = []

    var body: some View {
        YouScreen(title: "How to use CutSell", subtitle: "Step-by-step guides for every feature", back: back) {
            ForEach(guideGroups) { group in
                YouSection(label: group.label) {
                    ForEach(Array(group.topics.enumerated()), id: \.element.id) { i, topic in
                        topicRow(topic, divider: i < group.topics.count - 1)
                    }
                }
            }
        }
        .accessibilityIdentifier("guidePage")
    }

    private func topicRow(_ topic: GuideTopic, divider: Bool) -> some View {
        let expanded = open.contains(topic.id)
        return VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.2)) {
                    if expanded { open.remove(topic.id) } else { open.insert(topic.id) }
                }
            } label: {
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(L10n.tr(topic.title)).font(.system(size: 14)).foregroundStyle(YouStyle.primary)
                        Text(L10n.tr(topic.detail)).font(.system(size: 11)).foregroundStyle(YouStyle.muted)
                    }
                    Spacer(minLength: 8)
                    Text(verbatim: "›").font(.system(size: 20)).foregroundStyle(YouStyle.muted)
                        .rotationEffect(.degrees(expanded ? 90 : 0))
                }
                .frame(minHeight: 50)
                .padding(.vertical, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("guide_\(topic.id)")
            if expanded {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(Array(topic.steps.enumerated()), id: \.offset) { n, step in
                        HStack(alignment: .top, spacing: 10) {
                            Text(verbatim: "\(n + 1)")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundStyle(.white)
                                .frame(width: 20, height: 20)
                                .background(Circle().fill(CS.ctaGradient))
                            Text(L10n.tr(step))
                                .font(.system(size: 13))
                                .foregroundStyle(YouStyle.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .padding(.bottom, 14)
                .accessibilityIdentifier("guide_steps_\(topic.id)")
            }
            if divider { Rectangle().fill(YouStyle.border).frame(height: 1) }
        }
    }
}
