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

/// Figma 12_Prototype · Intro 1–4 (1145:2773 / 2802 / 2826 / 2858). Positions are the Figma ones on a 390-wide
/// canvas: the art block (y 150–520) and the text, dots and button block (from y 560) keep their Figma spacing;
/// taller iPhones get the extra room between them.
struct IntroView: View {
    let onFinish: () -> Void
    @State private var index = 0
    private let count = 4

    private var isLast: Bool { index == count - 1 }

    private static let titles = ["Upload your raw video", "Edit in seconds", "Make many videos at once", "Plan your week"]
    private static let bodies = [
        "CutSell removes silences, mistakes and repeats for you.",
        "Add captions, zoom, sounds and your voice. Everything is editable.",
        "Use Bulk or Mix to turn a few clips into dozens of videos.",
        "Pick your times. We remind you and have each video ready to post.",
    ]

    var body: some View {
        ZStack {
            LinearGradient(stops: [.init(color: Color(fixed: 0x03050D), location: 0),
                                   .init(color: Color(fixed: 0x050B1C), location: 0.55),
                                   .init(color: Color(fixed: 0x030409), location: 1)],
                           startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            VStack(spacing: 0) {
                // Logo centred at y 50 (status bar 47 + 3), Skip at 338,62 · Inter SemiBold 15 #99A3BD
                ZStack {
                    BrandLogo()
                    HStack {
                        Spacer()
                        if !isLast {
                            Button(action: onFinish) {
                                Text(L10n.tr("Skip"))
                                    .font(CS.inter(15, .semibold))
                                    .foregroundStyle(Color(fixed: 0x99A3BD))
                                    .frame(minWidth: 44, minHeight: 44)
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("intro_skip")
                            .padding(.trailing, 9)
                        }
                    }
                }
                .frame(height: 44)
                .padding(.top, 3)

                TabView(selection: $index) {
                    ForEach(0..<count, id: \.self) { i in
                        slide(i)
                            .tag(i)
                            .accessibilityElement(children: .contain)
                            .accessibilityIdentifier("intro_page_\(i + 1)")
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .never))

                // Dots at y 700: active 22×8 gradient, others 8×8 #33405C, 8 apart
                HStack(spacing: 8) {
                    ForEach(0..<count, id: \.self) { i in
                        RoundedRectangle(cornerRadius: 4)
                            .fill(i == index ? AnyShapeStyle(CS.brandStroke) : AnyShapeStyle(Color(fixed: 0x33405C)))
                            .frame(width: i == index ? 22 : 8, height: 8)
                    }
                }
                .animation(.easeOut(duration: 0.2), value: index)
                .frame(height: 8)

                // Button at y 736: 350×50, radius 25, #1969FF → #7C3AED 55 % → #19E6FF, shadow 0 6 18 #1969FF 25 %
                Button {
                    if isLast { onFinish() } else { withAnimation { index += 1 } }
                } label: {
                    Text(L10n.tr(isLast ? "Get started" : "Next"))
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .background(Self.buttonGradient, in: RoundedRectangle(cornerRadius: 25, style: .continuous))
                        .shadow(color: Color(fixed: 0x1969FF, opacity: 0.25), radius: 9, x: 0, y: 6)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("intro_next")
                .padding(.horizontal, 20)
                .padding(.top, 28)
                .padding(.bottom, 24)
            }
        }
        .environment(\.colorScheme, .dark)
        .preferredColorScheme(.dark)
        // .contain keeps each control's own id (intro_next, intro_skip).
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("introView")
    }

    static let buttonGradient = LinearGradient(stops: [.init(color: Color(fixed: 0x1969FF), location: 0),
                                                        .init(color: Color(fixed: 0x7C3AED), location: 0.55),
                                                        .init(color: Color(fixed: 0x19E6FF), location: 1)],
                                               startPoint: .leading, endPoint: .trailing)

    private func slide(_ i: Int) -> some View {
        VStack(spacing: 0) {
            // Art: Figma y 150–520 (390-wide canvas), glow ellipse 300 at 45,150
            ZStack(alignment: .topLeading) {
                Image("IntroGlow").resizable().frame(width: 420, height: 420).offset(x: 45 - 60, y: -60)
                    .allowsHitTesting(false)
                switch i {
                case 0: IntroArt.upload
                case 1: IntroArt.edit
                case 2: IntroArt.mix
                default: IntroArt.plan
                }
            }
            .frame(width: 390, height: 370, alignment: .topLeading)
            .padding(.top, 53)   // y 150 − (logo 50 + 44 + 3)
            Spacer(minLength: 8)
            VStack(spacing: 0) {
                Text(L10n.tr(Self.titles[i]))
                    .font(CS.inter(28, .bold))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .frame(width: 350)
                    .fixedSize(horizontal: false, vertical: true)
                Text(L10n.tr(Self.bodies[i]))
                    .font(CS.inter(15))
                    .foregroundStyle(Color(fixed: 0x99A3BD))
                    .multilineTextAlignment(.center)
                    .frame(width: 310)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 12)
            }
            .frame(height: 112, alignment: .top)   // y 560 → dots at 700 (28 above the dots)
            .padding(.bottom, 28)
        }
        .frame(maxWidth: .infinity)
    }
}

/// The four illustrations, drawn from the Figma layers (positions relative to y 150).
private enum IntroArt {
    static let panel = Color(fixed: 0x0F1726)
    static let line = Color(fixed: 0x33405C)
    static let cyan = Color(fixed: 0x19E6FF)

    private static func at(_ x: CGFloat, _ y: CGFloat) -> CGSize { CGSize(width: x, height: y - 150) }

    /// 1145:2773 · RAW 1:46 → CUT 0:40 → the cut
    static var upload: some View {
        ZStack(alignment: .topLeading) {
            card(title: "RAW · 1:46", titleColor: Color(fixed: 0x99A3BD), stroke: AnyShapeStyle(line), width: 1) {
                ForEach(Array([(13, 40, 0), (57, 20, 1), (81, 50, 0), (135, 30, 1), (169, 60, 0), (233, 24, 1), (261, 34, 0)]
                    .enumerated()), id: \.offset) { _, b in
                    RoundedRectangle(cornerRadius: 6).fill(b.2 == 1 ? Color(fixed: 0x591F2E) : Color(fixed: 0x404D6B))
                        .frame(width: CGFloat(b.1), height: 24).offset(x: CGFloat(b.0), y: 31)
                }
            }
            .offset(at(40, 200))
            Text(verbatim: "↓").font(CS.inter(26, .bold)).foregroundStyle(cyan).offset(at(183, 282))
            card(title: "CUT · 0:40", titleColor: cyan, stroke: AnyShapeStyle(Color(fixed: 0x19E5FF)), width: 1.5) {
                ForEach(Array([(12.5, 70.0), (86.5, 90.0), (180.5, 114.0)].enumerated()), id: \.offset) { _, b in
                    RoundedRectangle(cornerRadius: 6).fill(CS.brandStroke)
                        .frame(width: b.1, height: 24).offset(x: b.0, y: 30.5)
                }
            }
            .offset(at(40, 326))
            photo(width: 110, height: 96, radius: 14).offset(at(140, 412))
        }
    }

    /// 1145:2802 · the phone with a caption and four tools around it
    static var edit: some View {
        ZStack(alignment: .topLeading) {
            ZStack(alignment: .topLeading) {
                Image("IntroPerson").resizable().scaledToFill().frame(width: 160, height: 284).clipped()
                Text(L10n.tr("Look at this!"))
                    .font(CS.inter(13, .bold)).foregroundStyle(.white)
                    .frame(width: 132, height: 28)
                    .background(Color.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
                    .offset(x: 12, y: 198)
            }
            .frame(width: 160, height: 284)
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(CS.brandStroke, lineWidth: 2))
            .offset(at(115, 170))
            tool("T").offset(at(52, 200))
            tool("♪").offset(at(290, 230))
            tool("◇").offset(at(52, 320))
            tool("🎙").offset(at(290, 350))
        }
    }

    /// 1145:2826 · Hook / Body / Benefit / CTA × clips = 24 videos
    static var mix: some View {
        ZStack(alignment: .topLeading) {
            ForEach(Array([("Hook", 190, 3), ("Body", 252, 2), ("Benefit", 314, 2), ("CTA", 376, 2)].enumerated()),
                    id: \.offset) { _, row in
                Text(L10n.tr(row.0)).font(CS.inter(12, .bold)).foregroundStyle(.white)
                    .frame(width: 70, height: 30)
                    .background(CS.brandStroke, in: RoundedRectangle(cornerRadius: 15))
                    .offset(at(40, CGFloat(row.1)))
                ForEach(0..<row.2, id: \.self) { k in
                    photo(width: 40, height: 50, radius: 8).offset(at(122 + CGFloat(k) * 46, CGFloat(row.1) - 10))
                }
            }
            Text(L10n.tr("= 24 videos")).font(CS.inter(20, .bold)).foregroundStyle(.white)
                .frame(width: 198, height: 58)
                .background(panel, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.brandStroke, lineWidth: 1.5))
                .offset(at(96, 440))
        }
    }

    /// 1145:2858 · the week, a reminder and Post to TikTok
    static var plan: some View {
        ZStack(alignment: .topLeading) {
            ForEach(0..<5, id: \.self) { d in
                VStack(spacing: 0) {
                    Text(verbatim: (L10nState.code == "es" ? ["L", "M", "M", "J", "V"] : ["M", "T", "W", "T", "F"])[d])
                        .font(CS.inter(13, .bold)).foregroundStyle(.white).padding(.top, 11)
                    Spacer()
                    HStack(spacing: 3) {
                        Circle().fill(Color(fixed: 0x19E6FF)).frame(width: 7, height: 7)
                        Circle().fill(Color(fixed: 0xFF21D6)).frame(width: 7, height: 7)
                    }
                    .padding(.bottom, 18)
                }
                .frame(width: 54, height: 70)
                .background {
                    if d == 1 { RoundedRectangle(cornerRadius: 16, style: .continuous).fill(CS.brandStroke) }
                    else {
                        RoundedRectangle(cornerRadius: 16, style: .continuous).fill(panel)
                            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(line, lineWidth: 1))
                    }
                }
                .offset(at(45 + CGFloat(d) * 62, 190))
            }
            HStack(spacing: 12) {
                Text(verbatim: "C").font(CS.inter(18, .bold)).foregroundStyle(.white)
                    .frame(width: 40, height: 40)
                    .background(CS.brandStroke, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.tr("CutSell · now")).font(CS.inter(11, .semibold)).foregroundStyle(Color(fixed: 0x99A3BD))
                    Text(L10n.tr("Time to post: Rhino Straps 🔥")).font(CS.inter(14, .semibold)).foregroundStyle(.white)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            .padding(.leading, 13)
            .frame(width: 310, height: 70)
            .background(Color(fixed: 0x1A1F2E), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(line, lineWidth: 1))
            .offset(at(40, 290))
            Text(L10n.tr("Post to TikTok")).font(CS.inter(16, .bold)).foregroundStyle(.white)
                .frame(width: 200, height: 52)
                .background(CS.brandStroke, in: RoundedRectangle(cornerRadius: 26))
                .offset(at(95, 384))
        }
    }

    private static func card<Content: View>(title: String, titleColor: Color, stroke: AnyShapeStyle, width: CGFloat,
                                            @ViewBuilder content: () -> Content) -> some View {
        ZStack(alignment: .topLeading) {
            Text(L10n.tr(title)).font(CS.inter(10, .bold)).foregroundStyle(titleColor).offset(x: 13, y: 9)
            content()
        }
        .frame(width: 310, height: 70, alignment: .topLeading)
        .background(panel, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(stroke, lineWidth: width))
    }

    private static func photo(width: CGFloat, height: CGFloat, radius: CGFloat) -> some View {
        Image("IntroPerson").resizable().scaledToFill()
            .frame(width: width, height: height)
            .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).stroke(line, lineWidth: 1))
    }

    private static func tool(_ glyph: String) -> some View {
        Text(verbatim: glyph).font(CS.inter(18, .bold)).foregroundStyle(cyan)
            .frame(width: 48, height: 48)
            .background(panel, in: Circle())
            .overlay(Circle().stroke(line, lineWidth: 1))
    }
}

// MARK: - Router: "Try … now" in a guide opens that part of the app

@MainActor
final class AppRouter: ObservableObject {
    static let shared = AppRouter()
    enum Destination: String { case upload, mix, record, script, edits, products, plan, planWeek }
    /// Set by a guide; MainTabView picks the tab, the screen that owns it opens the feature and clears it.
    @Published var pending: Destination?
    private init() {}

    var tab: MainTab {
        switch pending {
        case .upload, .mix, .record, .script, nil: return .create
        case .edits, .products: return .edits
        case .plan, .planWeek: return .plan
        }
    }
}

// MARK: - How to use CutSell (Figma 1145:2898 list · 1145:2958 one guide)

struct GuideStep: Hashable {
    let title: String
    let body: String
}

struct Guide: Identifiable, Hashable {
    let id: String
    let glyph: String
    let gradientIcon: Bool
    let title: String          // list row
    let name: String           // guide page title
    let subtitle: String
    let steps: [GuideStep]
    let tryTitle: String
    let destination: AppRouter.Destination

    /// The 9 guides, in the Figma order. Steps follow how the app works today.
    static let all: [Guide] = [
        Guide(id: "upload", glyph: "⇪", gradientIcon: false, title: "Upload: Single, Bulk, Combine", name: "Upload",
              subtitle: "From your gallery to a ready cut.", steps: [
                GuideStep(title: "Pick a mode",
                          body: "In Create, tap Single for one cut, Bulk to make one cut per video, or Combine to join them in the order you pick."),
                GuideStep(title: "Choose videos and style",
                          body: "Select your videos, then pick the caption style, Auto Zoom AI and Overlap audio."),
                GuideStep(title: "Let CutSell cut",
                          body: "Tap Create. You can tap Continue in background; the cut waits for you in My videos."),
              ], tryTitle: "Try Upload now", destination: .upload),
        Guide(id: "mix", glyph: "⊞", gradientIcon: true, title: "Mix: many videos from a few clips", name: "Mix",
              subtitle: "Turn a few clips into many videos.", steps: [
                GuideStep(title: "Pick your product", body: "Tap Mix in Create or New Mix in Plan, then choose the product."),
                GuideStep(title: "Record or add clips", body: "Add 1–10 clips to each part: Hook, Body, Benefit and CTA."),
                GuideStep(title: "Check the count", body: "We show how many videos come out, like 3 × 2 × 2 × 2 = 24."),
                GuideStep(title: "Pick how many", body: "Make them all or choose 6, 12 or 18. We keep them different."),
                GuideStep(title: "Make videos", body: "They appear in My videos › Processing and move to Ready one by one."),
              ], tryTitle: "Try Mix now", destination: .mix),
        Guide(id: "editor", glyph: "✂", gradientIcon: false, title: "Editor basics", name: "Editor basics",
              subtitle: "Cut, fix and export a video.", steps: [
                GuideStep(title: "Open a cut", body: "In My videos, tap a cut to open it in the editor."),
                GuideStep(title: "Move through it", body: "Slide the timeline. The white line in the middle is where you are."),
                GuideStep(title: "Split and delete",
                          body: "Tap a track to select it, then Split or Delete at the white line. Undo and Redo sit above the timeline."),
                GuideStep(title: "Export", body: "Tap Export and choose the quality. The video is saved in Photos."),
              ], tryTitle: "Try the editor now", destination: .edits),
        Guide(id: "keyframes", glyph: "◇", gradientIcon: false, title: "Keyframes and zoom", name: "Keyframes and zoom",
              subtitle: "Move and zoom, like in CapCut.", steps: [
                GuideStep(title: "Pick the spot",
                          body: "Select the video or an overlay and move the white line to where the movement starts."),
                GuideStep(title: "Add a keyframe",
                          body: "Tap ◇. Then move the white line and pinch, drag or turn the picture: a new ◇ appears there."),
                GuideStep(title: "Jump and smooth", body: "Use ‹ › to jump between ◇. Tap a ◇ on the timeline to choose how it moves."),
                GuideStep(title: "Auto Zoom AI",
                          body: "CutSell adds soft zooms on key moments. Delete any you don’t want; it won’t come back."),
              ], tryTitle: "Try keyframes now", destination: .edits),
        Guide(id: "voiceover", glyph: "🎙", gradientIcon: false, title: "Voice over", name: "Voice over",
              subtitle: "Add your voice to any cut.", steps: [
                GuideStep(title: "Start where the voice goes",
                          body: "In the editor, put the white line where you want to talk and tap Voice Over."),
                GuideStep(title: "Record", body: "Tap the mic. After 3·2·1, speak. Tap the red square when you’re done."),
                GuideStep(title: "Listen and adjust", body: "Use Lower video, Clean up or Retake, and set the voice volume."),
                GuideStep(title: "Close", body: "Swipe the sheet down or tap outside it. Tap the voice track to change it later."),
              ], tryTitle: "Try Voice over now", destination: .edits),
        Guide(id: "teleprompter", glyph: "≡", gradientIcon: false, title: "Teleprompter", name: "Teleprompter",
              subtitle: "Read your script while you record.", steps: [
                GuideStep(title: "Write or paste",
                          body: "In Record with CutSell, tap Script. Paste or write it; it’s saved for that product. ✦ Split for Mix sends each part to its Mix section."),
                GuideStep(title: "Set speed and size", body: "Choose the scroll speed and text size, and keep Start after 3·2·1 on."),
                GuideStep(title: "Record with script",
                          body: "The text scrolls near the lens and pauses when you pause. Tap it to move, resize or edit it."),
              ], tryTitle: "Try the teleprompter now", destination: .script),
        Guide(id: "products", glyph: "▤", gradientIcon: false, title: "Products in My videos", name: "Products",
              subtitle: "Your videos, by brand and product.", steps: [
                GuideStep(title: "Open Products", body: "In My videos, tap Products. Your cuts are grouped by brand and product."),
                GuideStep(title: "Assign a product",
                          body: "If CutSell couldn’t tell the product, tap Assign and pick it or add a new one."),
                GuideStep(title: "Move videos", body: "Open a product, select videos and tap Move to change their product."),
              ], tryTitle: "Try Products now", destination: .products),
        Guide(id: "planweek", glyph: "▦", gradientIcon: false, title: "Plan my week", name: "Plan my week",
              subtitle: "A video ready at each time you pick.", steps: [
                GuideStep(title: "Open Plan my week", body: "In Plan, tap Plan my week."),
                GuideStep(title: "Pick how many and when", body: "Choose how many videos and at what times, then tap Build my plan."),
                GuideStep(title: "Check your week",
                          body: "We spread your ready videos so every day has variety. Tap one to move it to another day or time."),
                GuideStep(title: "Get reminded",
                          body: "At each time you get a reminder with the caption and video ready."),
              ], tryTitle: "Try Plan my week now", destination: .planWeek),
        Guide(id: "tiktok", glyph: "↗", gradientIcon: false, title: "Post to TikTok", name: "Post to TikTok",
              subtitle: "From CutSell to TikTok in a few taps.", steps: [
                GuideStep(title: "Open a ready video",
                          body: "From the reminder or in Plan, open the video and tap Post to TikTok."),
                GuideStep(title: "Paste and post",
                          body: "We save the video, copy your caption and open TikTok. Pick the video, paste the caption and post."),
                GuideStep(title: "Tell us you posted", body: "Back in CutSell, answer “Did you post it?” so Plan marks it as posted."),
              ], tryTitle: "Try it now", destination: .plan),
    ]
}

/// Figma 1145:2898: "‹ Help", title, "Short guides, 3–5 steps each.", search, one row per guide.
struct GuidePage: View {
    let back: () -> Void
    @State private var query = ""
    @State private var open: Guide?

    private var shown: [Guide] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return Guide.all }
        return Guide.all.filter { guide in
            ([guide.title, guide.subtitle] + guide.steps.flatMap { [$0.title, $0.body] })
                .contains { L10n.tr($0).lowercased().contains(q) || $0.lowercased().contains(q) }
        }
    }

    var body: some View {
        ZStack {
            if let open {
                GuideDetailPage(guide: open) { withAnimation(.easeOut(duration: 0.2)) { self.open = nil } }
                    .transition(.move(edge: .trailing))
            } else {
                list.transition(.move(edge: .leading))
            }
        }
        .animation(.easeOut(duration: 0.2), value: open)
    }

    private var list: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                GuideBack(title: "Help", id: "guideBack", action: back)
                Text(L10n.tr("How to use CutSell")).font(CS.inter(28, .bold)).foregroundStyle(Color.ink).padding(.top, 9)
                Text(L10n.tr("Short guides, 3–5 steps each.")).font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD))
                    .padding(.top, 4)
                HStack(spacing: 8) {
                    Text(verbatim: "⌕").font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD))
                    TextField("", text: $query, prompt: Text(L10n.tr("Search guides")).foregroundStyle(Color(hex: 0x99A3BD)))
                        .font(CS.inter(13))
                        .foregroundStyle(Color.ink)
                        .autocorrectionDisabled()
                        .accessibilityIdentifier("guideSearch")
                }
                .padding(.horizontal, 13)
                .frame(height: 40)
                .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
                .dayCardShadow()
                .padding(.top, 16)
                VStack(spacing: 8) {
                    ForEach(shown) { guide in row(guide) }
                    if shown.isEmpty {
                        Text(L10n.tr("No guide matches that."))
                            .font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 12)
                    }
                }
                .padding(.top, 14)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 30)
        }
        .scrollBounceBehavior(.basedOnSize)
        .scrollDismissesKeyboard(.interactively)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("guidePage")
    }

    private func row(_ guide: Guide) -> some View {
        Button { withAnimation(.easeOut(duration: 0.2)) { open = guide } } label: {
            HStack(spacing: 10) {
                Text(verbatim: guide.glyph).font(CS.inter(15, .bold)).foregroundStyle(guide.gradientIcon ? Color.white : Color.ink)
                    .frame(width: 36, height: 36)
                    .background {
                        if guide.gradientIcon {
                            RoundedRectangle(cornerRadius: 10, style: .continuous).fill(CS.brandStroke)
                        } else {
                            RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(hex: 0x1F2638))
                        }
                    }
                VStack(alignment: .leading, spacing: 3) {
                    Text(L10n.tr(guide.title)).font(CS.inter(14, .semibold)).foregroundStyle(Color.ink).lineLimit(1)
                        .minimumScaleFactor(0.85)
                    Text(L10n.tr("\(guide.steps.count) steps")).font(CS.inter(11)).foregroundStyle(Color(hex: 0x99A3BD))
                }
                Spacer(minLength: 6)
                Text(verbatim: "›").font(CS.inter(20, .medium)).foregroundStyle(Color(hex: 0x99A3BD))
            }
            .padding(.leading, 9)
            .padding(.trailing, 14)
            .frame(height: 52)
            .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
            .dayCardShadow()
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("guide_\(guide.id)")
    }
}

/// Figma 1145:2958: "‹ Guides", the guide's name and line, numbered gradient steps joined by a line, "Try … now".
struct GuideDetailPage: View {
    let guide: Guide
    let back: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                GuideBack(title: "Guides", id: "guideDetailBack", action: back)
                Text(L10n.tr(guide.name)).font(CS.inter(28, .bold)).foregroundStyle(Color.ink).padding(.top, 9)
                Text(L10n.tr(guide.subtitle)).font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 4)
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(guide.steps.enumerated()), id: \.offset) { n, step in
                        HStack(alignment: .top, spacing: 12) {
                            VStack(spacing: 0) {
                                Text(verbatim: "\(n + 1)").font(CS.inter(14, .bold)).foregroundStyle(.white)
                                    .frame(width: 32, height: 32)
                                    .background(CS.brandStroke, in: Circle())
                                if n < guide.steps.count - 1 {
                                    Rectangle().fill(Color(hex: 0x33405C)).frame(width: 2).frame(minHeight: 52)
                                        .padding(.top, 6)
                                }
                            }
                            .frame(width: 32)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(L10n.tr(step.title)).font(CS.inter(15, .semibold)).foregroundStyle(Color.ink)
                                Text(L10n.tr(step.body)).font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD))
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .padding(.top, 4)
                            .padding(.bottom, n < guide.steps.count - 1 ? 14 : 0)
                        }
                        .accessibilityElement(children: .combine)
                        .accessibilityIdentifier("guideStep_\(n + 1)")
                    }
                }
                .padding(.top, 26)
                Button {
                    // Close Perfil and open the feature.
                    AppRouter.shared.pending = guide.destination
                    dismiss()
                } label: {
                    Text(L10n.tr(guide.tryTitle))
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .background(IntroView.buttonGradient, in: RoundedRectangle(cornerRadius: 25, style: .continuous))
                        .shadow(color: Color(fixed: 0x1969FF, opacity: 0.25), radius: 9, x: 0, y: 6)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("guideTry")
                .padding(.top, 50)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 30)
        }
        .scrollBounceBehavior(.basedOnSize)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("guideDetail_\(guide.id)")
    }
}

/// "‹ Help" / "‹ Guides": Inter SemiBold 16, cyan (#1769FF in day mode).
private struct GuideBack: View {
    let title: String
    let id: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Text(verbatim: "‹ " + L10n.tr(title)).font(CS.inter(16, .semibold)).foregroundStyle(CS.cyan)
                .frame(minHeight: 44, alignment: .leading)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
        .padding(.top, 2)
    }
}
