import SwiftUI

/// Processing → Ready (Figma 08: "Prototype/Master/Single/Processing · Canonical" 670:409 and
/// "Ready · Canonical" 672:405). Step 1 is the upload (its real %); the rest follows the cut's
/// REAL state on the server (CutTracker) until it is ready. Then "Review & edit" opens the editor.
struct CutProgressView: View {
    /// Set while the video is still uploading from this iPhone (Create flow).
    var submitter: CutSubmitter?
    /// The project, once the server created it (always set when opened from Edits).
    var projectID: String?
    var fileName: String
    var thumbnail: UIImage?
    /// Back arrow / "Continue in background".
    let onClose: () -> Void
    /// "Edits" in the top right.
    let onOpenEdits: () -> Void

    @EnvironmentObject private var appState: AppState
    @ObservedObject private var tracker = CutTracker.shared
    @ObservedObject private var videos = BackgroundVideoUploads.shared
    @State private var openEditor = false

    private var cut: CutTracker.Cut? { projectID.flatMap { tracker.cut($0) } }

    var body: some View {
        ZStack {
            CS.wiPage.ignoresSafeArea()
            if let cut, cut.isReady, let project = readyProject(cut) {
                ReadyContent(cut: cut, project: project, onBack: onClose, onReview: { openEditor = true })
            } else {
                ProcessingContent(
                    stage: stage,
                    // The cut's name ("Cut 14 Oct"), never the phone's file name ("copy_9E49…"); nothing yet
                    // while it has no name.
                    fileName: CutName.display(cut?.title),
                    thumbnail: uploading ? thumbnail : (cut?.thumbnail ?? thumbnail),
                    failed: !uploading && cut?.isFailed == true,
                    uploading: uploading,
                    stalled: uploading ? submitter?.stalled : nil,
                    background: projectID.flatMap { videos.status($0)?.text },
                    onTryAgain: { submitter?.restartStuckParts() },
                    onClose: onClose,
                    onOpenEdits: onOpenEdits
                )
            }
        }
        .navigationDestination(isPresented: $openEditor) {
            if let cut, let project = readyProject(cut) {
                DraftEditorView(project: project)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .onAppear { tracker.visibleProjectID = projectID }
        .onChange(of: projectID) { _, id in tracker.visibleProjectID = id }
        .onDisappear { if tracker.visibleProjectID == projectID { tracker.visibleProjectID = nil } }
    }

    /// Videos are still going up from this iPhone (also cuts 2..N of a Bulk).
    private var uploading: Bool { submitter?.isSubmitting == true || (projectID == nil && submitter != nil) }

    private var stage: CutTracker.Stage {
        if uploading, let submitter {
            // Build 19: the real step and ITS real % ("Downloading from iCloud 40%", "Preparing your
            // video 12%", "Uploading your video 37%"); "…" for a step without a %.
            let label = submitter.statusText.components(separatedBy: " · ").first ?? ""
            let bulkLabel = label.hasPrefix("Cut ") ? label + " · " : ""
            let step = submitter.stageTitle == "Starting your cut" ? 2 : 1
            return CutTracker.Stage(step: step, title: bulkLabel + submitter.stageTitle, percent: submitter.stagePercent ?? -1)
        }
        if let cut { return CutTracker.serverStage(cut) }
        return CutTracker.Stage(step: 1, title: "Uploading your video", percent: 0)
    }

    private func readyProject(_ cut: CutTracker.Cut) -> Project? {
        if let known = appState.projects.first(where: { $0.projectID == cut.id }) {
            var project = known
            project.state = "draft_ready"
            return project
        }
        guard let userID = appState.session?.userID else { return nil }
        return Project(projectID: cut.id, userID: userID, title: cut.title, state: "draft_ready",
                       latestJobID: cut.jobID, sourceCount: nil, updatedAt: nil)
    }
}

// MARK: - Processing (670:409)

/// The name shown for a cut: its title, unless that is a technical file name.
enum CutName {
    static func display(_ title: String?) -> String {
        let name = (title ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return isTechnical(name) ? "" : name
    }

    /// File names from the phone or the server ("copy_9E4975E5-…", "IMG_1234.MOV", "trim.ABC.mp4").
    static func isTechnical(_ name: String) -> Bool {
        if name.isEmpty { return true }
        let lower = name.lowercased()
        if lower.hasPrefix("copy_") || lower.hasPrefix("img_") || lower.hasPrefix("trim.") || lower.hasPrefix("rpreplay") { return true }
        if [".mov", ".mp4", ".m4v", ".heic", ".jpg", ".png"].contains(where: { lower.hasSuffix($0) }) { return true }
        // A long hexadecimal run (UUIDs and hashes).
        return name.range(of: "[0-9A-Fa-f]{8}-?[0-9A-Fa-f]{4}", options: .regularExpression) != nil
    }
}

struct ProcessingContent: View {
    let stage: CutTracker.Stage
    let fileName: String
    let thumbnail: UIImage?
    var failed = false
    /// While the video uploads from this iPhone CutSell must stay open, so there is no way out.
    var uploading = false
    /// Build 21: "No data sent for 90 seconds…" (the upload keeps trying; nothing was cancelled).
    var stalled: String? = nil
    /// Build 23: "Uploading video in background · 40 % · about 1 min left" (audio first).
    var background: String? = nil
    /// "Try again": restarts only the parts in flight.
    var onTryAgain: () -> Void = {}
    let onClose: () -> Void
    let onOpenEdits: () -> Void

    /// White Indigo (Figma 1179:2 "Single · Processing ✅ APROBADO").
    var body: some View {
        VStack(spacing: 0) {
            WIBackBar(title: failed ? "Processing issue" : "Processing", onBack: onClose, trailingTitle: "Edits", onTrailing: onOpenEdits)

            VStack(spacing: 16) {
                VideoFirstFrame(image: thumbnail, width: 100, height: 156, cornerRadius: 20)
                Text(fileName)
                    .accessibilityIdentifier("processingCutName")
                    .font(CS.inter(13, .medium))
                    .foregroundStyle(CS.wiSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 260)
            }
            .padding(.top, 14)

            VStack(spacing: 6) {
                Text(L10n.tr(failed ? "This cut needs another try" : "Hold on…"))
                    .font(CS.inter(failed ? 26 : 30, .heavy))
                    .tracking(failed ? -0.52 : -0.6)
                    .foregroundStyle(CS.wiInk)
                Text(L10n.tr(failed ? "Something went wrong on our side." : "We’re saving you hours per day…"))
                    .font(CS.inter(15, .medium))
                    .foregroundStyle(CS.wiSecondary)
            }
            .multilineTextAlignment(.center)
            .padding(.top, 14)

            if failed {
                // White Indigo error (Figma 1176:2858): red mark in a circle and the green "original is safe" card.
                WIStatusBadge(kind: .error).padding(.top, 22)
                WIOriginalSafeCard().padding(.top, 22)
            } else {
                BrandRing(animating: true)
                    .frame(width: 150, height: 150)
                    .padding(.top, 16)
            }

            if !failed {
            VStack(spacing: 4) {
                Text(L10n.tr(stage.title))
                    .accessibilityIdentifier("processingStage")
                    .font(CS.inter(17, .bold))
                    .foregroundStyle(CS.wiInk)
                    .multilineTextAlignment(.center)
                Text(L10n.tr(stage.percent >= 0 ? "\(stage.percent)%" : "…"))
                    .font(CS.inter(26, .heavy))
                    .foregroundStyle(CS.accent)
                    .contentTransition(.numericText())
                    .animation(.easeOut, value: stage.percent)
                    .accessibilityIdentifier("processingPercent")
                if let background {
                    Text(L10n.tr(background))
                        .font(CS.inter(13, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.top, 6)
                        .padding(.horizontal, 24)
                        .accessibilityIdentifier("backgroundVideoUpload")
                }
                if let stalled {
                    VStack(spacing: 8) {
                        Text(L10n.tr(stalled))
                            .font(CS.inter(13, .medium))
                            .foregroundStyle(Color(fixed: 0xB26A00))
                            .multilineTextAlignment(.center)
                            .accessibilityIdentifier("uploadStalled")
                        Button(L10n.tr("Try again"), action: onTryAgain)
                            .font(CS.inter(15, .bold))
                            .foregroundStyle(CS.accent)
                            .accessibilityIdentifier("uploadTryAgain")
                    }
                    .padding(.top, 8)
                    .padding(.horizontal, 24)
                }
            }
            .padding(.top, 14)
            }

            Spacer(minLength: 12)

            VStack(spacing: 12) {
                // Figma 1179:22 "Note": 52 high, radius 18, info icon + Medium 13.
                HStack(spacing: 8) {
                    Image("WIInfo").renderingMode(.template).resizable().frame(width: 18, height: 18)
                        .foregroundStyle(CS.wiInk)
                    Text(L10n.tr(uploading ? "Your video keeps uploading if you leave this screen or CutSell." : "Processing continues if you leave this screen."))
                        .font(CS.inter(13, .medium))
                        .foregroundStyle(CS.wiInk)
                        .lineLimit(2).minimumScaleFactor(0.85)
                }
                .padding(.horizontal, 16)
                .frame(maxWidth: .infinity)
                .frame(minHeight: 52)
                .wiCard(radius: 18, shadow: false)
                Button(action: onClose) {
                    Text(L10n.tr("Continue in background"))
                        .font(CS.inter(16, .bold))
                        .foregroundStyle(Color(fixed: 0xFFFFFF))
                        .frame(maxWidth: .infinity)
                        .frame(height: 52)
                        .background(CS.neutralButton, in: Capsule())
                }
                .buttonStyle(.plain)
                // Build 24: active at every step (uploading, analysing, cutting, saving).
                .accessibilityIdentifier("continueInBackground")
                Text(L10n.tr("We’ll notify you when your cut is ready."))
                    .font(CS.inter(12, .medium))
                    .foregroundStyle(CS.wiSecondary)
            }
            .padding(.bottom, 8)
        }
        .padding(.horizontal, 20)
    }
}

/// 9:16 first frame (White Indigo, Figma 1179:8): 3 pt white border, soft blue shadow (0 12 32, #1769FF at 28 %).
struct VideoFirstFrame: View {
    let image: UIImage?
    let width: CGFloat
    let height: CGFloat
    let cornerRadius: CGFloat

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                CS.accentSoft
                BrandIcon(size: width * 0.5)
            }
        }
        .frame(width: width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous).stroke(Color(fixed: 0xFFFFFF), lineWidth: 3))
        .shadow(color: Color(fixed: 0x1769FF, opacity: 0.28), radius: 16, y: 12)
    }
}

/// White Indigo brand ring (Figma 1186:2 "Processing/BrandRing · White", note 1193:2): four arcs with the brand
/// gradient (cyan → blue → purple → pink) and their neon glow, turning without stopping while it processes (one turn
/// every 2.5 s); TRANSPARENT centre with the logo icon (Logo/Horizontal without letters) and its own glow.
/// Reduce Motion: the arcs stay still and the glow breathes softly. Used by Single / Combine Processing and Export.
struct BrandRing: View {
    var animating = true
    /// Ring diameter; everything scales with it (Figma: 120).
    var size: CGFloat = 120
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    static let stops: [Gradient.Stop] = [
        .init(color: Color(fixed: 0x19E6FF), location: 0), .init(color: Color(fixed: 0x1769FF), location: 0.28),
        .init(color: Color(fixed: 0x7C3AED), location: 0.55), .init(color: Color(fixed: 0xFF1493), location: 0.78),
        .init(color: Color(fixed: 0x19E6FF), location: 1)]
    static let colors: [Color] = stops.map(\.color)
    static let secondsPerTurn = 2.5

    /// Ring angle (degrees) at `time`: one full linear turn every 2.5 s.
    static func angle(at time: TimeInterval) -> Double {
        (time.truncatingRemainder(dividingBy: secondsPerTurn) / secondsPerTurn) * 360
    }

    var body: some View {
        let scale = size / 120
        TimelineView(.animation(minimumInterval: nil, paused: !animating)) { context in
            let time = context.date.timeIntervalSinceReferenceDate
            let turning = animating && !reduceMotion
            let angle = turning ? Self.angle(at: time) : 0
            let glowOpacity = (animating && reduceMotion) ? 0.35 + 0.25 * (1 + sin(time * 2 * .pi / 2.4)) / 2 : 0.52
            ZStack {
                // Processing/RingGlow: the same arcs, blurred behind (opacity 52 %).
                arcs(scale).blur(radius: 12 * scale).opacity(glowOpacity)
                    .rotationEffect(.degrees(angle))
                // Processing/RingGradient with its two neon shadows (cyan up-right, magenta down-left).
                arcs(scale)
                    .shadow(color: Color(fixed: 0x00E0FF, opacity: 0.76), radius: 3.5 * scale, x: 3 * scale, y: -1 * scale)
                    .shadow(color: Color(fixed: 0xE800FF, opacity: 0.58), radius: 4 * scale, x: -3 * scale, y: 3 * scale)
                    .rotationEffect(.degrees(angle))
                BrandIcon(size: 66 * scale)
            }
        }
        .frame(width: size + 30 * scale, height: size + 30 * scale)
        .accessibilityHidden(true)
    }

    /// Four 60° arcs, 30° apart, 10 pt wide with round ends (one centred on each side), the gradient along them.
    private func arcs(_ scale: CGFloat) -> some View {
        BrandArcs()
            .stroke(AngularGradient(stops: Self.stops, center: .center),
                    style: StrokeStyle(lineWidth: 10 * scale, lineCap: .round))
            .frame(width: size, height: size)
    }
}

/// The ring's four arcs (Figma RingGradient path: each spans ±30° around 0°, 90°, 180° and 270°).
struct BrandArcs: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) / 2
        for side in 0..<4 {
            let middle = Double(side) * 90
            path.move(to: CGPoint(x: center.x + radius * cos((middle - 30) * .pi / 180),
                                  y: center.y + radius * sin((middle - 30) * .pi / 180)))
            path.addArc(center: center, radius: radius, startAngle: .degrees(middle - 30), endAngle: .degrees(middle + 30),
                        clockwise: false)
        }
        return path
    }
}

struct StageDots: View {
    let current: Int
    let total: Int

    var body: some View {
        HStack(spacing: 11) {
            ForEach(1...total, id: \.self) { index in
                Circle()
                    .fill(index < current ? CS.cyan : index == current ? Color(hex: 0x7C3AED) : Color(hex: 0x2A3D63))
                    .frame(width: index == current ? 9 : 7, height: index == current ? 9 : 7)
                    .shadow(color: index == current ? Color(hex: 0x7C3AED).opacity(0.8) : .clear, radius: 4)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L10n.tr("Step \(current) of \(total)"))
    }
}

// MARK: - Ready (672:405)

struct ReadyContent: View {
    let cut: CutTracker.Cut
    let project: Project
    let onBack: () -> Void
    let onReview: () -> Void

    @EnvironmentObject private var appState: AppState
    @State private var finalSeconds: Double?
    @State private var keptClips: Int?
    @State private var footageSeconds: Double?
    @State private var totalSaved: Double?

    /// White Indigo (Figma 1179:30 "Single · Ready"). Combine · Ready uses the same screen.
    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                WIBackBar(onBack: onBack)

                VideoFirstFrame(image: cut.thumbnail, width: 132, height: 206, cornerRadius: 20)
                    .overlay {
                        Image("WIPlay").renderingMode(.template).resizable().frame(width: 24, height: 24)
                            .foregroundStyle(Color(fixed: 0xFFFFFF))
                            .offset(x: 1)
                            .frame(width: 52, height: 52)
                            .background(Color(fixed: 0x0B1020, opacity: 0.55), in: Circle())
                    }
                    .padding(.top, -46)
                    .onTapGesture(perform: onReview)

                Text(L10n.tr("Your cut is ready"))
                    .font(CS.inter(26, .heavy))
                    .tracking(-0.52)
                    .foregroundStyle(CS.wiInk)
                    .padding(.top, 24)
                Text(CutName.display(cut.title))
                    .font(CS.inter(13, .medium))
                    .foregroundStyle(CS.wiSecondary)
                    .lineLimit(1)
                    .padding(.top, 4)

                if let footage = footageSeconds ?? cut.footageSeconds, footage > 0 {
                    Text(L10n.tr("YOU JUST SAVED"))
                        .font(CS.inter(11, .bold))
                        .tracking(1.1)
                        .foregroundStyle(CS.accent)
                        .padding(.top, 18)
                    Text(Self.clock(TimeSavedLedger.saved(forFootage: footage)))
                        .font(CS.inter(54, .heavy))
                        .tracking(-1.62)
                        .foregroundStyle(CS.wiInk)
                        .accessibilityIdentifier("timeSaved")
                    Text(L10n.tr("of editing time back in your day"))
                        .font(CS.inter(14, .medium))
                        .foregroundStyle(CS.wiSecondary)
                }

                // Figma 1179:43 "Stats": three columns with dividers.
                HStack(spacing: 0) {
                    metric("WIStatFootage", footage, "your footage")
                    Rectangle().fill(CS.wiBorder).frame(width: 1, height: 60)
                    metric("WIStatTrims", keptClips.map { "\($0)" } ?? "–", "clips kept")
                    Rectangle().fill(CS.wiBorder).frame(width: 1, height: 60)
                    metric("WIStatFinal", finalSeconds.map(Self.clock) ?? "–", "your final cut")
                }
                .frame(height: 96)
                .wiCard(shadow: false)
                .padding(.top, 20)

                if let totalSaved, totalSaved > 0 {
                    // Figma 1179:65 "TotalSaved": blue gradient, clock in a light circle, total and the workdays badge.
                    HStack(alignment: .center, spacing: 14) {
                        Image("WIClockWhite").renderingMode(.template).resizable().frame(width: 24, height: 24)
                            .foregroundStyle(Color(fixed: 0xFFFFFF))
                            .frame(width: 48, height: 48)
                            .background(Color(fixed: 0xFFFFFF, opacity: 0.18), in: Circle())
                        VStack(alignment: .leading, spacing: 2) {
                            Text(TimeSavedLedger.hoursMinutes(totalSaved))
                                .font(CS.inter(24, .heavy))
                                .foregroundStyle(Color(fixed: 0xFFFFFF))
                            Text(L10n.tr("Your total editing time saved"))
                                .font(CS.inter(12, .medium))
                                .foregroundStyle(Color(fixed: 0xFFFFFF, opacity: 0.85))
                                .lineLimit(1).minimumScaleFactor(0.8)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 16)
                    .frame(height: 84)
                    .frame(maxWidth: .infinity)
                    .overlay(alignment: .topTrailing) {
                        if let days = TimeSavedLedger.workdays(totalSaved) {
                            Text(L10n.tr("\(days)+ WORKDAY\(days == 1 ? "" : "S")"))
                                .font(CS.inter(10, .bold))
                                .tracking(0.6)
                                .foregroundStyle(Color(fixed: 0xFFFFFF))
                                .padding(.horizontal, 8)
                                .frame(height: 22)
                                .background(Color(fixed: 0xFFFFFF, opacity: 0.2), in: Capsule())
                                .padding(.top, 14).padding(.trailing, 14)
                        }
                    }
                    .background(CS.accentGradient, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .padding(.top, 12)
                    .accessibilityIdentifier("totalTimeSaved")
                }

                HStack(spacing: 10) {
                    Button(action: onReview) {
                        Text(L10n.tr("Review & edit"))
                            .font(CS.inter(15, .bold))
                            .foregroundStyle(CS.wiInk)
                            .lineLimit(1).minimumScaleFactor(0.8)
                            .frame(maxWidth: .infinity)
                            .frame(height: 52)
                            .background(CS.wiPage, in: Capsule())
                            .overlay(Capsule().stroke(CS.wiInk, lineWidth: 1.5))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("readyReviewButton")
                    Button(action: onReview) {
                        Text(L10n.tr("Export video"))
                            .font(CS.inter(15, .bold))
                            .foregroundStyle(Color(fixed: 0xFFFFFF))
                            .lineLimit(1).minimumScaleFactor(0.8)
                            .frame(maxWidth: .infinity)
                            .frame(height: 52)
                            .background(CS.accent, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.top, 22)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 24)
        }
        .task { await loadNumbers() }
    }

    private var footage: String {
        (footageSeconds ?? cut.footageSeconds).map(Self.clock) ?? "–"
    }

    private func metric(_ icon: String, _ value: String, _ label: String) -> some View {
        VStack(spacing: 4) {
            Image(icon).renderingMode(.template).resizable().frame(width: 20, height: 20).foregroundStyle(CS.accent)
            Text(value).font(CS.inter(20, .heavy)).foregroundStyle(CS.wiInk)
            Text(L10n.tr(label)).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary).lineLimit(1).minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
    }

    /// Real numbers from the saved draft: kept clips, final length, and the footage length.
    private func loadNumbers() async {
        guard let session = appState.session else {
            if let footage = cut.footageSeconds { totalSaved = TimeSavedLedger.saved(forFootage: footage) }
            return
        }
        defer {
            if let footage = footageSeconds ?? cut.footageSeconds {
                totalSaved = TimeSavedLedger.record(userID: session.userID, projectID: project.projectID, footageSeconds: footage)
            } else {
                totalSaved = TimeSavedLedger.total(userID: session.userID)
            }
        }
        guard let snapshot: DraftSnapshot = try? await APIClient.shared.request(
            "/v1/projects/\(project.projectID)/draft",
            query: [URLQueryItem(name: "user_id", value: session.userID)]
        ) else { return }
        // The cut is ready: keep its draft (and, on Wi-Fi, its video) so the editor opens at once.
        Task { await EditorPrewarm.prewarm(projectID: project.projectID, session: session, snapshot: snapshot) }
        let selected = snapshot.draft["selected"]?.arrayValue?.compactMap(\.objectValue) ?? []
        keptClips = selected.count
        finalSeconds = selected.reduce(0) { $0 + max(0, ($1["end"]?.doubleValue ?? 0) - ($1["start"]?.doubleValue ?? 0)) }
        let sourceSeconds = snapshot.sources.compactMap { $0.objectValue?["duration_sec"]?.doubleValue }
        if !sourceSeconds.isEmpty { footageSeconds = sourceSeconds.reduce(0, +) }
    }

    static func clock(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
