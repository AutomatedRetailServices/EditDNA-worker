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
    @State private var openEditor = false

    private var cut: CutTracker.Cut? { projectID.flatMap { tracker.cut($0) } }

    var body: some View {
        ZStack {
            CS.base.ignoresSafeArea()
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
        .preferredColorScheme(.dark)
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
    /// "Try again": restarts only the parts in flight.
    var onTryAgain: () -> Void = {}
    let onClose: () -> Void
    let onOpenEdits: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Button(action: onClose) {
                    Image("ControlBack").resizable().frame(width: 36, height: 36)
                }
                .accessibilityLabel("Back")
                .opacity(uploading ? 0.3 : 1)
                .disabled(uploading)
                Spacer()
                Text("Processing")
                    .font(CS.inter(22, .semibold))
                    .foregroundStyle(Color(hex: 0xF2F7FF))
                Spacer()
                Button("Edits", action: onOpenEdits)
                    .font(CS.inter(15, .semibold))
                    .foregroundStyle(CS.cyan)
                    .frame(width: 44, alignment: .trailing)
                    .opacity(uploading ? 0.3 : 1)
                    .disabled(uploading)
            }
            .frame(height: 46)

            VStack(spacing: 14) {
                VideoFirstFrame(image: thumbnail, width: 88, height: 156, cornerRadius: 16)
                Text(fileName)
                    .accessibilityIdentifier("processingCutName")
                    .font(CS.sf(14, .semibold))
                    .foregroundStyle(CS.textPrimaryF)
                    .lineLimit(1)
                    .frame(maxWidth: 240)
            }
            .padding(.top, 4)

            VStack(spacing: 0) {
                Text(failed ? "This cut needs another try" : "Hold on…")
                    .font(CS.sf(failed ? 26 : 32, .semibold))
                    .foregroundStyle(Color(hex: 0xC4D6F7))
                Text(failed ? "Something went wrong on our side." : "We’re saving you hours per day…")
                    .font(CS.sf(18, .semibold))
                    .foregroundStyle(Color(hex: 0xC4D6F7))
            }
            .multilineTextAlignment(.center)
            .padding(.top, 8)

            BrandRing(animating: !failed)
                .frame(width: 150, height: 140)
                .padding(.vertical, 6)

            VStack(spacing: 6) {
                Text(stage.title)
                    .accessibilityIdentifier("processingStage")
                    .font(CS.sf(16, .semibold))
                    .foregroundStyle(Color(hex: 0xF2F7FF))
                Text(stage.percent >= 0 ? "\(stage.percent)%" : "…")
                    .font(CS.sf(24, .semibold))
                    .foregroundStyle(CS.cyan)
                    .contentTransition(.numericText())
                    .animation(.easeOut, value: stage.percent)
                    .accessibilityIdentifier("processingPercent")
                Text("Step \(stage.step) of \(CutTracker.stepCount)")
                    .font(CS.sf(13))
                    .foregroundStyle(CS.textSecondaryF)
                StageDots(current: stage.step, total: CutTracker.stepCount)
                    .padding(.top, 6)
                if let stalled {
                    VStack(spacing: 8) {
                        Text(stalled)
                            .font(CS.sf(13))
                            .foregroundStyle(Color(hex: 0xFFC48A))
                            .multilineTextAlignment(.center)
                            .accessibilityIdentifier("uploadStalled")
                        Button("Try again", action: onTryAgain)
                            .font(CS.sf(15, .semibold))
                            .foregroundStyle(CS.cyan)
                            .accessibilityIdentifier("uploadTryAgain")
                    }
                    .padding(.top, 8)
                    .padding(.horizontal, 24)
                }
            }

            Spacer(minLength: 8)

            VStack(spacing: 12) {
                Text(uploading ? "Your video keeps uploading if you leave CutSell or lock the screen." : "Processing continues if you leave this screen.")
                    .font(CS.sf(13))
                    .foregroundStyle(Color(hex: 0xB8D1F5))
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .background(Color(hex: 0x060E1C), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.cyan.opacity(0.18), lineWidth: 1))
                Button(action: onClose) {
                    Text("Continue in background")
                        .font(CS.sf(15, .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .background(
                            LinearGradient(
                                stops: [.init(color: Color(hex: 0x1969FF), location: 0),
                                        .init(color: Color(hex: 0x7C3AED), location: 0.55),
                                        .init(color: Color(hex: 0x19E6FF), location: 1)],
                                startPoint: .leading, endPoint: .trailing),
                            in: Capsule()
                        )
                        .shadow(color: Color(hex: 0x1969FF, opacity: 0.25), radius: 9, y: 6)
                }
                .buttonStyle(.plain)
                .disabled(uploading)
                .opacity(uploading ? 0.4 : 1)
                .accessibilityIdentifier("continueInBackground")
                Text("We’ll let you know in CutSell when your cut is ready.")
                    .font(CS.sf(12))
                    .foregroundStyle(Color(hex: 0x94ADD6))
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        .padding(.horizontal, 16)
    }
}

/// 9:16 first frame with the neon border (Processing/VideoFirstFrame9x16).
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
                LinearGradient(colors: [Color(hex: 0x2A3F6B), Color(hex: 0x0C1730)], startPoint: .top, endPoint: .bottom)
                Image("ProcessingLogo").resizable().scaledToFit().frame(width: width * 0.6).blendMode(.screen)
            }
        }
        .frame(width: width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous).stroke(Color(hex: 0x00E8FF), lineWidth: 2.5))
        .shadow(color: Color(hex: 0x331FFF, opacity: 0.46), radius: 10, y: 1)
        .shadow(color: Color(hex: 0xFF00BF, opacity: 0.64), radius: 7, x: -4, y: 5)
        .shadow(color: Color(hex: 0x00DBFF, opacity: 0.72), radius: 6, x: 4, y: -2)
    }
}

/// The brand ring around the CutSell bag (Processing/BrandRingMotion): one continuous circle
/// (no gaps) filled with the logo's colours as an angular gradient that closes on itself, turning
/// once every 2.5 s, linear and smooth, with a coloured glow behind it that turns with it.
/// Reduce Motion: the ring stays still and the glow breathes softly. Used on every loading screen
/// that shows the logo.
struct BrandRing: View {
    var animating = true
    /// Ring diameter; the logo and the inner disc scale with it.
    var size: CGFloat = 120
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    static let colors: [Color] = [Color(hex: 0x19E6FF), Color(hex: 0x1769FF), Color(hex: 0x7C3AED),
                                  Color(hex: 0xFF21D7), Color(hex: 0x19E6FF)]
    static let secondsPerTurn = 2.5
    static let lineWidth: CGFloat = 4.5

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
            // Reduce Motion: a soft breathing glow instead of the turn.
            let glowOpacity = (animating && reduceMotion) ? 0.4 + 0.25 * (1 + sin(time * 2 * .pi / 2.4)) / 2 : 0.6
            ZStack {
                // The glow: the same ring behind, blurred, turning with it (light, not a grey shadow).
                ring
                    .rotationEffect(.degrees(angle))
                    .blur(radius: 14 * scale)
                    .opacity(glowOpacity)
                    .blendMode(.plusLighter)
                ring
                    .rotationEffect(.degrees(angle))
                Circle()
                    .fill(Color(hex: 0x050813))
                    .frame(width: size - 26 * scale, height: size - 26 * scale)
                Image("ProcessingLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 112 * scale, height: 112 * scale)
                    .blendMode(.screen)
            }
        }
        .frame(width: size + 30 * scale, height: size + 30 * scale)
        .accessibilityHidden(true)
    }

    private var ring: some View {
        Circle()
            .stroke(AngularGradient(colors: Self.colors, center: .center), lineWidth: Self.lineWidth)
            .frame(width: size, height: size)
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
        .accessibilityLabel("Step \(current) of \(total)")
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

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                ZStack {
                    BrandLogo()
                    HStack {
                        Button(action: onBack) {
                            Image("ControlBack").resizable().frame(width: 36, height: 36)
                                .frame(width: 44, height: 44)
                        }
                        .accessibilityLabel("Back")
                        Spacer()
                    }
                }
                .padding(.top, 4)

                ZStack {
                    RoundedRectangle(cornerRadius: 25)
                        .fill(AngularGradient(colors: [Color(hex: 0x00EDFF), Color(hex: 0x1445FF), Color(hex: 0xB800FF), Color(hex: 0xFF0D75), Color(hex: 0x00EDFF)], center: .center))
                        .frame(width: 134, height: 220)
                        .blur(radius: 10)
                        .opacity(0.9)
                    VideoFirstFrame(image: cut.thumbnail, width: 112, height: 199, cornerRadius: 20)
                        .overlay {
                            Circle().fill(Color.black.opacity(0.45)).frame(width: 42, height: 42)
                                .overlay(Image(systemName: "play.fill").font(.system(size: 18)).foregroundStyle(.white))
                        }
                }
                .padding(.top, 8)
                .onTapGesture(perform: onReview)

                Text("Your cut is ready")
                    .font(CS.sf(28, .semibold))
                    .foregroundStyle(Color(hex: 0xFAFCFF))
                    .padding(.top, 18)
                Text(CutName.display(cut.title))
                    .font(CS.sf(13))
                    .foregroundStyle(Color(hex: 0x8594B2))
                    .lineLimit(1)
                    .padding(.top, 4)

                if let footage = footageSeconds ?? cut.footageSeconds, footage > 0 {
                    Text("YOU JUST SAVED")
                        .font(CS.sf(11, .semibold))
                        .tracking(1.4)
                        .foregroundStyle(Color(hex: 0x4DDBFF))
                        .padding(.top, 14)
                    Text(Self.clock(TimeSavedLedger.saved(forFootage: footage)))
                        .font(CS.sf(54, .semibold))
                        .tracking(-1.8)
                        .foregroundStyle(.white)
                        .accessibilityIdentifier("timeSaved")
                    Text("of editing time back in your day")
                        .font(CS.sf(14))
                        .foregroundStyle(Color(hex: 0xB2BDD1))
                }

                HStack(spacing: 0) {
                    metric("ReadyFootage", footage, "your footage")
                    Rectangle().fill(Color(hex: 0x384766, opacity: 0.7)).frame(width: 1, height: 56)
                    metric("ReadyTrims", keptClips.map { "\($0)" } ?? "–", "clips kept")
                    Rectangle().fill(Color(hex: 0x384766, opacity: 0.7)).frame(width: 1, height: 56)
                    metric("ReadyFinalCut", finalSeconds.map(Self.clock) ?? "–", "your final cut")
                }
                .frame(height: 88)
                .background(Color(hex: 0x0E111A, opacity: 0.94), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Color(hex: 0x33456B, opacity: 0.65), lineWidth: 1))
                .padding(.top, 20)

                if let totalSaved, totalSaved > 0 {
                    HStack(alignment: .top, spacing: 16) {
                        Image(systemName: "clock")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundStyle(Color(hex: 0x4DEBFF))
                            .frame(width: 46, height: 46)
                            .background(Circle().fill(Color(hex: 0x0A1A33)))
                            .overlay(Circle().stroke(Color(hex: 0x4DEBFF).opacity(0.6), lineWidth: 1.5))
                        VStack(alignment: .leading, spacing: 4) {
                            Text(TimeSavedLedger.hoursMinutes(totalSaved))
                                .font(CS.sf(28, .semibold))
                                .foregroundStyle(.white)
                            Text("Your total editing time saved")
                                .font(CS.sf(12))
                                .foregroundStyle(Color(hex: 0xADB8D1))
                        }
                        Spacer(minLength: 0)
                        if let days = TimeSavedLedger.workdays(totalSaved) {
                            Text("\(days)+ WORKDAY\(days == 1 ? "" : "S")")
                                .font(CS.sf(10, .semibold))
                                .tracking(1.1)
                                .foregroundStyle(Color(hex: 0x73D9FF))
                                .padding(.top, 4)
                        }
                    }
                    .padding(17)
                    .background(
                        LinearGradient(stops: [.init(color: Color(hex: 0x0A172B), location: 0),
                                               .init(color: Color(hex: 0x120E29), location: 0.55),
                                               .init(color: Color(hex: 0x1F0929), location: 1)],
                                       startPoint: .leading, endPoint: .trailing),
                        in: RoundedRectangle(cornerRadius: 20, style: .continuous)
                    )
                    .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Color(hex: 0x6B38C7, opacity: 0.72), lineWidth: 1))
                    .padding(.top, 14)
                    .accessibilityIdentifier("totalTimeSaved")
                }

                HStack(spacing: 10) {
                    Button(action: onReview) {
                        Text("Review & edit")
                            .font(CS.inter(16, .semibold))
                            .foregroundStyle(Color(hex: 0xFAFCFF))
                            .frame(maxWidth: .infinity)
                            .frame(height: 54)
                            .background(Color(hex: 0x02060F), in: Capsule())
                            .overlay(Capsule().stroke(Color(hex: 0x265EA3), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("readyReviewButton")
                    Button(action: onReview) {
                        Text("Export video")
                            .font(CS.sf(16, .semibold))
                            .foregroundStyle(Color(hex: 0x03050D))
                            .frame(maxWidth: .infinity)
                            .frame(height: 54)
                            .background(LinearGradient(
                                stops: [.init(color: Color(hex: 0x14E0FF), location: 0),
                                        .init(color: Color(hex: 0x1F70FF), location: 0.48),
                                        .init(color: Color(hex: 0xFF14D9), location: 1)],
                                startPoint: .leading, endPoint: .trailing), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.top, 28)
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
            Image(icon).resizable().frame(width: 20, height: 20)
            Text(value).font(CS.sf(19, .semibold)).foregroundStyle(.white)
            Text(label).font(CS.sf(11)).foregroundStyle(Color(hex: 0x8F9CB8))
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
