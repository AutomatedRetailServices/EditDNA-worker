import AVFoundation
import CoreImage
import SwiftUI

/// Clon (beta, PO 29 Oct; was "Split"): two "yous" in the same room from one fixed shot, joined on the iPhone
/// with the SILHOUETTE method (part 2 whole + the cut-out person of part 1; halves as the fallback). The cameras
/// are full screen: a dotted guide in the middle, part 1 as a 35 % ghost while part 2 is recorded.
/// "Use this video" hands the joined video to the same flow as any Record video (the engine is unchanged).
struct SplitCloneView: View {
    enum Step: Equatable { case setup, part1, part1Ready, part2, joining, result }

    let onClose: () -> Void
    let onUse: (LocalVideo) -> Void
    /// Design screenshots / probes: show this step with sample content, no camera.
    var previewStep: Step? = nil

    @StateObject private var camera = SplitCaptureController()
    @ObservedObject private var prompter = TeleprompterStore.shared
    @State private var step: Step = .setup
    @State private var part1: URL?
    @State private var part1Seconds: Double = 0
    @State private var part1First: CIImage?
    @State private var part2: URL?
    @State private var elapsed: TimeInterval = 0
    @State private var timer: Timer?
    @State private var starting = false
    @State private var editingScript = false
    @State private var countingDown = false
    @State private var problem: String?

    // Part 2
    @State private var ghost = AVPlayer()
    @State private var offset: SplitOffset = .zero
    @State private var aligned: Bool?
    @State private var headphones = SplitCloneView.headphonesConnected()
    @State private var alignTask: Task<Void, Never>?

    // Joining / result
    @State private var joinProgress: Double = 0
    @State private var joined: URL?
    @State private var usedBlend: SplitBlend = .silhouette
    @State private var fallbackNotice: String?
    @State private var joinTask: Task<Void, Never>?
    @State private var review = AVPlayer()

    private var isPreview: Bool { previewStep != nil }

    var body: some View {
        ZStack {
            switch step {
            case .setup: setupScreen
            case .part1, .part2: cameraScreen
            case .part1Ready: part1ReadyScreen
            case .joining: joiningScreen
            case .result: resultScreen
            }
        }
        .onAppear { if let previewStep { step = previewStep; applyPreviewState() } }
        .onDisappear { leave() }
        .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.routeChangeNotification)) { _ in
            headphones = Self.headphonesConnected()
            if !camera.isRecording { ghost.isMuted = !headphones }
        }
        .sheet(isPresented: $editingScript) {
            TeleprompterSheet(onRecord: { toggleRecording() })
        }
        .alert(L10n.tr("Something went wrong"), isPresented: Binding(get: { problem != nil }, set: { if !$0 { problem = nil } })) {
            Button(L10n.tr("OK"), role: .cancel) { problem = nil }
        } message: {
            Text(L10n.tr(problem ?? ""))
        }
    }

    // MARK: 1 · Clon start screen (white)

    private var setupScreen: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: onClose) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left").font(.system(size: 14, weight: .semibold))
                    Text(L10n.tr("Record")).font(CS.inter(15, .semibold))
                }
                .foregroundStyle(CS.accent)
                .frame(height: 44)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("splitBack")
            .padding(.top, 6)
            HStack(spacing: 10) {
                Text(L10n.tr("Clon")).font(CS.inter(28, .bold)).foregroundStyle(CS.wiInk)
                CloneBetaTag(size: 12)
            }
            .padding(.top, 4)
            Text(L10n.tr("Record yourself twice. We put both of you in the same room."))
                .font(CS.inter(15, .medium)).foregroundStyle(CS.wiSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 6)
            tipsCard.padding(.top, 24)
            Spacer(minLength: 16)
            Button { start() } label: {
                Text(L10n.tr("Start part 1")).font(CS.inter(16, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(maxWidth: .infinity).frame(height: 52)
                    .background(CS.accent, in: Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("splitStart")
            .padding(.bottom, 16)
        }
        .padding(.horizontal, 20)
        .background(CS.wiPage.ignoresSafeArea())
        .preferredColorScheme(.light)
    }

    private static let tips = [
        "Put your phone on a stand — keep it still",
        "Part 1: stand on the LEFT · Part 2: stand on the RIGHT",
        "Stay on your side — don't walk in front of yourself",
        "When you finish talking, stay and react",
        "Use headphones for part 2",
    ]

    private var tipsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(L10n.tr("Tips")).font(CS.inter(14, .bold)).foregroundStyle(CS.wiInk)
            ForEach(Self.tips, id: \.self) { tip in
                HStack(alignment: .top, spacing: 8) {
                    Circle().fill(CS.accent).frame(width: 5, height: 5).padding(.top, 6)
                    Text(L10n.tr(tip)).font(CS.inter(13, .medium)).foregroundStyle(CS.wiSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(fixed: 0xFFFFFF), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("splitTips")
    }

    // MARK: 2 + 4 · Cameras (full screen, dark)

    private var cameraScreen: some View {
        let isPart2 = step == .part2
        return ZStack {
            Color.black.ignoresSafeArea()
            if !isPreview {
                CameraPreview(session: camera.session, mirrored: camera.mirrorOn, position: camera.cameraPosition)
                    .ignoresSafeArea()
            } else {
                PreviewRoom(side: isPart2 ? .right : .left).ignoresSafeArea()
            }
            if isPart2 {
                Group {
                    if isPreview { PreviewRoom(side: .left).ignoresSafeArea() } else { PlayerLayerView(player: ghost, fill: true).ignoresSafeArea() }
                }
                .opacity(0.35)
                .allowsHitTesting(false)
                .accessibilityIdentifier("splitGhost")
            }
            MiddleGuide().padding(.top, 64).ignoresSafeArea(edges: .bottom).allowsHitTesting(false)

            VStack(spacing: 0) {
                topBar(title: isPart2 ? "Part 2 of 2" : "Part 1 of 2", dark: true, enabled: !camera.isRecording) {
                    if isPart2 { backToPart1Ready() } else { onClose() }
                }
                if camera.isRecording || (isPreview && step == .part1) {
                    recBadge(part: isPart2 ? 2 : 1).padding(.top, 8)
                }
                // The label sits in the middle of its own half.
                HStack(spacing: 0) {
                    if isPart2 { Color.clear.frame(maxWidth: .infinity, maxHeight: 1) }
                    sideLabel(isPart2 ? "Now stand on the RIGHT" : "Stand on the LEFT")
                        .frame(maxWidth: .infinity)
                    if !isPart2 { Color.clear.frame(maxWidth: .infinity, maxHeight: 1) }
                }
                .padding(.top, 14)
                if isPart2 && !camera.isRecording {
                    VStack(spacing: 8) {
                        alignmentChip
                        if !headphones {
                            notice("Use headphones to hear part 1", icon: "headphones")
                                .accessibilityIdentifier("splitHeadphones")
                        }
                    }
                    .padding(.top, 12)
                }
                Spacer()
                ZStack {
                    recordButton
                    HStack {
                        Spacer()
                        rail
                    }
                }
                .padding(.bottom, 26)
            }
            if prompter.isOn && prompter.hasScript {
                TeleprompterOverlay(running: camera.isRecording, onEdit: { if !camera.isRecording { editingScript = true } })
                    .transition(.opacity)
            }
            if countingDown {
                ScriptCountdown { countingDown = false; beginRecording() }
            }
        }
        .preferredColorScheme(.dark)
    }

    private func topBar(title: String, dark: Bool, enabled: Bool, close: @escaping () -> Void) -> some View {
        ZStack {
            Text(L10n.tr(title)).font(CS.inter(15, .semibold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                .accessibilityIdentifier("splitStepTitle")
            HStack {
                Button(action: close) {
                    Image(systemName: "xmark").font(.system(size: 15, weight: .semibold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                        .frame(width: 40, height: 40)
                        .background(Color(fixed: 0xFFFFFF).opacity(0.16), in: Circle())
                }
                .buttonStyle(.plain)
                .disabled(!enabled)
                .opacity(enabled ? 1 : 0.4)
                .accessibilityLabel(L10n.tr("Close"))
                .accessibilityIdentifier("splitClose")
                Spacer()
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 6)
    }

    private func recBadge(part: Int) -> some View {
        HStack(spacing: 6) {
            Circle().fill(Color(fixed: 0xFFFFFF)).frame(width: 8, height: 8)
            Text(L10n.tr("REC \(Self.clock(isPreview ? 6 : elapsed)) · Part \(part)"))
                .font(CS.inter(13, .bold)).monospacedDigit()
        }
        .foregroundStyle(Color(fixed: 0xFFFFFF))
        .padding(.horizontal, 12).frame(height: 30)
        .background(Color(fixed: 0xFF3B5C), in: Capsule())
        .accessibilityIdentifier("splitRecBadge")
    }

    private func sideLabel(_ text: String) -> some View {
        Text(L10n.tr(text)).font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
            .padding(.horizontal, 14).frame(height: 32)
            .background(Color.black.opacity(0.5), in: Capsule())
            .overlay(Capsule().stroke(Color(fixed: 0xFFFFFF).opacity(0.25), lineWidth: 1))
            .accessibilityIdentifier("splitSideLabel")
    }

    @ViewBuilder
    private var alignmentChip: some View {
        switch aligned {
        case .some(true):
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(Color(fixed: 0x33D98C))
                Text(L10n.tr("Lined up")).foregroundStyle(Color(fixed: 0xFFFFFF))
            }
            .font(CS.inter(13, .bold))
            .padding(.horizontal, 12).frame(height: 30)
            .background(Color.black.opacity(0.55), in: Capsule())
            .accessibilityIdentifier("splitAligned")
        case .some(false):
            notice("Your phone moved — line it up with the ghost", icon: "exclamationmark.triangle.fill", warning: true)
                .accessibilityIdentifier("splitMoved")
        case .none:
            notice("Checking the position…", icon: "viewfinder")
                .accessibilityIdentifier("splitChecking")
        }
    }

    private func notice(_ text: String, icon: String, warning: Bool = false) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon).foregroundStyle(warning ? Color(fixed: 0xFFB020) : Color(fixed: 0xFFFFFF))
            Text(L10n.tr(text)).foregroundStyle(Color(fixed: 0xFFFFFF)).lineLimit(2).multilineTextAlignment(.leading)
        }
        .font(CS.inter(13, .semibold))
        .padding(.horizontal, 12).padding(.vertical, 7)
        .background(Color.black.opacity(0.6), in: Capsule())
        .padding(.horizontal, 24)
    }

    /// The same record button as Record.
    private var recordButton: some View {
        Button { toggleRecording() } label: {
            ZStack {
                Circle().fill(Color(hex: 0x45474F)).frame(width: 104, height: 104)
                Circle().fill(CS.accentGradient).frame(width: 90, height: 90)
                    .shadow(color: Color(hex: 0x1769FF, opacity: 0.45), radius: 9)
                Circle().fill(CS.background).frame(width: 78, height: 78)
                if camera.isRecording || (isPreview && step == .part1) {
                    RoundedRectangle(cornerRadius: 8).fill(Color(hex: 0xFF2E5E)).frame(width: 32, height: 32)
                } else {
                    Circle().fill(.white).frame(width: 64, height: 64)
                }
            }
            .frame(width: 116, height: 116)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled((!camera.isConfigured && !isPreview) || starting)
        .accessibilityLabel(camera.isRecording ? "Stop recording" : "Record")
        .accessibilityIdentifier("splitRecord")
    }

    /// Flip (part 1, before recording) and the Record teleprompter.
    private var rail: some View {
        VStack(spacing: 18) {
            if step == .part1 && !camera.isRecording && !camera.isLocked {
                railButton(title: "Flip", system: "arrow.triangle.2.circlepath.camera") { Task { await camera.flipCamera() } }
                    .accessibilityIdentifier("splitFlip")
            }
            let on = prompter.isOn && prompter.hasScript
            railButton(title: "Script", system: "text.alignleft", highlighted: on) {
                if prompter.hasScript { prompter.isOn.toggle() } else { editingScript = true }
            }
            .simultaneousGesture(LongPressGesture(minimumDuration: 0.5).onEnded { _ in editingScript = true })
            .accessibilityIdentifier("splitScript")
        }
        .padding(.trailing, 14)
    }

    private func railButton(title: String, system: String, highlighted: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: system).font(.system(size: 20, weight: .semibold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(width: 44, height: 44)
                    .background(highlighted ? AnyShapeStyle(CS.accent) : AnyShapeStyle(Color.black.opacity(0.5)), in: Circle())
                Text(L10n.tr(title)).font(CS.inter(12, .semibold)).foregroundStyle(Color(fixed: 0xFFFFFF))
            }
            .frame(width: 62)
        }
        .buttonStyle(.plain)
    }

    // MARK: 3 · Part 1 ready (dark)

    private var part1ReadyScreen: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if isPreview { PreviewRoom(side: .left).ignoresSafeArea() } else { PlayerLayerView(player: review, fill: true).ignoresSafeArea() }
            MiddleGuide().padding(.top, 64).ignoresSafeArea(edges: .bottom).allowsHitTesting(false)
            VStack(spacing: 0) {
                topBar(title: "Part 1 of 2", dark: true, enabled: true) { onClose() }
                HStack {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark").font(.system(size: 11, weight: .bold))
                        Text(L10n.tr("Part 1 · \(Self.clock(isPreview ? 8 : part1Seconds))")).font(CS.inter(12, .bold)).monospacedDigit()
                    }
                    .foregroundStyle(Color(fixed: 0xFFFFFF))
                    .padding(.horizontal, 10).frame(height: 26)
                    .background(Color(fixed: 0x1FA971), in: Capsule())
                    .accessibilityIdentifier("splitPart1Badge")
                    Spacer()
                }
                .padding(.horizontal, 16).padding(.top, 14)
                Spacer()
                VStack(spacing: 6) {
                    Text(L10n.tr("Now record part 2")).font(CS.inter(18, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                    Text(L10n.tr("Part 1 plays as a ghost while you record. Stand on the RIGHT."))
                        .font(CS.inter(13, .medium)).foregroundStyle(Color(fixed: 0xFFFFFF).opacity(0.75))
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal, 28).padding(.vertical, 14)
                .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                .padding(.bottom, 20)
                HStack(spacing: 20) {
                    Button { retakePart1() } label: {
                        Text(L10n.tr("Retake part 1")).font(CS.inter(16, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                            .frame(maxWidth: .infinity).frame(height: 52)
                            .background(Color(fixed: 0x1C2233), in: Capsule())
                            .overlay(Capsule().stroke(Color(fixed: 0xFFFFFF).opacity(0.85), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("splitRetake")
                    Button { goToPart2() } label: {
                        Text(L10n.tr("Record part 2")).font(CS.inter(16, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                            .frame(maxWidth: .infinity).frame(height: 52)
                            .background(CS.accent, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("splitRecordPart2")
                }
                .padding(.horizontal, 20).padding(.bottom, 24)
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: 5 · Creating the clone + result (white)

    private var joiningScreen: some View {
        VStack(spacing: 18) {
            Spacer()
            BrandRing(size: 110)
            Text(L10n.tr("Creating your clone…")).font(CS.inter(20, .bold)).foregroundStyle(CS.wiInk)
            Text(verbatim: "\(Int((joinProgress * 100).rounded()))%").font(CS.inter(15, .semibold)).foregroundStyle(CS.wiSecondary)
                .monospacedDigit()
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .background(CS.wiPage.ignoresSafeArea())
        .preferredColorScheme(.light)
        .accessibilityIdentifier("splitJoining")
    }

    private var resultScreen: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text(L10n.tr("Your clone")).font(CS.inter(24, .bold)).foregroundStyle(CS.wiInk)
                CloneBetaTag(size: 11)
            }
            .padding(.top, 16)
            Group {
                if isPreview {
                    PreviewRoom(side: .both)
                } else {
                    PlayerLayerView(player: review, fill: true)
                }
            }
            .frame(width: 250, height: 444)
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
            .frame(maxWidth: .infinity)
            .padding(.top, 14)
            .accessibilityIdentifier("splitResultVideo")
            if let fallbackNotice {
                Text(L10n.tr(fallbackNotice)).font(CS.inter(13, .medium)).foregroundStyle(CS.wiSecondary)
                    .frame(maxWidth: .infinity).multilineTextAlignment(.center).padding(.top, 12)
                    .accessibilityIdentifier("splitFallbackNotice")
            }
            Spacer(minLength: 12)
            HStack(spacing: 12) {
                Button { backToPart1Ready() } label: {
                    Text(L10n.tr("Retake")).font(CS.inter(16, .bold)).foregroundStyle(CS.wiInk)
                        .frame(maxWidth: .infinity).frame(height: 52)
                        .background(Color(fixed: 0xFFFFFF), in: Capsule())
                        .overlay(Capsule().stroke(CS.wiBorder, lineWidth: 1.5))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("splitResultRetake")
                Button { useVideo() } label: {
                    Text(L10n.tr("Use this video")).font(CS.inter(16, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                        .frame(maxWidth: .infinity).frame(height: 52)
                        .background(CS.accent, in: Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("splitUse")
            }
            .padding(.bottom, 16)
        }
        .padding(.horizontal, 20)
        .background(CS.wiPage.ignoresSafeArea())
        .preferredColorScheme(.light)
    }

    // MARK: Actions

    private func start() {
        step = .part1
        guard !isPreview else { return }
        Task { await camera.prepare() }
    }

    private func toggleRecording() {
        guard !isPreview else { return }
        if camera.isRecording { Task { await finishPart() }; return }
        if countingDown || starting { return }
        if prompter.isOn && prompter.hasScript && prompter.countdown { countingDown = true; return }
        beginRecording()
    }

    private func beginRecording() {
        guard !camera.isRecording, !starting else { return }
        starting = true
        if step == .part1 {
            camera.lockCamera()
            do { try camera.startRecording() } catch { problem = "The recording could not start." }
            starting = false
            if camera.isRecording { startTimer() }
            return
        }
        // Part 2: part 1 starts playing at the very instant the recording starts.
        alignTask?.cancel()
        Task {
            ghost.isMuted = !headphones
            ghost.automaticallyWaitsToMinimizeStalling = false
            await ghost.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
            for _ in 0..<40 where ghost.currentItem?.status != .readyToPlay { try? await Task.sleep(for: .milliseconds(50)) }
            _ = await ghost.preroll(atRate: 1)
            let at = SplitCaptureController.hostTime(after: 0.25)
            do {
                try camera.startRecording(startHostTime: at)
                ghost.setRate(1, time: .zero, atHostTime: at)
                startTimer()
            } catch {
                problem = "The recording could not start."
            }
            starting = false
        }
    }

    private func finishPart() async {
        stopTimer()
        do {
            let url = try await camera.stopRecording()
            if step == .part1 {
                part1 = url
                part1Seconds = (try? await AVURLAsset(url: url).load(.duration).seconds) ?? elapsed
                part1First = await SplitAligner.firstFrame(of: url)
                review.replaceCurrentItem(with: AVPlayerItem(url: url))
                review.isMuted = false
                loop(review)
                review.play()
                step = .part1Ready
            } else {
                ghost.pause()
                part2 = url
                await join()
            }
        } catch {
            problem = "The recording could not be saved."
        }
    }

    private func retakePart1() {
        review.pause()
        if let part1 { try? FileManager.default.removeItem(at: part1) }
        part1 = nil; part1First = nil
        camera.unlockCamera()
        elapsed = 0
        step = .part1
    }

    private func goToPart2() {
        review.pause()
        guard let part1 else { return }
        ghost.replaceCurrentItem(with: AVPlayerItem(url: part1))
        ghost.isMuted = !headphones
        elapsed = 0
        aligned = nil
        step = .part2
        startAlignmentCheck()
    }

    private func backToPart1Ready() {
        joinTask?.cancel(); joinTask = nil
        alignTask?.cancel()
        ghost.pause(); review.pause()
        for url in [part2, joined].compactMap({ $0 }) { try? FileManager.default.removeItem(at: url) }
        part2 = nil; joined = nil; fallbackNotice = nil
        if let part1 { review.replaceCurrentItem(with: AVPlayerItem(url: part1)); loop(review); review.play() }
        step = .part1Ready
    }

    /// Before part 2: compare the camera with part 1's first frame twice a second.
    private func startAlignmentCheck() {
        alignTask?.cancel()
        guard let reference = part1First else { return }
        alignTask = Task {
            while !Task.isCancelled {
                if !camera.isRecording, let live = await camera.latestFrame() {
                    let measured = await Task.detached(priority: .userInitiated) {
                        SplitAligner.offset(part1: reference, live: live)
                    }.value
                    if let measured, !Task.isCancelled {
                        offset = measured
                        aligned = measured.magnitude <= 0.02
                    }
                }
                try? await Task.sleep(for: .milliseconds(500))
            }
        }
    }

    private func join() async {
        guard let part1, let part2 else { return }
        step = .joining
        joinProgress = 0
        fallbackNotice = nil
        let started = Date()
        // The shift between the two recorded parts (more exact than the live check); the live one otherwise.
        let measured = await SplitCompositor.measuredOffset(part1: part1, part2: part2) ?? offset
        let gains = await SplitCompositor.colourGains(part1: part1, part2: part2)
        var blend: SplitBlend = .silhouette
        var url: URL?
        var reason = ""
        if measured.magnitude <= SplitCompositor.tooFarForSilhouette {
            let stats = SplitStats()
            let work = Task {
                try await SplitCompositor.join(part1: part1, part2: part2, offset: measured, blend: .silhouette,
                                               gains: gains, stats: stats) { value in
                    Task { @MainActor in joinProgress = value * 0.97 }
                }
            }
            joinTask = Task { _ = await work.result }
            // A 30 s clone should take well under this; past it, the simpler blend.
            let watchdog = Task { try await Task.sleep(for: .seconds(240)); work.cancel() }
            let result = await work.result
            watchdog.cancel()
            switch result {
            case .success(let made): url = made
            case .failure(let error): reason = String(describing: error).prefix(160).description
            }
            reason += " " + stats.summary
        } else {
            reason = String(format: "moved %.3f", measured.magnitude)
        }
        if url == nil {
            if Task.isCancelled || step != .joining { return }
            blend = .halves
            joinProgress = 0
            do {
                url = try await SplitCompositor.join(part1: part1, part2: part2, offset: measured, blend: .halves, gains: gains) { value in
                    Task { @MainActor in joinProgress = value }
                }
                fallbackNotice = "We used a simpler blend for this video"
            } catch {
                problem = "The video could not be joined. Try recording part 2 again."
                step = .part1Ready
                return
            }
        }
        guard let url, step == .joining else { return }
        let seconds = Date().timeIntervalSince(started)
        let length = (try? await AVURLAsset(url: url).load(.duration).seconds) ?? 0
        CutSellDiagnostics.log("clone_joined", ["method": blend.rawValue, "seconds": String(format: "%.1f", seconds),
                                                "video_s": String(format: "%.1f", length),
                                                "shift": String(format: "%.3f", measured.magnitude), "note": reason])
        joined = url
        usedBlend = blend
        review.replaceCurrentItem(with: AVPlayerItem(url: url))
        review.isMuted = false
        loop(review)
        review.play()
        step = .result
    }

    private func useVideo() {
        guard let chosen = joined else { return }
        review.pause()
        // Only the joined video stays on the iPhone.
        for url in [part1, part2].compactMap({ $0 }) where url != chosen {
            try? FileManager.default.removeItem(at: url)
        }
        Task {
            let seconds = (try? await AVURLAsset(url: chosen).load(.duration).seconds) ?? 0
            CutSellDiagnostics.log("clone_used", ["method": usedBlend.rawValue, "seconds": String(format: "%.1f", seconds)])
            onUse(LocalVideo(url: chosen, name: "Clon video", duration: max(0, seconds), contentType: "video/quicktime"))
        }
    }

    private func leave() {
        stopTimer()
        alignTask?.cancel()
        joinTask?.cancel()
        ghost.pause(); review.pause()
        guard !isPreview else { return }
        camera.unlockCamera()
        camera.stopSession()
    }

    private func loop(_ player: AVPlayer) {
        NotificationCenter.default.removeObserver(player)
        guard let item = player.currentItem else { return }
        NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main) { [weak player] _ in
            player?.seek(to: .zero)
            player?.play()
        }
    }

    private func startTimer() {
        timer?.invalidate()
        elapsed = 0
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            Task { @MainActor in elapsed += 0.1 }
        }
    }

    private func stopTimer() { timer?.invalidate(); timer = nil }

    private func applyPreviewState() {
        switch previewStep {
        case .part2: aligned = false; headphones = false
        case .joining: joinProgress = 0.42
        default: break
        }
    }

    static func clock(_ seconds: TimeInterval) -> String {
        let whole = max(0, Int(seconds.rounded(.down)))
        return "\(whole / 60):" + String(format: "%02d", whole % 60)
    }

    /// Wired or Bluetooth headphones (part 1's sound goes there; otherwise it stays silent so the
    /// microphone doesn't record it again).
    static func headphonesConnected() -> Bool {
        let kinds: [AVAudioSession.Port] = [.headphones, .bluetoothA2DP, .bluetoothLE, .bluetoothHFP, .usbAudio]
        return AVAudioSession.sharedInstance().currentRoute.outputs.contains { kinds.contains($0.portType) }
    }
}

/// The dotted guide in the middle of the camera (white, 40 %).
private struct MiddleGuide: View {
    var body: some View {
        GeometryReader { geo in
            Path { path in
                path.move(to: CGPoint(x: geo.size.width / 2, y: 0))
                path.addLine(to: CGPoint(x: geo.size.width / 2, y: geo.size.height))
            }
            .stroke(Color(fixed: 0xFFFFFF).opacity(0.4), style: StrokeStyle(lineWidth: 2, dash: [8, 8]))
        }
        .accessibilityIdentifier("splitMiddleGuide")
    }
}

/// Screenshots only: a stand-in for the camera (a room with a person on one side, or both for the result).
private struct PreviewRoom: View {
    enum Side { case left, right, both }
    let side: Side

    var body: some View {
        GeometryReader { geo in
            ZStack {
                LinearGradient(colors: [Color(fixed: 0x3A4256), Color(fixed: 0x161B29)], startPoint: .top, endPoint: .bottom)
                if side != .right { person(geo, x: 0.27, color: 0xC9B6A6) }
                if side != .left { person(geo, x: 0.73, color: 0xA6B8C9) }
            }
        }
    }

    private func person(_ geo: GeometryProxy, x: CGFloat, color: UInt32) -> some View {
        Image(systemName: "person.fill")
            .resizable().scaledToFit()
            .frame(width: geo.size.width * 0.32)
            .foregroundStyle(Color(fixed: color))
            .position(x: geo.size.width * x, y: geo.size.height * 0.58)
    }
}

/// The small "BETA" tag next to Clon: #1769FF, white text, rounded corners.
struct CloneBetaTag: View {
    var size: CGFloat = 10
    var body: some View {
        Text(verbatim: "BETA")
            .font(CS.inter(size, .heavy))
            .tracking(0.4)
            .foregroundStyle(Color(fixed: 0xFFFFFF))
            .padding(.horizontal, size * 0.55).padding(.vertical, size * 0.22)
            .background(Color(fixed: 0x1769FF), in: RoundedRectangle(cornerRadius: size * 0.45, style: .continuous))
            .accessibilityLabel(L10n.tr("Beta"))
    }
}
