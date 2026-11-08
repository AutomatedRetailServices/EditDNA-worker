@preconcurrency import ARKit
@preconcurrency import AVFoundation
import AVKit
import RealityKit
import SwiftUI

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

private enum CloneColor {
    static let blue = Color(fixed: 0x1769FF)
    static let ink = Color(fixed: 0x0B1020)
    static let gray = Color(fixed: 0x6B7385)
    static let line = Color(fixed: 0xE6E9EF)
    static let red = Color(fixed: 0xE5484D)
    static let white = Color(fixed: 0xFFFFFF)
}

private struct CloneButton: View {
    let title: String
    var primary = false
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Text(L10n.tr(title))
                .font(CS.inter(16, .bold))
                .foregroundStyle(primary ? CloneColor.white : CloneColor.ink)
                .frame(maxWidth: .infinity).frame(height: 52)
                .background(primary ? CloneColor.blue : CloneColor.white,
                            in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(primary ? Color.clear : CloneColor.line, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}

// MARK: Start (A2)

struct CloneStartView: View {
    let onClose: () -> Void
    let onStart: () -> Void

    static let tips = ["Move your phone slowly — small moves are OK", "Stand in a different spot each time",
                       "Don't walk around your clones", "When you finish talking, stay and react",
                       "Use headphones to hear your clones"]

    var body: some View {
        ZStack {
            CloneColor.white.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Button(action: onClose) {
                        Image(systemName: "xmark").font(.system(size: 18, weight: .bold)).foregroundStyle(CloneColor.ink)
                            .frame(width: 44, height: 44)
                    }
                    .accessibilityLabel(L10n.tr("Close"))
                    Spacer()
                }
                HStack(spacing: 10) {
                    Text(L10n.tr("Clon")).font(CS.jakarta(32, .heavy)).foregroundStyle(CloneColor.ink)
                    CloneBetaTag(size: 12)
                }
                .padding(.top, 12)
                Text(L10n.tr("Record yourself up to 4 times. All of you, in the same room."))
                    .font(CS.jakarta(15, .medium)).foregroundStyle(CloneColor.gray)
                    .padding(.top, 8)
                VStack(alignment: .leading, spacing: 14) {
                    ForEach(Self.tips, id: \.self) { tip in
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(CloneColor.blue)
                            Text(L10n.tr(tip)).font(CS.jakarta(15, .medium)).foregroundStyle(CloneColor.ink)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .padding(18)
                .background(CloneColor.white, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(CloneColor.line, lineWidth: 1))
                .padding(.top, 24)
                Spacer()
                CloneButton(title: "Start clone 1", primary: true, action: onStart)
                    .accessibilityIdentifier("cloneStart")
                    .padding(.bottom, 12)
            }
            .padding(.horizontal, 20)
        }
        .preferredColorScheme(.light)
    }
}

// MARK: AR model

@MainActor
final class CloneARModel: ObservableObject {
    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    let engine = CloneCaptureEngine()
    let clones = CloneLiveClones()
    @Published var tracking: CloneCaptureEngine.Tracking = .scanning
    @Published var recording = false
    @Published var elapsed: Double = 0
    @Published var headphones = CloneSupport.headphonesConnected
    private var started = false
    private var routeObserver: NSObjectProtocol?
    private var lockedCamera = false

    func start() {
        guard !started else { return }
        started = true
        try? AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .videoRecording,
                                                         options: [.allowBluetoothA2DP, .defaultToSpeaker])
        try? AVAudioSession.sharedInstance().setActive(true)
        engine.onTracking = { [weak self] t in self?.tracking = t }
        engine.onElapsed = { [weak self] s in self?.elapsed = s }
        engine.session = arView.session
        arView.session.delegateQueue = engine.queue
        arView.session.delegate = engine
        arView.session.run(CloneSupport.configuration())
        clones.attach(to: arView)
        routeObserver = NotificationCenter.default.addObserver(forName: AVAudioSession.routeChangeNotification,
                                                               object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.headphones = CloneSupport.headphonesConnected
                self?.clones.refreshSound()
            }
        }
    }

    func show(_ takes: [CloneTake]) { clones.show(takes) }

    func beginTake() throws {
        if !lockedCamera { CloneSupport.lockCamera(true); lockedCamera = true }
        try engine.begin()
        elapsed = 0
        recording = true
        clones.play()
    }

    func endTake() async -> CloneTake? {
        clones.stop()
        recording = false
        return await engine.end()
    }

    func stop() {
        if let routeObserver { NotificationCenter.default.removeObserver(routeObserver) }
        routeObserver = nil
        clones.clear()
        arView.session.pause()
        if lockedCamera { CloneSupport.lockCamera(false); lockedCamera = false }
        started = false
    }
}

private struct ARViewHost: UIViewRepresentable {
    let view: ARView
    func makeUIView(context: Context) -> ARView { view }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

// MARK: Flow (A2-A6)

struct CloneFlowView: View {
    let onClose: () -> Void
    let onUse: (LocalVideo) -> Void

    private enum Step: Equatable {
        case start, record, preview, processing, result, failed(String)
    }

    @StateObject private var model = CloneARModel()
    @ObservedObject private var prompter = TeleprompterStore.shared
    @State private var step: Step = .start
    @State private var takes: [CloneTake] = []
    @State private var pending: CloneTake?
    @State private var finalURL: URL?
    @State private var progress: Double = 0
    @State private var preparing = false

    var body: some View {
        ZStack {
            switch step {
            case .start:
                CloneStartView(onClose: close, onStart: { model.start(); model.show(takes); step = .record })
            case .record:
                recordScreen
            case .preview:
                if let pending { previewScreen(pending) }
            case .processing:
                processingScreen
            case .result:
                if let finalURL { resultScreen(finalURL) }
            case .failed(let message):
                failedScreen(message)
            }
        }
        .onDisappear { model.stop() }
    }

    private var takeNumber: Int { takes.count + 1 }

    // A3: recording with the earlier clones in the room.
    private var recordScreen: some View {
        ZStack {
            ARViewHost(view: model.arView).ignoresSafeArea()
            VStack(spacing: 10) {
                HStack {
                    Button(action: close) {
                        Image(systemName: "xmark").font(.system(size: 18, weight: .bold)).foregroundStyle(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.35), in: Circle())
                    }
                    .disabled(model.recording)
                    .accessibilityLabel(L10n.tr("Close"))
                    Spacer()
                    badge
                    Spacer()
                    Color.clear.frame(width: 44, height: 44)
                }
                .padding(.horizontal, 12)
                if let message { hint(message) }
                if !takes.isEmpty && !model.headphones { hint("Use headphones to hear your clones") }
                Spacer()
                recordButton.padding(.bottom, 28)
            }
            if prompter.isOn && prompter.hasScript {
                TeleprompterOverlay(running: model.recording, onEdit: {})
            }
        }
        .preferredColorScheme(.dark)
    }

    private var message: String? {
        if model.recording { return model.tracking == .normal ? nil : "Slow down — keep the room in view" }
        return model.tracking == .normal ? nil : "Move your phone slowly to scan the room"
    }

    /// "● REC 0:06 · Clone 2" while recording, "Clone 2" before.
    private var badge: some View {
        HStack(spacing: 6) {
            if model.recording {
                Circle().fill(CloneColor.red).frame(width: 8, height: 8)
                Text(verbatim: "REC \(Self.clock(model.elapsed)) · \(L10n.tr("Clone")) \(takeNumber)")
            } else {
                Text(verbatim: "\(L10n.tr("Clone")) \(takeNumber)")
            }
        }
        .font(CS.inter(14, .bold))
        .foregroundStyle(.white)
        .padding(.horizontal, 12).padding(.vertical, 7)
        .background(Color.black.opacity(0.45), in: Capsule())
        .accessibilityIdentifier("cloneBadge")
    }

    private func hint(_ text: String) -> some View {
        Text(L10n.tr(text))
            .font(CS.inter(13, .semibold)).foregroundStyle(.white)
            .padding(.horizontal, 12).padding(.vertical, 7)
            .background(Color.black.opacity(0.5), in: Capsule())
    }

    /// The same record button as Record.
    private var recordButton: some View {
        Button { Task { await toggleRecording() } } label: {
            ZStack {
                Circle().fill(Color(hex: 0x45474F)).frame(width: 104, height: 104)
                Circle().fill(CS.accentGradient).frame(width: 90, height: 90)
                    .shadow(color: Color(hex: 0x1769FF, opacity: 0.45), radius: 9)
                Circle().fill(CS.background).frame(width: 78, height: 78)
                if model.recording {
                    RoundedRectangle(cornerRadius: 8).fill(Color(hex: 0xFF2E5E)).frame(width: 32, height: 32)
                } else {
                    Circle().fill(.white).frame(width: 64, height: 64)
                }
            }
            .frame(width: 116, height: 116)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        // Before a take: the room must be tracked first (REC waits for it).
        .disabled(!model.recording && model.tracking != .normal)
        .opacity(!model.recording && model.tracking != .normal ? 0.5 : 1)
        .accessibilityLabel(model.recording ? "Stop recording" : "Record")
        .accessibilityIdentifier("cloneRecord")
    }

    private func toggleRecording() async {
        if model.recording {
            guard let take = await model.endTake() else {
                step = .failed("The take could not be saved. Try again.")
                return
            }
            pending = take
            step = .preview
            // Its cut-out for the live preview of the next take, made while you look at it.
            preparing = true
            Task {
                try? await CloneRender.makeLiveClone(take)
                preparing = false
            }
        } else {
            do { try model.beginTake() } catch { step = .failed("The camera could not start. Try again.") }
        }
    }

    // A4: after each take.
    private func previewScreen(_ take: CloneTake) -> some View {
        ZStack {
            CloneColor.white.ignoresSafeArea()
            VStack(spacing: 14) {
                Text(verbatim: "\(L10n.tr("Clone")) \(takeNumber)").font(CS.jakarta(20, .heavy)).foregroundStyle(CloneColor.ink)
                    .padding(.top, 12)
                LoopingPlayer(url: take.videoURL)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .padding(.horizontal, 20)
                HStack(spacing: 10) {
                    CloneButton(title: "Retake") {
                        CloneFolder.remove(take); pending = nil
                        model.show(takes); step = .record
                    }
                    .accessibilityIdentifier("cloneRetake")
                    if takes.count + 1 < CloneMath.maxTakes {
                        CloneButton(title: "Add clone") { Task { await addClone(take) } }
                            .accessibilityIdentifier("cloneAdd")
                    }
                }
                .padding(.horizontal, 20)
                CloneButton(title: "Finish", primary: true) { Task { await finish(take) } }
                    .accessibilityIdentifier("cloneFinish")
                    .padding(.horizontal, 20).padding(.bottom, 12)
            }
        }
        .preferredColorScheme(.light)
    }

    private func addClone(_ take: CloneTake) async {
        while preparing { try? await Task.sleep(nanoseconds: 100_000_000) }
        takes.append(take); pending = nil
        model.show(takes)
        step = .record
    }

    private func finish(_ take: CloneTake) async {
        while preparing { try? await Task.sleep(nanoseconds: 100_000_000) }
        takes.append(take); pending = nil
        await render()
    }

    private func render() async {
        step = .processing
        progress = 0
        do {
            let url = try await CloneRender.render(takes) { p in Task { @MainActor in progress = p } }
            finalURL = url
            step = .result
        } catch {
            step = .failed("We couldn't create your clones. Try again.")
        }
    }

    // A5: waiting.
    private var processingScreen: some View {
        ZStack {
            CloneColor.white.ignoresSafeArea()
            VStack(spacing: 18) {
                BrandRing()
                Text(L10n.tr("Creating your clones…")).font(CS.jakarta(17, .bold)).foregroundStyle(CloneColor.ink)
                Text(verbatim: "\(Int(progress * 100)) %").font(CS.jakarta(13, .medium)).foregroundStyle(CloneColor.gray)
            }
        }
        .preferredColorScheme(.light)
    }

    // A6: result.
    private func resultScreen(_ url: URL) -> some View {
        ZStack {
            CloneColor.white.ignoresSafeArea()
            VStack(spacing: 14) {
                HStack(spacing: 8) {
                    Text(L10n.tr("Your clones")).font(CS.jakarta(20, .heavy)).foregroundStyle(CloneColor.ink)
                    CloneBetaTag(size: 9)
                }
                .padding(.top, 12)
                LoopingPlayer(url: url)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .padding(.horizontal, 20)
                HStack(spacing: 10) {
                    CloneButton(title: "Retake last") {
                        try? FileManager.default.removeItem(at: url)
                        finalURL = nil
                        if let last = takes.popLast() { CloneFolder.remove(last) }
                        model.start(); model.show(takes); step = .record
                    }
                    .accessibilityIdentifier("cloneRetakeLast")
                    CloneButton(title: "Use this video", primary: true) { Task { await use(url) } }
                        .accessibilityIdentifier("cloneUse")
                }
                .padding(.horizontal, 20).padding(.bottom, 12)
            }
        }
        .preferredColorScheme(.light)
    }

    private func use(_ url: URL) async {
        let seconds = (try? await AVURLAsset(url: url).load(.duration).seconds) ?? (takes.last?.duration ?? 0)
        for take in takes { CloneFolder.remove(take) }
        takes = []
        model.stop()
        // Same path as Record: product sheet, upload, engine.
        onUse(LocalVideo(url: url, name: "Clon video", duration: max(0, seconds), contentType: "video/quicktime"))
    }

    private func failedScreen(_ text: String) -> some View {
        ZStack {
            CloneColor.white.ignoresSafeArea()
            VStack(spacing: 16) {
                Text(L10n.tr(text)).font(CS.jakarta(16, .bold)).foregroundStyle(CloneColor.ink)
                    .multilineTextAlignment(.center)
                CloneButton(title: "Try again", primary: true) { model.show(takes); step = .record }
                CloneButton(title: "Close", action: close)
            }
            .padding(24)
        }
        .preferredColorScheme(.light)
    }

    private func close() {
        model.stop()
        for take in takes { CloneFolder.remove(take) }
        if let pending { CloneFolder.remove(pending) }
        if let finalURL { try? FileManager.default.removeItem(at: finalURL) }
        onClose()
    }

    static func clock(_ seconds: Double) -> String {
        let s = max(0, Int(seconds))
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

/// A video that plays in a loop (take preview, result).
private struct LoopingPlayer: View {
    let url: URL
    @State private var player: AVQueuePlayer?
    @State private var looper: AVPlayerLooper?

    var body: some View {
        VideoPlayer(player: player)
            .aspectRatio(9 / 16, contentMode: .fit)
            .onAppear {
                let item = AVPlayerItem(url: url)
                let queue = AVQueuePlayer()
                looper = AVPlayerLooper(player: queue, templateItem: item)
                player = queue
                queue.play()
            }
            .onDisappear { player?.pause() }
    }
}
