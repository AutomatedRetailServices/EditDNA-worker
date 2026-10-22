import AVFoundation
import SwiftUI

/// Create tab: the camera the app opens on (Figma 03 Create, "Create/Camera/01 Start", 625:86).
/// Records with the same `CameraController` the app already used.
struct CreateCameraView: View {
    enum RecordingPreset: Int, CaseIterable, Identifiable {
        case tenMinutes = 600
        case sixtySeconds = 60
        case threeMinutes = 180

        var id: Int { rawValue }
        var title: String {
            switch self {
            case .sixtySeconds: return "60s"
            case .threeMinutes: return "3m"
            case .tenMinutes: return "10m"
            }
        }
    }

    private enum Gate: Equatable {
        case checking
        case camera(denied: Bool)
        case microphone(denied: Bool)
        case ready
    }

    let onBack: () -> Void
    let onUpload: () -> Void
    /// A finished recording, ready for "Choose your style".
    let onRecorded: (LocalVideo) -> Void
    /// false while another screen covers the camera (gallery, upload): camera and light off.
    var isActive = true
    /// Design screenshots only: show the camera screen without a real camera.
    var previewMode = false

    @StateObject private var camera = CameraController()
    @State private var gate: Gate = .checking
    @State private var preset: RecordingPreset = .sixtySeconds
    @State private var elapsed: TimeInterval = 0
    @State private var timer: Timer?
    @State private var torchOn = false
    @State private var showGrid = false
    @State private var recordingFailed = false

    var body: some View {
        ZStack {
            switch gate {
            case .checking:
                CS.background.ignoresSafeArea()
            case .camera(let denied):
                PermissionScreen(kind: .camera, deniedBefore: denied, onClose: onBack) {
                    Task {
                        _ = await AVCaptureDevice.requestAccess(for: .video)
                        await evaluateGate()
                    }
                }
            case .microphone(let denied):
                PermissionScreen(kind: .microphone, deniedBefore: denied, onClose: onBack) {
                    Task {
                        _ = await AVCaptureDevice.requestAccess(for: .audio)
                        await evaluateGate()
                    }
                }
            case .ready:
                cameraScreen
            }
        }
        .preferredColorScheme(.dark)
        .task {
            if previewMode { gate = .ready; return }
            await evaluateGate()
        }
        .onChange(of: isActive) { _, active in
            guard !previewMode, gate == .ready else { return }
            if active {
                camera.startSession()
            } else {
                if camera.isRecording { stopTimer(); camera.stopRecording() }
                if torchOn { camera.setTorch(false); torchOn = false }
                camera.stopSession()
            }
        }
        .onDisappear {
            stopTimer()
            if camera.isRecording { camera.stopRecording() }
            if torchOn { camera.setTorch(false) }
            camera.stopSession()
        }
        .fullScreenCover(isPresented: $recordingFailed) {
            CreateProblemView(
                problem: .couldNotAdd,
                onBack: { recordingFailed = false },
                onRetry: { recordingFailed = false }
            )
        }
    }

    private var cameraScreen: some View {
        ZStack {
            CS.screenBackground.ignoresSafeArea()
            if !previewMode {
                CameraPreview(session: camera.session, mirrored: camera.mirrorOn, position: camera.cameraPosition)
                    .ignoresSafeArea()
            }
            if showGrid {
                GridOverlay().ignoresSafeArea().allowsHitTesting(false)
            }

            VStack(spacing: 0) {
                ZStack {
                    BrandLogo()
                    HStack {
                        Button {
                            onBack()
                        } label: {
                            Image("IconBack")
                                .resizable()
                                .frame(width: 18, height: 28)
                                .frame(width: 44, height: 44)
                        }
                        .accessibilityLabel(L10n.tr("Back"))
                        .disabled(camera.isRecording)
                        Spacer()
                    }
                    .padding(.leading, 11)
                }
                .padding(.top, 4)

                if camera.isRecording || elapsed > 0 {
                    Text(L10n.tr("\(timeString(elapsed)) / \(preset.title)"))
                        .font(.system(size: 15, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 7)
                        .background(Color.black.opacity(0.45), in: Capsule())
                        .padding(.top, 8)
                }

                HStack {
                    Spacer()
                    // Hidden while recording (Figma 645:23 / 626:361); back between clips.
                    if !camera.isRecording {
                        controlsColumn
                    }
                }
                .padding(.top, camera.isRecording || elapsed > 0 ? 6 : 20)

                Spacer()

                if let error = camera.errorMessage {
                    Text(L10n.tr(error))
                        .font(CS.sf(13, .medium))
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                        .padding(12)
                        .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14))
                        .padding(.horizontal, 24)
                        .padding(.bottom, 12)
                }

                durationChips
                    .padding(.bottom, 12)

                ZStack {
                    recordButton
                    HStack {
                        uploadButton
                        Spacer()
                    }
                    .padding(.leading, 22)
                    .offset(y: 14)
                }
                .padding(.bottom, 26)
            }
        }
    }

    private var controlsColumn: some View {
        VStack(spacing: 24) {
            controlButton(title: "Flip", image: "IconFlip", size: CGSize(width: 42, height: 34)) {
                Task {
                    if torchOn { camera.setTorch(false); torchOn = false }
                    await camera.flipCamera()
                }
            }
            .disabled(camera.isRecording)
            controlButton(title: "Flash", image: torchOn ? "IconFlashOn" : "IconFlash", size: CGSize(width: 34, height: 38)) {
                torchOn.toggle()
                camera.setTorch(torchOn)
            }
            .disabled(!camera.hasTorch && !previewMode)
            .opacity(camera.hasTorch || previewMode ? 1 : 0.5)
            controlButton(title: "Grid", image: "IconGrid", size: CGSize(width: 38, height: 38)) {
                showGrid.toggle()
            }
            Button {
                camera.mirrorOn.toggle()
            } label: {
                VStack(spacing: 4) {
                    Image("IconMirror").renderingMode(.template).resizable()
                        .foregroundStyle(camera.mirrorOn ? CS.cyan : .white)
                        .frame(width: 36, height: 36)
                    Text(L10n.tr("Mirror"))
                        .font(CS.sf(13, .semibold))
                        .foregroundStyle(camera.mirrorOn ? CS.cyan : .white)
                }
                .frame(width: 62)
            }
            .buttonStyle(.plain)
            .accessibilityValue(camera.mirrorOn ? "On" : "Off")
            .accessibilityIdentifier("cameraMirror")
        }
        .frame(width: 62)
        .padding(.trailing, 8)
    }

    private func controlButton(title: String, image: String, size: CGSize, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(image).resizable().frame(width: size.width, height: size.height)
                Text(L10n.tr(title))
                    .font(CS.sf(13, .semibold))
                    .foregroundStyle(.white)
            }
            .frame(width: 62)
        }
        .buttonStyle(.plain)
    }

    private var durationChips: some View {
        HStack(spacing: 0) {
            ForEach(RecordingPreset.allCases) { item in
                let isSelected = item == preset
                Button {
                    preset = item
                } label: {
                    Text(L10n.tr(item.title))
                        .font(CS.sf(isSelected ? 16 : 15, isSelected ? .bold : .semibold))
                        .foregroundStyle(.white)
                        .frame(width: isSelected ? 84 : 54, height: 42)
                        .background {
                            if isSelected {
                                Capsule().fill(CS.brandGradient)
                                    .shadow(color: Color(hex: 0x1769FF, opacity: 0.28), radius: 6)
                            }
                        }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 6)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .disabled(camera.isRecording)
    }

    private var recordButton: some View {
        Button {
            toggleRecording()
        } label: {
            ZStack {
                Circle().fill(Color(hex: 0x45474F)).frame(width: 104, height: 104)
                Circle().fill(CS.brandGradient).frame(width: 90, height: 90)
                    .shadow(color: Color(hex: 0x1769FF, opacity: 0.45), radius: 9)
                Circle().fill(CS.background).frame(width: 78, height: 78)
                if camera.isRecording {
                    RoundedRectangle(cornerRadius: 8).fill(Color(hex: 0xFF2E5E)).frame(width: 32, height: 32)
                } else {
                    Circle().fill(.white).frame(width: 64, height: 64)
                }
            }
            .frame(width: 116, height: 116)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(!camera.isConfigured && !previewMode)
        .accessibilityLabel(camera.isRecording ? "Stop recording" : "Record")
        .accessibilityIdentifier("recordButton")
    }

    private var uploadButton: some View {
        Button(action: onUpload) {
            VStack(spacing: 8) {
                Image("IconUpload")
                    .resizable()
                    .frame(width: 34, height: 34)
                    .frame(width: 54, height: 58)
                    .overlay(RoundedRectangle(cornerRadius: 9).stroke(CS.cyan, lineWidth: 2))
                    .shadow(color: CS.cyan.opacity(0.24), radius: 5)
                Text(L10n.tr("Upload"))
                    .font(CS.sf(14, .bold))
                    .foregroundStyle(.white)
            }
        }
        .buttonStyle(.plain)
        .disabled(camera.isRecording)
        .accessibilityIdentifier("uploadButton")
    }

    // MARK: Permissions

    @MainActor
    private func evaluateGate() async {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .notDetermined: gate = .camera(denied: false); return
        case .denied, .restricted: gate = .camera(denied: true); return
        default: break
        }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .notDetermined: gate = .microphone(denied: false); return
        case .denied, .restricted: gate = .microphone(denied: true); return
        default: break
        }
        gate = .ready
        await camera.prepare()
    }

    // MARK: Recording

    private func toggleRecording() {
        if camera.isRecording {
            stopTimer()
            camera.stopRecording()
            return
        }
        elapsed = 0
        camera.record { result in
            stopTimer()
            switch result {
            case .success(let url):
                // D-133: bounded, diagnostics-only capture-metadata log.
                let side = camera.cameraPosition == .front ? "front" : "back"
                Task {
                    let snapshot = await MediaDiagnostics.capture(fileURL: url, mirroredHint: camera.mirrorOn)
                    var fields: [String: String] = [
                        "device_model": CutSellDiagnostics.deviceModel,
                        "ios_version": CutSellDiagnostics.iosVersion,
                        "camera_side": side,
                    ]
                    fields.merge(snapshot?.logFields ?? [:]) { current, _ in current }
                    CutSellDiagnostics.log("capture_completed", fields)
                }
                Task {
                    let seconds = (try? await AVURLAsset(url: url).load(.duration).seconds) ?? elapsed
                    elapsed = 0
                    onRecorded(LocalVideo(
                        url: url,
                        name: "Recorded clip",
                        duration: max(0, seconds),
                        contentType: "video/quicktime"
                    ))
                }
            case .failure:
                elapsed = 0
                recordingFailed = true
            }
        }
        // `record` sets isRecording only when the recording really started.
        if camera.isRecording { startTimer(limit: TimeInterval(preset.rawValue)) }
    }

    private func startTimer(limit: TimeInterval) {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            Task { @MainActor in
                elapsed += 0.1
                if elapsed >= limit {
                    elapsed = limit
                    stopTimer()
                    if camera.isRecording { camera.stopRecording() }
                }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func timeString(_ seconds: TimeInterval) -> String {
        let whole = max(0, Int(seconds.rounded(.down)))
        return String(format: "%02d:%02d", whole / 60, whole % 60)
    }
}

/// Rule-of-thirds lines (the camera "Grid" button).
private struct GridOverlay: View {
    var body: some View {
        GeometryReader { proxy in
            Path { path in
                let w = proxy.size.width, h = proxy.size.height
                for i in 1...2 {
                    let x = w * CGFloat(i) / 3, y = h * CGFloat(i) / 3
                    path.move(to: CGPoint(x: x, y: 0)); path.addLine(to: CGPoint(x: x, y: h))
                    path.move(to: CGPoint(x: 0, y: y)); path.addLine(to: CGPoint(x: w, y: y))
                }
            }
            .stroke(Color.white.opacity(0.35), lineWidth: 1)
        }
    }
}
