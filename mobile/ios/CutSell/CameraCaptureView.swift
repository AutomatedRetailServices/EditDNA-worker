import AVFoundation
import SwiftUI

struct CameraCaptureView: View {
    private enum RecordingPreset: Int, CaseIterable, Identifiable {
        case sixtySeconds = 60
        case threeMinutes = 180
        case tenMinutes = 600

        var id: Int { rawValue }
        var title: String {
            switch self {
            case .sixtySeconds: return "60s"
            case .threeMinutes: return "3m"
            case .tenMinutes: return "10m"
            }
        }
    }

    @Environment(\.dismiss) private var dismiss
    @StateObject private var camera = CameraController()
    @State private var elapsed: TimeInterval = 0
    @State private var timer: Timer?
    @State private var preset: RecordingPreset = .sixtySeconds

    let onCapture: (URL) -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            VStack {
                HStack {
                    Button {
                        camera.stopSession()
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.title3.bold())
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    Spacer()
                    Text("\(timeString(elapsed)) / \(preset.title)")
                        .font(.system(.body, design: .monospaced).bold())
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                    Button {
                        Task { await camera.flipCamera() }
                    } label: {
                        Image(systemName: "camera.rotate")
                            .font(.title3.bold())
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .disabled(camera.isRecording)
                }
                .padding()

                Spacer()

                if let error = camera.errorMessage {
                    Text(error)
                        .font(.footnote)
                        .multilineTextAlignment(.center)
                        .padding()
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                        .padding()
                }

                Picker("Recording length", selection: $preset) {
                    ForEach(RecordingPreset.allCases) { item in
                        Text(item.title).tag(item)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 48)
                .padding(.bottom, 20)
                .disabled(camera.isRecording)

                Button {
                    if camera.isRecording {
                        stopTimer()
                        camera.stopRecording()
                    } else {
                        elapsed = 0
                        startTimer(limit: TimeInterval(preset.rawValue))
                        camera.record { result in
                            stopTimer()
                            switch result {
                            case .success(let url):
                                onCapture(url)
                                dismiss()
                            case .failure(let error):
                                camera.errorMessage = error.localizedDescription
                            }
                        }
                    }
                } label: {
                    ZStack {
                        Circle()
                            .stroke(.white, lineWidth: 5)
                            .frame(width: 82, height: 82)
                        RoundedRectangle(cornerRadius: camera.isRecording ? 8 : 35)
                            .fill(.red)
                            .frame(width: camera.isRecording ? 34 : 66, height: camera.isRecording ? 34 : 66)
                    }
                }
                .padding(.bottom, 34)
                .disabled(!camera.isConfigured)
            }
        }
        .foregroundStyle(.white)
        .task { await camera.prepare() }
        .onDisappear {
            stopTimer()
            if camera.isRecording { camera.stopRecording() }
            camera.stopSession()
        }
    }

    private func startTimer(limit: TimeInterval) {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            elapsed += 0.1
            if elapsed >= limit {
                elapsed = limit
                stopTimer()
                if camera.isRecording {
                    camera.stopRecording()
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
