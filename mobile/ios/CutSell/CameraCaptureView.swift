import AVFoundation
import SwiftUI

struct CameraCaptureView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var camera = CameraController()
    @State private var elapsed: TimeInterval = 0
    @State private var timer: Timer?

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
                    Text(timeString(elapsed))
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

                Button {
                    if camera.isRecording {
                        stopTimer()
                        camera.stopRecording()
                    } else {
                        elapsed = 0
                        startTimer()
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

    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            elapsed += 0.1
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func timeString(_ seconds: TimeInterval) -> String {
        let tenths = max(0, Int(seconds * 10))
        return String(format: "%02d:%02d.%d", tenths / 600, (tenths / 10) % 60, tenths % 10)
    }
}
