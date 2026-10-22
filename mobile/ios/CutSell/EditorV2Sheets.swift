import AVFoundation
import SwiftUI

/// "Add Captions" (Figma 04_Editor, Captions 894:303).
struct AddCaptionsSheet: View {
    @ObservedObject var model: DraftEditorViewModel
    let playhead: Double

    @Environment(\.dismiss) private var dismiss
    @State private var auto = true
    @State private var askingText = false
    @State private var manualText = ""
    @State private var working = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Color(hex: 0x3A5280)).frame(width: 40, height: 4).padding(.top, 10)
            Text(L10n.tr("Add Captions"))
                .font(CS.inter(26, .bold))
                .foregroundStyle(.white)
                .padding(.top, 18)
            Text(L10n.tr("Turn speech into editable captions."))
                .font(CS.inter(15))
                .foregroundStyle(CS.textSecondaryF)
                .padding(.top, 6)

            Button { auto.toggle() } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(L10n.tr("Auto captions")).font(CS.inter(16, .semibold)).foregroundStyle(.white)
                        Text(L10n.tr("Generate from your video’s speech.")).font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
                    }
                    Spacer()
                    ZStack {
                        Circle().fill(auto ? Color(hex: 0x1677FF) : .clear)
                            .overlay(Circle().stroke(auto ? .clear : CS.borderDefault, lineWidth: 1.5))
                        if auto { Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(.white) }
                    }
                    .frame(width: 22, height: 22)
                }
                .padding(16)
                .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 16))
                .overlay(RoundedRectangle(cornerRadius: 16).stroke(
                    auto ? AnyShapeStyle(LinearGradient(colors: [Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)], startPoint: .leading, endPoint: .trailing)) : AnyShapeStyle(CS.borderDefault),
                    lineWidth: auto ? 1.5 : 1))
            }
            .buttonStyle(.plain)
            .padding(.top, 20)

            Button { askingText = true } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(L10n.tr("Add text manually")).font(CS.inter(16, .semibold)).foregroundStyle(.white)
                        Text(L10n.tr("Create text at the playhead.")).font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
                    }
                    Spacer()
                    Image(systemName: "plus").font(.system(size: 14, weight: .heavy)).foregroundStyle(Color(hex: 0x0B1020))
                        .frame(width: 24, height: 24).background(Color(hex: 0xE6E9EF), in: RoundedRectangle(cornerRadius: 6))
                }
                .padding(16)
                .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 16))
                .overlay(RoundedRectangle(cornerRadius: 16).stroke(CS.borderDefault, lineWidth: 1))
            }
            .buttonStyle(.plain)
            .padding(.top, 10)

            HStack {
                Text(L10n.tr("Language")).font(CS.inter(15)).foregroundStyle(.white)
                Spacer()
                Text(L10n.tr("Auto-detect")).font(CS.inter(15, .semibold)).foregroundStyle(CS.cyan)
            }
            .padding(16)
            .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(CS.borderDefault, lineWidth: 1))
            .padding(.top, 10)

            PrimaryButton(title: model.captionsEnabled && auto ? "Captions on · Done" : "Generate Captions",
                          height: 54, cornerRadius: 18, fontSize: 17, gradient: CS.alertButtonGradient,
                          textColor: Color(hex: 0x0B1A40), isEnabled: !working) {
                Task {
                    working = true
                    if auto != model.captionsEnabled { await model.setCaptionSettings(enabled: auto) }
                    working = false
                    dismiss()
                }
            }
            .padding(.top, 18)
            .accessibilityIdentifier("generateCaptions")

            Text(L10n.tr("Audio is processed only when you generate captions."))
                .font(CS.inter(11))
                .foregroundStyle(CS.textMutedF)
                .frame(maxWidth: .infinity)
                .padding(.top, 12)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(hex: 0x060B18).ignoresSafeArea())
        .presentationDetents([.height(500)])
        .presentationDragIndicator(.hidden)
        .presentationBackground(Color(hex: 0x060B18))
        .preferredColorScheme(.dark)
        .alert(L10n.tr("Add text"), isPresented: $askingText) {
            TextField(L10n.tr("Text"), text: $manualText)
            Button(L10n.tr("Add")) {
                let text = manualText.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { return }
                Task {
                    await model.addTextOverlay(text: text, start: playhead, end: playhead + 3)
                    dismiss()
                }
            }
            Button(L10n.tr("Cancel"), role: .cancel) {}
        } message: {
            Text(L10n.tr("It appears on the video at the playhead for 3 seconds."))
        }
    }
}

/// "Recording voice-over" (Figma 04_Editor, VoiceOver/Record 408:58).
struct VoiceOverRecordSheet: View {
    @ObservedObject var model: DraftEditorViewModel
    let start: Double
    let onDone: (DraftEditorViewModel.RecordedVoiceOverOutcome) -> Void

    @Environment(\.dismiss) private var dismiss
    @StateObject private var recorder = VoiceRecorder()
    @State private var player: AVAudioPlayer?
    @State private var playing = false
    @State private var saving = false

    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(Color(hex: 0x3A5280)).frame(width: 40, height: 4).padding(.top, 10)
            HStack(spacing: 10) {
                Image("EdMic").resizable().renderingMode(.template).foregroundStyle(CS.magenta).frame(width: 20, height: 20)
                Text(L10n.tr(recordedURL == nil ? "Recording voice-over" : "Voice-over"))
                    .font(CS.inter(19, .bold)).foregroundStyle(.white)
            }
            .padding(.top, 20)
            Text(L10n.tr(recordedURL == nil ? "Add your voice to this video" : "Preview your recording before saving."))
                .font(CS.inter(13, .semibold))
                .foregroundStyle(CS.textSecondaryF)
                .padding(.top, 6)

            if recordedURL == nil {
                recordControls
            } else {
                previewControls
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
        .frame(maxWidth: .infinity)
        .background(Color(hex: 0x060B18).ignoresSafeArea())
        .presentationDetents([.height(430)])
        .presentationBackground(Color(hex: 0x060B18))
        .preferredColorScheme(.dark)
        .onDisappear {
            player?.stop()
            if recorder.state == .recording { recorder.stop() }
        }
    }

    private var recordedURL: URL? { recorder.recordedURL }

    private var recordControls: some View {
        VStack(spacing: 16) {
            Text(VoiceRecorder.formatted(recorder.elapsed))
                .font(CS.sf(15, .semibold)).monospacedDigit().foregroundStyle(.white)
                .padding(.top, 22)
            HStack(spacing: 36) {
                circleButton(icon: recorder.isPaused ? "play.fill" : "pause.fill", label: recorder.isPaused ? "Tap to resume" : "Tap to pause",
                             size: 48, fill: Color(hex: 0x0B1020), stroke: .white) {
                    if recorder.isPaused { recorder.resume() } else { recorder.pause() }
                }
                .disabled(recorder.state != .recording)
                .opacity(recorder.state == .recording ? 1 : 0.4)

                VStack(spacing: 8) {
                    Button {
                        Task {
                            if recorder.state == .recording { recorder.stop() } else { await recorder.start() }
                        }
                    } label: {
                        ZStack {
                            Circle().fill(LinearGradient(colors: [Color(hex: 0x1769FF), Color(hex: 0x0E46D9)], startPoint: .top, endPoint: .bottom))
                                .frame(width: 76, height: 76)
                            Circle().stroke(LinearGradient(colors: [CS.cyan, Color(hex: 0x7C3AED), CS.magenta], startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 4)
                                .frame(width: 84, height: 84)
                            Image(systemName: recorder.state == .recording ? "stop.fill" : "mic.fill")
                                .font(.system(size: 26, weight: .bold)).foregroundStyle(.white)
                        }
                        .shadow(color: CS.cyan.opacity(0.4), radius: 12)
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("voiceRecordButton")
                    Text(L10n.tr(recorder.state == .recording ? "Tap to stop" : "Tap to record"))
                        .font(CS.inter(11)).foregroundStyle(CS.textSecondaryF)
                }

                circleButton(icon: "stop.fill", label: "Tap to finish", size: 48, fill: CS.error.opacity(0.85), stroke: CS.error) {
                    recorder.stop()
                }
                .disabled(recorder.state != .recording)
                .opacity(recorder.state == .recording ? 1 : 0.4)
            }
            .padding(.top, 6)
            switch recorder.state {
            case .permissionDenied:
                Text(L10n.tr("Allow the microphone in Settings to record.")).font(CS.inter(12)).foregroundStyle(CS.error)
            case .failed(let message):
                Text(L10n.tr(message)).font(CS.inter(12)).foregroundStyle(CS.error)
            default:
                EmptyView()
            }
        }
    }

    private var previewControls: some View {
        VStack(spacing: 14) {
            Image("EdWave").resizable().frame(height: 40).padding(.top, 24)
            HStack {
                Text(L10n.tr("00:00"))
                Spacer()
                Text(VoiceRecorder.formatted(recorder.elapsed))
            }
            .font(CS.sf(12)).foregroundStyle(CS.textSecondaryF)
            HStack(spacing: 10) {
                pill(playing ? "Pause" : "Play", icon: playing ? "pause.fill" : "play.fill", border: CS.magenta) { togglePlay() }
                pill("Retake", icon: "arrow.counterclockwise", border: Color(hex: 0x7C3AED)) {
                    player?.stop(); playing = false
                    recorder.discardTake()
                }
                Button {
                    Task { await save() }
                } label: {
                    Label(saving ? "Saving…" : "Save", systemImage: "checkmark")
                        .font(CS.inter(15, .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity).frame(height: 46)
                        .background(LinearGradient(colors: [CS.cyan, Color(hex: 0x1769FF), CS.magenta], startPoint: .leading, endPoint: .trailing), in: RoundedRectangle(cornerRadius: 14))
                }
                .buttonStyle(.plain)
                .disabled(saving)
                .accessibilityIdentifier("voiceSave")
            }
            .padding(.top, 8)
        }
    }

    private func circleButton(icon: String, label: String, size: CGFloat, fill: Color, stroke: Color, action: @escaping () -> Void) -> some View {
        VStack(spacing: 8) {
            Button(action: action) {
                Image(systemName: icon).font(.system(size: 16, weight: .bold)).foregroundStyle(.white)
                    .frame(width: size, height: size)
                    .background(Circle().fill(fill))
                    .overlay(Circle().stroke(stroke, lineWidth: 2))
            }
            .buttonStyle(.plain)
            Text(L10n.tr(label)).font(CS.inter(11)).foregroundStyle(CS.textSecondaryF)
        }
    }

    private func pill(_ title: String, icon: String, border: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(L10n.tr(title), systemImage: icon)
                .font(CS.inter(15, .semibold))
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity).frame(height: 46)
                .background(Color(hex: 0x0B1020), in: RoundedRectangle(cornerRadius: 14))
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(border, lineWidth: 1.5))
        }
        .buttonStyle(.plain)
    }

    private func togglePlay() {
        if playing { player?.pause(); playing = false; return }
        guard let url = recordedURL else { return }
        if player == nil || player?.url != url {
            try? AVAudioSession.sharedInstance().setCategory(.playback)
            player = try? AVAudioPlayer(contentsOf: url)
        }
        player?.play()
        playing = true
    }

    private func save() async {
        guard let url = recordedURL else { return }
        player?.stop(); playing = false
        saving = true
        let outcome = await model.addRecordedVoiceOver(fileURL: url, at: start)
        saving = false
        if outcome != .failed {
            recorder.finishTake()
            onDone(outcome)
            dismiss()
        }
    }
}

/// Export: started when Export is tapped. Shows what it is doing and the real % from the server,
/// then Save / Share. Leaving the editor is fine: the creator is told when the video is ready.
struct ExportSheet: View {
    @ObservedObject var model: DraftEditorViewModel

    var body: some View {
        VStack(spacing: 16) {
            Capsule().fill(Color(hex: 0x3A5280)).frame(width: 40, height: 4).padding(.top, 10)
            Text(L10n.tr("Export")).font(CS.inter(22, .bold)).foregroundStyle(.white)
            if let problem = model.exportError {
                // Build 27: an export that could not start says why, in plain words, with Try again.
                Text(L10n.tr(problem)).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF).multilineTextAlignment(.center)
                    .accessibilityIdentifier("exportError")
            } else if let preparing = model.exportPreparing {
                Text(L10n.tr(preparing)).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
                ProgressView(value: 0, total: 100).tint(CS.cyan)
            } else if let job = model.exportJob {
                if job.state == "finished" {
                    Text(L10n.tr("Your video is ready.")).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
                } else if job.state == "failed" || job.state == "canceled" {
                    Text(L10n.tr("The export didn't finish. Please try again.")).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
                } else {
                    let percent = Self.shownPercent(job)
                    HStack {
                        Text(L10n.tr(Self.stageText(job))).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
                        Spacer()
                        Text(L10n.tr("\(percent)%")).font(CS.inter(16, .bold)).monospacedDigit().foregroundStyle(CS.cyan)
                            .accessibilityIdentifier("exportPercent")
                    }
                    ProgressView(value: Double(percent), total: 100).tint(CS.cyan)
                    Text(L10n.tr("You can leave this screen. We'll let you know when it's ready."))
                        .font(CS.inter(12)).foregroundStyle(CS.textMutedF).multilineTextAlignment(.center)
                }
                if let urlString = job.result?["download_url"]?.stringValue, let url = URL(string: urlString) {
                    FinishedExportActionsView(remoteURL: url, projectID: model.project.projectID, session: model.session)
                }
            } else {
                Text(L10n.tr("Getting your video ready…")).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
                ProgressView().tint(CS.cyan)
            }
            if model.exportError != nil || model.exportJob.map({ $0.state == "failed" || $0.state == "canceled" }) == true {
                PrimaryButton(title: "Try again", height: 54, cornerRadius: 18, fontSize: 17) {
                    Task { await model.export() }
                }
                .accessibilityIdentifier("exportStart")
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
        .frame(maxWidth: .infinity)
        .background(Color(hex: 0x060B18).ignoresSafeArea())
        .presentationDetents([.medium])
        .presentationBackground(Color(hex: 0x060B18))
        .preferredColorScheme(.dark)
    }

    /// The server's own % (never shown going backwards).
    static func shownPercent(_ job: JobStatus) -> Int { max(1, min(99, job.progress ?? 1)) }

    static func stageText(_ job: JobStatus) -> String {
        let progress = job.progress ?? 0
        switch job.state {
        case "queued", "uploaded": return "Preparing your export…"
        default:
            if progress < 25 { return "Getting your clips…" }
            if progress < 85 { return "Making your video…" }
            return "Finishing…"
        }
    }
}
