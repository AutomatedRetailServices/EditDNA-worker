import Photos
import SwiftUI

struct FinishedExportActionsView: View {
    let remoteURL: URL
    var projectID: String? = nil
    var session: CutSellSession? = nil
    @ObservedObject private var poster = TikTokPoster.shared
    @State private var notice: String?

    @State private var localURL: URL?
    @State private var isDownloading = false
    @State private var isSaving = false
    @State private var saved = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 10) {
            if let localURL {
                if let projectID {
                    // Post to TikTok: video to Photos, caption copied, TikTok opens (Share Kit once TikTok approves).
                    PostToTikTokButton(title: poster.busy ? "Opening TikTok…" : "Post to TikTok") {
                        Task {
                            let caption = await PlanWeekStore.shared.caption(for: projectID, session: session)
                            await poster.post(video: localURL, caption: caption, projectID: projectID,
                                              onNotice: { notice = $0 }) {
                                try? await PlanWeekStore.shared.markPosted(projectID: projectID, session: session)
                            }
                        }
                    }
                    .disabled(poster.busy)
                }
                HStack(spacing: 10) {
                    ShareLink(item: localURL) {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button {
                        Task { await saveToPhotos(localURL) }
                    } label: {
                        if isSaving {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Label(saved ? "Saved" : "Save", systemImage: saved ? "checkmark.circle.fill" : "square.and.arrow.down")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isSaving || saved)
                }
            } else {
                Button {
                    Task { await download() }
                } label: {
                    if isDownloading {
                        HStack { ProgressView(); Text("Preparing video…") }
                            .frame(maxWidth: .infinity)
                    } else {
                        Label("Prepare to share", systemImage: "arrow.down.circle")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(isDownloading)
            }

            if let notice {
                Text(notice).font(.caption).foregroundStyle(CS.cyan).accessibilityIdentifier("exportNotice")
            }
            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
                if localURL == nil {
                    Button("Retry") { self.errorMessage = nil; Task { await download() } }
                        .accessibilityIdentifier("exportDownloadRetry")
                }
            }
        }
        .task { await download() }
    }

    @MainActor
    private func download() async {
        guard localURL == nil, !isDownloading else { return }
        isDownloading = true
        defer { isDownloading = false }
        do {
            let (temporary, response) = try await URLSession.shared.download(from: remoteURL)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                throw ExportActionError.downloadFailed
            }
            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent("CutSell-\(UUID().uuidString).mp4")
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: temporary, to: destination)
            localURL = destination
            #if DEBUG
            // Simulator probe only: the workflow reads the finished file to check the voice-over in the export.
            if ProcessInfo.processInfo.environment["CUTSELL_PROBE_STATE"] != nil { NSLog("CUTSELLEXPORTURL %@", remoteURL.absoluteString) }
            #endif
            // D-133: bounded, diagnostics-only "result received" flag --
            // never the actual (possibly signed) result URL string.
            CutSellDiagnostics.log("result_received", ["received": "true"])
            CutSellDiagnostics.log("export_downloaded", ["mb": String(format: "%.1f", Double((try? FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? Int) ?? 0) / 1e6)])
        } catch {
            errorMessage = error.localizedDescription
            CutSellDiagnostics.log("result_received", ["received": "false"])
            CutSellDiagnostics.log("export_failed", Retry.fields(error, ["step": "download"]))
        }
    }

    @MainActor
    private func saveToPhotos(_ url: URL) async {
        guard !isSaving else { return }
        isSaving = true
        defer { isSaving = false }
        do {
            let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard status == .authorized || status == .limited else {
                throw ExportActionError.photosPermissionDenied
            }
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            }
            saved = true
            CutSellDiagnostics.log("save_to_photos_completed", ["succeeded": "true"])
            CutSellDiagnostics.log("export_saved_to_photos", [:])
        } catch {
            errorMessage = error.localizedDescription
            CutSellDiagnostics.log("save_to_photos_completed", ["succeeded": "false"])
            CutSellDiagnostics.log("export_failed", Retry.fields(error, ["step": "save_to_photos"]))
        }
    }
}

private enum ExportActionError: LocalizedError {
    case downloadFailed
    case photosPermissionDenied

    var errorDescription: String? {
        switch self {
        case .downloadFailed: return "CutSell could not download the finished video."
        case .photosPermissionDenied: return "Allow CutSell to add the finished video to Photos."
        }
    }
}
