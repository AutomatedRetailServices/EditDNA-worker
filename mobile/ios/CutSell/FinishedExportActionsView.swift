import Photos
import SwiftUI

struct FinishedExportActionsView: View {
    let remoteURL: URL

    @State private var localURL: URL?
    @State private var isDownloading = false
    @State private var isSaving = false
    @State private var saved = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 10) {
            if let localURL {
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

            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
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
        } catch {
            errorMessage = error.localizedDescription
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
        } catch {
            errorMessage = error.localizedDescription
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
