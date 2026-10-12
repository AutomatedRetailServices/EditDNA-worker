import Photos
import SwiftUI

/// The Create tab: camera → (Upload → gallery) → "Choose your style" → upload and start the cut.
/// Uploading and starting the cut use `CutSubmitter`, the same pipeline as before.
struct CreateFlowView: View {
    let onBack: () -> Void
    /// Called after the cut(s) started, to show them in Edits.
    let onStarted: () -> Void

    @EnvironmentObject private var appState: AppState
    @StateObject private var submitter = CutSubmitter()
    @State private var showingGallery = false
    @State private var recorded: LocalVideo?
    @State private var job: CreateJob?
    @State private var problem: CreateProblem?
    @State private var uploadFailed = false

    var body: some View {
        CreateCameraView(
            onBack: onBack,
            onUpload: { showingGallery = true },
            onRecorded: { recorded = $0 }
        )
        .sheet(item: $recorded) { video in
            StyleSheetView(ctaTitle: CreateMode.single.ctaTitle(1)) { style in
                recorded = nil
                start(CreateJob(mode: .single, source: .recorded(video), style: style))
            }
        }
        .fullScreenCover(isPresented: $showingGallery) {
            GalleryPickerView(
                onCancel: { showingGallery = false },
                onCreate: { mode, assets, style in
                    showingGallery = false
                    start(CreateJob(mode: mode, source: .library(assets), style: style))
                }
            )
        }
        .fullScreenCover(item: $job) { current in
            SubmittingView(submitter: submitter, job: current)
                .fullScreenCover(item: $problem) { shown in
                    CreateProblemView(
                        problem: shown,
                        onBack: {
                            problem = nil
                            job = nil
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { showingGallery = true }
                        },
                        onShowFormats: { problem = .supportedFormats }
                    )
                }
                .fullScreenCover(isPresented: $uploadFailed) {
                    ProblemScreen(
                        screenTitle: "Upload paused",
                        heading: "Something went wrong",
                        message: "Your video is saved on this iPhone. Open Edits to resume the upload.",
                        primaryTitle: "Go to Edits",
                        primaryAction: { uploadFailed = false; job = nil; onStarted() }
                    )
                }
        }
    }

    private func start(_ newJob: CreateJob) {
        CutSellDiagnostics.log("create_style_chosen", ["pace": newJob.style.pace.rawValue, "zoom": newJob.style.zoom.rawValue, "mode": newJob.mode.rawValue])
        // Let the style sheet / gallery finish closing before the progress screen opens.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            job = newJob
            Task { await run(newJob) }
        }
    }

    @MainActor
    private func run(_ current: CreateJob) async {
        do {
            switch current.source {
            case .recorded(let video):
                try await submitter.submit(
                    videos: [video], title: nil, mode: CutSubmitter.singleMode,
                    audioOverlap: false, resumeProjectID: nil, appState: appState
                )
            case .library(let assets):
                if current.mode == .combine {
                    var videos: [LocalVideo] = []
                    for (index, asset) in assets.enumerated() {
                        videos.append(try await export(asset, name: "Video \(index + 1)", index: index, count: assets.count))
                    }
                    try await submitter.submit(
                        videos: videos, title: nil, mode: CutSubmitter.multipleMode,
                        audioOverlap: false, resumeProjectID: nil, appState: appState
                    )
                } else {
                    // Single: one asset. Bulk: one separate cut per video, one after another.
                    let count = assets.count
                    for (index, asset) in assets.enumerated() {
                        let video = try await export(asset, name: "Video 1", index: index, count: count)
                        let lower = Double(index) / Double(count)
                        try await submitter.submit(
                            videos: [video], title: nil, mode: CutSubmitter.singleMode,
                            audioOverlap: false, resumeProjectID: nil, appState: appState,
                            progressRange: lower...(lower + 1 / Double(count)),
                            label: count > 1 ? "Cut \(index + 1) of \(count)" : ""
                        )
                    }
                }
            }
            job = nil
            onStarted()
        } catch let failure as GalleryExport.Failure {
            switch failure {
            case .unsupported: problem = .unsupportedMedia
            case .tooLarge(let bytes):
                problem = .tooLarge(
                    detail: "\(CreateLimits.sizeText(bytes)) / 1 GB limit",
                    sizeText: CreateLimits.sizeText(bytes)
                )
            case .unavailable: problem = .couldNotAdd
            }
        } catch {
            uploadFailed = true
        }
    }

    private func export(_ asset: PHAsset, name: String, index: Int, count: Int) async throws -> LocalVideo {
        submitter.announce(count > 1 ? "Getting video \(index + 1) of \(count) from Photos…" : "Getting your video from Photos…")
        return try await GalleryExport.export(asset, name: name)
    }
}

struct CreateJob: Identifiable {
    enum Source {
        case recorded(LocalVideo)
        case library([PHAsset])
    }

    let id = UUID()
    let mode: CreateMode
    let source: Source
    let style: StyleChoice
}

/// Shown while the videos upload and the cut starts.
struct SubmittingView: View {
    @ObservedObject var submitter: CutSubmitter
    let job: CreateJob

    var body: some View {
        ZStack {
            ScreenBackground()
            VStack(alignment: .leading, spacing: 0) {
                BrandLogo(width: 112).padding(.top, 8)
                Spacer()
                VStack(spacing: 16) {
                    Text(job.mode == .bulk ? "Uploading your videos" : "Uploading your video")
                        .font(CS.inter(22, .bold))
                        .foregroundStyle(.white)
                    GeometryReader { proxy in
                        ZStack(alignment: .leading) {
                            Capsule().fill(CS.fieldBackground)
                            Capsule().fill(CS.brandGradient)
                                .frame(width: max(8, proxy.size.width * submitter.progress))
                        }
                    }
                    .frame(height: 8)
                    Text(submitter.statusText.isEmpty ? "Getting ready…" : submitter.statusText)
                        .font(CS.inter(13))
                        .foregroundStyle(CS.textSecondary)
                        .multilineTextAlignment(.center)
                    Text("Keep CutSell open until the upload finishes.")
                        .font(CS.inter(12))
                        .foregroundStyle(CS.textMuted)
                }
                .padding(.horizontal, 24)
                Spacer()
                Spacer()
            }
            .padding(.horizontal, 16)
        }
        .preferredColorScheme(.dark)
        .interactiveDismissDisabled()
    }
}
