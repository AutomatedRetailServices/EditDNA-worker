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
    /// Chosen in the gallery; started once the gallery has finished closing.
    @State private var jobAfterGallery: CreateJob?
    @State private var reopenGalleryAfterJob = false
    @State private var problem: CreateProblem?
    @State private var failure: SubmitFailure?
    /// Bulk: cuts already started before a failure (not repeated on "Try again").
    @State private var startedInJob = 0

    enum SubmitFailure: Identifiable {
        /// The upload record is saved on the phone: Edits shows "Resume upload".
        case savedForResume
        /// Nothing was saved yet (e.g. no connection): the same videos can be sent again.
        case notSaved
        var id: Int { self == .savedForResume ? 1 : 2 }
    }

    var body: some View {
        ZStack {
            CreateCameraView(
                onBack: onBack,
                onUpload: { showingGallery = true },
                onRecorded: { recorded = $0 },
                isActive: !showingGallery && job == nil && jobAfterGallery == nil
            )
            BottomSheetOverlay(isPresented: Binding(
                get: { recorded != nil },
                set: { if !$0, let video = recorded { recorded = nil; discardRecording(video) } }
            )) {
                StyleSheetView(ctaTitle: CreateMode.single.ctaTitle(1)) { style in
                    guard let video = recorded else { return }
                    recorded = nil
                    launch(CreateJob(mode: .single, source: .recorded(video), style: style))
                }
            }
        }
        .fullScreenCover(isPresented: $showingGallery, onDismiss: {
            if let next = jobAfterGallery {
                jobAfterGallery = nil
                launch(next)
            }
        }) {
            GalleryPickerView(
                onCancel: { showingGallery = false },
                onCreate: { mode, assets, style in
                    jobAfterGallery = CreateJob(mode: mode, source: .library(assets), style: style)
                    showingGallery = false
                }
            )
        }
        .fullScreenCover(item: $job, onDismiss: {
            if reopenGalleryAfterJob {
                reopenGalleryAfterJob = false
                showingGallery = true
            }
        }) { current in
            SubmittingView(submitter: submitter, job: current)
                .fullScreenCover(item: $problem) { shown in
                    CreateProblemView(
                        problem: shown,
                        onBack: {
                            reopenGalleryAfterJob = true
                            problem = nil
                            job = nil
                        },
                        onShowFormats: { problem = .supportedFormats }
                    )
                }
                .fullScreenCover(item: $failure) { shown in
                    switch shown {
                    case .savedForResume:
                        ProblemScreen(
                            screenTitle: "Upload paused",
                            heading: "Something went wrong",
                            message: "Your video is saved on this iPhone. Open Edits to resume the upload.",
                            primaryTitle: "Go to Edits",
                            primaryAction: { failure = nil; job = nil; onStarted() }
                        )
                    case .notSaved:
                        ProblemScreen(
                            screenTitle: "Couldn’t start the cut",
                            heading: "Something went wrong",
                            message: "Check your connection and try again. Your video is still here.",
                            primaryTitle: "Try again",
                            primaryAction: {
                                failure = nil
                                Task { await run(current) }
                            },
                            secondaryTitle: "Cancel",
                            secondaryAction: {
                                failure = nil
                                job = nil
                                if case .recorded(let video) = current.source { discardRecording(video) }
                            }
                        )
                    }
                }
        }
    }

    private func launch(_ newJob: CreateJob) {
        CutSellDiagnostics.log("create_style_chosen", ["pace": newJob.style.pace.rawValue, "zoom": newJob.style.zoom.rawValue, "mode": newJob.mode.rawValue])
        startedInJob = 0
        job = newJob
        Task { await run(newJob) }
    }

    /// A recording the person closed without making a cut: remove it from the phone.
    private func discardRecording(_ video: LocalVideo) {
        try? FileManager.default.removeItem(at: video.url)
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
                    for (index, asset) in assets.enumerated() where index >= startedInJob {
                        let video = try await export(asset, name: "Video 1", index: index, count: count)
                        let lower = Double(index) / Double(count)
                        try await submitter.submit(
                            videos: [video], title: nil, mode: CutSubmitter.singleMode,
                            audioOverlap: false, resumeProjectID: nil, appState: appState,
                            progressRange: lower...(lower + 1 / Double(count)),
                            label: count > 1 ? "Cut \(index + 1) of \(count)" : ""
                        )
                        startedInJob = index + 1
                    }
                }
            }
            job = nil
            onStarted()
        } catch let exportFailure as GalleryExport.Failure {
            switch exportFailure {
            case .unsupported: problem = .unsupportedMedia
            case .tooLarge(let bytes):
                problem = .tooLarge(
                    detail: "\(CreateLimits.sizeText(bytes)) / 1 GB limit",
                    sizeText: CreateLimits.sizeText(bytes)
                )
            case .unavailable: problem = .couldNotAdd
            }
        } catch let submitError as CutSubmitError {
            failure = submitError.savedForResume ? .savedForResume : .notSaved
        } catch {
            failure = .notSaved
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
                BrandLogo().padding(.top, 8)
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
