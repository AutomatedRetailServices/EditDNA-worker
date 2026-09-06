import SwiftUI
import PhotosUI
import AVFoundation
import UniformTypeIdentifiers

struct NewCutView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case single = "Single clip"
        case multiple = "Multiple clips"
        var id: String { rawValue }
    }

    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var mode: Mode
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var selected: [LocalVideo]
    @State private var title: String
    @State private var audioOverlap: Bool
    @State private var isSubmitting = false
    @State private var progress = 0.0
    @State private var statusText = ""
    @State private var errorMessage: String?
    @State private var showCamera = false
    @State private var isImportingFromPhotos = false
    @State private var importStatusText = ""

    private let resumeProjectID: String?
    private let maxMultiClipSources = 10

    init(pending: PendingCutRecord? = nil) {
        resumeProjectID = pending?.projectID
        _mode = State(initialValue: Mode(rawValue: pending?.mode ?? "") ?? .single)
        _selected = State(initialValue: (pending?.videos ?? []).map {
            LocalVideo(url: $0.fileURL, name: $0.name, duration: $0.duration, contentType: $0.contentType)
        })
        _title = State(initialValue: pending?.title ?? "")
        _audioOverlap = State(initialValue: pending?.audioOverlap ?? false)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Create") {
                    Picker("Mode", selection: $mode) {
                        ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(resumeProjectID != nil || isImportingFromPhotos)
                    TextField("Project name (optional)", text: $title)
                        .disabled(resumeProjectID != nil)
                    if resumeProjectID != nil {
                        Label("Resuming interrupted upload", systemImage: "arrow.clockwise.icloud")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Footage") {
                    Button { showCamera = true } label: {
                        Label("Record in CutSell", systemImage: "camera.fill")
                    }
                    .disabled(resumeProjectID != nil || isImportingFromPhotos || (mode == .multiple && selected.count >= maxMultiClipSources))

                    PhotosPicker(
                        selection: $pickerItems,
                        maxSelectionCount: mode == .single ? 1 : max(1, maxMultiClipSources - selected.count),
                        matching: .videos
                    ) {
                        Label(selected.isEmpty ? "Choose from Photos" : "Add from Photos", systemImage: "photo.on.rectangle")
                    }
                    .disabled(resumeProjectID != nil || isImportingFromPhotos || (mode == .multiple && selected.count >= maxMultiClipSources))
                    .onChange(of: pickerItems) { _, items in Task { await load(items) } }

                    if isImportingFromPhotos {
                        HStack(spacing: 10) {
                            ProgressView()
                            Text(importStatusText)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if selected.isEmpty && !isImportingFromPhotos {
                        Text("Record here or choose existing footage. Both use the same CutSell AI edit pipeline.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if mode == .multiple {
                        Text("\(selected.count) of \(maxMultiClipSources) clips selected")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    ForEach(Array(selected.enumerated()), id: \.element.id) { index, item in
                        HStack {
                            Image(systemName: mode == .multiple ? "line.3.horizontal" : "video.fill")
                                .foregroundStyle(.secondary)
                            VStack(alignment: .leading) {
                                Text(item.name).lineLimit(1)
                                Text(duration(item.duration)).font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text("\(index + 1)").font(.caption.monospacedDigit())
                        }
                    }
                    .onMove { source, destination in
                        guard resumeProjectID == nil && !isImportingFromPhotos else { return }
                        selected.move(fromOffsets: source, toOffset: destination)
                    }
                    .onDelete {
                        guard resumeProjectID == nil && !isImportingFromPhotos else { return }
                        selected.remove(atOffsets: $0)
                    }
                }

                if mode == .multiple {
                    Section("Cut behavior") {
                        Toggle("Natural audio overlap", isOn: $audioOverlap)
                            .disabled(resumeProjectID != nil)
                        Text("Off is the safest default. You can change audio later in the editor.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                if isSubmitting {
                    Section {
                        ProgressView(value: progress)
                        Text(statusText).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .environment(\.editMode, .constant(mode == .multiple && resumeProjectID == nil && !isImportingFromPhotos ? .active : .inactive))
            .navigationTitle(resumeProjectID == nil ? "New Cut" : "Resume Cut")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button(resumeProjectID == nil ? "Create" : "Resume") { Task { await submit() } }
                        .disabled(selected.isEmpty || isSubmitting || isImportingFromPhotos)
                }
            }
            .fullScreenCover(isPresented: $showCamera) {
                CameraCaptureView { url in Task { await addRecording(url) } }
            }
            .alert("Couldn’t create cut", isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )) { Button("OK", role: .cancel) {} } message: { Text(errorMessage ?? "") }
        }
    }

    private func addRecording(_ url: URL) async {
        do {
            let asset = AVURLAsset(url: url)
            let seconds = try await asset.load(.duration).seconds
            let clip = LocalVideo(
                url: url,
                name: mode == .single ? "Recorded clip" : "Take \(selected.count + 1)",
                duration: max(0, seconds),
                contentType: "video/quicktime"
            )
            if mode == .single {
                selected = [clip]
            } else if selected.count < maxMultiClipSources {
                selected.append(clip)
            } else {
                errorMessage = "A CutSell multi-clip project can contain up to \(maxMultiClipSources) source clips."
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    @MainActor
    private func load(_ items: [PhotosPickerItem]) async {
        let available = mode == .single ? 1 : max(0, maxMultiClipSources - selected.count)
        let requested = Array(items.prefix(available))
        guard !requested.isEmpty else {
            pickerItems = []
            return
        }

        isImportingFromPhotos = true
        defer {
            isImportingFromPhotos = false
            importStatusText = ""
            pickerItems = []
        }

        var failures = 0
        for (index, item) in requested.enumerated() {
            importStatusText = "Retrieving \(index + 1) of \(requested.count) from Photos / iCloud…"
            do {
                guard let imported = try await item.loadTransferable(type: ImportedVideoFile.self) else {
                    failures += 1
                    continue
                }
                let asset = AVURLAsset(url: imported.url)
                let seconds = try await asset.load(.duration).seconds
                let type = item.supportedContentTypes.first(where: { $0.conforms(to: .movie) })
                let contentType = type?.preferredMIMEType ?? videoContentType(for: imported.url)
                let clip = LocalVideo(
                    url: imported.url,
                    name: "Video \(selected.count + 1)",
                    duration: max(0, seconds),
                    contentType: contentType
                )
                if mode == .single {
                    selected = [clip]
                    break
                }
                if selected.count < maxMultiClipSources {
                    selected.append(clip)
                }
            } catch {
                failures += 1
            }
        }

        if failures > 0 {
            errorMessage = failures == 1
                ? "One selected video could not be retrieved. The videos already loaded were kept."
                : "\(failures) selected videos could not be retrieved. The videos already loaded were kept."
        }
    }

    private func submit() async {
        guard let session = appState.session else { return }
        isSubmitting = true
        progress = 0
        var transientPreparedFiles: [URL] = []
        var pendingSaved = resumeProjectID != nil

        do {
            let uploadVideos: [LocalVideo]
            if resumeProjectID != nil {
                uploadVideos = selected
                statusText = "Resuming upload…"
            } else {
                var preparedVideos: [LocalVideo] = []
                for (index, video) in selected.enumerated() {
                    statusText = "Preparing \(index + 1) of \(selected.count)…"
                    let prepared = try await VideoPreparation.shared.prepare(
                        fileURL: video.url,
                        contentType: video.contentType
                    )
                    if prepared.wasTranscoded { transientPreparedFiles.append(prepared.url) }
                    preparedVideos.append(LocalVideo(
                        url: prepared.url,
                        name: video.name,
                        duration: video.duration,
                        contentType: prepared.contentType
                    ))
                }
                uploadVideos = preparedVideos
            }

            let projectID: String
            if let resumeProjectID {
                projectID = resumeProjectID
            } else {
                statusText = "Creating project…"
                let project = try await appState.createProject(
                    title: title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : title
                )
                projectID = project.projectID
                let pending = PendingCutRecord(
                    projectID: projectID,
                    title: project.title,
                    mode: mode.rawValue,
                    audioOverlap: audioOverlap,
                    videos: uploadVideos.map {
                        PendingCutVideo(
                            filePath: $0.url.path,
                            name: $0.name,
                            duration: $0.duration,
                            contentType: $0.contentType
                        )
                    },
                    createdAt: Date()
                )
                await PendingCutStore.shared.save(pending)
                pendingSaved = true
                transientPreparedFiles.removeAll()
            }

            var sources: [SourceInput] = []
            for (index, video) in uploadVideos.enumerated() {
                statusText = "Uploading \(index + 1) of \(uploadVideos.count)…"
                let base = Double(index) / Double(uploadVideos.count)
                let result = try await MultipartUploadManager.shared.upload(
                    fileURL: video.url,
                    projectID: projectID,
                    session: session,
                    contentType: video.contentType
                ) { partProgress in
                    Task { @MainActor in progress = base + partProgress / Double(uploadVideos.count) }
                }
                sources.append(SourceInput(
                    originalName: video.url.lastPathComponent,
                    uri: result.sourceURI,
                    sourceOrder: index,
                    durationSec: video.duration
                ))
            }

            statusText = "Starting AI edit…"
            struct Body: Encodable {
                let project_id: String
                let user_id: String
                let sources: [SourceInput]
                let language_hint: String?
                let audio_overlap: Bool
            }
            let _: FlowBSubmitResponse = try await APIClient.shared.request(
                "/v1/flow-b/jobs",
                method: "POST",
                body: Body(
                    project_id: projectID,
                    user_id: session.userID,
                    sources: sources,
                    language_hint: nil,
                    audio_overlap: audioOverlap
                )
            )
            await PendingCutStore.shared.remove(projectID: projectID, deleteLocalFiles: true)
            progress = 1
            try await appState.refreshProjects()
            dismiss()
        } catch {
            if !pendingSaved {
                for url in transientPreparedFiles { try? FileManager.default.removeItem(at: url) }
            }
            errorMessage = error.localizedDescription
            isSubmitting = false
        }
    }

    private func videoContentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "mp4": return "video/mp4"
        case "m4v": return "video/x-m4v"
        case "webm": return "video/webm"
        default: return "video/quicktime"
        }
    }

    private func duration(_ seconds: Double) -> String {
        let value = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", value / 60, value % 60)
    }
}

struct LocalVideo: Identifiable, Hashable {
    let id = UUID()
    let url: URL
    let name: String
    let duration: Double
    let contentType: String
}
