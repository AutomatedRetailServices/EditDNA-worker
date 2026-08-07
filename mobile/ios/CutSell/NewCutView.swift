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
    @State private var mode: Mode = .single
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var selected: [LocalVideo] = []
    @State private var title = ""
    @State private var audioOverlap = false
    @State private var isSubmitting = false
    @State private var progress = 0.0
    @State private var statusText = ""
    @State private var errorMessage: String?
    @State private var showCamera = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Create") {
                    Picker("Mode", selection: $mode) {
                        ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    TextField("Project name (optional)", text: $title)
                }

                Section("Footage") {
                    Button { showCamera = true } label: {
                        Label("Record in CutSell", systemImage: "camera.fill")
                    }

                    PhotosPicker(
                        selection: $pickerItems,
                        maxSelectionCount: mode == .single ? 1 : 20,
                        matching: .videos
                    ) {
                        Label(selected.isEmpty ? "Choose from Photos" : "Add from Photos", systemImage: "photo.on.rectangle")
                    }
                    .onChange(of: pickerItems) { _, items in Task { await load(items) } }

                    if selected.isEmpty {
                        Text("Record here or choose existing footage. Both use the same CutSell AI edit pipeline.")
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
                    .onMove { source, destination in selected.move(fromOffsets: source, toOffset: destination) }
                    .onDelete { selected.remove(atOffsets: $0) }
                }

                if mode == .multiple {
                    Section("Cut behavior") {
                        Toggle("Natural audio overlap", isOn: $audioOverlap)
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
            .environment(\.editMode, .constant(mode == .multiple ? .active : .inactive))
            .navigationTitle("New Cut")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") { Task { await submit() } }
                        .disabled(selected.isEmpty || isSubmitting)
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
            if mode == .single { selected = [clip] } else { selected.append(clip) }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func load(_ items: [PhotosPickerItem]) async {
        do {
            var output: [LocalVideo] = []
            for (index, item) in items.enumerated() {
                guard let imported = try await item.loadTransferable(type: ImportedVideoFile.self) else { continue }
                let asset = AVURLAsset(url: imported.url)
                let seconds = try await asset.load(.duration).seconds
                let type = item.supportedContentTypes.first(where: { $0.conforms(to: .movie) })
                let contentType = type?.preferredMIMEType ?? videoContentType(for: imported.url)
                output.append(LocalVideo(
                    url: imported.url,
                    name: "Video \(index + 1)",
                    duration: max(0, seconds),
                    contentType: contentType
                ))
            }
            if mode == .single { selected = Array(output.prefix(1)) } else { selected.append(contentsOf: output) }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func submit() async {
        guard let session = appState.session else { return }
        isSubmitting = true
        progress = 0
        do {
            statusText = "Creating project…"
            let project = try await appState.createProject(title: title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : title)
            var sources: [SourceInput] = []
            for (index, video) in selected.enumerated() {
                statusText = "Uploading \(index + 1) of \(selected.count)…"
                let base = Double(index) / Double(selected.count)
                let result = try await MultipartUploadManager.shared.upload(
                    fileURL: video.url,
                    projectID: project.projectID,
                    session: session,
                    contentType: video.contentType
                ) { partProgress in
                    Task { @MainActor in progress = base + partProgress / Double(selected.count) }
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
                    project_id: project.projectID,
                    user_id: session.userID,
                    sources: sources,
                    language_hint: nil,
                    audio_overlap: audioOverlap
                )
            )
            progress = 1
            try await appState.refreshProjects()
            dismiss()
        } catch {
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
