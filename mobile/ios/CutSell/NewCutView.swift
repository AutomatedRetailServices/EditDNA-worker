import SwiftUI
import PhotosUI
import AVFoundation

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
                    PhotosPicker(
                        selection: $pickerItems,
                        maxSelectionCount: mode == .single ? 1 : 20,
                        matching: .videos
                    ) {
                        Label(selected.isEmpty ? "Choose video" : "Add videos", systemImage: "photo.on.rectangle")
                    }
                    .onChange(of: pickerItems) { _, items in Task { await load(items) } }

                    ForEach(Array(selected.enumerated()), id: \.element.id) { index, item in
                        HStack {
                            Image(systemName: "line.3.horizontal")
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
            .alert("Couldn’t create cut", isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )) { Button("OK", role: .cancel) {} } message: { Text(errorMessage ?? "") }
        }
    }

    private func load(_ items: [PhotosPickerItem]) async {
        do {
            var output: [LocalVideo] = []
            for (index, item) in items.enumerated() {
                guard let data = try await item.loadTransferable(type: Data.self) else { continue }
                let url = FileManager.default.temporaryDirectory
                    .appending(path: "cutsell-picked-\(UUID().uuidString)-\(index).mov")
                try data.write(to: url, options: .atomic)
                let asset = AVURLAsset(url: url)
                let duration = try await asset.load(.duration).seconds
                output.append(LocalVideo(url: url, name: "Video \(index + 1)", duration: max(0, duration)))
            }
            if mode == .single { selected = Array(output.prefix(1)) }
            else { selected = output }
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
                    contentType: "video/quicktime"
                ) { partProgress in
                    Task { @MainActor in
                        progress = base + partProgress / Double(selected.count)
                    }
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
}
