import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct EditorExtrasView: View {
    @ObservedObject var model: DraftEditorViewModel
    @State private var newText = ""
    @State private var pickerItem: PhotosPickerItem?
    @State private var isAddingMedia = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Text & overlays").font(.headline)

            HStack {
                TextField("Add text", text: $newText)
                    .textFieldStyle(.roundedBorder)
                Button("Add") {
                    let value = newText.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !value.isEmpty else { return }
                    newText = ""
                    Task { await model.addTextOverlay(text: value) }
                }
                .buttonStyle(.borderedProminent)
                .disabled(newText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }

            if !model.textOverlays.isEmpty {
                VStack(spacing: 8) {
                    ForEach(Array(model.textOverlays.enumerated()), id: \.offset) { _, item in
                        TextOverlayRow(model: model, item: item)
                    }
                }
            }

            PhotosPicker(selection: $pickerItem, matching: .any(of: [.images, .videos])) {
                Label(isAddingMedia ? "Adding overlay…" : "Add photo or video", systemImage: "photo.badge.plus")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(isAddingMedia)
            .onChange(of: pickerItem) { _, item in
                guard let item else { return }
                Task { await importOverlay(item) }
            }

            if !model.mediaOverlays.isEmpty {
                VStack(spacing: 8) {
                    ForEach(Array(model.mediaOverlays.enumerated()), id: \.offset) { _, item in
                        MediaOverlayRow(model: model, item: item)
                    }
                }
            }
        }
        .padding(.horizontal)
    }

    @MainActor
    private func importOverlay(_ item: PhotosPickerItem) async {
        isAddingMedia = true
        defer {
            isAddingMedia = false
            pickerItem = nil
        }
        do {
            let isVideo = item.supportedContentTypes.contains(where: { $0.conforms(to: .movie) })
            if isVideo {
                guard let imported = try await item.loadTransferable(type: ImportedVideoFile.self) else { return }
                await model.addMediaOverlay(fileURL: imported.url)
            } else {
                guard let imported = try await item.loadTransferable(type: ImportedImageFile.self) else { return }
                await model.addMediaOverlay(fileURL: imported.url)
            }
        } catch {
            model.errorMessage = error.localizedDescription
        }
    }
}

private struct TextOverlayRow: View {
    @ObservedObject var model: DraftEditorViewModel
    let item: [String: JSONValue]

    private var id: String { item["overlay_id"]?.stringValue ?? "" }
    private var text: String { item["text"]?.stringValue ?? "Text" }
    private var x: Double { item["x"]?.doubleValue ?? 0.5 }
    private var y: Double { item["y"]?.doubleValue ?? 0.2 }
    private var scale: Double { item["scale"]?.doubleValue ?? 1 }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "textformat")
            Text(text).lineLimit(1)
            Spacer()
            Menu {
                Button("Top") { Task { await model.updateTextOverlay(id: id, x: 0.5, y: 0.15) } }
                Button("Center") { Task { await model.updateTextOverlay(id: id, x: 0.5, y: 0.5) } }
                Button("Bottom") { Task { await model.updateTextOverlay(id: id, x: 0.5, y: 0.82) } }
                Divider()
                Button("Smaller") { Task { await model.updateTextOverlay(id: id, scale: max(0.5, scale - 0.2)) } }
                Button("Larger") { Task { await model.updateTextOverlay(id: id, scale: min(3, scale + 0.2)) } }
                Divider()
                Button("Remove", role: .destructive) { Task { await model.removeTextOverlay(id: id) } }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
        }
        .padding(10)
        .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityLabel("Text overlay \(text), position \(x), \(y)")
    }
}

private struct MediaOverlayRow: View {
    @ObservedObject var model: DraftEditorViewModel
    let item: [String: JSONValue]

    private var id: String { item["overlay_id"]?.stringValue ?? "" }
    private var kind: String { item["kind"]?.stringValue ?? "media" }
    private var width: Double { item["width"]?.doubleValue ?? 0.4 }
    private var muted: Bool { item["mute_audio"]?.boolValue ?? true }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: kind == "photo" ? "photo" : "video")
            Text(kind.capitalized + " overlay")
            Spacer()
            Menu {
                Button("Top left") { Task { await model.updateMediaOverlay(id: id, x: 0.25, y: 0.25) } }
                Button("Center") { Task { await model.updateMediaOverlay(id: id, x: 0.5, y: 0.5) } }
                Button("Bottom right") { Task { await model.updateMediaOverlay(id: id, x: 0.75, y: 0.75) } }
                Divider()
                Button("Smaller") { Task { await model.updateMediaOverlay(id: id, width: max(0.1, width - 0.1)) } }
                Button("Larger") { Task { await model.updateMediaOverlay(id: id, width: min(1, width + 0.1)) } }
                if kind == "video" {
                    Divider()
                    Button(muted ? "Unmute overlay" : "Mute overlay") {
                        Task { await model.updateMediaOverlay(id: id, muted: !muted) }
                    }
                }
                Divider()
                Button("Remove", role: .destructive) { Task { await model.removeMediaOverlay(id: id) } }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
        }
        .padding(10)
        .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }
}
