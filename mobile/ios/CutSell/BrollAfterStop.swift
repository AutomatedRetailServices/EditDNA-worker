import AVFoundation
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

// B-roll after STOP (PO 10 Nov 2026, Figma page "📱 App build 78 · 01 Inicio, Record y Upload", row E):
// E1 / E1b  "+ B-roll" switch on the Record camera (3023:324 / 3024:331)
// E3        the B-roll camera: short shots, nothing cut (3023:387)
// E4        "Ready to edit" summary (3023:439) -> "Choose your style" (3023:468, unchanged)
// The talking part is always recorded with Auto-pause as before; it is tagged clip_type "primary". Every shot taken
// or picked on E3 / E4 is tagged clip_type "broll" (the same tag as the gallery's per-clip "B-roll" switch).

/// The rules of the flow, without views (unit-tested).
enum BrollAfterStop {
    /// After STOP on the Record camera: "Choose your style" (switch off) or the B-roll camera (switch on).
    enum NextStep: Equatable { case chooseStyle, brollCamera }

    static func afterStop(brollAfter: Bool) -> NextStep { brollAfter ? .brollCamera : .chooseStyle }

    /// A shot recorded or picked on the B-roll camera / summary: always B-roll, never the voice.
    static func shot(_ video: LocalVideo) -> LocalVideo {
        var out = video
        out.isBroll = true
        out.isPrimaryVoice = false
        return out
    }

    /// The talking part: the voice, edited as today.
    static func talk(_ video: LocalVideo) -> LocalVideo {
        var out = video
        out.isBroll = false
        out.isPrimaryVoice = true
        return out
    }

    /// What goes to the cut, in this order: the voice first, then the B-roll shots (one project).
    static func uploadVideos(talk: LocalVideo, shots: [LocalVideo]) -> [LocalVideo] {
        [Self.talk(talk)] + shots.map(Self.shot)
    }

    /// "● Shot 3 · 0:03" while a shot records (N = shots already taken + 1).
    static func shotPill(taken: Int, seconds: Double) -> String {
        "● " + String(format: L10n.tr("Shot %d"), taken + 1) + " · " + clock(seconds)
    }

    /// "Talking · 0:24"
    static func talkingLine(seconds: Double) -> String {
        String(format: L10n.tr("Talking · %@"), clock(seconds))
    }

    /// "B-roll · 3 shots" ("B-roll · 1 shot")
    static func shotsTitle(_ count: Int) -> String {
        count == 1 ? L10n.tr("B-roll · 1 shot") : String(format: L10n.tr("B-roll · %d shots"), count)
    }

    static func clock(_ seconds: Double) -> String {
        let s = max(0, Int(seconds.rounded(.down)))
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

// MARK: - E1 / E1b: the switch on the Record camera

/// Off: "+ B-roll" (black 45 %, white 35 % border). On: "✓ B-roll after" (blue 3D) and, above it,
/// "After STOP you'll add product shots".
struct BrollAfterSwitch: View {
    @Binding var isOn: Bool

    var body: some View {
        VStack(spacing: 8) {
            if isOn {
                Text(L10n.tr("After STOP you'll add product shots"))
                    .font(CS.inter(12, .medium))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .frame(width: 250, height: 30)
                    .background(Color.black.opacity(0.55), in: Capsule())
                    .accessibilityIdentifier("brollAfterNotice")
            }
            Button {
                isOn.toggle()
            } label: {
                Group {
                    if isOn {
                        Text(L10n.tr("✓ B-roll after"))
                            .font(CS.inter(15, .bold))
                            .foregroundStyle(.white)
                            .frame(width: 150, height: 34)
                            .background(
                                Capsule().fill(LinearGradient(colors: [Color(fixed: 0x2F7BFF), Color(fixed: 0x0A55F0)],
                                                              startPoint: .top, endPoint: .bottom))
                                    .shadow(color: Color(fixed: 0x1769FF, opacity: 0.4), radius: 5, x: 0, y: 4)
                            )
                    } else {
                        Text(L10n.tr("+ B-roll"))
                            .font(CS.inter(15, .bold))
                            .foregroundStyle(.white)
                            .frame(width: 120, height: 34)
                            .background(Color.black.opacity(0.45), in: Capsule())
                            .overlay(Capsule().stroke(Color.white.opacity(0.35), lineWidth: 1))
                    }
                }
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(isOn ? .isSelected : [])
            .accessibilityIdentifier("brollAfterSwitch")
        }
    }
}

// MARK: - E3: pieces of the B-roll camera

/// Green "B-roll" label, top centre.
struct BrollCameraLabel: View {
    var body: some View {
        Text(L10n.tr("B-roll"))
            .font(CS.inter(16, .bold))
            .foregroundStyle(.white)
            .frame(width: 110, height: 34)
            .background(Color(fixed: 0x1FA971), in: Capsule())
            .accessibilityIdentifier("brollCameraLabel")
    }
}

/// Red "● Shot 3 · 0:03" while a shot records.
struct BrollShotPill: View {
    let text: String
    var body: some View {
        Text(verbatim: text)
            .font(CS.inter(13, .bold))
            .foregroundStyle(.white)
            .lineLimit(1)
            .padding(.horizontal, 14)
            .frame(minWidth: 150, alignment: .leading)
            .frame(height: 30)
            .background(Color(fixed: 0xE5484D), in: Capsule())
            .accessibilityIdentifier("brollShotPill")
    }
}

/// The "×" on a shot's thumbnail (#0B1020 circle).
private struct RemoveBadge: View {
    var size: CGFloat
    var body: some View {
        Text(verbatim: "×")
            .font(CS.inter(12, .bold))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(Color(fixed: 0x0B1020), in: Circle())
    }
}

/// First frame of a clip, for the thumbnails.
struct ClipThumbnail: View {
    let url: URL
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Color(fixed: 0x1A2236)
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            }
        }
        .task(id: url) { image = await Self.firstFrame(url) }
    }

    static func firstFrame(_ url: URL) async -> UIImage? {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 240, height: 320)
        guard let image = try? await generator.image(at: .zero).image else { return nil }
        return UIImage(cgImage: image)
    }
}

/// Shots taken on the B-roll camera: 54×72 thumbnails with a white border and "×", then a dashed "+" for another.
struct BrollShotsStrip: View {
    let shots: [LocalVideo]
    let onRemove: (Int) -> Void
    let onAdd: () -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(Array(shots.enumerated()), id: \.element.url) { index, shot in
                    ClipThumbnail(url: shot.url)
                        .frame(width: 54, height: 72)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(.white, lineWidth: 2))
                        .overlay(alignment: .topTrailing) {
                            Button { onRemove(index) } label: { RemoveBadge(size: 18) }
                                .buttonStyle(.plain)
                                .offset(x: 4, y: -6)
                                .accessibilityLabel(L10n.tr("Remove"))
                                .accessibilityIdentifier("brollShotRemove")
                        }
                }
                Button(action: onAdd) {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(.white, style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
                        .frame(width: 54, height: 72)
                        .overlay(Text(verbatim: "+").font(CS.inter(24, .bold)).foregroundStyle(.white))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.tr("Add a shot"))
                .accessibilityIdentifier("brollShotAdd")
            }
            .padding(.top, 8)
            .padding(.horizontal, 24)
        }
    }
}

/// Blue 3D "Done" (92×46) → summary.
struct BrollDoneButton: View {
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Text(L10n.tr("Done"))
                .font(CS.inter(16, .bold))
                .foregroundStyle(.white)
                .frame(width: 92, height: 46)
                .softPuff(.blue, in: Capsule())
        }
        .buttonStyle(BubblePress())
        .accessibilityIdentifier("brollDone")
    }
}

// MARK: - Picking clips you already have (Upload on E3, "+" on E4)

/// The iPhone's photo library, videos only, several at a time. Each picked clip is copied into CutSell's own folder.
struct BrollClipsPicker: UIViewControllerRepresentable {
    let onPicked: ([URL]) -> Void
    let onCancel: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration()
        configuration.filter = .videos
        configuration.selectionLimit = 0
        configuration.preferredAssetRepresentationMode = .current
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {
        context.coordinator.parent = self
    }

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        var parent: BrollClipsPicker
        init(_ parent: BrollClipsPicker) { self.parent = parent }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            guard !results.isEmpty else { parent.onCancel(); return }
            let parent = self.parent
            Task {
                var urls: [URL] = []
                for result in results {
                    if let url = await Self.copy(result.itemProvider) { urls.append(url) }
                }
                await MainActor.run { parent.onPicked(urls) }
            }
        }

        static func copy(_ provider: NSItemProvider) async -> URL? {
            await withCheckedContinuation { continuation in
                provider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { url, _ in
                    guard let url else { continuation.resume(returning: nil); return }
                    let ext = url.pathExtension.isEmpty ? "mov" : url.pathExtension
                    let target = FileManager.default.temporaryDirectory
                        .appendingPathComponent("broll-\(UUID().uuidString).\(ext)")
                    do {
                        try FileManager.default.copyItem(at: url, to: target)
                        continuation.resume(returning: target)
                    } catch {
                        continuation.resume(returning: nil)
                    }
                }
            }
        }
    }
}

extension LocalVideo {
    /// A clip picked from Photos for B-roll (already copied into CutSell's folder).
    static func brollClip(url: URL) async -> LocalVideo {
        let seconds = (try? await AVURLAsset(url: url).load(.duration).seconds) ?? 0
        let type = url.pathExtension.lowercased() == "mp4" ? "video/mp4" : "video/quicktime"
        return BrollAfterStop.shot(LocalVideo(url: url, name: url.lastPathComponent, duration: max(0, seconds),
                                              contentType: type))
    }
}

// MARK: - E4: "Ready to edit"

struct BrollSummaryView: View {
    let talk: LocalVideo
    /// Auto-pause was on for the talking part (then "Silences removed").
    let silencesRemoved: Bool
    @Binding var shots: [LocalVideo]
    let onBack: () -> Void
    let onAddMore: () -> Void
    let onContinue: () -> Void

    var body: some View {
        ZStack(alignment: .top) {
            HomeBackdrop()
            VStack(alignment: .leading, spacing: 0) {
                Button(action: onBack) {
                    Text(verbatim: "‹")
                        .font(CS.inter(22, .bold))
                        .foregroundStyle(Color(fixed: 0x0B1020))
                        .frame(width: 40, height: 40)
                        .softPuff(.white, in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.tr("Back"))
                .padding(.leading, 20)
                .padding(.top, 8)

                Text(L10n.tr("Ready to edit"))
                    .font(CS.inter(29, .heavy))
                    .foregroundStyle(Color(fixed: 0x0B1020))
                    .padding(.top, 16)
                    .padding(.leading, 24)

                ScrollView {
                    VStack(spacing: 20) {
                        videoCard
                        brollCard
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 18)
                    .padding(.bottom, 20)
                }

                Button(action: onContinue) {
                    Text(L10n.tr("Continue"))
                        .font(CS.inter(16, .bold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 52)
                        .softPuff(.blue, in: Capsule())
                }
                .buttonStyle(BubblePress())
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
                .accessibilityIdentifier("brollSummaryContinue")
            }
        }
        .accessibilityIdentifier("brollSummary")
    }

    private var videoCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(L10n.tr("Your video"))
                .font(CS.inter(13, .bold))
                .foregroundStyle(Color(fixed: 0x6B7385))
            HStack(spacing: 16) {
                ClipThumbnail(url: talk.url)
                    .frame(width: 64, height: 94)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                VStack(alignment: .leading, spacing: 6) {
                    Text(verbatim: BrollAfterStop.talkingLine(seconds: talk.duration))
                        .font(CS.inter(16, .bold))
                        .foregroundStyle(Color(fixed: 0x0B1020))
                    if silencesRemoved {
                        Text(L10n.tr("Silences removed"))
                            .font(CS.inter(13, .medium))
                            .foregroundStyle(Color(fixed: 0x6B7385))
                    }
                }
                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 16)
        .padding(.bottom, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .modifier(SummaryCard())
    }

    private var brollCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(verbatim: BrollAfterStop.shotsTitle(shots.count))
                .font(CS.inter(13, .bold))
                .foregroundStyle(Color(fixed: 0x6B7385))
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(Array(shots.enumerated()), id: \.element.url) { index, shot in
                        ClipThumbnail(url: shot.url)
                            .frame(width: 64, height: 94)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                            .overlay(alignment: .topTrailing) {
                                Button { shots.remove(at: index) } label: { RemoveBadge(size: 20) }
                                    .buttonStyle(.plain)
                                    .offset(x: 4, y: -6)
                                    .accessibilityLabel(L10n.tr("Remove"))
                                    .accessibilityIdentifier("brollSummaryRemove")
                            }
                    }
                    Button(action: onAddMore) {
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Color(fixed: 0xF3F5F9))
                            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .stroke(Color(fixed: 0xE6E9EF), lineWidth: 1))
                            .frame(width: 64, height: 94)
                            .overlay(Text(verbatim: "+").font(CS.inter(26, .bold)).foregroundStyle(Color(fixed: 0x6B7385)))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(L10n.tr("Add a shot"))
                    .accessibilityIdentifier("brollSummaryAdd")
                }
                .padding(.top, 12)
                .padding(.trailing, 6)
            }
            Text(L10n.tr("We'll place them over your voice at the right moments."))
                .font(CS.inter(13, .medium))
                .foregroundStyle(Color(fixed: 0x6B7385))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 14)
        }
        .padding(.horizontal, 18)
        .padding(.top, 16)
        .padding(.bottom, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .modifier(SummaryCard())
    }
}

/// The E4 cards: white → #F7F9FF, radius 24, a #D3DDF5 base 3 pt below and a soft shadow.
private struct SummaryCard: ViewModifier {
    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: 24, style: .continuous)
        return content
            .background {
                shape.fill(LinearGradient(colors: [Color(fixed: 0xFFFFFF), Color(fixed: 0xF7F9FF)],
                                          startPoint: .top, endPoint: .bottom))
                    .shadow(color: Color(fixed: 0xD3DDF5), radius: 0, x: 0, y: 3)
                    .shadow(color: Color(fixed: 0x0D1433, opacity: 0.08), radius: 12, x: 0, y: 10)
            }
    }
}
