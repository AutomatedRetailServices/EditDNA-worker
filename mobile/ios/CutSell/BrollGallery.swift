import Photos
import SwiftUI
import UIKit

// B-roll v2 (PO 11 Nov 2026, Figma row F): with "+ B-roll" on, Upload on the Record camera opens the gallery in
// 2 steps. F2 (3026:366) "Your talking video" · "1 of 2": one or more videos, numbered in the order tapped (joined
// in that order). F3 (3026:416) "B-roll shots" · "2 of 2": videos AND photos, numbered 1, 2, 3, all marked
// "✓ B-roll". "Add N shots" → F4 "Ready to edit"; "Record shots instead" → the B-roll camera.
// ✕: step 1 closes the gallery; step 2 goes back to step 1 keeping the videos already picked (PO 11 Nov).

/// The rules of the 2-step pick, without views (unit-tested).
struct BrollGalleryPick: Equatable {
    enum Step: Equatable { case talking, shots }

    var step: Step = .talking
    /// Photos ids in the order they were tapped.
    var talking: [String] = []
    var shots: [String] = []

    /// Tap on a tile: picks it (next number) or un-picks it (the numbers after it move up).
    mutating func toggle(_ id: String) {
        switch step {
        case .talking: Self.toggle(id, in: &talking)
        case .shots: Self.toggle(id, in: &shots)
        }
    }

    /// The number shown on a tile (1, 2, 3…), nil when not picked.
    func number(of id: String) -> Int? {
        let list = step == .talking ? talking : shots
        return list.firstIndex(of: id).map { $0 + 1 }
    }

    var canGoOn: Bool { step == .talking ? !talking.isEmpty : !shots.isEmpty }

    /// "Next: add B-roll" → step 2.
    mutating func next() { if step == .talking, !talking.isEmpty { step = .shots } }

    /// ✕: step 2 → step 1 (keeps the talking videos); step 1 → closes (returns true).
    mutating func close() -> Bool {
        if step == .shots { step = .talking; return false }
        return true
    }

    private static func toggle(_ id: String, in list: inout [String]) {
        if let index = list.firstIndex(of: id) { list.remove(at: index) } else { list.append(id) }
    }
}

/// Photos for the 2-step pick: videos (step 1) or videos and photos (step 2), newest first.
@MainActor
final class BrollMediaLibrary: ObservableObject {
    @Published private(set) var status: PHAuthorizationStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @Published private(set) var videos: [PHAsset] = []
    @Published private(set) var media: [PHAsset] = []
    let imageManager = PHCachingImageManager()

    func start() async {
        if status == .notDetermined { status = await PHPhotoLibrary.requestAuthorization(for: .readWrite) }
        guard status == .authorized || status == .limited else { return }
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        videos = Self.list(PHAsset.fetchAssets(with: .video, options: options))
        let both = PHFetchOptions()
        both.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        both.predicate = NSPredicate(format: "mediaType == %d OR mediaType == %d",
                                     PHAssetMediaType.video.rawValue, PHAssetMediaType.image.rawValue)
        media = Self.list(PHAsset.fetchAssets(with: both))
    }

    private static func list(_ result: PHFetchResult<PHAsset>) -> [PHAsset] {
        var out: [PHAsset] = []
        out.reserveCapacity(result.count)
        result.enumerateObjects { asset, _, _ in out.append(asset) }
        return out
    }

    /// A photo copied into CutSell's folder as a JPEG (at most 1920 px long side), ready to upload as B-roll.
    static func exportPhoto(_ asset: PHAsset) async throws -> LocalVideo {
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .highQualityFormat
        options.version = .current
        let data: Data = try await withCheckedThrowingContinuation { continuation in
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: options) { data, _, _, _ in
                if let data { continuation.resume(returning: data) }
                else { continuation.resume(throwing: GalleryExport.Failure.unavailable) }
            }
        }
        guard let image = UIImage(data: data) else { throw GalleryExport.Failure.unsupported }
        let longSide = max(image.size.width, image.size.height)
        let scale = min(1, 1920 / max(1, longSide))
        let size = CGSize(width: (image.size.width * scale).rounded(), height: (image.size.height * scale).rounded())
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let resized = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
        guard let jpeg = resized.jpegData(compressionQuality: 0.88) else { throw GalleryExport.Failure.unsupported }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("broll-\(UUID().uuidString).jpg")
        try jpeg.write(to: url)
        return BrollAfterStop.shot(LocalVideo(url: url, name: url.lastPathComponent, duration: 0, contentType: "image/jpeg"))
    }
}

struct BrollGalleryFlow: View {
    /// The picks, kept by the camera so "‹" on "Ready to edit" comes back to them.
    @Binding var pick: BrollGalleryPick
    let onClose: () -> Void
    /// "Add N shots": the talking videos (in order) and the shots (in order).
    let onAdd: ([LocalVideo], [LocalVideo]) -> Void
    /// "Record shots instead": the talking videos; the B-roll camera opens.
    let onRecordInstead: ([LocalVideo]) -> Void

    @StateObject private var library = BrollMediaLibrary()
    @State private var working = false
    @State private var failed = false

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 6), count: 3)

    var body: some View {
        VStack(spacing: 0) {
            header
            Rectangle().fill(Color(fixed: 0xE6E9EF)).frame(height: 1)
            if library.status == .denied || library.status == .restricted {
                PermissionScreen(kind: .photos, deniedBefore: true, onClose: onClose) {}
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 6) {
                        ForEach(pick.step == .talking ? library.videos : library.media, id: \.localIdentifier) { asset in
                            BrollGalleryTile(asset: asset, manager: library.imageManager,
                                             number: pick.number(of: asset.localIdentifier),
                                             brollChip: pick.step == .shots)
                                .onTapGesture { if !working { pick.toggle(asset.localIdentifier) } }
                        }
                    }
                    .padding(.horizontal, 6)
                    .padding(.top, 8)
                    .padding(.bottom, 220)
                }
            }
        }
        .background(Color(fixed: 0xFFFFFF).ignoresSafeArea())
        .overlay(alignment: .bottom) { bottomPanel }
        .task { await library.start() }
        .alert(L10n.tr("Couldn’t add this clip"), isPresented: $failed) {
            Button(L10n.tr("OK"), role: .cancel) {}
        }
        .accessibilityIdentifier(pick.step == .talking ? "brollGalleryTalking" : "brollGalleryShots")
    }

    private var header: some View {
        ZStack {
            Text(L10n.tr(pick.step == .talking ? "Your talking video" : "B-roll shots"))
                .font(CS.inter(22, .semibold))
                .foregroundStyle(Color(fixed: 0x0B1020))
            HStack {
                Button {
                    if pick.close() { onClose() }
                } label: {
                    Text(verbatim: "✕")
                        .font(CS.inter(18, .semibold))
                        .foregroundStyle(Color(fixed: 0x0B1020))
                        .frame(width: 36, height: 36, alignment: .leading)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.tr(pick.step == .talking ? "Close" : "Back"))
                .accessibilityIdentifier("brollGalleryClose")
                Spacer()
                Text(L10n.tr(pick.step == .talking ? "1 of 2" : "2 of 2"))
                    .font(CS.inter(12, .bold))
                    .foregroundStyle(Color(fixed: 0x1769FF))
                    .frame(width: 64, height: 24)
                    .background(Color(fixed: 0xEEF3FF), in: Capsule())
            }
            .padding(.horizontal, 20)
        }
        .frame(height: 52)
        .padding(.top, 6)
    }

    private var bottomPanel: some View {
        let count = pick.step == .talking ? pick.talking.count : pick.shots.count
        return VStack(spacing: 0) {
            Text(verbatim: BrollAfterStop.selectedText(count, shots: pick.step == .shots))
                .font(CS.inter(14, .regular))
                .foregroundStyle(Color(fixed: 0x6B7385))
                .padding(.top, 14)
            Text(L10n.tr(pick.step == .talking ? "Pick one or more. We'll join them in this order."
                                               : "Tap them in the order they'll appear: 1, 2, 3."))
                .font(CS.inter(13, .medium))
                .foregroundStyle(Color(fixed: 0x6B7385))
                .multilineTextAlignment(.center)
                .padding(.top, pick.step == .talking ? 18 : 14)
                .padding(.horizontal, 20)
            if pick.step == .shots {
                Button { recordInstead() } label: {
                    Text(L10n.tr("Record shots instead"))
                        .font(CS.inter(13, .bold))
                        .foregroundStyle(Color(fixed: 0x1769FF))
                }
                .buttonStyle(.plain)
                .disabled(working)
                .padding(.top, 10)
                .accessibilityIdentifier("brollRecordInstead")
            }
            Spacer(minLength: 0)
            Button { primary() } label: {
                ZStack {
                    if working {
                        BrandRing(size: 22)
                    } else {
                        Text(pick.step == .talking ? L10n.tr("Next: add B-roll") : BrollAfterStop.addShotsTitle(count))
                            .font(CS.inter(16, .bold))
                            .foregroundStyle(.white)
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 52)
                .softPuff(.blue, in: Capsule())
            }
            .buttonStyle(BubblePress())
            .disabled(!pick.canGoOn || working)
            .opacity(pick.canGoOn ? 1 : 0.45)
            .padding(.horizontal, 20)
            .padding(.bottom, 10)
            .accessibilityIdentifier(pick.step == .talking ? "brollGalleryNext" : "brollGalleryAdd")
        }
        .frame(maxWidth: .infinity)
        .frame(height: 204)
        .background(
            Color(fixed: 0xFFFFFF)
                .shadow(color: Color(fixed: 0x0D1433, opacity: 0.08), radius: 8, x: 0, y: -6)
                .ignoresSafeArea(edges: .bottom)
        )
    }

    private func primary() {
        if pick.step == .talking { pick.next(); return }
        working = true
        Task {
            defer { working = false }
            do {
                let talks = try await exportTalks()
                var shots: [LocalVideo] = []
                for id in pick.shots {
                    guard let asset = library.media.first(where: { $0.localIdentifier == id }) else { continue }
                    if asset.mediaType == .image {
                        shots.append(try await BrollMediaLibrary.exportPhoto(asset))
                    } else {
                        shots.append(BrollAfterStop.shot(try await GalleryExport.export(asset, name: nil)))
                    }
                }
                onAdd(talks, shots)
            } catch {
                failed = true
            }
        }
    }

    private func recordInstead() {
        working = true
        Task {
            defer { working = false }
            do { onRecordInstead(try await exportTalks()) } catch { failed = true }
        }
    }

    private func exportTalks() async throws -> [LocalVideo] {
        var out: [LocalVideo] = []
        for id in pick.talking {
            guard let asset = library.videos.first(where: { $0.localIdentifier == id }) else { continue }
            var video = try await GalleryExport.export(asset, name: nil)
            video.albumHint = GalleryExport.albumName(of: asset)
            out.append(BrollAfterStop.talk(video))
        }
        return out
    }
}

/// A gallery tile of F2 / F3: 122×158.6 (radius 12), the duration (or "PHOTO") bottom right, the order number top
/// right, a #1769FF 2 pt border when picked and, on F3, "✓ B-roll" bottom left.
struct BrollGalleryTile: View {
    let asset: PHAsset
    let manager: PHCachingImageManager
    let number: Int?
    let brollChip: Bool
    @State private var image: UIImage?
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        Color(fixed: 0x1A2236)
            .aspectRatio(122 / 158.6, contentMode: .fit)
            .overlay {
                if let image { Image(uiImage: image).resizable().scaledToFill() }
            }
            .overlay(alignment: .bottomTrailing) {
                Text(asset.mediaType == .image ? L10n.tr("PHOTO") : CreateLimits.durationText(asset.duration))
                    .font(CS.inter(12, .semibold))
                    .foregroundStyle(.white)
                    .shadow(color: .black.opacity(0.6), radius: 2)
                    .padding(8)
            }
            .overlay(alignment: .bottomLeading) {
                if brollChip, number != nil {
                    Text(L10n.tr("✓ B-roll"))
                        .font(CS.inter(11, .bold))
                        .foregroundStyle(.white)
                        .frame(width: 70, height: 24)
                        .background(Color(fixed: 0x1769FF), in: Capsule())
                        .padding(6)
                }
            }
            .overlay(alignment: .topTrailing) {
                if let number { BrollNumberBadge(number: number).padding(6) }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay {
                if number != nil {
                    RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color(fixed: 0x1769FF), lineWidth: 2)
                }
            }
            .contentShape(Rectangle())
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(asset.mediaType == .image ? L10n.tr("Photo") : L10n.tr("Video, \(CreateLimits.durationText(asset.duration))"))
            .accessibilityAddTraits(number == nil ? .isButton : [.isButton, .isSelected])
            .task(id: asset.localIdentifier) { await load() }
    }

    private func load() async {
        let side = 160 * displayScale
        let options = PHImageRequestOptions()
        options.deliveryMode = .opportunistic
        options.isNetworkAccessAllowed = true
        options.resizeMode = .fast
        let stream = AsyncStream<UIImage> { continuation in
            let id = manager.requestImage(for: asset, targetSize: CGSize(width: side, height: side * 1.3),
                                          contentMode: .aspectFill, options: options) { picture, info in
                if let picture { continuation.yield(picture) }
                if !((info?[PHImageResultIsDegradedKey] as? Bool) ?? false) { continuation.finish() }
            }
            continuation.onTermination = { [manager] _ in manager.cancelImageRequest(id) }
        }
        for await picture in stream { image = picture }
    }
}
