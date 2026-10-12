import Photos
import SwiftUI
import UIKit

/// How the chosen videos become cuts (Figma 03 Create gallery: 69:90, 473:77, 473:120).
enum CreateMode: String, CaseIterable, Hashable {
    case single, bulk, combine

    var title: String {
        switch self {
        case .single: return "Single"
        case .bulk: return "Bulk"
        case .combine: return "Combine"
        }
    }

    var helper: String {
        switch self {
        case .single: return "One video → one edit."
        case .bulk: return "Each video becomes a separate edit."
        case .combine: return "One edit in the order you selected."
        }
    }

    func selectedText(_ count: Int) -> String {
        let noun = self == .combine ? "clip" : "video"
        return "\(count) \(noun)\(count == 1 ? "" : "s") selected"
    }

    func ctaTitle(_ count: Int) -> String {
        switch self {
        case .single: return "Create 1 cut"
        case .bulk: return count == 1 ? "Create 1 cut" : "Create \(max(count, 0)) cuts"
        case .combine: return "Combine \(max(count, 0)) clips"
        }
    }
}

/// Reads the iPhone's video library (Photos) for the gallery grid.
@MainActor
final class GalleryLibrary: NSObject, ObservableObject, PHPhotoLibraryChangeObserver {
    @Published private(set) var status: PHAuthorizationStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @Published private(set) var assets: [PHAsset] = []
    let imageManager = PHCachingImageManager()
    private var observing = false

    func start() async {
        if status == .notDetermined {
            status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        }
        guard status == .authorized || status == .limited else { return }
        if !observing {
            PHPhotoLibrary.shared().register(self)
            observing = true
        }
        reload()
    }

    func requestAccess() async {
        status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        await start()
    }

    func reload() {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        let result = PHAsset.fetchAssets(with: .video, options: options)
        var list: [PHAsset] = []
        list.reserveCapacity(result.count)
        result.enumerateObjects { asset, _, _ in list.append(asset) }
        assets = list
    }

    nonisolated func photoLibraryDidChange(_ changeInstance: PHChange) {
        Task { @MainActor in
            self.status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
            self.reload()
        }
    }

    func stop() {
        if observing {
            PHPhotoLibrary.shared().unregisterChangeObserver(self)
            observing = false
        }
    }
}

/// Copies a library video into the app so it can be uploaded (downloads it from iCloud if needed).
enum GalleryExport {
    enum Failure: Error {
        case unsupported
        case tooLarge(Int64)
        case unavailable
    }

    static func export(_ asset: PHAsset, name: String) async throws -> LocalVideo {
        let resources = PHAssetResource.assetResources(for: asset)
        guard let resource = resources.first(where: { $0.type == .fullSizeVideo })
            ?? resources.first(where: { $0.type == .video }) else {
            throw Failure.unavailable
        }
        let ext = (resource.originalFilename as NSString).pathExtension.lowercased()
        guard CreateLimits.supportedExtensions.contains(ext) else { throw Failure.unsupported }
        let destination = try persistentMediaURL(extension: ext)
        let options = PHAssetResourceRequestOptions()
        options.isNetworkAccessAllowed = true
        do {
            try await PHAssetResourceManager.default().writeData(for: resource, toFile: destination, options: options)
        } catch {
            try? FileManager.default.removeItem(at: destination)
            throw Failure.unavailable
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: destination.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard size <= CreateLimits.maxFileBytes else {
            try? FileManager.default.removeItem(at: destination)
            throw Failure.tooLarge(size)
        }
        let contentType: String
        switch ext {
        case "mp4": contentType = "video/mp4"
        case "m4v": contentType = "video/x-m4v"
        default: contentType = "video/quicktime"
        }
        return LocalVideo(url: destination, name: name, duration: asset.duration, contentType: contentType)
    }
}

/// Fixed content for the design screenshots (no photo library needed).
struct GalleryPreview {
    var durations: [Double]
    var mode: CreateMode
    var selected: [Int]
    var showStyle = false
    var showLimitedCard = false
}

struct GalleryPickerView: View {
    let onCancel: () -> Void
    /// Called after "Choose your style": the mode, the videos in selection order and the style.
    let onCreate: (CreateMode, [PHAsset], StyleChoice) -> Void
    var preview: GalleryPreview?

    @StateObject private var library = GalleryLibrary()
    @State private var mode: CreateMode = .single
    @State private var selection: [String] = []
    @State private var showingStyle = false
    @State private var showingLimitedCard = false
    @State private var problem: CreateProblem?

    private var columns: [GridItem] { Array(repeating: GridItem(.flexible(), spacing: 6), count: 3) }

    var body: some View {
        ZStack {
            CS.background.ignoresSafeArea()
            switch preview != nil ? PHAuthorizationStatus.authorized : library.status {
            case .authorized, .limited:
                grid
            case .notDetermined:
                ProgressView().tint(.white)
            default:
                PermissionScreen(kind: .photos, deniedBefore: true, onClose: onCancel) {}
            }
            BottomSheetOverlay(isPresented: $showingStyle) {
                StyleSheetView(ctaTitle: mode.ctaTitle(selection.count)) { style in
                    showingStyle = false
                    let chosen = selection.compactMap { id in library.assets.first { $0.localIdentifier == id } }
                    onCreate(mode, chosen, style)
                }
            }
            if showingLimitedCard {
                GalleryAccessCard(
                    onSelectMore: {
                        showingLimitedCard = false
                        if let root = UIApplication.shared.connectedScenes
                            .compactMap({ ($0 as? UIWindowScene)?.keyWindow?.rootViewController }).first {
                            var top = root
                            while let presented = top.presentedViewController { top = presented }
                            PHPhotoLibrary.shared().presentLimitedLibraryPicker(from: top)
                        }
                    },
                    onKeep: { showingLimitedCard = false }
                )
            }
        }
        .preferredColorScheme(.dark)
        .task {
            if let preview {
                mode = preview.mode
                selection = preview.selected.map { "preview-\($0)" }
                showingLimitedCard = preview.showLimitedCard
                if preview.showStyle {
                    try? await Task.sleep(nanoseconds: 300_000_000)
                    showingStyle = true
                }
                return
            }
            await library.start()
            if library.status == .limited { showingLimitedCard = true }
        }
        .onDisappear { library.stop() }
        .fullScreenCover(item: $problem) { current in
            CreateProblemView(
                problem: current,
                onBack: { problem = nil },
                onShowFormats: { problem = .supportedFormats },
                onRemoveClip: {
                    if !selection.isEmpty { selection.removeLast() }
                    problem = nil
                }
            )
        }
        .onChange(of: mode) { _, newMode in
            if newMode == .single, selection.count > 1 { selection = Array(selection.prefix(1)) }
        }
    }

    private var grid: some View {
        VStack(spacing: 0) {
            ZStack {
                Text("Recents")
                    .font(CS.inter(22, .semibold))
                    .foregroundStyle(.white)
                HStack {
                    Button(action: onCancel) {
                        Image(systemName: "xmark")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(.white)
                            .frame(width: 44, height: 44)
                    }
                    .accessibilityLabel("Close")
                    Spacer()
                }
            }
            .padding(.horizontal, 10)
            .frame(height: 56)
            Rectangle().fill(CS.divider).frame(height: 1)

            if let preview {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 6) {
                        ForEach(Array(preview.durations.enumerated()), id: \.offset) { index, duration in
                            GalleryPreviewTile(index: index, duration: duration, badge: badge(forID: "preview-\(index)"))
                        }
                    }
                    .padding(.horizontal, 4)
                    .padding(.top, 6)
                    .padding(.bottom, 12)
                }
            } else if library.assets.isEmpty {
                Spacer()
                Text("No videos in your library yet.")
                    .font(CS.inter(14))
                    .foregroundStyle(CS.textSecondary)
                Spacer()
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 6) {
                        ForEach(library.assets, id: \.localIdentifier) { asset in
                            GalleryTile(
                                asset: asset,
                                manager: library.imageManager,
                                badge: badge(for: asset)
                            )
                            .onTapGesture { toggle(asset) }
                        }
                    }
                    .padding(.horizontal, 4)
                    .padding(.top, 6)
                    .padding(.bottom, 12)
                }
            }

            bottomPanel
        }
    }

    private var bottomPanel: some View {
        VStack(spacing: 0) {
            Text(mode.selectedText(selection.count))
                .font(CS.inter(14))
                .foregroundStyle(CS.textCaption)
                .padding(.top, 16)
            SegmentedPill(options: CreateMode.allCases, selection: $mode) { $0.title }
                .padding(.top, 14)
                .accessibilityIdentifier("createModePicker")
            Text(mode.helper)
                .font(CS.inter(13))
                .foregroundStyle(CS.textSecondary)
                .padding(.top, 14)
            PrimaryButton(title: mode.ctaTitle(max(selection.count, mode == .single ? 1 : selection.count)), isEnabled: !selection.isEmpty) {
                showingStyle = true
            }
            .padding(.top, 14)
            .padding(.bottom, 6)
            .accessibilityIdentifier("galleryCreateButton")
        }
        .padding(.horizontal, 18)
        .frame(maxWidth: .infinity)
        .background(CS.panel.ignoresSafeArea(edges: .bottom))
    }

    private func badge(for asset: PHAsset) -> GalleryTile.Badge {
        badge(forID: asset.localIdentifier)
    }

    private func badge(forID id: String) -> GalleryTile.Badge {
        guard let index = selection.firstIndex(of: id) else { return .none }
        return mode == .bulk ? .check : .number(index + 1)
    }

    private func toggle(_ asset: PHAsset) {
        let id = asset.localIdentifier
        if let index = selection.firstIndex(of: id) {
            selection.remove(at: index)
            return
        }
        if asset.duration > CreateLimits.maxDurationSeconds + 0.5 {
            problem = .tooLong(detail: CreateLimits.durationText(asset.duration))
            return
        }
        if mode == .single {
            selection = [id]
        } else if selection.count >= CreateLimits.maxClips {
            problem = .clipLimit
        } else {
            selection.append(id)
        }
    }
}

/// A colored placeholder tile used only by the design screenshots.
struct GalleryPreviewTile: View {
    let index: Int
    let duration: Double
    let badge: GalleryTile.Badge

    private static let palette: [(UInt32, UInt32)] = [
        (0x2B4A5E, 0x1A2A3A), (0x4A2A55, 0x2A1A35), (0x1E2E5A, 0x3A2A6A), (0x22343E, 0x18242E),
        (0x2E5A2E, 0x1A3A1E), (0x5A2A45, 0x7A2A3A), (0x2A2A35, 0x1E1E28), (0x4A4A1E, 0x2E3A1E),
        (0x4A4A55, 0x2E2E38), (0x2A2E5A, 0x3A2245), (0x4A2A2E, 0x2E1A1E), (0x2E3A4A, 0x1E2834),
    ]

    var body: some View {
        let colors = Self.palette[index % Self.palette.count]
        return LinearGradient(colors: [Color(hex: colors.0), Color(hex: colors.1)], startPoint: .topLeading, endPoint: .bottomTrailing)
            .aspectRatio(1, contentMode: .fit)
            .overlay(GalleryTileChrome(duration: duration, badge: badge))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay {
                if badge != .none {
                    RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.selection, lineWidth: 2)
                }
            }
    }
}

/// Duration label and selection badge drawn on top of a gallery tile.
struct GalleryTileChrome: View {
    let duration: Double
    let badge: GalleryTile.Badge

    var body: some View {
        ZStack {
            Color.clear
            Text(CreateLimits.durationText(duration))
                .font(CS.inter(12, .semibold))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.6), radius: 2)
                .padding(8)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
            Group {
                switch badge {
                case .none:
                    EmptyView()
                case .check:
                    circle { Image("IconCheck").resizable().frame(width: 14, height: 14) }
                case .number(let value):
                    circle { Text("\(value)").font(CS.inter(12, .semibold)).foregroundStyle(.white) }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
        }
    }

    private func circle<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .frame(width: 24, height: 24)
            .background(Circle().fill(Color(hex: 0x0D9EFF)))
            .overlay(Circle().stroke(Color(hex: 0x1AF2FF), lineWidth: 1.5))
            .padding(6)
    }
}

struct GalleryTile: View {
    enum Badge: Equatable { case none, check, number(Int) }

    let asset: PHAsset
    let manager: PHCachingImageManager
    let badge: Badge
    @State private var image: UIImage?
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        Color(hex: 0x1A2236)
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if let image {
                    Image(uiImage: image).resizable().scaledToFill()
                }
            }
            .overlay(GalleryTileChrome(duration: asset.duration, badge: badge))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay {
                if badge != .none {
                    RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.selection, lineWidth: 2)
                }
            }
            .contentShape(Rectangle())
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Video, \(CreateLimits.durationText(asset.duration))")
            .accessibilityAddTraits(badge == .none ? .isButton : [.isButton, .isSelected])
            .task(id: asset.localIdentifier) { await loadThumbnail() }
    }

    private func loadThumbnail() async {
        let side = 130 * displayScale
        let options = PHImageRequestOptions()
        options.deliveryMode = .opportunistic
        options.isNetworkAccessAllowed = true
        options.resizeMode = .fast
        for await picture in thumbnails(size: CGSize(width: side, height: side), options: options) {
            image = picture
        }
    }

    private func thumbnails(size: CGSize, options: PHImageRequestOptions) -> AsyncStream<UIImage> {
        AsyncStream { continuation in
            let id = manager.requestImage(for: asset, targetSize: size, contentMode: .aspectFill, options: options) { picture, info in
                if let picture { continuation.yield(picture) }
                let degraded = (info?[PHImageResultIsDegradedKey] as? Bool) ?? false
                if !degraded { continuation.finish() }
            }
            continuation.onTermination = { [manager] _ in manager.cancelImageRequest(id) }
        }
    }
}
