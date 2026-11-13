import Combine
import SwiftUI
import UIKit

// Shop boards saved on the iPhone (Edits › Shop boards, Figma 3e 3055:516). Every board saves itself while it is
// being made: its products (cut-outs as PNG), how they were moved by hand, the template, letters, background, link
// and caption, plus a small picture for the grid. Nothing leaves the phone.
// Folder: Application Support/ShopBoards/<board id>/ (board.json, item PNGs, background photo, thumb.jpg).

struct ShopBoardRecord: Codable, Identifiable, Equatable {
    struct Item: Codable, Equatable {
        var id: UUID
        var source: ShopItemSource
        var name: String
        var affiliateLink: URL?
        var productID: String?
        var kind: ShopItemKind
        var shaped: Bool
        var backgroundRemoved: Bool
        var cutoutFile: String?
        var flatFile: String?
    }

    struct Adjust: Codable, Equatable {
        var dx: Double, dy: Double, scale: Double, rotation: Double
    }

    enum Background: Codable, Equatable {
        case color(ShopRGB), classic(String), texture(String), photo
    }

    var id: UUID
    var createdAt: Date
    var updatedAt: Date
    var template: String
    var bigTitle: String
    var scriptLine: String
    var fontPair: String
    var variant: Int
    var matchBackground: Bool
    var background: Background
    var matchTitle: Bool
    var titleColor: ShopRGB
    var affiliateLink: String
    var caption: String
    var items: [Item]
    var adjusts: [UUID: Adjust]
    var raised: [UUID]
    var hasPhotoBackground: Bool

    /// "CUTSELL · Everyday chic" under the tile.
    var displayTitle: String {
        [bigTitle, scriptLine].map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.joined(separator: " · ")
    }
}

extension ShopBoardRecord.Background {
    init(_ choice: ShopBackgroundChoice) {
        switch choice {
        case .color(let rgb): self = .color(rgb)
        case .classic(let c): self = .classic(c.rawValue)
        case .texture(let t): self = .texture(t.rawValue)
        case .photo: self = .photo
        }
    }

    var choice: ShopBackgroundChoice {
        switch self {
        case .color(let rgb): return .color(rgb)
        case .classic(let raw): return .classic(ShopClassicColor(rawValue: raw) ?? .cream)
        case .texture(let raw): return .texture(ShopTexture(rawValue: raw) ?? .linen)
        case .photo: return .photo
        }
    }
}

@MainActor
final class ShopBoardStore: ObservableObject {
    static let shared = ShopBoardStore()

    /// Newest first.
    @Published private(set) var boards: [ShopBoardRecord] = []
    /// Bumped when a board's picture changes (the grid reloads it).
    @Published private(set) var thumbnailVersion: [UUID: Int] = [:]

    let root: URL
    private let io = DispatchQueue(label: "ai.cutsell.shopboards", qos: .utility)

    init(root: URL? = nil) {
        let base = root ?? (FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("ShopBoards", isDirectory: true))
        self.root = base
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        boards = Self.readAll(base)
    }

    func folder(_ id: UUID) -> URL { root.appendingPathComponent(id.uuidString, isDirectory: true) }

    // MARK: Save

    /// Saves the board (only once it has at least one product). Pictures are written once per version of a product.
    func save(_ model: ShopBoardModel, thumbnail: UIImage? = nil, synchronously: Bool = false) {
        guard !model.items.isEmpty else { return }
        let dir = folder(model.boardID)
        var files: [String: Data] = [:]
        var items: [ShopBoardRecord.Item] = []
        for item in model.items {
            var cutoutFile: String?, flatFile: String?
            if let cutout = item.cutout {
                let name = "cutout-\(item.id.uuidString)-\(item.shaped ? "shaped" : "flat").png"
                cutoutFile = name
                if !FileManager.default.fileExists(atPath: dir.appendingPathComponent(name).path),
                   let data = UIImage(cgImage: cutout).pngData() { files[name] = data }
            }
            if item.shaped, let flat = item.flat {
                let name = "flat-\(item.id.uuidString).png"
                flatFile = name
                if !FileManager.default.fileExists(atPath: dir.appendingPathComponent(name).path),
                   let data = UIImage(cgImage: flat).pngData() { files[name] = data }
            }
            items.append(.init(id: item.id, source: item.source, name: item.name, affiliateLink: item.affiliateLink,
                               productID: item.productID, kind: item.kind, shaped: item.shaped,
                               backgroundRemoved: item.backgroundRemoved, cutoutFile: cutoutFile, flatFile: flatFile))
        }
        if let photo = model.photoBackground, let data = photo.jpegData(compressionQuality: 0.85) {
            files["background.jpg"] = data
        }
        if let thumbnail, let data = thumbnail.jpegData(compressionQuality: 0.8) { files["thumb.jpg"] = data }
        let now = Date()
        let created = boards.first(where: { $0.id == model.boardID })?.createdAt ?? now
        let record = ShopBoardRecord(
            id: model.boardID, createdAt: created, updatedAt: now, template: model.template.rawValue,
            bigTitle: model.bigTitle, scriptLine: model.scriptLine, fontPair: model.fontPair.rawValue, variant: model.variant,
            matchBackground: model.style.matchBackground, background: .init(model.style.background),
            matchTitle: model.style.matchTitle, titleColor: model.style.titleColor,
            affiliateLink: model.affiliateLink, caption: model.caption, items: items,
            adjusts: model.adjusts.mapValues { .init(dx: $0.offset.width, dy: $0.offset.height, scale: $0.scale, rotation: $0.rotation) },
            raised: model.raised, hasPhotoBackground: model.photoBackground != nil)
        boards.removeAll { $0.id == record.id }
        boards.insert(record, at: 0)
        if thumbnail != nil { thumbnailVersion[record.id, default: 0] += 1 }
        let json = try? JSONEncoder().encode(record)
        let keep = Set(items.flatMap { [$0.cutoutFile, $0.flatFile].compactMap { $0 } } + ["board.json", "thumb.jpg", "background.jpg"])
        let hasPhoto = record.hasPhotoBackground
        let write = {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for (name, data) in files { try? data.write(to: dir.appendingPathComponent(name), options: .atomic) }
            if let json { try? json.write(to: dir.appendingPathComponent("board.json"), options: .atomic) }
            // Old versions of products (removed, shaped again) and a removed background photo go away.
            for name in (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
            where !keep.contains(name) || (name == "background.jpg" && !hasPhoto) {
                try? FileManager.default.removeItem(at: dir.appendingPathComponent(name))
            }
        }
        if synchronously { io.sync(execute: write) } else { io.async(execute: write) }
    }

    // MARK: Open, delete

    /// The saved board as a board being made again (opens on "Your board").
    func open(_ id: UUID) -> ShopBoardModel? {
        io.sync {}                                     // pending writes first
        guard let record = boards.first(where: { $0.id == id }) else { return nil }
        let dir = folder(id)
        func image(_ name: String?) -> CGImage? {
            guard let name, let ui = UIImage(contentsOfFile: dir.appendingPathComponent(name).path) else { return nil }
            return ui.cgImage
        }
        let model = ShopBoardModel(boardID: id)
        model.template = ShopTemplate(rawValue: record.template) ?? .everydayChic
        model.bigTitle = record.bigTitle
        model.scriptLine = record.scriptLine
        model.fontPair = ShopFontPair(rawValue: record.fontPair) ?? .classic
        model.variant = record.variant
        model.style = ShopBoardStyle(matchBackground: record.matchBackground, background: record.background.choice,
                                     matchTitle: record.matchTitle, titleColor: record.titleColor)
        model.affiliateLink = record.affiliateLink
        model.caption = record.caption
        if record.hasPhotoBackground {
            model.photoBackground = UIImage(contentsOfFile: dir.appendingPathComponent("background.jpg").path)
        }
        model.items = record.items.map { saved in
            var item = ShopItem(id: saved.id, source: saved.source, name: saved.name, affiliateLink: saved.affiliateLink,
                                productID: saved.productID, cutout: image(saved.cutoutFile), kind: saved.kind)
            item.shaped = saved.shaped
            item.backgroundRemoved = saved.backgroundRemoved
            if saved.shaped { item.flat = image(saved.flatFile) ?? item.cutout }
            return item
        }
        model.adjusts = record.adjusts.mapValues {
            ShopAdjust(offset: CGSize(width: $0.dx, height: $0.dy), scale: CGFloat($0.scale), rotation: $0.rotation)
        }
        model.raised = record.raised
        return model
    }

    func thumbnail(_ id: UUID) -> UIImage? {
        UIImage(contentsOfFile: folder(id).appendingPathComponent("thumb.jpg").path)
    }

    func delete(_ id: UUID) {
        boards.removeAll { $0.id == id }
        let dir = folder(id)
        io.async { try? FileManager.default.removeItem(at: dir) }
    }

    func flush() { io.sync {} }

    private static func readAll(_ root: URL) -> [ShopBoardRecord] {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return dirs.compactMap { dir in
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("board.json")) else { return nil }
            return try? JSONDecoder().decode(ShopBoardRecord.self, from: data)
        }
        .sorted { $0.updatedAt > $1.updatedAt }
    }
}

enum ShopBoardThumbnail {
    /// The 270 × 480 picture of the grid tile.
    @MainActor
    static func image(_ snapshot: ShopBoardSnapshot) -> UIImage? {
        let renderer = ImageRenderer(content: ShopBoardCanvas(snapshot: snapshot, width: 270).environment(\.colorScheme, .light))
        renderer.scale = 1
        renderer.isOpaque = true
        return renderer.uiImage
    }
}

// MARK: - Edits › Shop boards (Figma 3e 3055:516)

/// "+ New Shop board" tile (dashed #1769FF on #EEF3FF, radius 16) and the saved boards (110 × 195 9:16 picture,
/// radius 16, "•••" menu, Bold 12 title, Medium 11 #6B7385 date), three per row.
struct ShopBoardsGrid: View {
    @ObservedObject var store = ShopBoardStore.shared
    let onNew: () -> Void
    let onOpen: (UUID) -> Void
    @State private var deleting: ShopBoardRecord?

    private let columns = [GridItem(.flexible(), spacing: 10, alignment: .top), GridItem(.flexible(), spacing: 10, alignment: .top),
                           GridItem(.flexible(), spacing: 10, alignment: .top)]

    var body: some View {
        LazyVGrid(columns: columns, spacing: 14) {
            Button(action: onNew) {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color(fixed: 0xEEF3FF))
                    .overlay {
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .strokeBorder(Color(fixed: 0x1769FF), style: StrokeStyle(lineWidth: 1.5, dash: [6, 4]))
                    }
                    .overlay {
                        VStack(spacing: 10) {
                            Text(verbatim: "+").font(CS.inter(28, .bold))
                            Text(L10n.tr("New Shop board")).font(CS.inter(12, .bold))
                        }
                        .foregroundStyle(Color(fixed: 0x1769FF))
                    }
                    .aspectRatio(9 / 16, contentMode: .fit)
            }
            .buttonStyle(BubblePress())
            .accessibilityIdentifier("shopBoardsNew")
            ForEach(store.boards) { board in tile(board) }
        }
        .padding(.top, 22)
        .confirmationDialog(L10n.tr("Delete this Shop board?"), isPresented: Binding(get: { deleting != nil },
                                                                                       set: { if !$0 { deleting = nil } }),
                            titleVisibility: .visible) {
            Button(L10n.tr("Delete"), role: .destructive) {
                if let board = deleting { store.delete(board.id) }
                deleting = nil
            }
        } message: {
            Text(L10n.tr("It is only removed from CutSell. Pictures you saved to Photos stay."))
        }
    }

    private func tile(_ board: ShopBoardRecord) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Button { onOpen(board.id) } label: {
                ShopBoardTilePicture(id: board.id, version: store.thumbnailVersion[board.id, default: 0])
                    .aspectRatio(9 / 16, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .shadow(color: Color(fixed: 0x0D1433, opacity: 0.08), radius: 7, x: 0, y: 6)
                    .overlay(alignment: .topTrailing) {
                        Menu {
                            Button(L10n.tr("Delete"), role: .destructive) { deleting = board }
                        } label: {
                            Text(verbatim: "•••").font(CS.inter(9, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                                .frame(width: 26, height: 26).background(Color.white, in: Circle())
                                .padding(6)
                        }
                        .accessibilityIdentifier("shopBoardMenu")
                    }
            }
            .buttonStyle(BubblePress())
            .accessibilityIdentifier("shopBoardTile")
            Text(verbatim: board.displayTitle).font(CS.inter(12, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                .lineLimit(1).padding(.top, 2)
            Text(verbatim: ShopBoardsGrid.dateText(board.updatedAt)).font(CS.inter(11, .medium))
                .foregroundStyle(Color(fixed: 0x6B7385))
        }
    }

    /// "Today", "Yesterday" or "Nov 10".
    static func dateText(_ date: Date, now: Date = Date(), calendar: Calendar = .current) -> String {
        if calendar.isDate(date, inSameDayAs: now) { return L10n.tr("Today") }
        if let y = calendar.date(byAdding: .day, value: -1, to: now), calendar.isDate(date, inSameDayAs: y) {
            return L10n.tr("Yesterday")
        }
        let f = DateFormatter()
        f.locale = L10n.locale
        f.setLocalizedDateFormatFromTemplate("MMM d")
        return f.string(from: date)
    }
}

private struct ShopBoardTilePicture: View {
    let id: UUID
    let version: Int
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Color(fixed: 0xF1ECE4)
            if let image { Image(uiImage: image).resizable().scaledToFill() }
        }
        .task(id: version) { image = ShopBoardStore.shared.thumbnail(id) }
    }
}
