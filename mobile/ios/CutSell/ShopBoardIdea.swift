import SwiftUI
import UIKit

// Shop board · "My idea" (PO 18 Nov 2026: "up load an idea es para replicar EXACT SAME THING"; Figma S3i-C 3078:418,
// S3i-D 3078:499, carousel S3p-B 3078:319). She uploads a screenshot of a board she likes; the server's Claude
// Sonnet looks at it once (POST /v1/shop/idea, ~2 cents, nothing stored there) and returns an exact map: background,
// every text (words, place, font kind, colour) and every product place (kind, place, size, turn). Her products go
// in those places, by kind. The map is one of HER styles, saved on this iPhone only (PO: "cargando esto a su memoria
// del celular no a nuestro server cloud"): Application Support/ShopStyles/<id>.json + <id>.jpg (her screenshot).

struct ShopIdeaStyle: Codable, Equatable, Identifiable {
    struct Background: Codable, Equatable { var top: String; var bottom: String }

    struct Text: Codable, Equatable {
        var text: String
        var role: String
        var x: Double, y: Double, w: Double, h: Double
        var color: String
        var font: String
        var weight: String
        var align: String
        var uppercase: Bool
        var rotation: Double
    }

    struct Slot: Codable, Equatable {
        var kind: String
        var x: Double, y: Double, w: Double, h: Double
        var rotation: Double
        var front: Int
    }

    var id = UUID()
    var name = "My style"
    var createdAt = Date()
    var background: Background
    var texts: [Text]
    var slots: [Slot]

    enum CodingKeys: String, CodingKey { case id, name, createdAt, background, texts, slots }

    init(id: UUID = UUID(), name: String = "My style", createdAt: Date = Date(), background: Background, texts: [Text], slots: [Slot]) {
        self.id = id; self.name = name; self.createdAt = createdAt
        self.background = background; self.texts = texts; self.slots = slots
    }

    /// The server's answer has no id / name / date yet: they are added here.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "My style"
        createdAt = try c.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        background = try c.decode(Background.self, forKey: .background)
        texts = try c.decodeIfPresent([Text].self, forKey: .texts) ?? []
        slots = try c.decode([Slot].self, forKey: .slots)
    }

    var top: ShopRGB { Self.rgb(background.top) ?? ShopRGB(hex: 0xF4EFE6) }
    var bottom: ShopRGB { Self.rgb(background.bottom) ?? top }

    static func rgb(_ hex: String) -> ShopRGB? {
        let digits = hex.trimmingCharacters(in: CharacterSet(charactersIn: "# "))
        guard digits.count == 6, let value = UInt32(digits, radix: 16) else { return nil }
        return ShopRGB(hex: value)
    }

    /// The first text with that role (title / subtitle / brand): the one she can rewrite on her board.
    func firstText(_ role: String) -> String? { texts.first { $0.role == role }?.text }
}

// MARK: - Her products in the idea's places

enum ShopIdeaMatch {
    /// The idea's product kind -> the app's product kind.
    static func kind(_ raw: String) -> ShopItemKind {
        switch raw {
        case "top", "outerwear": .top
        case "bottom": .bottom
        case "dress": .clothing
        case "shoes": .shoes
        case "bag": .bag
        case "accessory", "jewelry": .accessory
        default: .unknown
        }
    }

    /// Each of her products gets a place of the same kind (the biggest such place first); then the rest by size
    /// (her first product into the biggest free place); more products than places = small places along the bottom;
    /// fewer = the extra places are not shown. Drawn back to front as in the idea ("front" 1 = on top).
    static func placements(_ style: ShopIdeaStyle, kinds: [ShopItemKind]) -> [ShopPlacement] {
        let design = ShopLayout.designSize
        let order = style.slots.indices.sorted { area(style.slots[$0]) > area(style.slots[$1]) }
        var slotFor: [Int: Int] = [:]          // item -> slot
        var taken = Set<Int>()
        for (item, kind) in kinds.enumerated() where kind != .unknown {
            if let s = order.first(where: { !taken.contains($0) && Self.kind(style.slots[$0].kind) == kind }) {
                slotFor[item] = s; taken.insert(s)
            }
        }
        for item in kinds.indices where slotFor[item] == nil {
            if let s = order.first(where: { !taken.contains($0) }) { slotFor[item] = s; taken.insert(s) }
        }
        var out: [(front: Int, placement: ShopPlacement)] = []
        var extra = 0
        for item in kinds.indices {
            if let s = slotFor[item] {
                let slot = style.slots[s]
                out.append((slot.front, ShopPlacement(itemIndex: item, slot: ShopSlot(
                    center: CGPoint(x: (slot.x + slot.w / 2) * design.width, y: (slot.y + slot.h / 2) * design.height),
                    size: CGSize(width: slot.w * design.width, height: slot.h * design.height),
                    rotation: slot.rotation, prefers: [kinds[item]]))))
            } else {
                // More products than the idea: small, in a row low on the board (she can move them).
                let column = extra % 4, row = extra / 4
                extra += 1
                out.append((0, ShopPlacement(itemIndex: item, slot: ShopSlot(
                    center: CGPoint(x: 60 + CGFloat(column) * 90, y: design.height - 140 - CGFloat(row) * 90),
                    size: CGSize(width: 78, height: 78), rotation: 0, prefers: [kinds[item]]))))
            }
        }
        // Back to front: the biggest "front" number first; extras (0) last, on top.
        return out.sorted { ($0.front == 0 ? -1 : $0.front) > ($1.front == 0 ? -1 : $1.front) }.map(\.placement)
    }

    private static func area(_ s: ShopIdeaStyle.Slot) -> Double { s.w * s.h }
}

// MARK: - Its texts on the board

enum ShopIdeaFonts {
    /// The idea's kind of font -> the closest face the app has (all bundled, all free).
    static func face(_ kind: String, weight: String) -> ShopFontFace? {
        switch kind {
        case "serif": weight == "light" ? ShopFontPair.soft.titleFace : ShopFontPair.editorial.titleFace
        case "script": ShopFontPair.classic.scriptFace
        case "handwritten": ShopFontPair.bold.scriptFace
        case "condensed": ShopFontPair.bold.titleFace
        default: nil                                    // sans / display / mono: Plus Jakarta Sans
        }
    }

    static func font(_ text: ShopIdeaStyle.Text, size: CGFloat) -> Font {
        if let face = face(text.font, weight: text.weight) { return ShopFonts.font(face, size: size) }
        if text.font == "mono" { return .system(size: size, weight: weight(text.weight), design: .monospaced) }
        return CS.inter(size, weight(text.weight))
    }

    static func weight(_ raw: String) -> Font.Weight {
        switch raw {
        case "light": .light
        case "bold": .bold
        case "black": .heavy
        default: .regular
        }
    }
}

struct ShopIdeaTexts: View {
    let texts: [ShopIdeaStyle.Text]
    /// Her own words for the title / subtitle / brand line (she can rewrite them on her board).
    let title: String
    let subtitle: String
    let brand: String
    let width: CGFloat

    var body: some View {
        let height = width * 16 / 9
        var seen = Set<String>()
        let words: [String] = texts.map { t in
            guard ["title", "subtitle", "brand"].contains(t.role), !seen.contains(t.role) else { return t.text }
            seen.insert(t.role)
            switch t.role {
            case "title": return title
            case "subtitle": return subtitle
            default: return brand
            }
        }
        return ZStack(alignment: .topLeading) {
            ForEach(texts.indices, id: \.self) { i in
                let t = texts[i]
                let box = CGRect(x: t.x * width, y: t.y * height, width: max(8, t.w * width), height: max(6, t.h * height))
                let lines = max(1, words[i].split(separator: "\n").count)
                Text(verbatim: t.uppercase ? words[i].uppercased() : words[i])
                    .font(ShopIdeaFonts.font(t, size: box.height / CGFloat(lines) * 0.82))
                    .foregroundStyle((ShopIdeaStyle.rgb(t.color) ?? ShopRGB(hex: 0x0B1020)).color)
                    .multilineTextAlignment(t.align == "left" ? .leading : t.align == "right" ? .trailing : .center)
                    .lineLimit(lines + 1)
                    .minimumScaleFactor(0.3)
                    .frame(width: box.width, height: box.height,
                           alignment: t.align == "left" ? .leading : t.align == "right" ? .trailing : .center)
                    .rotationEffect(.degrees(t.rotation))
                    .position(x: box.midX, y: box.midY)
            }
        }
        .frame(width: width, height: height, alignment: .topLeading)
        .allowsHitTesting(false)
    }
}

// MARK: - Her styles, on this iPhone only

@MainActor
final class ShopIdeaStore: ObservableObject {
    static let shared = ShopIdeaStore()
    @Published private(set) var styles: [ShopIdeaStyle] = []

    private let folder: URL

    init(folder: URL? = nil) {
        let base = folder ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ShopStyles", isDirectory: true)
        self.folder = base
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        reload()
    }

    func reload() {
        let decoder = JSONDecoder()
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        styles = files.filter { $0.pathExtension == "json" }
            .compactMap { try? decoder.decode(ShopIdeaStyle.self, from: Data(contentsOf: $0)) }
            .sorted { $0.createdAt > $1.createdAt }
    }

    /// "My style 1", "My style 2"...
    var nextName: String { "My style \(styles.count + 1)" }

    func save(_ style: ShopIdeaStyle, picture: UIImage?) {
        if let data = try? JSONEncoder().encode(style) {
            try? data.write(to: folder.appendingPathComponent("\(style.id.uuidString).json"), options: .atomic)
        }
        if let jpeg = picture?.jpegData(compressionQuality: 0.7) {
            try? jpeg.write(to: folder.appendingPathComponent("\(style.id.uuidString).jpg"), options: .atomic)
        }
        reload()
    }

    func rename(_ id: UUID, to name: String) {
        guard var style = styles.first(where: { $0.id == id }) else { return }
        style.name = String(name.trimmingCharacters(in: .whitespacesAndNewlines).prefix(40))
        save(style, picture: nil)
    }

    func delete(_ id: UUID) {
        for ext in ["json", "jpg"] { try? FileManager.default.removeItem(at: folder.appendingPathComponent("\(id.uuidString).\(ext)")) }
        reload()
    }

    func picture(_ id: UUID) -> UIImage? { UIImage(contentsOfFile: folder.appendingPathComponent("\(id.uuidString).jpg").path) }
}

// MARK: - Reading her screenshot (POST /v1/shop/idea)

enum ShopIdeaReader {
    struct Body: Encodable { let user_id: String; let image_b64: String }
    struct Answer: Decodable { let style: ShopIdeaStyle }

    /// Her screenshot, at most 1280 px tall, as JPEG (about 150-300 KB).
    static func jpeg(_ image: UIImage) -> Data? {
        let longest = max(image.size.width, image.size.height)
        let scale = min(1, 1280 / max(longest, 1))
        let size = CGSize(width: (image.size.width * scale).rounded(), height: (image.size.height * scale).rounded())
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let small = UIGraphicsImageRenderer(size: size, format: format).image { _ in image.draw(in: CGRect(origin: .zero, size: size)) }
        return small.jpegData(compressionQuality: 0.8)
    }

    static func read(_ image: UIImage, userID: String) async throws -> ShopIdeaStyle {
        guard let data = jpeg(image) else { throw URLError(.cannotDecodeContentData) }
        let answer: Answer = try await APIClient.shared.request(
            "/v1/shop/idea", method: "POST", body: Body(user_id: userID, image_b64: data.base64EncodedString()), timeout: 90)
        return answer.style
    }
}

// MARK: - The two screens (Figma S3i-C 3078:418 reading it · S3i-D 3078:499 your new board)

struct ShopIdeaPicture: Identifiable {
    let id = UUID()
    let image: UIImage
}

struct ShopIdeaFlowView: View {
    @ObservedObject var model: ShopBoardModel
    let picture: UIImage
    let onClose: () -> Void
    let onUse: (ShopIdeaStyle) -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var store = ShopIdeaStore.shared
    @State private var style: ShopIdeaStyle?
    @State private var failed = false
    @State private var name = ""

    var body: some View {
        ZStack {
            HomeBackdrop()
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 12) {
                    ShopBackButton(action: onClose)
                    Text(L10n.tr(style == nil ? "Copying your idea" : "Your new board"))
                        .font(CS.inter(24, .heavy)).foregroundStyle(Color(fixed: 0x0B1020))
                    Spacer()
                }
                .padding(.horizontal, 20).padding(.top, 8)
                if let style { result(style) } else { readingView }
            }
        }
        .environment(\.colorScheme, .light)
        .task { await read() }
        .accessibilityIdentifier("shopIdeaFlow")
    }

    private func read() async {
        guard let session = appState.session else { failed = true; return }
        failed = false
        do {
            var answer = try await ShopIdeaReader.read(picture, userID: session.userID)
            answer.name = store.nextName
            name = answer.name
            withAnimation(.snappy) { style = answer }
            CutSellDiagnostics.log("shop_idea_read", ["ok": "1", "slots": String(answer.slots.count), "texts": String(answer.texts.count)])
        } catch {
            failed = true
            CutSellDiagnostics.log("shop_idea_read", ["ok": "0", "error": String(describing: error).prefix(80).description])
        }
    }

    // C · her screenshot while it is read
    private var readingView: some View {
        VStack(spacing: 16) {
            Image(uiImage: picture).resizable().scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .shadow(color: Color(fixed: 0x1769FF).opacity(0.16), radius: 12, x: 0, y: 10)
                .frame(maxHeight: 420)
                .overlay { if !failed { ProgressView().controlSize(.large).tint(Color(fixed: 0x1769FF)) } }
                .padding(.top, 18)
            VStack(alignment: .leading, spacing: 10) {
                row("Background colors")
                row("Title · same words, font and place")
                row("Product places · sizes and turns")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(L10n.tr(failed ? "We couldn't read this picture. Try another screenshot with the products easy to see."
                                : "Saved on this iPhone only"))
                .font(CS.inter(12, .medium))
                .foregroundStyle(Color(fixed: failed ? 0xE5484D : 0x6B7385))
                .multilineTextAlignment(.center).frame(maxWidth: .infinity)
            Spacer(minLength: 8)
            ShopBigButton(title: failed ? "Try again" : "Reading…", enabled: failed, id: "shopIdeaRead") {
                Task { await read() }
            }
            .padding(.bottom, 24)
        }
        .padding(.horizontal, 24)
    }

    private func row(_ text: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: failed ? "xmark.circle" : "ellipsis.circle")
                .foregroundStyle(Color(fixed: failed ? 0xE5484D : 0x1769FF))
            Text(L10n.tr(text)).font(CS.inter(13, .semibold)).foregroundStyle(Color(fixed: 0x0B1020))
        }
    }

    // D · the idea, small -> her board in the same layout, and where each of her products went
    private func result(_ style: ShopIdeaStyle) -> some View {
        let usable = model.items.filter { $0.cutout != nil }
        let placed = ShopIdeaMatch.placements(style, kinds: usable.map(\.kind))
        return ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .center, spacing: 10) {
                    VStack(spacing: 6) {
                        Image(uiImage: picture).resizable().scaledToFit().frame(width: 96)
                            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        Text(L10n.tr("Your idea")).font(CS.inter(11, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
                    }
                    Image(systemName: "arrow.right").font(.system(size: 20, weight: .bold)).foregroundStyle(Color(fixed: 0x1769FF))
                    VStack(spacing: 6) {
                        ShopBoardCanvas(snapshot: model.snapshot(idea: style), width: 196)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                            .shadow(color: Color(fixed: 0x1769FF).opacity(0.18), radius: 12, x: 0, y: 10)
                        Text(L10n.tr("Your board")).font(CS.inter(11, .bold)).foregroundStyle(Color(fixed: 0x1769FF))
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.top, 16)
                Text(L10n.tr("Where your products went")).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                ForEach(placed.indices, id: \.self) { i in
                    let item = usable[placed[i].itemIndex]
                    Text(verbatim: "\(Self.kindName(placed[i].slot.prefers.first ?? item.kind)) → \(Self.productName(item))")
                        .font(CS.inter(12, .semibold)).foregroundStyle(Color(fixed: 0x0B1020))
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Capsule().fill(.white).shadow(color: Color(fixed: 0x1769FF).opacity(0.08), radius: 5, x: 0, y: 3))
                }
                TextField(L10n.tr("Name"), text: $name)
                    .font(CS.inter(14, .semibold))
                    .padding(12)
                    .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.white))
                    .accessibilityIdentifier("shopIdeaName")
                Text(L10n.tr("Saved on this iPhone only · tap any word on your board to change it"))
                    .font(CS.inter(11, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                ShopBigButton(title: "Use this style", id: "shopIdeaUse") {
                    var saved = style
                    let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !clean.isEmpty { saved.name = String(clean.prefix(40)) }
                    store.save(saved, picture: picture)
                    onUse(saved)
                }
                .padding(.top, 6).padding(.bottom, 24)
            }
            .padding(.horizontal, 24)
        }
    }

    static func kindName(_ kind: ShopItemKind) -> String {
        switch kind {
        case .top: L10n.tr("Top")
        case .bottom: L10n.tr("Bottom")
        case .clothing: L10n.tr("Clothing")
        case .shoes: L10n.tr("Shoes")
        case .bag: L10n.tr("Bag")
        case .accessory: L10n.tr("Accessory")
        case .unknown: L10n.tr("Product")
        }
    }

    static func productName(_ item: ShopItem) -> String {
        let n = item.name.trimmingCharacters(in: .whitespaces)
        return n.isEmpty ? L10n.tr("your") + " " + kindName(item.kind).lowercased() : n
    }
}
