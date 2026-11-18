import Photos
import SwiftUI
import UIKit

// Shop board: the board being made (items, template, letters, look), the board drawing itself (9:16), its
// 1080 × 1920 export and the share actions. Screens: ShopBoardViews.swift.

// MARK: - The board being made

@MainActor
final class ShopBoardModel: ObservableObject {
    /// The saved board's id (Edits › Shop boards).
    let boardID: UUID

    init(boardID: UUID = UUID()) { self.boardID = boardID }

    /// The title block moves by hand like a product (PO 13 Nov); its hand edit lives in `adjusts` under this id.
    static let titleID = UUID(uuidString: "7171E000-0000-4000-8000-000000000001")!

    @Published var items: [ShopItem] = [] { didSet { if oldValue.map(\.cutout) != items.map(\.cutout) { refreshPalette() } } }
    /// Items whose photo is being cut out now.
    @Published var busy: Set<UUID> = []
    @Published var template: ShopTemplate = .everydayChic
    /// "My idea" (PO 18 Nov): her saved style -- an exact map of a board she liked; nil = Collage / Wear it.
    @Published var idea: ShopIdeaStyle?
    @Published var bigTitle = ShopTemplate.everydayChic.bigTitle
    @Published var scriptLine = ShopTemplate.everydayChic.scriptLine
    /// The line at the bottom of the board ("YOUR BRAND" by default; PO 14 Nov, Figma S4d2 3062:193). Empty = none.
    @Published var bottomLine = ShopTemplate.footer
    @Published var fontPair: ShopFontPair = .classic
    @Published var style = ShopBoardStyle()
    @Published var variant = 0
    /// Products moved, resized or turned by hand on "Your board" (cleared by Shuffle and a new template).
    @Published var adjusts: [UUID: ShopAdjust] = [:]
    /// Products brought to the front by a tap, last on top.
    @Published var raised: [UUID] = []
    @Published var photoBackground: UIImage?
    /// Wear it with a video (Figma W1/W2): her video file and its length; `photoBackground` is then its first picture
    /// (thumbnails, the style tile). Only Wear it uses a video.
    @Published var videoBackground: URL?
    @Published var videoDuration: Double = 0

    /// The board is a video (Wear it, her video chosen as the background).
    var isVideoBoard: Bool {
        guard template == .wearIt, videoBackground != nil, !style.matchBackground, case .photo = style.background else { return false }
        return true
    }

    func useVideo(_ loaded: ShopVideoBackground.Loaded) {
        videoBackground = loaded.url
        videoDuration = loaded.duration
        photoBackground = loaded.poster
        style.pick(.photo)
    }

    func usePhoto(_ image: UIImage) {
        videoBackground = nil
        videoDuration = 0
        photoBackground = image
        style.pick(.photo)
    }
    @Published private(set) var palette: [ShopRGB] = []
    @Published var affiliateLink = ""
    @Published var caption = ""

    private func refreshPalette() {
        let images = items.compactMap(\.cutout)
        Task.detached(priority: .userInitiated) {
            let pixels = images.map { ShopCutout.opaquePixels($0) }
            let palette = ShopPalette.palette(perItem: pixels)
            await MainActor.run { self.palette = palette }
        }
    }

    // MARK: Items

    /// "From photos": every picked photo becomes an item; its background is removed on the iPhone.
    func addPhotos(_ images: [UIImage]) {
        for image in images.prefix(ShopBoardRules.remaining(items)) {
            let item = ShopItem(source: .photos)
            items.append(item)
            attach(image, to: item.id)
        }
    }

    /// Photos picked in the picker: each gets its tile (with the spinner) at once and loads on its own.
    /// A photo that cannot load in time is removed and counted in `photoFailures` (the screen says so).
    func addPhotos(_ providers: [NSItemProvider]) {
        for provider in providers.prefix(ShopBoardRules.remaining(items)) {
            let item = ShopItem(source: .photos)
            items.append(item)
            attach(provider, to: item.id, removeIfFails: true)
        }
    }

    /// Bumped each time a picked photo could not be loaded.
    @Published var photoFailures = 0

    func attach(_ provider: NSItemProvider, to id: UUID, removeIfFails: Bool = false) {
        busy.insert(id)
        let started = Date()
        CutSellDiagnostics.log("shop_photo_load_start", ["types": provider.registeredTypeIdentifiers.joined(separator: ",")])
        Task {
            guard let image = await ShopPhotoLoader.load(provider) else {
                CutSellDiagnostics.log("shop_photo_load_failed", ["ms": String(Int(Date().timeIntervalSince(started) * 1000))])
                busy.remove(id)
                if removeIfFails { remove(id) }
                photoFailures += 1
                return
            }
            CutSellDiagnostics.log("shop_photo_loaded", [
                "ms": String(Int(Date().timeIntervalSince(started) * 1000)),
                "w": String(Int(image.size.width * image.scale)), "h": String(Int(image.size.height * image.scale)),
            ])
            attach(image, to: id, removeIfFails: removeIfFails)
        }
    }

    /// "From my products": the product (its name, its id). Its photo comes from Photos next.
    @discardableResult
    func addProduct(name: String, productID: String) -> UUID? {
        guard ShopBoardRules.remaining(items) > 0 else { return nil }
        let item = ShopItem(source: .myProducts, name: name, productID: productID)
        items.append(item)
        return item.id
    }

    /// "Paste a link": the link is kept as the item's affiliate link; nothing is downloaded from it.
    @discardableResult
    func addLink(_ text: String, handler: ShopLinkHandler = ShopLinkHandler()) -> UUID? {
        guard ShopBoardRules.remaining(items) > 0, let item = handler.item(fromPasted: text) else { return nil }
        items.append(item)
        if affiliateLink.isEmpty, let url = item.affiliateLink { affiliateLink = url.absoluteString }
        return item.id
    }

    /// The photo of an item (picked in Photos): background removed, kind read by Vision.
    func attach(_ image: UIImage, to id: UUID, removeIfFails: Bool = false) {
        busy.insert(id)
        let started = Date()
        Task {
            let result = await Self.processWithTimeout(image)
            do {
                self.busy.remove(id)
                CutSellDiagnostics.log(result.map { _ in "shop_photo_processed" } ?? "shop_photo_process_failed", [
                    "ms": String(Int(Date().timeIntervalSince(started) * 1000)),
                    "removed": String(result?.removed ?? false),
                ])
                guard let index = self.items.firstIndex(where: { $0.id == id }) else { return }
                guard let result else {
                    // Never leave an empty tile behind (PO 14 Nov).
                    if removeIfFails { self.remove(id) }
                    self.photoFailures += 1
                    return
                }
                self.items[index].cutout = result.image
                self.items[index].flat = result.image
                self.items[index].backgroundRemoved = result.removed
                self.items[index].kind = result.kind
                self.items[index].shaped = false
            }
        }
    }

    /// `process` with a time limit: after 25 s the photo is used as it is (background kept), so a tile never spins
    /// for ever (PO 14 Nov, build 90: photos still not loading).
    ///
    /// Build 90 on the PO's iPhone: ten picked photos all spun for ever. Each photo ran Vision synchronously on a
    /// Swift-concurrency thread; ten at once used up those few threads, so nothing else (not even the time limits)
    /// could run. The heavy work now runs on its own queue, one photo at a time (`ShopWork`), and the time limit is
    /// a plain timer, so it always fires.
    nonisolated static func processWithTimeout(_ image: UIImage, seconds: Double = 25)
        async -> (image: CGImage, removed: Bool, kind: ShopItemKind)? {
        await ShopWork.run(timeout: seconds, onTimeout: {
            CutSellDiagnostics.log("shop_photo_process_timeout", [:])
            return ShopCutout.normalized(image).map { ($0, false, ShopItemKind.unknown) }
        }) { process(image) }
    }

    nonisolated static func process(_ image: UIImage) -> (image: CGImage, removed: Bool, kind: ShopItemKind)? {
        guard let photo = ShopCutout.normalized(image) else { return nil }
        // A PNG that already has no background (transparent): kept as it is, no Vision pass.
        if ShopCutout.hasTransparentBackground(photo), let trimmed = ShopCutout.trimmed(photo) {
            return (trimmed, true, ShopVision.kind(of: trimmed))
        }
        if let cut = try? ShopCutout.removeBackground(photo) {
            return (cut, true, ShopVision.kind(of: cut))
        }
        CutSellDiagnostics.log("shop_background_kept", [:])
        return (photo, false, ShopVision.kind(of: photo))
    }

    func remove(_ id: UUID) {
        items.removeAll { $0.id == id }
        adjusts[id] = nil
        raised.removeAll { $0 == id }
    }

    /// Your board's × (Figma S4x): the product leaves the board and every other product STAYS where it is now
    /// (the layout for one product fewer is different, so each one gets the hand edit that keeps it in place).
    /// PO 17 Nov ("Add another photo or item" on Your board): the products already on the board stay exactly where she
    /// put them while new ones come in. `pinPlaces()` remembers where each one is drawn; `reapplyPins()` puts them
    /// back there each time the layout changes (a new photo ready); the pins are let go once nothing is loading.
    private var pinned: [UUID: (rect: CGRect, rotation: Double)] = [:]

    func pinPlaces() {
        let board = ShopLayout.designSize
        let now = snapshot()
        pinned = [:]
        for index in now.placements.indices {
            if let placed = now.frame(at: index, board: board) { pinned[now.itemIDs[index]] = placed }
        }
    }

    func reapplyPins() {
        guard !pinned.isEmpty else { return }
        let board = ShopLayout.designSize
        let after = snapshot()
        for index in after.placements.indices {
            let itemID = after.itemIDs[index]
            guard let old = pinned[itemID] else { continue }
            let placement = after.placements[index]
            let image = after.images[placement.itemIndex]
            let aspect = CGFloat(image.width) / CGFloat(max(1, image.height))
            let base = ShopLayout.frame(for: placement.slot, aspect: aspect, board: board)
            let keep = ShopAdjust.keeping(old.rect, rotation: old.rotation, base: base, baseRotation: placement.slot.rotation)
            if adjusts[itemID] != keep { adjusts[itemID] = keep }
        }
        if busy.isEmpty { pinned = [:] }
    }

    func removeKeepingPlaces(_ id: UUID) {
        let board = ShopLayout.designSize
        let before = snapshot()
        var kept: [UUID: (rect: CGRect, rotation: Double)] = [:]
        for index in before.placements.indices where before.itemIDs[index] != id {
            if let placed = before.frame(at: index, board: board) { kept[before.itemIDs[index]] = placed }
        }
        remove(id)
        let after = snapshot()
        for index in after.placements.indices {
            let itemID = after.itemIDs[index]
            guard let old = kept[itemID] else { continue }
            let placement = after.placements[index]
            let image = after.images[placement.itemIndex]
            let aspect = CGFloat(image.width) / CGFloat(max(1, image.height))
            let base = ShopLayout.frame(for: placement.slot, aspect: aspect, board: board)
            adjusts[itemID] = ShopAdjust.keeping(old.rect, rotation: old.rotation, base: base,
                                                 baseRotation: placement.slot.rotation)
        }
    }

    func useShaped(_ image: CGImage, for id: UUID) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].cutout = ShopCutout.trimmed(image) ?? image
        items[index].shaped = true
    }

    func keepFlat(_ id: UUID) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].cutout = items[index].flat
        items[index].shaped = false
    }

    // MARK: Template, title, look

    /// One of her styles (an exact copy of a board she liked): its places, colours and texts, with her products.
    func use(_ idea: ShopIdeaStyle) {
        if template == .wearIt { choose(.everydayChic) }
        self.idea = idea
        template = .everydayChic
        bigTitle = idea.firstText("title") ?? ""
        scriptLine = idea.firstText("subtitle") ?? ""
        if let brand = idea.firstText("brand") { bottomLine = brand }
        style = ShopBoardStyle(matchBackground: false, background: .color(idea.top), matchTitle: false,
                               titleColor: ShopIdeaStyle.rgb(idea.texts.first?.color ?? "") ?? ShopRGB(hex: 0x0B1020))
        variant = 0
        clearHandEdits()
    }

    func choose(_ template: ShopTemplate) {
        let leavingWearIt = self.template == .wearIt && template != .wearIt
        if idea != nil {
            idea = nil
            style = ShopBoardStyle(background: .classic(template.background))
        }
        self.template = template
        bigTitle = template.bigTitle
        scriptLine = template.scriptLine
        if !style.matchBackground, case .classic = style.background { style.background = .classic(template.background) }
        if template == .wearIt {
            // Her photo is the background; the title is white over it (she can change both).
            style.matchBackground = false
            style.background = .photo
            style.matchTitle = false
            style.titleColor = ShopTemplate.wearIt.ink
        } else if leavingWearIt {
            style = ShopBoardStyle(background: .classic(template.background))
            if videoBackground != nil {                 // a video is only for Wear it
                videoBackground = nil
                videoDuration = 0
                photoBackground = nil
            }
        }
        variant = 0
        clearHandEdits()
    }

    /// Shuffle: another arrangement of the same items (what was moved by hand starts over).
    func shuffle() {
        variant += 1
        clearHandEdits()
    }

    func clearHandEdits() {
        adjusts = [:]
        raised = []
    }

    func raise(_ id: UUID) {
        raised.removeAll { $0 == id }
        raised.append(id)
    }

    func prepareShare() {
        if caption.isEmpty { caption = ShopCaption.defaultCaption(scriptLine: scriptLine) }
        if affiliateLink.isEmpty, let link = items.compactMap(\.affiliateLink).first { affiliateLink = link.absoluteString }
    }

    /// What is posted.
    var finalCaption: String { ShopCaption.compliant(caption, title: bigTitle, link: affiliateLink) }

    /// The name under "Background" on the board screen.
    var backgroundName: String {
        if style.matchBackground {
            if case .texture(let t) = style.background { return L10n.tr(t.title) + " · " + L10n.tr("✨ Match") }
            return L10n.tr("✨ Match my products")
        }
        switch style.background {
        case .color(let c): return L10n.tr(ShopPalette.name(of: c))
        case .classic(let c): return L10n.tr(c.title)
        case .texture(let t): return L10n.tr(t.title)
        case .photo: return L10n.tr("My photo")
        }
    }

    // MARK: Drawing

    /// A style of hers on the carousel (its own words, no hand edits), or her board in it.
    func snapshot(idea: ShopIdeaStyle, handEdited: Bool = false) -> ShopBoardSnapshot {
        let usable = items.filter { $0.cutout != nil }
        let placements = ShopIdeaMatch.placements(idea, kinds: usable.map(\.kind))
        let ids = usable.map(\.id)
        let placedIDs = placements.map { ids[$0.itemIndex] }
        // Her own background choice wins once she changes it; until then, the idea's exact top / bottom colours.
        var top = idea.top, bottom = idea.bottom
        var texture: ShopTexture?
        if handEdited, style.background != .color(idea.top) {
            let colors = style.backgroundColors(palette: palette)
            top = colors.top; bottom = colors.bottom
            if case .texture(let t) = style.background { texture = t }
        }
        return ShopBoardSnapshot(
            images: usable.compactMap(\.cutout), placements: placements, itemIDs: placedIDs,
            adjusts: handEdited ? placedIDs.map { adjusts[$0] ?? ShopAdjust() } : placedIDs.map { _ in ShopAdjust() },
            drawOrder: ShopDrawOrder.order(itemIDs: placedIDs, raised: handEdited ? raised : []),
            titleAdjust: ShopAdjust(), top: top, bottom: bottom, texture: texture,
            textureMatched: false, photo: nil,
            bigTitle: handEdited ? bigTitle : (idea.firstText("title") ?? ""),
            scriptLine: handEdited ? scriptLine : (idea.firstText("subtitle") ?? ""),
            bottomLine: handEdited ? bottomLine : (idea.firstText("brand") ?? ""),
            wearIt: false, pair: fontPair, titleColor: style.titleColor, ideaTexts: idea.texts)
    }

    func snapshot(template override: ShopTemplate? = nil) -> ShopBoardSnapshot {
        if override == nil, let idea { return snapshot(idea: idea, handEdited: true) }
        let tmpl = override ?? template
        var look = style
        var title = bigTitle, script = scriptLine
        if override != nil {           // "Choose a style" previews: the template's own look and texts
            look = ShopBoardStyle(matchBackground: false, background: .classic(tmpl.background), matchTitle: false, titleColor: tmpl.ink)
            title = tmpl.bigTitle
            script = tmpl.scriptLine
        }
        let usable = items.filter { $0.cutout != nil }
        let wear = tmpl == .wearIt
        let placements = wear ? ShopLayout.wearItPlacements(count: usable.count, variant: override == nil ? variant : 0)
            : ShopLayout.placements(kinds: usable.map(\.kind), variant: override == nil ? variant : 0)
        let ids = usable.map(\.id)
        let handEdited = override == nil
        let placedIDs = placements.map { ids[$0.itemIndex] }
        var colors = look.backgroundColors(palette: palette)
        var texture: ShopTexture?
        if case .texture(let t) = look.background { texture = t }
        var photo: UIImage? = { if case .photo = look.background, !look.matchBackground { return photoBackground }; return nil }()
        if wear {
            if override != nil { photo = photoBackground }       // the style tile shows her photo once she has one
            if photo == nil, case .photo = look.background { colors = ShopTemplate.wearItPlaceholder }
            if override != nil, photo == nil { colors = ShopTemplate.wearItPlaceholder }
        }
        return ShopBoardSnapshot(
            images: usable.compactMap(\.cutout), placements: placements,
            itemIDs: placedIDs,
            adjusts: handEdited ? placedIDs.map { adjusts[$0] ?? ShopAdjust() } : placedIDs.map { _ in ShopAdjust() },
            drawOrder: ShopDrawOrder.order(itemIDs: placedIDs, raised: handEdited ? raised : []),
            titleAdjust: handEdited ? (adjusts[Self.titleID] ?? ShopAdjust()) : ShopAdjust(),
            top: colors.top, bottom: colors.bottom, texture: texture, textureMatched: look.matchBackground && texture != nil,
            photo: photo, bigTitle: title, scriptLine: script, bottomLine: bottomLine, wearIt: wear, pair: fontPair,
            titleColor: look.resolvedTitleColor(palette: palette, template: tmpl))
    }
}

// MARK: - The board drawing (9:16)

/// Everything the board shows, as plain values (also used for the 1080 × 1920 export).
struct ShopBoardSnapshot {
    var images: [CGImage]
    var placements: [ShopPlacement]
    /// Per place (same order as `placements`): the product's id, its hand edit, and the drawing order.
    var itemIDs: [UUID] = []
    var adjusts: [ShopAdjust] = []
    var drawOrder: [Int] = []
    /// The title block moved, resized or turned by hand.
    var titleAdjust = ShopAdjust()
    var top: ShopRGB
    var bottom: ShopRGB
    var texture: ShopTexture?
    var textureMatched = false
    var photo: UIImage?
    var bigTitle: String
    var scriptLine: String
    var bottomLine: String = ShopTemplate.footer
    /// Wear it (Figma 3064:197): title in the lower third, a soft shade under it, bottom line just below.
    var wearIt = false
    var pair: ShopFontPair
    var titleColor: ShopRGB
    /// "My idea": the texts exactly where the idea had them (instead of the title block and the brand line).
    var ideaTexts: [ShopIdeaStyle.Text] = []

    var isDark: Bool { top.mixed(bottom, 0.5).luminance < 0.25 }

    /// Wear it: the title block is 596 design units lower (Figma: "Stylist-approved" at y 610).
    var titleShift: CGFloat { wearIt ? 596 : 0 }

    var order: [Int] { drawOrder.count == placements.count ? drawOrder : Array(placements.indices) }

    /// Where the title block is on a board of `size` (after the hand edit), and its tilt.
    func titleFrame(board size: CGSize) -> (rect: CGRect, rotation: Double) {
        let sx = size.width / ShopLayout.designSize.width, sy = size.height / ShopLayout.designSize.height
        let r = ShopLayout.titleRect
        let base = CGRect(x: r.minX * sx, y: (r.minY + titleShift) * sy, width: r.width * sx, height: r.height * sy)
        return (titleAdjust.frame(base: base, board: size), titleAdjust.rotation)
    }

    /// Is `point` on the title block (its turned frame)?
    func isOnTitle(_ point: CGPoint, board size: CGSize) -> Bool {
        let placed = titleFrame(board: size)
        return Self.contains(placed.rect, rotation: placed.rotation, point, minimum: 0)
    }

    static func contains(_ rect: CGRect, rotation: Double, _ point: CGPoint, minimum: CGFloat) -> Bool {
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let r = -rotation * Double.pi / 180
        let dx = Double(point.x - c.x), dy = Double(point.y - c.y)
        let lx = dx * Darwin.cos(r) - dy * Darwin.sin(r), ly = dx * Darwin.sin(r) + dy * Darwin.cos(r)
        return abs(lx) <= Double(max(rect.width, minimum)) / 2 && abs(ly) <= Double(max(rect.height, minimum)) / 2
    }

    /// The top product under `point` (drawn last wins), testing its turned frame.
    func item(at point: CGPoint, board size: CGSize) -> Int? {
        for index in order.reversed() {
            guard let placed = frame(at: index, board: size) else { continue }
            let c = CGPoint(x: placed.rect.midX, y: placed.rect.midY)
            let r = CGFloat(-placed.rotation * Double.pi / 180)
            let dx = point.x - c.x, dy = point.y - c.y
            let cr = CGFloat(Darwin.cos(Double(r))), sr = CGFloat(Darwin.sin(Double(r)))
            let local = CGPoint(x: dx * cr - dy * sr, y: dx * sr + dy * cr)
            // A little room around small products so they are easy to grab.
            let half = CGSize(width: max(placed.rect.width, 44) / 2, height: max(placed.rect.height, 44) / 2)
            if abs(local.x) <= half.width && abs(local.y) <= half.height { return index }
        }
        return nil
    }

    /// Place `index` before any hand edit, in board design units (390 × 844): for the size limits (PO 17 Nov).
    func baseSize(at index: Int) -> CGSize? {
        guard placements.indices.contains(index) else { return nil }
        let placement = placements[index]
        guard images.indices.contains(placement.itemIndex) else { return nil }
        let image = images[placement.itemIndex]
        let aspect = CGFloat(image.width) / CGFloat(max(1, image.height))
        return ShopLayout.frame(for: placement.slot, aspect: aspect, board: ShopLayout.designSize).size
    }

    /// Where place `index` is drawn on a board of `size` (after the hand edit), and its tilt.
    func frame(at index: Int, board size: CGSize) -> (rect: CGRect, rotation: Double)? {
        let placement = placements[index]
        guard images.indices.contains(placement.itemIndex) else { return nil }
        let image = images[placement.itemIndex]
        let aspect = CGFloat(image.width) / CGFloat(max(1, image.height))
        let base = ShopLayout.frame(for: placement.slot, aspect: aspect, board: size)
        let adjust = adjusts.indices.contains(index) ? adjusts[index] : ShopAdjust()
        return (adjust.frame(base: base, board: size), placement.slot.rotation + adjust.rotation)
    }
}

/// The board itself (Figma result 3031:4), drawn at any width; its height is width × 16 / 9. Positions come from the
/// Figma board (390 × 844) scaled to 9:16.
struct ShopBoardCanvas: View {
    let snapshot: ShopBoardSnapshot
    let width: CGFloat
    /// Wear it with a video: everything but the background (her video plays under it / is laid under it on export).
    var transparent = false

    var body: some View {
        let height = width * 16 / 9
        let sx = width / ShopLayout.designSize.width, sy = height / ShopLayout.designSize.height
        let size = CGSize(width: width, height: height)
        ZStack(alignment: .top) {
            background(size: size)
            ForEach(snapshot.order, id: \.self) { index in
                if let placed = snapshot.frame(at: index, board: size) {
                    let frame = placed.rect
                    let rotation = placed.rotation
                    Image(decorative: snapshot.images[snapshot.placements[index].itemIndex], scale: 1)
                        .resizable()
                        .interpolation(.high)
                        .frame(width: frame.width, height: frame.height)
                        .rotationEffect(.degrees(rotation))
                        // Figma: drop shadow 0 8, blur 14, rgba(13,13,26,0.18).
                        .shadow(color: Color(fixed: 0x0D0D1A, opacity: 0.18), radius: 7 * sx, x: 0, y: 8 * sx)
                        .position(x: frame.midX, y: frame.midY)
                }
            }
            if snapshot.ideaTexts.isEmpty {
                texts(sx: sx, sy: sy)
            } else {
                ShopIdeaTexts(texts: snapshot.ideaTexts, title: snapshot.bigTitle, subtitle: snapshot.scriptLine,
                              brand: snapshot.bottomLine, width: width)
            }
        }
        .frame(width: width, height: height)
        .clipped()
    }

    @ViewBuilder
    private func background(size: CGSize) -> some View {
        if transparent {
            Color.clear.frame(width: size.width, height: size.height)
                .overlay(alignment: .bottom) { if snapshot.wearIt { wearItShade(size: size) } }
        } else if let photo = snapshot.photo {
            Image(uiImage: photo).resizable().scaledToFill().frame(width: size.width, height: size.height).clipped()
                .overlay(alignment: .bottom) { if snapshot.wearIt { wearItShade(size: size) } }
        } else if snapshot.wearIt {
            LinearGradient(colors: [snapshot.top.color, snapshot.bottom.color], startPoint: .top, endPoint: .bottom)
                .frame(width: size.width, height: size.height)
                .overlay(alignment: .bottom) { wearItShade(size: size) }
        } else if let texture = snapshot.texture {
            Image(uiImage: ShopTextureRenderer.image(texture, size: size,
                                                     base: snapshot.textureMatched ? (snapshot.top, snapshot.bottom) : nil))
                .resizable()
                .frame(width: size.width, height: size.height)
        } else {
            LinearGradient(colors: [snapshot.top.color, snapshot.bottom.color], startPoint: .top, endPoint: .bottom)
                .frame(width: size.width, height: size.height)
        }
    }

    /// Figma 3064:197 "Sombra abajo": clear → 45 % black over the lower 420 of 844, so white text always reads.
    private func wearItShade(size: CGSize) -> some View {
        LinearGradient(colors: [Color.black.opacity(0), Color.black.opacity(0.45)], startPoint: .top, endPoint: .bottom)
            .frame(width: size.width, height: size.height * 420 / 844)
            .allowsHitTesting(false)
    }

    private func texts(sx: CGFloat, sy: CGFloat) -> some View {
        let ink = snapshot.titleColor.color
        let footerInk = snapshot.isDark || snapshot.photo != nil ? Color(fixed: 0xFFFFFF, opacity: 0.85) : ShopTemplate.footerInk.color
        let titleSize = 44 * sx * snapshot.pair.titleScale
        let t = snapshot.titleAdjust
        return ZStack(alignment: .top) {
            titleBlock(sx: sx, sy: sy, ink: ink, titleSize: titleSize)
                // Moved by hand (PO 13 Nov): around the title block's centre, in design units.
                .scaleEffect(t.scale, anchor: UnitPoint(x: ShopLayout.titleRect.midX / ShopLayout.designSize.width,
                                                        y: ShopLayout.titleRect.midY / 140))
                .rotationEffect(.degrees(t.rotation), anchor: UnitPoint(x: ShopLayout.titleRect.midX / ShopLayout.designSize.width,
                                                                        y: ShopLayout.titleRect.midY / 140))
                .offset(x: t.offset.width * sx, y: (t.offset.height + snapshot.titleShift) * sy)
            Text(verbatim: snapshot.bottomLine.trimmingCharacters(in: .whitespaces).uppercased())
                .font(ShopFonts.font(ShopFontPair.footerFace, size: 14 * sx))
                .tracking(4.2 * sx)
                .foregroundStyle(snapshot.wearIt ? Color.white : footerInk)
                .padding(.top, (snapshot.wearIt ? 712 : 810) * sy)
        }
        .frame(width: width, alignment: .top)
    }

    /// "CUTSELL" + "Everyday chic", in a 140-design-unit-high block from the top of the board.
    @ViewBuilder
    private func titleBlock(sx: CGFloat, sy: CGFloat, ink: Color, titleSize: CGFloat) -> some View {
        if snapshot.wearIt {
            // Figma 3064:197: both lines in the script face, 30 and 34, centred (a light shadow keeps them clear).
            ZStack(alignment: .top) {
                Text(verbatim: snapshot.bigTitle)
                    .font(ShopFonts.font(snapshot.pair.scriptFace, size: 30 * sx))
                    .foregroundStyle(ink)
                    .lineLimit(1).minimumScaleFactor(0.4)
                    .frame(width: width - 24 * sx)
                    .padding(.top, 14 * sy)
                Text(verbatim: snapshot.scriptLine)
                    .font(ShopFonts.font(snapshot.pair.scriptFace, size: 34 * sx))
                    .foregroundStyle(ink)
                    .lineLimit(1).minimumScaleFactor(0.4)
                    .frame(width: width - 24 * sx)
                    .padding(.top, 50 * sy)
            }
            .shadow(color: Color.black.opacity(0.25), radius: 4 * sx, x: 0, y: 1 * sx)
            .frame(width: width, height: 140 * sy, alignment: .top)
        } else {
            classicTitleBlock(sx: sx, sy: sy, ink: ink, titleSize: titleSize)
        }
    }

    private func classicTitleBlock(sx: CGFloat, sy: CGFloat, ink: Color, titleSize: CGFloat) -> some View {
        ZStack(alignment: .top) {
            Text(verbatim: snapshot.bigTitle)
                .font(ShopFonts.font(snapshot.pair.titleFace, size: titleSize, opticalSize: 44 * snapshot.pair.titleScale))
                .tracking(3.52 * sx)
                .foregroundStyle(ink)
                .lineLimit(1).minimumScaleFactor(0.4)
                .frame(width: width - 24 * sx)
                .padding(.top, 24 * sy)
            Text(verbatim: snapshot.scriptLine)
                .font(ShopFonts.font(snapshot.pair.scriptFace, size: 46 * sx))
                .foregroundStyle(ink)
                .lineLimit(1).minimumScaleFactor(0.4)
                .frame(width: width - 24 * sx)
                .padding(.top, 84 * sy)
        }
        .frame(width: width, height: 140 * sy, alignment: .top)
    }
}

enum ShopBoardExport {
    /// Wear it with a video: the board without its background, transparent, 1080 × 1920 (laid over her video).
    @MainActor
    static func overlay(_ snapshot: ShopBoardSnapshot) -> UIImage? {
        let renderer = ImageRenderer(content: ShopBoardCanvas(snapshot: snapshot, width: ShopLayout.exportSize.width, transparent: true)
            .environment(\.colorScheme, .light))
        renderer.scale = 1
        renderer.proposedSize = ProposedViewSize(ShopLayout.exportSize)
        renderer.isOpaque = false
        return renderer.uiImage
    }

    /// The board as a 1080 × 1920 picture (9:16) for Save / Share.
    @MainActor
    static func image(_ snapshot: ShopBoardSnapshot) -> UIImage? {
        let renderer = ImageRenderer(content: ShopBoardCanvas(snapshot: snapshot, width: ShopLayout.exportSize.width)
            .environment(\.colorScheme, .light))
        renderer.scale = 1
        renderer.proposedSize = ProposedViewSize(ShopLayout.exportSize)
        renderer.isOpaque = true
        return renderer.uiImage
    }
}

// MARK: - Shape it (server)

enum ShopShapeClient {
    /// POST /v1/shop/shape with the signed-in session (APIClient: same headers as every call). Never throws: every
    /// failure is a `ShopShapeFailure` with its own message.
    @MainActor
    static func shape(_ image: CGImage, kind: String) async -> Result<CGImage, ShopShapeFailure> {
        // Photos go to an AI service: the same consent as the cuts (App Review 5.1.2).
        guard await AIConsentCenter.shared.ensure() else { return .failure(.failed) }
        guard let png = ShopCutout.shapePNG(image) else { return .failure(.failed) }
        let body = ShopShapeRequestBody(image_png_base64: png.base64EncodedString(), kind: kind)
        do {
            let answer: ShopShapeResponseBody = try await APIClient.shared.request("/v1/shop/shape", method: "POST", body: body,
                                                                                    timeout: 120)
            guard let data = Data(base64Encoded: answer.image_png_base64), let shaped = UIImage(data: data)?.cgImage else {
                return .failure(.failed)
            }
            CutSellDiagnostics.log("shop_shape_ok", ["provider": answer.provider ?? "", "kind": kind])
            return .success(shaped)
        } catch APIError.http(let code, let message) {
            CutSellDiagnostics.log("shop_shape_failed", ["code": String(code)])
            return .failure(.from(status: code, body: message))
        } catch {
            CutSellDiagnostics.log("shop_shape_failed", ["code": "network"])
            return .failure(.failed)
        }
    }
}

// MARK: - Share

enum ShopShare {
    /// "Save to Photos": the 9:16 picture.
    static func saveToPhotos(_ image: UIImage) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { throw CocoaError(.userCancelled) }
        guard let png = image.pngData() else { throw CocoaError(.fileWriteUnknown) }
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetCreationRequest.forAsset().addResource(with: .photo, data: png, options: nil)
        }
    }

    /// The Facebook App ID Instagram asks for (Info.plist "FacebookAppID"); CutSell has none yet.
    static var facebookAppID: String? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: "FacebookAppID") as? String else { return nil }
        let clean = value.trimmingCharacters(in: .whitespaces)
        return clean.isEmpty || clean.hasPrefix("$(") ? nil : clean
    }

    /// "Instagram story": Instagram's Stories composer with the board as the background (instagram-stories://share
    /// + the pasteboard). Needs a Facebook App ID (source_application); without one: false (the caller saves the
    /// picture, copies the link and opens Instagram instead).
    @MainActor
    static func instagramStory(_ image: UIImage, link: String) -> Bool {
        guard let appID = facebookAppID, let png = image.pngData(),
              let url = URL(string: "instagram-stories://share?source_application=\(appID)"),
              UIApplication.shared.canOpenURL(url) else { return false }
        var item: [String: Any] = ["com.instagram.sharedSticker.backgroundImage": png]
        if !link.isEmpty { item["com.instagram.sharedSticker.contentURL"] = link }
        UIPasteboard.general.setItems([item], options: [.expirationDate: Date().addingTimeInterval(300)])
        UIApplication.shared.open(url)
        return true
    }

    /// Instagram's camera (stories) or the app; its App Store page when Instagram is not installed.
    @MainActor
    static func openInstagram() {
        for scheme in ["instagram://story-camera", "instagram://app"] {
            if let url = URL(string: scheme), UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url)
                return
            }
        }
        if let store = URL(string: "https://apps.apple.com/app/id389801252") { UIApplication.shared.open(store) }
    }
}
