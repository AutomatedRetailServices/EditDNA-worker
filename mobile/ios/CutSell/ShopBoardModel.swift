import Photos
import SwiftUI
import UIKit

// Shop board: the board being made (items, template, letters, look), the board drawing itself (9:16), its
// 1080 × 1920 export and the share actions. Screens: ShopBoardViews.swift.

// MARK: - The board being made

@MainActor
final class ShopBoardModel: ObservableObject {
    @Published var items: [ShopItem] = [] { didSet { if oldValue.map(\.cutout) != items.map(\.cutout) { refreshPalette() } } }
    /// Items whose photo is being cut out now.
    @Published var busy: Set<UUID> = []
    @Published var template: ShopTemplate = .everydayChic
    @Published var bigTitle = ShopTemplate.everydayChic.bigTitle
    @Published var scriptLine = ShopTemplate.everydayChic.scriptLine
    @Published var fontPair: ShopFontPair = .classic
    @Published var style = ShopBoardStyle()
    @Published var variant = 0
    @Published var photoBackground: UIImage?
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
    func attach(_ image: UIImage, to id: UUID) {
        busy.insert(id)
        Task.detached(priority: .userInitiated) {
            let result = Self.process(image)
            await MainActor.run {
                self.busy.remove(id)
                guard let index = self.items.firstIndex(where: { $0.id == id }), let result else { return }
                self.items[index].cutout = result.image
                self.items[index].flat = result.image
                self.items[index].backgroundRemoved = result.removed
                self.items[index].kind = result.kind
                self.items[index].shaped = false
            }
        }
    }

    nonisolated static func process(_ image: UIImage) -> (image: CGImage, removed: Bool, kind: ShopItemKind)? {
        guard let photo = ShopCutout.normalized(image) else { return nil }
        if let cut = try? ShopCutout.removeBackground(photo) {
            return (cut, true, ShopVision.kind(of: cut))
        }
        CutSellDiagnostics.log("shop_background_kept", [:])
        return (photo, false, ShopVision.kind(of: photo))
    }

    func remove(_ id: UUID) { items.removeAll { $0.id == id } }

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

    func choose(_ template: ShopTemplate) {
        self.template = template
        bigTitle = template.bigTitle
        scriptLine = template.scriptLine
        if !style.matchBackground, case .classic = style.background { style.background = .classic(template.background) }
        variant = 0
    }

    /// Shuffle: another arrangement of the same items.
    func shuffle() { variant += 1 }

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

    func snapshot(template override: ShopTemplate? = nil) -> ShopBoardSnapshot {
        let tmpl = override ?? template
        var look = style
        var title = bigTitle, script = scriptLine
        if override != nil {           // "Choose a style" previews: the template's own look and texts
            look = ShopBoardStyle(matchBackground: false, background: .classic(tmpl.background), matchTitle: false, titleColor: tmpl.ink)
            title = tmpl.bigTitle
            script = tmpl.scriptLine
        }
        let usable = items.filter { $0.cutout != nil }
        let placements = ShopLayout.placements(kinds: usable.map(\.kind), variant: override == nil ? variant : 0)
        let colors = look.backgroundColors(palette: palette)
        var texture: ShopTexture?
        if case .texture(let t) = look.background { texture = t }
        let photo: UIImage? = { if case .photo = look.background, !look.matchBackground { return photoBackground }; return nil }()
        return ShopBoardSnapshot(
            images: usable.compactMap(\.cutout), placements: placements,
            top: colors.top, bottom: colors.bottom, texture: texture, textureMatched: look.matchBackground && texture != nil,
            photo: photo, bigTitle: title, scriptLine: script, pair: fontPair,
            titleColor: look.resolvedTitleColor(palette: palette, template: tmpl))
    }
}

// MARK: - The board drawing (9:16)

/// Everything the board shows, as plain values (also used for the 1080 × 1920 export).
struct ShopBoardSnapshot {
    var images: [CGImage]
    var placements: [ShopPlacement]
    var top: ShopRGB
    var bottom: ShopRGB
    var texture: ShopTexture?
    var textureMatched = false
    var photo: UIImage?
    var bigTitle: String
    var scriptLine: String
    var pair: ShopFontPair
    var titleColor: ShopRGB

    var isDark: Bool { top.mixed(bottom, 0.5).luminance < 0.25 }
}

/// The board itself (Figma result 3031:4), drawn at any width; its height is width × 16 / 9. Positions come from the
/// Figma board (390 × 844) scaled to 9:16.
struct ShopBoardCanvas: View {
    let snapshot: ShopBoardSnapshot
    let width: CGFloat

    var body: some View {
        let height = width * 16 / 9
        let sx = width / ShopLayout.designSize.width, sy = height / ShopLayout.designSize.height
        let size = CGSize(width: width, height: height)
        ZStack(alignment: .top) {
            background(size: size)
            ForEach(Array(snapshot.placements.enumerated()), id: \.offset) { _, placement in
                if snapshot.images.indices.contains(placement.itemIndex) {
                    let image = snapshot.images[placement.itemIndex]
                    let aspect = CGFloat(image.width) / CGFloat(max(1, image.height))
                    let frame = ShopLayout.frame(for: placement.slot, aspect: aspect, board: size)
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .interpolation(.high)
                        .frame(width: frame.width, height: frame.height)
                        .rotationEffect(.degrees(placement.slot.rotation))
                        // Figma: drop shadow 0 8, blur 14, rgba(13,13,26,0.18).
                        .shadow(color: Color(fixed: 0x0D0D1A, opacity: 0.18), radius: 7 * sx, x: 0, y: 8 * sx)
                        .position(x: frame.midX, y: frame.midY)
                }
            }
            texts(sx: sx, sy: sy)
        }
        .frame(width: width, height: height)
        .clipped()
    }

    @ViewBuilder
    private func background(size: CGSize) -> some View {
        if let photo = snapshot.photo {
            Image(uiImage: photo).resizable().scaledToFill().frame(width: size.width, height: size.height).clipped()
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

    private func texts(sx: CGFloat, sy: CGFloat) -> some View {
        let ink = snapshot.titleColor.color
        let footerInk = snapshot.isDark || snapshot.photo != nil ? Color(fixed: 0xFFFFFF, opacity: 0.85) : ShopTemplate.footerInk.color
        let titleSize = 44 * sx * snapshot.pair.titleScale
        return ZStack(alignment: .top) {
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
            Text(verbatim: ShopTemplate.footer)
                .font(ShopFonts.font(ShopFontPair.footerFace, size: 14 * sx))
                .tracking(4.2 * sx)
                .foregroundStyle(footerInk)
                .padding(.top, 810 * sy)
        }
        .frame(width: width, alignment: .top)
    }
}

enum ShopBoardExport {
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
