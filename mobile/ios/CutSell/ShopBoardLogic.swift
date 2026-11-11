import CoreGraphics
import Foundation

// Shop board (PO 11 Nov 2026, Figma page "💡 10 · Shop board (propuesta)"): the rules of the feature, without views
// (unit-tested in CutSellUnitTests/ShopBoardTests.swift). Screens: ShopBoardViews.swift; pictures: ShopBoardImaging.swift.

// MARK: - Items

/// What an item is, read on the iPhone from Vision's labels. Only clothing and shoes get "✦ Shape it".
enum ShopItemKind: String, Codable, Equatable, CaseIterable {
    case top, bottom, clothing, shoes, bag, accessory, unknown

    var isClothing: Bool { self == .top || self == .bottom || self == .clothing }
    /// "✦ Shape it" is offered for clothing and shoes only (bags, accessories and anything unsure: no).
    var canShape: Bool { isClothing || self == .shoes }
    /// The `kind` sent to POST /v1/shop/shape.
    var shapeKind: String? { isClothing ? "clothing" : (self == .shoes ? "shoes" : nil) }
}

/// Where an item came from (Figma 2: From my products · From photos · Paste a link).
enum ShopItemSource: String, Codable, Equatable { case myProducts, photos, link }

/// One product on the board.
struct ShopItem: Identifiable, Equatable {
    let id: UUID
    var source: ShopItemSource
    /// The product's name (From my products) or the store's name (Paste a link).
    var name: String
    /// The link pasted for this product: kept as its affiliate link. Nothing is ever downloaded from it.
    var affiliateLink: URL?
    /// The catalog product (From my products).
    var productID: String?
    /// The cut-out (PNG with alpha), nil until a photo was added from Photos.
    var cutout: CGImage?
    /// The original cut-out before "✦ Shape it" (to go back to flat).
    var flat: CGImage?
    var kind: ShopItemKind = .unknown
    var shaped = false
    /// Vision removed the background ("✓ No background").
    var backgroundRemoved = false

    init(id: UUID = UUID(), source: ShopItemSource, name: String = "", affiliateLink: URL? = nil,
         productID: String? = nil, cutout: CGImage? = nil, kind: ShopItemKind = .unknown) {
        self.id = id
        self.source = source
        self.name = name
        self.affiliateLink = affiliateLink
        self.productID = productID
        self.cutout = cutout
        self.flat = cutout
        self.kind = kind
    }

    /// A product picked from a link or the product list still needs its photo from Photos.
    var needsPhoto: Bool { cutout == nil }

    static func == (a: ShopItem, b: ShopItem) -> Bool {
        a.id == b.id && a.name == b.name && a.affiliateLink == b.affiliateLink && a.kind == b.kind && a.shaped == b.shaped
            && (a.cutout == nil) == (b.cutout == nil) && a.cutout.map(ObjectIdentifier.init) == b.cutout.map(ObjectIdentifier.init)
    }
}

enum ShopBoardRules {
    /// "Pick 3–8 items for your board."
    static let minItems = 3
    static let maxItems = 8

    /// "Choose a style" works with 3–8 items that all have their photo.
    static func canChooseStyle(_ items: [ShopItem]) -> Bool {
        (minItems...maxItems).contains(items.count) && items.allSatisfy { !$0.needsPhoto }
    }

    /// How many more photos can be picked now.
    static func remaining(_ items: [ShopItem]) -> Int { max(0, maxItems - items.count) }
}

// MARK: - Classification (Vision labels → kind)

/// Maps `VNClassifyImageRequest` labels (identifier, confidence) to an item kind. Pure: Vision runs elsewhere.
/// Word lists by kind; the best kind must reach `minConfidence` and beat every other kind by `margin`, else
/// `.unknown` (when unsure: no "Shape it").
enum ShopItemClassifier {
    static let minConfidence: Float = 0.3
    static let margin: Float = 0.1

    static let shoeWords: Set<String> = ["shoe", "shoes", "sneaker", "sneakers", "footwear", "boot", "boots", "sandal",
                                         "sandals", "heels", "heel", "loafer", "loafers", "slipper", "slippers", "clog",
                                         "clogs", "moccasin", "pump", "pumps", "espadrille", "flipflop", "trainers"]
    static let bagWords: Set<String> = ["bag", "bags", "handbag", "purse", "backpack", "tote", "clutch", "wallet",
                                        "luggage", "suitcase", "satchel", "briefcase", "pouch", "duffel"]
    static let accessoryWords: Set<String> = ["jewelry", "jewellery", "earring", "earrings", "necklace", "bracelet", "ring",
                                              "watch", "wristwatch", "sunglasses", "eyeglasses", "glasses", "eyewear",
                                              "hat", "cap", "beanie", "scarf", "belt", "tie", "necktie", "glove", "gloves",
                                              "headband", "accessory", "accessories", "brooch", "pendant", "hairclip",
                                              "umbrella", "keychain"]
    static let bottomWords: Set<String> = ["pants", "trousers", "jeans", "shorts", "skirt", "leggings", "joggers", "chinos",
                                           "sweatpants", "slacks"]
    static let topWords: Set<String> = ["shirt", "tshirt", "blouse", "top", "sweater", "sweatshirt", "hoodie", "cardigan",
                                        "jacket", "coat", "blazer", "vest", "polo", "jersey", "tank", "pullover", "jumper",
                                        "parka", "outerwear", "knitwear", "tunic", "camisole"]
    static let clothingWords: Set<String> = ["clothing", "clothes", "apparel", "garment", "dress", "gown", "jumpsuit",
                                             "romper", "suit", "denim", "swimwear", "swimsuit", "lingerie", "outfit",
                                             "fashion", "textile", "uniform"]

    /// "t-shirt", "tank_top", "Running Shoe" → ["t", "shirt", …] plus the whole word without separators.
    static func words(_ label: String) -> Set<String> {
        let lower = label.lowercased()
        var out = Set(lower.split(whereSeparator: { !$0.isLetter }).map(String.init))
        out.insert(lower.filter(\.isLetter))
        return out
    }

    /// The kind one label points to (shoes and bags before generic clothing), nil if none.
    static func kind(ofLabel label: String) -> ShopItemKind? {
        let w = words(label)
        if !w.isDisjoint(with: shoeWords) { return .shoes }
        if !w.isDisjoint(with: bagWords) { return .bag }
        if !w.isDisjoint(with: accessoryWords) { return .accessory }
        if !w.isDisjoint(with: bottomWords) { return .bottom }
        if !w.isDisjoint(with: topWords) { return .top }
        if !w.isDisjoint(with: clothingWords) { return .clothing }
        return nil
    }

    static func kind(fromLabels labels: [(identifier: String, confidence: Float)]) -> ShopItemKind {
        var best: [ShopItemKind: Float] = [:]
        for label in labels {
            guard let kind = kind(ofLabel: label.identifier) else { continue }
            best[kind] = max(best[kind] ?? 0, label.confidence)
        }
        // Generic "clothing" supports a more precise clothing kind (top / bottom) instead of competing with it.
        if let generic = best[.clothing] {
            for precise in [ShopItemKind.top, .bottom] where best[precise] != nil {
                best[precise] = max(best[precise] ?? 0, generic)
            }
            if best[.top] != nil || best[.bottom] != nil { best[.clothing] = nil }
        }
        let ranked = best.sorted { $0.value > $1.value }
        guard let first = ranked.first, first.value >= minConfidence else { return .unknown }
        if ranked.count > 1 {
            let second = ranked[1]
            // Two clothing kinds close together are still clothing (Shape it works on both).
            if first.key.isClothing && second.key.isClothing {
                return first.value - second.value >= margin ? first.key : .clothing
            }
            if first.value - second.value < margin { return .unknown }
        }
        return first.key
    }
}

// MARK: - Links (Paste a link)

/// "Paste a link": the link is kept as the product's affiliate link. Nothing is downloaded or opened; the photo
/// comes from Photos.
enum ShopLink {
    /// A pasted text → an http(s) link (adds https:// when missing), or nil.
    static func url(from text: String) -> URL? {
        var raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let found = raw.split(whereSeparator: { $0.isWhitespace }).first(where: { $0.contains(".") }) { raw = String(found) }
        guard !raw.isEmpty, !raw.contains(" ") else { return nil }
        if !raw.lowercased().hasPrefix("http://") && !raw.lowercased().hasPrefix("https://") { raw = "https://" + raw }
        guard let url = URL(string: raw), let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http",
              let host = url.host, host.contains("."), !host.hasPrefix("."), !host.hasSuffix(".") else { return nil }
        return url
    }

    /// amazon.* (amazon.com, amazon.co.uk, smile.amazon.de …), amzn.to, amzn.com, a.co.
    static func isAmazon(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased() else { return false }
        return isAmazonHost(host)
    }

    static func isAmazonHost(_ rawHost: String) -> Bool {
        let host = rawHost.hasSuffix(".") ? String(rawHost.dropLast()) : rawHost
        if ["amzn.to", "amzn.com", "a.co", "amzn.eu", "amzn.asia"].contains(host) { return true }
        if host.hasSuffix(".amzn.to") || host.hasSuffix(".a.co") { return true }
        let labels = host.split(separator: ".").map(String.init)
        guard let index = labels.lastIndex(of: "amazon") else { return false }
        let tld = labels[(index + 1)...]
        return (1...2).contains(tld.count) && tld.allSatisfy { (2...3).contains($0.count) }
    }

    /// The store's name for the item card ("amazon.com" → "Amazon", "www.walmart.com" → "Walmart").
    static func storeName(_ url: URL) -> String {
        if isAmazon(url) { return "Amazon" }
        guard var host = url.host?.lowercased() else { return "" }
        if host.hasPrefix("www.") { host.removeFirst(4) }
        let labels = host.split(separator: ".")
        let name = labels.count >= 2 ? labels[labels.count - 2] : labels.first ?? ""
        return name.prefix(1).uppercased() + name.dropFirst()
    }
}

/// Whatever loads data from the network. The link handler takes one so the tests can check it is never used.
protocol ShopNetworkLoading {
    func data(from url: URL) async throws -> Data
}

/// "Paste a link": makes the item that keeps the link. It never downloads anything (no page, no picture).
struct ShopLinkHandler {
    /// Never called (kept to make the promise testable: a spy here sees zero calls).
    let network: ShopNetworkLoading?

    init(network: ShopNetworkLoading? = nil) { self.network = network }

    /// The new item (no photo yet: the creator adds it from Photos), or nil for something that is not a link.
    func item(fromPasted text: String) -> ShopItem? {
        guard let url = ShopLink.url(from: text) else { return nil }
        return ShopItem(source: .link, name: ShopLink.storeName(url), affiliateLink: url)
    }
}

// MARK: - Caption (#ad, Amazon Associates)

enum ShopCaption {
    static let amazonDisclosure = "As an Amazon Associate I earn from qualifying purchases."

    /// Figma 5: the caption she starts from.
    static func defaultCaption(scriptLine: String) -> String {
        let line = scriptLine.trimmingCharacters(in: .whitespacesAndNewlines)
        return (line.isEmpty ? "" : line + " ") + "for fall 🍂 Shop the look — link in bio. #ad #affiliate"
    }

    /// True when "#ad" is already there as its own tag (not "#adorable").
    static func hasAdTag(_ caption: String) -> Bool {
        caption.range(of: "(?i)#ad(?![\\p{L}\\p{N}_])", options: .regularExpression) != nil
    }

    /// The title says Amazon, or the title / link is an Amazon link (amazon.* / amzn.to / a.co).
    static func mentionsAmazon(title: String, link: String) -> Bool {
        if title.range(of: "(?i)\\bamazon\\b", options: .regularExpression) != nil { return true }
        for text in [link, title] {
            for token in text.split(whereSeparator: { $0.isWhitespace }) where token.contains(".") {
                if let url = ShopLink.url(from: String(token)), ShopLink.isAmazon(url) { return true }
            }
        }
        return false
    }

    /// What is posted: the caption with "#ad" (added when missing) and, for Amazon, the Associates line.
    static func compliant(_ caption: String, title: String, link: String) -> String {
        var out = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        if !hasAdTag(out) { out = out.isEmpty ? "#ad" : out + " #ad" }
        if mentionsAmazon(title: title, link: link) && !out.contains(amazonDisclosure) {
            out += "\n\n" + amazonDisclosure
        }
        return out
    }

    /// "Copy link & caption".
    static func clipboard(caption: String, title: String, link: String) -> String {
        let text = compliant(caption, title: title, link: link)
        let clean = link.trimmingCharacters(in: .whitespacesAndNewlines)
        return clean.isEmpty ? text : text + "\n\n" + clean
    }
}

// MARK: - Colours (Match my products)

/// sRGB colour, 0…1.
struct ShopRGB: Equatable, Hashable, Codable {
    var r: Double, g: Double, b: Double

    init(_ r: Double, _ g: Double, _ b: Double) { self.r = r; self.g = g; self.b = b }
    init(hex: UInt32) {
        self.init(Double((hex >> 16) & 0xFF) / 255, Double((hex >> 8) & 0xFF) / 255, Double(hex & 0xFF) / 255)
    }

    var hex: UInt32 {
        func byte(_ v: Double) -> UInt32 { UInt32(max(0, min(255, (v * 255).rounded()))) }
        return byte(r) << 16 | byte(g) << 8 | byte(b)
    }

    /// WCAG relative luminance.
    var luminance: Double {
        func lin(_ c: Double) -> Double { c <= 0.03928 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
        return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    }

    /// Hue 0…1, saturation 0…1, lightness 0…1.
    var hsl: (h: Double, s: Double, l: Double) {
        let maxC = max(r, g, b), minC = min(r, g, b), l = (maxC + minC) / 2, d = maxC - minC
        guard d > 0.0001 else { return (0, 0, l) }
        let s = d / (1 - abs(2 * l - 1))
        var h: Double
        if maxC == r { h = ((g - b) / d).truncatingRemainder(dividingBy: 6) }
        else if maxC == g { h = (b - r) / d + 2 } else { h = (r - g) / d + 4 }
        h /= 6
        if h < 0 { h += 1 }
        return (h, min(1, s), l)
    }

    static func fromHSL(_ h: Double, _ s: Double, _ l: Double) -> ShopRGB {
        let c = (1 - abs(2 * l - 1)) * s
        let hh = (h - floor(h)) * 6
        let x = c * (1 - abs(hh.truncatingRemainder(dividingBy: 2) - 1))
        let m = l - c / 2
        let (r1, g1, b1): (Double, Double, Double)
        switch Int(hh) {
        case 0: (r1, g1, b1) = (c, x, 0)
        case 1: (r1, g1, b1) = (x, c, 0)
        case 2: (r1, g1, b1) = (0, c, x)
        case 3: (r1, g1, b1) = (0, x, c)
        case 4: (r1, g1, b1) = (x, 0, c)
        default: (r1, g1, b1) = (c, 0, x)
        }
        return ShopRGB(r1 + m, g1 + m, b1 + m)
    }

    func distance2(_ o: ShopRGB) -> Double { (r - o.r) * (r - o.r) + (g - o.g) * (g - o.g) + (b - o.b) * (b - o.b) }

    /// Mixed with `other` (0 = self, 1 = other).
    func mixed(_ other: ShopRGB, _ t: Double) -> ShopRGB {
        ShopRGB(r + (other.r - r) * t, g + (other.g - g) * t, b + (other.b - b) * t)
    }
}

enum ShopPalette {
    /// WCAG contrast ratio (1…21).
    static func contrast(_ a: ShopRGB, _ b: ShopRGB) -> Double {
        let la = a.luminance, lb = b.luminance
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    /// k-means on the opaque pixels of the cut-outs (transparent pixels are left out by the caller). Starts from
    /// the farthest points (deterministic), returns the clusters by weight, heaviest first.
    static func kMeans(_ pixels: [ShopRGB], k: Int, iterations: Int = 12) -> [(color: ShopRGB, weight: Double)] {
        guard !pixels.isEmpty, k > 0 else { return [] }
        // Deterministic start: the mean, then repeatedly the pixel farthest from the centres so far.
        let mean = ShopRGB(pixels.map(\.r).reduce(0, +) / Double(pixels.count), pixels.map(\.g).reduce(0, +) / Double(pixels.count),
                           pixels.map(\.b).reduce(0, +) / Double(pixels.count))
        var centres = [pixels.min { $0.distance2(mean) < $1.distance2(mean) }!]
        while centres.count < min(k, pixels.count) {
            guard let far = pixels.max(by: { a, b in
                centres.map { a.distance2($0) }.min()! < centres.map { b.distance2($0) }.min()!
            }), centres.map({ far.distance2($0) }).min()! > 1e-6 else { break }
            centres.append(far)
        }
        var counts = [Int](repeating: 0, count: centres.count)
        for _ in 0..<max(1, iterations) {
            var sums = [(Double, Double, Double)](repeating: (0, 0, 0), count: centres.count)
            counts = [Int](repeating: 0, count: centres.count)
            for p in pixels {
                var best = 0, bestD = Double.greatestFiniteMagnitude
                for (i, c) in centres.enumerated() {
                    let d = p.distance2(c)
                    if d < bestD { bestD = d; best = i }
                }
                sums[best].0 += p.r; sums[best].1 += p.g; sums[best].2 += p.b
                counts[best] += 1
            }
            var moved = false
            for i in centres.indices where counts[i] > 0 {
                let n = Double(counts[i])
                let next = ShopRGB(sums[i].0 / n, sums[i].1 / n, sums[i].2 / n)
                if next.distance2(centres[i]) > 1e-8 { moved = true }
                centres[i] = next
            }
            if !moved { break }
        }
        let total = Double(pixels.count)
        return zip(centres, counts).filter { $0.1 > 0 }.map { ($0.0, Double($0.1) / total) }.sorted { $0.weight > $1.weight }
    }

    /// The products' colours: a few clusters per product (every product counts the same), near-duplicates merged.
    static func palette(perItem: [[ShopRGB]], perItemColors: Int = 2, maxColors: Int = 5) -> [ShopRGB] {
        var weighted: [(ShopRGB, Double)] = []
        for pixels in perItem where !pixels.isEmpty {
            for cluster in kMeans(pixels, k: perItemColors) where cluster.weight >= 0.15 {
                weighted.append((cluster.color, cluster.weight))
            }
        }
        weighted.sort { $0.1 > $1.1 }
        var out: [ShopRGB] = []
        for (color, _) in weighted where !out.contains(where: { $0.distance2(color) < 0.012 }) {
            out.append(color)
            if out.count == maxColors { break }
        }
        return out
    }

    /// ✨ Match my products: a soft, light background in the products' overall hue, kept nearly neutral so every
    /// item stands out (top and bottom of the gentle board gradient).
    static func softBackground(for palette: [ShopRGB]) -> (top: ShopRGB, bottom: ShopRGB) {
        guard !palette.isEmpty else { return (ShopRGB(hex: 0xF1ECE4), ShopRGB(hex: 0xE4DDD2)) }   // Cream (Figma)
        // Average hue on the colour circle, weighted by how colourful each colour is.
        var x = 0.0, y = 0.0, sat = 0.0
        for c in palette {
            let hsl = c.hsl
            let w = max(0.05, hsl.s * (1 - abs(2 * hsl.l - 1)))
            x += cos(hsl.h * 2 * .pi) * w; y += sin(hsl.h * 2 * .pi) * w
            sat += hsl.s
        }
        var hue = atan2(y, x) / (2 * .pi)
        if hue < 0 { hue += 1 }
        let avgSat = sat / Double(palette.count)
        let s = min(0.16, 0.06 + avgSat * 0.12)        // soft: never a strong colour
        let top = ShopRGB.fromHSL(hue, s, 0.94)
        let bottom = ShopRGB.fromHSL(hue, s * 1.1, 0.885)
        return (top, bottom)
    }

    /// The title colour: the product colour with the best contrast on the background (at least 4.5 : 1); if none
    /// reaches it, that colour is darkened (or lightened on a dark background) until it does.
    static func titleColor(for palette: [ShopRGB], on background: ShopRGB) -> ShopRGB {
        let ink = ShopRGB(hex: 0x2A2320)                    // the Figma title colour
        guard let best = palette.max(by: { contrast($0, background) < contrast($1, background) }) else { return ink }
        if contrast(best, background) >= 4.5 { return best }
        let target = background.luminance > 0.4 ? ShopRGB(0, 0, 0) : ShopRGB(1, 1, 1)
        var c = best
        for step in 1...20 {
            c = best.mixed(target, Double(step) * 0.05)
            if contrast(c, background) >= 4.5 { return c }
        }
        return target
    }

    /// The name under a swatch ("Navy", "Sky", "Gray", "White", "Brown" as in Figma, and the nearest of a few more).
    static let names: [(String, UInt32)] = [
        ("Navy", 0x1E2A4A), ("Sky", 0x7DA4E6), ("Gray", 0x9AA0A8), ("White", 0xF4F4F2), ("Brown", 0x6E4E36),
        ("Black", 0x151515), ("Beige", 0xD8C7AE), ("Red", 0xB83A3A), ("Pink", 0xE8A0B4), ("Green", 0x4E7A4E),
        ("Olive", 0x7A7A3A), ("Yellow", 0xE8C84A), ("Orange", 0xE07B39), ("Purple", 0x6E4A8E), ("Blue", 0x2F5FBF),
        ("Teal", 0x2E8A8A), ("Cream", 0xF1ECE4), ("Burgundy", 0x6E1F2E),
    ]

    static func name(of color: ShopRGB) -> String {
        names.min { color.distance2(ShopRGB(hex: $0.1)) < color.distance2(ShopRGB(hex: $1.1)) }?.0 ?? ""
    }
}

// MARK: - Backgrounds

/// "Classic" colours (Figma 4b, gradient top → bottom).
enum ShopClassicColor: String, CaseIterable, Identifiable {
    case cream, beige, stone, white, blush, sage, sky, night
    var id: String { rawValue }
    var title: String { rawValue.prefix(1).uppercased() + rawValue.dropFirst() }
    var top: ShopRGB {
        switch self {
        case .cream: ShopRGB(hex: 0xF1ECE4)
        case .beige: ShopRGB(hex: 0xEFE7DA)
        case .stone: ShopRGB(hex: 0xE9E9E6)
        case .white: ShopRGB(hex: 0xFFFFFF)
        case .blush: ShopRGB(hex: 0xF7E8E6)
        case .sage: ShopRGB(hex: 0xE6ECE2)
        case .sky: ShopRGB(hex: 0xE7EEFF)
        case .night: ShopRGB(hex: 0x1B2133)
        }
    }
    var bottom: ShopRGB {
        switch self {
        case .cream: ShopRGB(hex: 0xE4DDD2)
        case .beige: ShopRGB(hex: 0xE2D6C4)
        case .stone: ShopRGB(hex: 0xD9D9D4)
        case .white: ShopRGB(hex: 0xF3F0EA)
        case .blush: ShopRGB(hex: 0xEFD6D2)
        case .sage: ShopRGB(hex: 0xD3DDCC)
        case .sky: ShopRGB(hex: 0xD6E2FF)
        case .night: ShopRGB(hex: 0x0B1020)
        }
    }
}

/// "Textures" (Figma 4c). Drawn in code on the iPhone (ShopTextureRenderer): no image files.
enum ShopTexture: String, CaseIterable, Identifiable {
    case linen, paper, marble, wood, knit, satin, concrete, velvet
    var id: String { rawValue }
    var title: String { rawValue.prefix(1).uppercased() + rawValue.dropFirst() }
    /// The Figma swatch gradient (top → bottom), the base the texture is drawn on.
    var top: ShopRGB {
        switch self {
        case .linen: ShopRGB(hex: 0xEDE6DA)
        case .paper: ShopRGB(hex: 0xF6F2EA)
        case .marble: ShopRGB(hex: 0xF2F2F0)
        case .wood: ShopRGB(hex: 0xC9A27E)
        case .knit: ShopRGB(hex: 0xE8E2DA)
        case .satin: ShopRGB(hex: 0xF4E9E4)
        case .concrete: ShopRGB(hex: 0xD9D9D6)
        case .velvet: ShopRGB(hex: 0x3A2340)
        }
    }
    var bottom: ShopRGB {
        switch self {
        case .linen: ShopRGB(hex: 0xDCD2C2)
        case .paper: ShopRGB(hex: 0xE9E2D4)
        case .marble: ShopRGB(hex: 0xDADAD6)
        case .wood: ShopRGB(hex: 0xA67C55)
        case .knit: ShopRGB(hex: 0xD4CBBE)
        case .satin: ShopRGB(hex: 0xE2CFC6)
        case .concrete: ShopRGB(hex: 0xBDBDB8)
        case .velvet: ShopRGB(hex: 0x22142A)
        }
    }
    var isDark: Bool { self == .velvet }
}

/// What is behind the products.
enum ShopBackgroundChoice: Equatable {
    case color(ShopRGB)                     // a product colour or Custom
    case classic(ShopClassicColor)
    case texture(ShopTexture)
    case photo
}

/// Board look: background + title colour, with ✨ Match my products (on by default).
struct ShopBoardStyle: Equatable {
    var matchBackground = true
    var background: ShopBackgroundChoice = .classic(.cream)
    var matchTitle = true
    var titleColor = ShopRGB(hex: 0x2A2320)

    /// Picking a colour (or a classic, a texture or a photo) by hand turns Match off.
    mutating func pick(_ choice: ShopBackgroundChoice) {
        background = choice
        matchBackground = false
    }

    /// Match back on. With a texture chosen, Match tints the texture; otherwise the matched colour shows.
    mutating func turnOnMatch() {
        matchBackground = true
        if case .texture = background { return }
        if case .photo = background { background = .classic(.cream) }
    }

    mutating func pickTitleColor(_ color: ShopRGB) {
        titleColor = color
        matchTitle = false
    }

    /// The gradient behind the items (photo: drawn by the view; this is its fallback).
    func backgroundColors(palette: [ShopRGB]) -> (top: ShopRGB, bottom: ShopRGB) {
        if matchBackground { return ShopPalette.softBackground(for: palette) }
        switch background {
        case .color(let c):
            let hsl = c.hsl
            return (c, ShopRGB.fromHSL(hsl.h, hsl.s, max(0, hsl.l - 0.05)))
        case .classic(let classic): return (classic.top, classic.bottom)
        case .texture(let t): return (t.top, t.bottom)
        case .photo: return (ShopRGB(hex: 0xF1ECE4), ShopRGB(hex: 0xE4DDD2))
        }
    }

    /// The title's colour now.
    func resolvedTitleColor(palette: [ShopRGB], template: ShopTemplate) -> ShopRGB {
        guard matchTitle else { return titleColor }
        let bg = backgroundColors(palette: palette)
        let middle = bg.top.mixed(bg.bottom, 0.5)
        if palette.isEmpty { return ShopPalette.contrast(template.ink, middle) >= 4.5 ? template.ink : ShopPalette.titleColor(for: [template.ink], on: middle) }
        return ShopPalette.titleColor(for: palette, on: middle)
    }
}

// MARK: - Letters (Title & fonts)

/// One face bundled for Shop board (Fonts/ShopBoard, SIL Open Font License). `weight` picks a weight of a
/// variable font; `opticalSize` its opsz axis.
struct ShopFontFace: Equatable, Hashable {
    let postScriptName: String
    let fileName: String
    var weight: CGFloat? = nil
    var variable = false
}

/// "Letter style · all free" (Figma 4a): a big title face + a script face.
enum ShopFontPair: String, CaseIterable, Identifiable {
    case classic, editorial, bold, elegant, soft, modern
    var id: String { rawValue }
    var title: String { rawValue.prefix(1).uppercased() + rawValue.dropFirst() }

    var titleFace: ShopFontFace {
        switch self {
        case .classic: ShopFontFace(postScriptName: "BodoniModa-Regular", fileName: "ShopBoard-BodoniModa-Variable", weight: 400, variable: true)
        case .editorial: ShopFontFace(postScriptName: "PlayfairDisplay-Regular", fileName: "ShopBoard-PlayfairDisplay-Variable", weight: 400, variable: true)
        case .bold: ShopFontFace(postScriptName: "BebasNeue-Regular", fileName: "ShopBoard-BebasNeue-Regular")
        case .elegant: ShopFontFace(postScriptName: "Cinzel-Regular", fileName: "ShopBoard-Cinzel-Variable", weight: 400, variable: true)
        case .soft: ShopFontFace(postScriptName: "CormorantGaramond-Light", fileName: "ShopBoard-CormorantGaramond-Variable", weight: 400, variable: true)
        case .modern: ShopFontFace(postScriptName: "Italiana-Regular", fileName: "ShopBoard-Italiana-Regular")
        }
    }

    var scriptFace: ShopFontFace {
        switch self {
        case .classic: ShopFontFace(postScriptName: "GreatVibes-Regular", fileName: "ShopBoard-GreatVibes-Regular")
        case .editorial: ShopFontFace(postScriptName: "Parisienne-Regular", fileName: "ShopBoard-Parisienne-Regular")
        case .bold: ShopFontFace(postScriptName: "DancingScript-Regular", fileName: "ShopBoard-DancingScript-Variable", weight: 400, variable: true)
        case .elegant: ShopFontFace(postScriptName: "PinyonScript-Regular", fileName: "ShopBoard-PinyonScript-Regular")
        case .soft: ShopFontFace(postScriptName: "Allura-Regular", fileName: "ShopBoard-Allura-Regular")
        case .modern: ShopFontFace(postScriptName: "Sacramento-Regular", fileName: "ShopBoard-Sacramento-Regular")
        }
    }

    /// Bebas Neue is drawn larger in Figma (24 vs 18 on the cards) because its capitals are narrow.
    var titleScale: CGFloat { self == .bold ? 24.0 / 18.0 : 1 }

    /// The footer face ("YOUR BRAND", Cormorant Garamond Medium in every Figma board).
    static let footerFace = ShopFontFace(postScriptName: "CormorantGaramond-Light", fileName: "ShopBoard-CormorantGaramond-Variable",
                                         weight: 500, variable: true)
}

/// Font credits (Profile › Help & Support › Font credits): every bundled family, its author and licence.
struct ShopFontCredit: Identifiable, Equatable {
    let family: String
    let designer: String
    let licenseFile: String
    var id: String { family }
    static let license = "SIL Open Font License 1.1"

    static let all: [ShopFontCredit] = [
        ShopFontCredit(family: "Bodoni Moda", designer: "Owen Earl", licenseFile: "OFL-BodoniModa"),
        ShopFontCredit(family: "Great Vibes", designer: "Robert Leuschke", licenseFile: "OFL-GreatVibes"),
        ShopFontCredit(family: "Playfair Display", designer: "Claus Eggers Sørensen", licenseFile: "OFL-PlayfairDisplay"),
        ShopFontCredit(family: "Parisienne", designer: "Astigmatic", licenseFile: "OFL-Parisienne"),
        ShopFontCredit(family: "Bebas Neue", designer: "Ryoichi Tsunekawa (Dharma Type)", licenseFile: "OFL-BebasNeue"),
        ShopFontCredit(family: "Dancing Script", designer: "Impallari Type", licenseFile: "OFL-DancingScript"),
        ShopFontCredit(family: "Cinzel", designer: "Natanael Gama", licenseFile: "OFL-Cinzel"),
        ShopFontCredit(family: "Pinyon Script", designer: "Nicole Fally", licenseFile: "OFL-PinyonScript"),
        ShopFontCredit(family: "Cormorant Garamond", designer: "Christian Thalmann", licenseFile: "OFL-CormorantGaramond"),
        ShopFontCredit(family: "Allura", designer: "Robert Leuschke", licenseFile: "OFL-Allura"),
        ShopFontCredit(family: "Italiana", designer: "Santiago Orozco", licenseFile: "OFL-Italiana"),
        ShopFontCredit(family: "Sacramento", designer: "Astigmatic", licenseFile: "OFL-Sacramento"),
    ]
}

// MARK: - Templates and layout

/// "Choose a style" (Figma 3 and the example results 3031:4 / 3031:29 / 3031:54).
enum ShopTemplate: String, CaseIterable, Identifiable {
    case everydayChic, trendyInspo, lookForLess
    var id: String { rawValue }

    var title: String {
        switch self {
        case .everydayChic: "Everyday chic"
        case .trendyInspo: "Trendy inspo"
        case .lookForLess: "Look for less"
        }
    }
    /// The board's own texts (Figma results).
    var bigTitle: String { self == .lookForLess ? "HOLIDAY" : "CUTSELL" }
    var scriptLine: String {
        switch self {
        case .everydayChic: "Everyday chic"
        case .trendyInspo: "trendy outfit inspo"
        case .lookForLess: "look for less"
        }
    }
    var background: ShopClassicColor {
        switch self {
        case .everydayChic: .cream
        case .trendyInspo: .beige
        case .lookForLess: .white
        }
    }
    /// Title colour of the template.
    var ink: ShopRGB {
        switch self {
        case .everydayChic: ShopRGB(hex: 0x2A2320)
        case .trendyInspo: ShopRGB(hex: 0x3A2416)
        case .lookForLess: ShopRGB(hex: 0x2F4A7A)
        }
    }
    static let footerInk = ShopRGB(hex: 0x4A403A)
    static let footer = "YOUR BRAND"
}

/// Where items go, in the Figma board's own coordinates (390 × 844): centre, box and rotation.
struct ShopSlot: Equatable {
    var center: CGPoint
    var size: CGSize
    var rotation: Double            // degrees
    var prefers: [ShopItemKind]
}

/// One item placed on the board.
struct ShopPlacement: Equatable {
    let itemIndex: Int
    let slot: ShopSlot
}

enum ShopLayout {
    static let designSize = CGSize(width: 390, height: 844)

    /// The six places of the Figma board (shirt, sweater, earrings, pants, bag, shoes), in Figma's drawing order.
    static let figmaSlots: [ShopSlot] = [
        ShopSlot(center: CGPoint(x: 277.82, y: 255.69), size: CGSize(width: 170, height: 200), rotation: 4, prefers: [.top, .clothing]),
        ShopSlot(center: CGPoint(x: 129.10, y: 265.67), size: CGSize(width: 190, height: 161.5), rotation: -3, prefers: [.top, .clothing]),
        ShopSlot(center: CGPoint(x: 72, y: 167), size: CGSize(width: 72, height: 54), rotation: 0, prefers: [.accessory]),
        ShopSlot(center: CGPoint(x: 243.75, y: 517.5), size: CGSize(width: 187.5, height: 375), rotation: 0, prefers: [.bottom, .clothing]),
        ShopSlot(center: CGPoint(x: 87.36, y: 559.07), size: CGSize(width: 172.5, height: 161), rotation: 6, prefers: [.bag]),
        ShopSlot(center: CGPoint(x: 268.65, y: 742.17), size: CGSize(width: 161.5, height: 76), rotation: 3, prefers: [.shoes]),
    ]
    /// Items 7 and 8 (Figma shows six): the two free areas of the same board, left middle and bottom left.
    static let extraSlots: [ShopSlot] = [
        ShopSlot(center: CGPoint(x: 75, y: 405), size: CGSize(width: 104, height: 96), rotation: -4, prefers: [.accessory, .bag]),
        ShopSlot(center: CGPoint(x: 96, y: 742), size: CGSize(width: 130, height: 92), rotation: 2, prefers: [.accessory, .shoes]),
    ]

    /// Places every item: first by kind (tops on the top places, pants on the long one, bag, shoes, accessory),
    /// then the rest in order. `variant` 0 is the Figma arrangement; Shuffle gives 1, 2, … (items swap places,
    /// odd variants mirror the board).
    static func placements(kinds: [ShopItemKind], variant: Int = 0) -> [ShopPlacement] {
        guard !kinds.isEmpty else { return [] }
        // The six Figma places always (each kind finds its own), the two extra ones only for items 7 and 8.
        let slots = figmaSlots + extraSlots.prefix(max(0, kinds.count - figmaSlots.count))
        var order = slots.indices.map { $0 }
        var assigned = [Int?](repeating: nil, count: slots.count)       // slot → item
        var free = Set(kinds.indices)
        if variant == 0 {
            // Pass 1: an item whose kind the slot prefers first.
            for rank in 0..<3 {
                for s in order where assigned[s] == nil {
                    guard rank < slots[s].prefers.count else { continue }
                    if let item = kinds.indices.first(where: { free.contains($0) && kinds[$0] == slots[s].prefers[rank] }) {
                        assigned[s] = item; free.remove(item)
                    }
                }
            }
        } else {
            var rng = ShopRandom(seed: UInt64(variant) &* 0x9E3779B97F4A7C15)
            order.shuffle(using: &rng)
        }
        // Pass 2: everything left, in order.
        for s in order where assigned[s] == nil {
            guard let item = kinds.indices.first(where: { free.contains($0) }) else { break }
            assigned[s] = item; free.remove(item)
        }
        let mirror = variant % 2 == 1
        // Figma's drawing order (later on top).
        return slots.indices.compactMap { s in
            guard let item = assigned[s] else { return nil }
            var slot = slots[s]
            if mirror {
                slot.center.x = designSize.width - slot.center.x
                slot.rotation = -slot.rotation
            }
            return ShopPlacement(itemIndex: item, slot: slot)
        }
    }

    /// The frame of an image (aspect `aspect` = width / height) fitted in a slot, on a board of `size`.
    static func frame(for slot: ShopSlot, aspect: CGFloat, board size: CGSize) -> CGRect {
        let sx = size.width / designSize.width, sy = size.height / designSize.height
        let box = CGSize(width: slot.size.width * sx, height: slot.size.height * sy)
        var w = box.width, h = w / max(aspect, 0.01)
        if h > box.height { h = box.height; w = h * aspect }
        let c = CGPoint(x: slot.center.x * sx, y: slot.center.y * sy)
        return CGRect(x: c.x - w / 2, y: c.y - h / 2, width: w, height: h)
    }

    /// The export size: 9:16, 1080 × 1920.
    static let exportSize = CGSize(width: 1080, height: 1920)
}

/// Small seeded generator (Shuffle and the textures look the same every time for the same seed).
struct ShopRandom: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed == 0 ? 0x2545F4914F6CDD1D : seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
}

// MARK: - Shape it (POST /v1/shop/shape)

/// Why Shape it did not work, each with its own message.
enum ShopShapeFailure: Equatable, Error {
    case dailyLimit(Int)
    case disabled
    case signedOut
    case tooLarge
    case failed

    /// From the server's answer: 429 shape_daily_limit, 503 shape_disabled, 401, 413; anything else (404 while the
    /// endpoint is not deployed, no connection) is the generic "Retry" state.
    static func from(status: Int, body: String) -> ShopShapeFailure {
        switch status {
        case 429:
            let limit = (body.data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] })?["limit"]
            return .dailyLimit((limit as? Int) ?? (limit as? NSNumber)?.intValue ?? 20)
        case 503: return .disabled
        case 401: return .signedOut
        case 413: return .tooLarge
        default: return .failed
        }
    }

    var message: String {
        switch self {
        case .dailyLimit(let limit): String(format: L10n.tr("You've used today's %d Shape it items. Try again tomorrow."), limit)
        case .disabled: L10n.tr("Shape it is turned off right now. Keep flat or try later.")
        case .signedOut: L10n.tr("Sign in again to use Shape it.")
        case .tooLarge: L10n.tr("This photo is too large to shape. Try a smaller one.")
        case .failed: L10n.tr("Shape it didn't work this time. Tap Retry.")
        }
    }
}

struct ShopShapeRequestBody: Encodable, Equatable {
    let image_png_base64: String
    let kind: String
}

struct ShopShapeResponseBody: Decodable, Equatable {
    let image_png_base64: String
    let provider: String?
    let cost_usd: Double?
}
