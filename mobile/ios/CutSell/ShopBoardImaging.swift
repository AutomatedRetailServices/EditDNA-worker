import CoreImage
import CoreImage.CIFilterBuiltins
import CoreText
import SwiftUI
import UIKit
import Vision

// Shop board pictures, all on the iPhone: background removal (Vision), item kind (Vision labels), the products'
// colours, the bundled letters and the procedural textures. Rules: ShopBoardLogic.swift.

// MARK: - Letters

/// Registers the twelve Shop board faces (Fonts/ShopBoard, SIL OFL) the same way the app registers Plus Jakarta Sans
/// (CTFontManagerRegisterFontsForURL, process scope), and makes the UIFonts (weight of the variable files by
/// font variation).
enum ShopFonts {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var didRegister = false

    static var allFaces: [ShopFontFace] {
        ShopFontPair.allCases.flatMap { [$0.titleFace, $0.scriptFace] } + [ShopFontPair.footerFace]
    }

    static func registerIfNeeded() {
        lock.lock()
        defer { lock.unlock() }
        guard !didRegister else { return }
        didRegister = true
        var files = Set<String>()
        for face in allFaces where files.insert(face.fileName).inserted {
            guard let url = Bundle.main.url(forResource: face.fileName, withExtension: "ttf") else {
                CutSellDiagnostics.log("shop_font_missing", ["font": face.fileName])
                continue
            }
            var error: Unmanaged<CFError>?
            if !CTFontManagerRegisterFontsForURL(url as CFURL, .process, &error) {
                // Bebas Neue is also one of the caption faces: "already registered" is fine.
                let code = error.map { CFErrorGetCode($0.takeRetainedValue()) } ?? 0
                if code != CTFontManagerError.alreadyRegistered.rawValue {
                    CutSellDiagnostics.log("shop_font_not_registered", ["font": face.fileName, "code": String(code)])
                }
            }
        }
    }

    /// The face at `size` points. `opticalSize` (Bodoni Moda's opsz axis) follows the size in the Figma board.
    static func uiFont(_ face: ShopFontFace, size: CGFloat, opticalSize: CGFloat? = nil) -> UIFont {
        registerIfNeeded()
        guard let base = UIFont(name: face.postScriptName, size: size) else { return .systemFont(ofSize: size) }
        guard face.variable else { return base }
        var variation: [NSNumber: NSNumber] = [:]
        if let weight = face.weight { variation[NSNumber(value: tag("wght"))] = NSNumber(value: Double(weight)) }
        if face.fileName.contains("BodoniModa") {
            variation[NSNumber(value: tag("opsz"))] = NSNumber(value: Double(min(96, max(6, opticalSize ?? size))))
        }
        guard !variation.isEmpty else { return base }
        let attributes: [UIFontDescriptor.AttributeName: Any] = [
            UIFontDescriptor.AttributeName(rawValue: kCTFontVariationAttribute as String): variation,
        ]
        let descriptor = base.fontDescriptor.addingAttributes(attributes)
        return UIFont(descriptor: descriptor, size: size)
    }

    static func font(_ face: ShopFontFace, size: CGFloat, opticalSize: CGFloat? = nil) -> Font {
        Font(uiFont(face, size: size, opticalSize: opticalSize) as CTFont)
    }

    private static func tag(_ s: String) -> UInt32 {
        s.unicodeScalars.reduce(0) { $0 << 8 | UInt32($1.value) }
    }
}

// MARK: - Background removal

enum ShopCutoutError: Error { case noSubject, unavailable }

enum ShopCutout {
    static let context = CIContext(options: [.useSoftwareRenderer: false])

    /// "From photos": Vision's foreground mask (iOS 17), applied to the photo, cropped to the subject.
    static func removeBackground(_ image: CGImage) throws -> CGImage {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])
        guard let observation = request.results?.first, !observation.allInstances.isEmpty else { throw ShopCutoutError.noSubject }
        let mask = try observation.generateScaledMaskForImage(forInstances: observation.allInstances, from: handler)
        guard let out = apply(mask: CIImage(cvPixelBuffer: mask), to: image), let trimmed = trimmed(out) else {
            throw ShopCutoutError.noSubject
        }
        return trimmed
    }

    /// The photo with the mask as its alpha (white = keep, black = remove). Pure Core Image, unit-tested.
    static func apply(mask: CIImage, to image: CGImage) -> CGImage? {
        let source = CIImage(cgImage: image)
        let extent = source.extent
        var maskImage = mask
        if mask.extent.size != extent.size {
            maskImage = mask.transformed(by: CGAffineTransform(scaleX: extent.width / mask.extent.width,
                                                                 y: extent.height / mask.extent.height))
        }
        let blend = CIFilter.blendWithMask()
        blend.inputImage = source
        blend.backgroundImage = CIImage(color: .clear).cropped(to: extent)
        blend.maskImage = maskImage
        guard let output = blend.outputImage?.cropped(to: extent) else { return nil }
        return context.createCGImage(output, from: extent, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }

    /// RGBA8 bytes of an image (premultiplied, sRGB), row by row.
    static func rgba(_ image: CGImage, maxSide: Int? = nil) -> (bytes: [UInt8], width: Int, height: Int)? {
        var w = image.width, h = image.height
        if let maxSide, max(w, h) > maxSide {
            let s = Double(maxSide) / Double(max(w, h))
            w = max(1, Int(Double(w) * s)); h = max(1, Int(Double(h) * s))
        }
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.clear(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.interpolationQuality = .medium
        // Row 0 of a bitmap context's memory is the top of the picture (as CGImage.cropping expects).
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { return nil }
        let stride = ctx.bytesPerRow
        let raw = data.bindMemory(to: UInt8.self, capacity: stride * h)
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<(w * 4) { bytes[y * w * 4 + x] = raw[y * stride + x] }
        }
        return (bytes, w, h)
    }

    /// Cropped to its opaque pixels (alpha > 8), with a small margin.
    static func trimmed(_ image: CGImage) -> CGImage? {
        guard let (bytes, w, h) = rgba(image) else { return nil }
        var minX = w, minY = h, maxX = -1, maxY = -1
        for y in 0..<h {
            for x in 0..<w where bytes[(y * w + x) * 4 + 3] > 8 {
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        guard maxX >= minX, maxY >= minY else { return nil }
        let pad = 4
        let rect = CGRect(x: max(0, minX - pad), y: max(0, minY - pad),
                          width: min(w, maxX + pad + 1) - max(0, minX - pad), height: min(h, maxY + pad + 1) - max(0, minY - pad))
        return image.cropping(to: rect)
    }

    /// Opaque pixels (alpha ≥ 50 %), un-premultiplied, from at most 64 × 64 — the input of Match my products.
    static func opaquePixels(_ image: CGImage, maxSide: Int = 64) -> [ShopRGB] {
        guard let (bytes, w, h) = rgba(image, maxSide: maxSide) else { return [] }
        var out: [ShopRGB] = []
        out.reserveCapacity(w * h)
        for i in stride(from: 0, to: w * h * 4, by: 4) {
            let a = Double(bytes[i + 3])
            guard a >= 128 else { continue }
            out.append(ShopRGB(Double(bytes[i]) / a, Double(bytes[i + 1]) / a, Double(bytes[i + 2]) / a))
        }
        return out
    }

    /// The PNG sent to Shape it: alpha kept, long side at most 1536 px.
    static func shapePNG(_ image: CGImage, maxSide: CGFloat = 1536) -> Data? {
        let long = CGFloat(max(image.width, image.height))
        let scale = min(1, maxSide / max(1, long))
        let size = CGSize(width: (CGFloat(image.width) * scale).rounded(), height: (CGFloat(image.height) * scale).rounded())
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = false
        let resized = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            UIImage(cgImage: image).draw(in: CGRect(origin: .zero, size: size))
        }
        return resized.pngData()
    }

    /// A photo from Photos, upright and at most 2048 px long side.
    static func normalized(_ image: UIImage, maxSide: CGFloat = 2048) -> CGImage? {
        let long = max(image.size.width, image.size.height)
        let scale = min(1, maxSide / max(1, long))
        let size = CGSize(width: (image.size.width * scale).rounded(), height: (image.size.height * scale).rounded())
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }.cgImage
    }
}

// MARK: - Item kind (Vision labels)

enum ShopVision {
    /// Vision's labels for the cut-out (drawn on white), mapped by `ShopItemClassifier`; `.unknown` if Vision fails.
    static func kind(of cutout: CGImage) -> ShopItemKind {
        let size = CGSize(width: cutout.width, height: cutout.height)
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        guard let flat = UIGraphicsImageRenderer(size: size, format: format).image(actions: { ctx in
            UIColor.white.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
            UIImage(cgImage: cutout).draw(in: CGRect(origin: .zero, size: size))
        }).cgImage else { return .unknown }
        let request = VNClassifyImageRequest()
        do {
            try VNImageRequestHandler(cgImage: flat, options: [:]).perform([request])
        } catch {
            return .unknown
        }
        let labels = (request.results ?? []).filter { $0.confidence > 0.05 }.prefix(40)
            .map { (identifier: $0.identifier, confidence: $0.confidence) }
        return ShopItemClassifier.kind(fromLabels: Array(labels))
    }
}

// MARK: - Textures (drawn in code: no image files)

/// Linen, Paper, Marble, Wood, Knit, Satin, Concrete and Velvet, drawn with Core Graphics + Core Image noise from a
/// fixed seed, on the Figma swatch gradients (or on the matched colour with ✨ Match my products).
enum ShopTextureRenderer {
    private static let cache = NSCache<NSString, UIImage>()

    static func image(_ texture: ShopTexture, size: CGSize, base: (top: ShopRGB, bottom: ShopRGB)? = nil) -> UIImage {
        let colors = base ?? (texture.top, texture.bottom)
        let key = "\(texture.rawValue)-\(Int(size.width))x\(Int(size.height))-\(colors.top.hex)-\(colors.bottom.hex)" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let image = UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            draw(texture, in: ctx.cgContext, size: size, top: colors.top, bottom: colors.bottom)
        }
        cache.setObject(image, forKey: key)
        return image
    }

    private static func cg(_ c: ShopRGB, _ a: CGFloat = 1) -> CGColor {
        CGColor(srgbRed: c.r, green: c.g, blue: c.b, alpha: a)
    }

    private static func white(_ a: CGFloat) -> CGColor { CGColor(gray: 1, alpha: a) }
    private static func black(_ a: CGFloat) -> CGColor { CGColor(gray: 0, alpha: a) }

    /// Grey noise from CIRandomGenerator, stretched by (sx, sy) and blurred.
    private static func noise(size: CGSize, sx: CGFloat, sy: CGFloat, blur: CGFloat, offset: CGFloat = 0) -> CGImage? {
        guard var image = CIFilter.randomGenerator().outputImage else { return nil }
        image = image.transformed(by: CGAffineTransform(translationX: offset, y: offset * 0.7))
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        let mono = CIFilter.colorControls()
        mono.inputImage = image
        mono.saturation = 0
        image = mono.outputImage ?? image
        if blur > 0 { image = image.clampedToExtent().applyingGaussianBlur(sigma: Double(blur)) }
        return ShopCutout.context.createCGImage(image.cropped(to: CGRect(origin: .zero, size: size)),
                                                from: CGRect(origin: .zero, size: size))
    }

    private static func overlay(_ image: CGImage?, in ctx: CGContext, size: CGSize, alpha: CGFloat, mode: CGBlendMode) {
        guard let image else { return }
        ctx.saveGState()
        ctx.setAlpha(alpha)
        ctx.setBlendMode(mode)
        ctx.draw(image, in: CGRect(origin: .zero, size: size))
        ctx.restoreGState()
    }

    static func draw(_ texture: ShopTexture, in ctx: CGContext, size: CGSize, top: ShopRGB, bottom: ShopRGB) {
        let w = size.width, h = size.height, u = max(w, h) / 844     // u: one Figma point on this size
        // Base: the swatch gradient, top → bottom (UIKit coordinates: y down).
        if let gradient = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: [cg(top), cg(bottom)] as CFArray,
                                     locations: [0, 1]) {
            ctx.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: 0, y: h), options: [])
        }
        var rng = ShopRandom(seed: UInt64(ShopTexture.allCases.firstIndex(of: texture)! + 1) * 7919)
        switch texture {
        case .linen:
            // Woven threads: fine noise stretched along x and along y, crossed.
            overlay(noise(size: size, sx: 1, sy: 60 * u, blur: 0), in: ctx, size: size, alpha: 0.10, mode: .softLight)
            overlay(noise(size: size, sx: 60 * u, sy: 1, blur: 0, offset: 13), in: ctx, size: size, alpha: 0.10, mode: .softLight)
            overlay(noise(size: size, sx: 1.5 * u, sy: 1.5 * u, blur: 0), in: ctx, size: size, alpha: 0.06, mode: .multiply)
        case .paper:
            overlay(noise(size: size, sx: 1, sy: 1, blur: 0.6 * u), in: ctx, size: size, alpha: 0.08, mode: .multiply)
            overlay(noise(size: size, sx: 40 * u, sy: 40 * u, blur: 30 * u, offset: 5), in: ctx, size: size, alpha: 0.10, mode: .softLight)
        case .marble:
            overlay(noise(size: size, sx: 50 * u, sy: 50 * u, blur: 40 * u), in: ctx, size: size, alpha: 0.12, mode: .softLight)
            // Veins: soft grey curves across the board.
            ctx.setLineCap(.round)
            for i in 0..<7 {
                let path = CGMutablePath()
                var p = CGPoint(x: -0.1 * w, y: CGFloat(rng.unit()) * h)
                path.move(to: p)
                for _ in 0..<6 {
                    let next = CGPoint(x: p.x + w * (0.2 + 0.15 * CGFloat(rng.unit())), y: p.y + h * (CGFloat(rng.unit()) - 0.4) * 0.18)
                    let c = CGPoint(x: (p.x + next.x) / 2, y: p.y + h * (CGFloat(rng.unit()) - 0.5) * 0.12)
                    path.addQuadCurve(to: next, control: c)
                    p = next
                }
                ctx.addPath(path)
                ctx.setStrokeColor(black(i % 3 == 0 ? 0.10 : 0.05))
                ctx.setLineWidth((i % 3 == 0 ? 2.2 : 1.0) * u)
                ctx.setShadow(offset: .zero, blur: 3 * u, color: black(0.08))
                ctx.strokePath()
            }
            ctx.setShadow(offset: .zero, blur: 0, color: nil)
        case .wood:
            // Long grain along the board plus a few darker rings.
            overlay(noise(size: size, sx: 1.2 * u, sy: 220 * u, blur: 0.8 * u), in: ctx, size: size, alpha: 0.28, mode: .multiply)
            overlay(noise(size: size, sx: 3 * u, sy: 400 * u, blur: 2 * u, offset: 21), in: ctx, size: size, alpha: 0.22, mode: .softLight)
            for _ in 0..<10 {
                let x = CGFloat(rng.unit()) * w
                ctx.setStrokeColor(cg(bottom.mixed(ShopRGB(0.2, 0.12, 0.05), 0.35), 0.25))
                ctx.setLineWidth((0.6 + CGFloat(rng.unit())) * 1.4 * u)
                ctx.move(to: CGPoint(x: x, y: 0))
                ctx.addCurve(to: CGPoint(x: x + 6 * u, y: h), control1: CGPoint(x: x - 10 * u, y: h * 0.3),
                             control2: CGPoint(x: x + 12 * u, y: h * 0.7))
                ctx.strokePath()
            }
        case .knit:
            // Rows of small "V" stitches.
            let stitchW = 10 * u, stitchH = 12 * u
            ctx.setLineWidth(1.4 * u)
            ctx.setLineCap(.round)
            var y: CGFloat = 0, row = 0
            while y < h + stitchH {
                var x: CGFloat = row % 2 == 0 ? 0 : stitchW / 2
                while x < w + stitchW {
                    ctx.move(to: CGPoint(x: x, y: y))
                    ctx.addQuadCurve(to: CGPoint(x: x + stitchW / 2, y: y + stitchH), control: CGPoint(x: x + stitchW * 0.1, y: y + stitchH * 0.75))
                    ctx.addQuadCurve(to: CGPoint(x: x + stitchW, y: y), control: CGPoint(x: x + stitchW * 0.9, y: y + stitchH * 0.75))
                    x += stitchW
                }
                ctx.setStrokeColor(white(0.22))
                ctx.strokePath()
                y += stitchH * 0.62
                row += 1
            }
            overlay(noise(size: size, sx: 2 * u, sy: 2 * u, blur: 0), in: ctx, size: size, alpha: 0.06, mode: .multiply)
        case .satin:
            // Soft diagonal sheen bands.
            for i in 0..<4 {
                let t = CGFloat(i) / 3
                let band = CGGradient(colorsSpace: nil, colors: [white(0), white(i % 2 == 0 ? 0.35 : 0.18), white(0)] as CFArray,
                                      locations: [0, 0.5, 1])!
                ctx.saveGState()
                ctx.translateBy(x: w * (t - 0.2), y: 0)
                ctx.rotate(by: .pi / 9)
                ctx.drawLinearGradient(band, start: CGPoint(x: 0, y: 0), end: CGPoint(x: w * 0.22, y: 0), options: [])
                ctx.restoreGState()
            }
            overlay(noise(size: size, sx: 1, sy: 1, blur: 0.4 * u), in: ctx, size: size, alpha: 0.03, mode: .multiply)
        case .concrete:
            overlay(noise(size: size, sx: 1, sy: 1, blur: 0.5 * u), in: ctx, size: size, alpha: 0.16, mode: .multiply)
            overlay(noise(size: size, sx: 30 * u, sy: 30 * u, blur: 18 * u, offset: 9), in: ctx, size: size, alpha: 0.22, mode: .softLight)
            for _ in 0..<Int(140 * (w * h) / (390 * 844)) {
                let r = (0.5 + CGFloat(rng.unit()) * 1.2) * u
                ctx.setFillColor(black(0.12 + CGFloat(rng.unit()) * 0.1))
                ctx.fillEllipse(in: CGRect(x: CGFloat(rng.unit()) * w, y: CGFloat(rng.unit()) * h, width: r * 2, height: r * 2))
            }
        case .velvet:
            // Deep pile: a wide soft sheen and fine fibres.
            let sheen = CGGradient(colorsSpace: nil, colors: [white(0.16), white(0)] as CFArray, locations: [0, 1])!
            ctx.drawRadialGradient(sheen, startCenter: CGPoint(x: w * 0.3, y: h * 0.25), startRadius: 0,
                                   endCenter: CGPoint(x: w * 0.3, y: h * 0.25), endRadius: max(w, h) * 0.6, options: [])
            overlay(noise(size: size, sx: 1, sy: 3 * u, blur: 0.4 * u), in: ctx, size: size, alpha: 0.10, mode: .screen)
            overlay(noise(size: size, sx: 25 * u, sy: 25 * u, blur: 20 * u, offset: 3), in: ctx, size: size, alpha: 0.18, mode: .multiply)
        }
    }
}

extension ShopRGB {
    var color: Color { Color(fixed: hex) }
    var uiColor: UIColor { UIColor(red: r, green: g, blue: b, alpha: 1) }
    init(_ color: UIColor) {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        let srgb = color.cgColor.converted(to: CGColorSpace(name: CGColorSpace.sRGB)!, intent: .defaultIntent, options: nil)
        if let c = srgb?.components, c.count >= 3 { r = c[0]; g = c[1]; b = c[2] } else { color.getRed(&r, green: &g, blue: &b, alpha: &a) }
        self.init(Double(min(max(r, 0), 1)), Double(min(max(g, 0), 1)), Double(min(max(b, 0), 1)))
    }
}
