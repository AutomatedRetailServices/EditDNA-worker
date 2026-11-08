import CoreGraphics
import Foundation
import simd

/// Clon AR (beta, PO 8 Nov 2026): the geometry and timing rules, with no ARKit / AVFoundation in them so the unit
/// tests can check them with synthetic cameras.
///
/// Camera convention (ARKit): `transform` maps camera space to world space; the camera looks down -Z, +X is to the
/// right and +Y is up of the SENSOR image (landscape). Pixel (u, v) of the captured image (u right, v down) is the
/// ray x = (u - cx) / fx, y = -(v - cy) / fy, z = -1.
enum CloneMath {
    struct Camera: Equatable {
        var transform: simd_float4x4     // camera -> world
        var fx: Float
        var fy: Float
        var cx: Float
        var cy: Float
        var width: Float                 // captured image size (landscape sensor pixels)
        var height: Float

        static func == (a: Camera, b: Camera) -> Bool {
            a.transform == b.transform && a.fx == b.fx && a.fy == b.fy && a.cx == b.cx && a.cy == b.cy
                && a.width == b.width && a.height == b.height
        }
    }

    /// World point -> camera space.
    static func toCamera(_ world: SIMD3<Float>, _ cam: Camera) -> SIMD3<Float> {
        let p = cam.transform.inverse * SIMD4<Float>(world, 1)
        return SIMD3<Float>(p.x, p.y, p.z)
    }

    /// World point -> pixel of `cam` (nil when the point is behind the camera).
    static func project(_ world: SIMD3<Float>, _ cam: Camera) -> CGPoint? {
        let p = toCamera(world, cam)
        guard p.z < -1e-4 else { return nil }
        let x = p.x / -p.z, y = p.y / -p.z
        return CGPoint(x: CGFloat(cam.fx * x + cam.cx), y: CGFloat(cam.cy - cam.fy * y))
    }

    /// Pixel of `cam` at `depth` metres in front of it (along its optical axis) -> world point.
    static func unproject(_ pixel: CGPoint, depth: Float, _ cam: Camera) -> SIMD3<Float> {
        let x = (Float(pixel.x) - cam.cx) / cam.fx, y = -(Float(pixel.y) - cam.cy) / cam.fy
        let p = cam.transform * SIMD4<Float>(x * depth, y * depth, -depth, 1)
        return SIMD3<Float>(p.x, p.y, p.z)
    }

    /// How far the anchor is in front of `cam`, along its optical axis (the depth of the clone's plane).
    static func depth(of anchor: SIMD3<Float>, from cam: Camera) -> Float {
        max(0.2, -toCamera(anchor, cam).z)
    }

    /// The clone frame seen by `clone` becomes a flat picture standing at the clone's anchor depth (a slice of
    /// its view). Its four corners (top-left, top-right, bottom-right, bottom-left of the clone image) in the
    /// pixels of `target`. nil when the clone is behind the target camera.
    static func cornersInTarget(clone: Camera, anchor: SIMD3<Float>, target: Camera) -> [CGPoint]? {
        let d = depth(of: anchor, from: clone)
        let corners = [CGPoint(x: 0, y: 0), CGPoint(x: CGFloat(clone.width), y: 0),
                       CGPoint(x: CGFloat(clone.width), y: CGFloat(clone.height)), CGPoint(x: 0, y: CGFloat(clone.height))]
        var out: [CGPoint] = []
        for c in corners {
            guard let p = project(unproject(c, depth: d, clone), target) else { return nil }
            out.append(p)
        }
        return out
    }

    /// Live preview: the plane that shows a clone frame in the room. Centre (world), size (metres) and the
    /// clone camera's rotation (the plane faces the camera that recorded it).
    struct Plane: Equatable {
        var transform: simd_float4x4      // unit plane in XY facing +Z -> world (rotation + centre)
        var width: Float
        var height: Float
    }

    static func plane(clone: Camera, anchor: SIMD3<Float>) -> Plane {
        let d = depth(of: anchor, from: clone)
        let width = clone.width / clone.fx * d, height = clone.height / clone.fy * d
        // Centre of the image at depth d, in camera space.
        let centre = SIMD3<Float>((clone.width / 2 - clone.cx) / clone.fx * d, -(clone.height / 2 - clone.cy) / clone.fy * d, -d)
        var local = matrix_identity_float4x4
        local.columns.3 = SIMD4<Float>(centre, 1)
        return Plane(transform: clone.transform * local, width: width, height: height)
    }

    /// Clones drawn farthest first (so a nearer clone covers a farther one) as seen from `target`.
    static func drawOrder(anchors: [SIMD3<Float>], target: Camera) -> [Int] {
        anchors.indices.sorted { depth(of: anchors[$0], from: target) > depth(of: anchors[$1], from: target) }
    }

    // MARK: Timing and sound

    /// Every take starts at 0 s. The video lasts as long as the LAST take; a shorter take keeps its last frame.
    static func frameIndex(at time: Double, frameTimes: [Double]) -> Int {
        guard !frameTimes.isEmpty else { return 0 }
        if time <= frameTimes[0] { return 0 }
        var lo = 0, hi = frameTimes.count - 1
        if time >= frameTimes[hi] { return hi }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if frameTimes[mid] <= time { lo = mid } else { hi = mid }
        }
        return lo
    }

    static func finalDuration(takeDurations: [Double]) -> Double { takeDurations.last ?? 0 }

    /// Gain that brings a take's loudness (RMS) to the target; kept between x0.5 and x3 so silence is not blown up.
    static func gain(rms: Float, target: Float = 0.1) -> Float {
        guard rms > 1e-5 else { return 1 }
        return min(3, max(0.5, target / rms))
    }

    /// The mixed sound: every take from 0 s, each with its gain, as long as the last take; clipped to [-1, 1].
    static func mix(takes: [[Float]], gains: [Float]) -> [Float] {
        guard let last = takes.last else { return [] }
        var out = [Float](repeating: 0, count: last.count)
        for (k, take) in takes.enumerated() {
            let g = k < gains.count ? gains[k] : 1
            for i in 0..<min(out.count, take.count) { out[i] += take[i] * g }
        }
        // Soft limit so several voices together never clip harshly.
        let peak = out.reduce(Float(0)) { max($0, abs($1)) }
        if peak > 1 { let s = 0.98 / peak; for i in out.indices { out[i] *= s } }
        return out
    }

    static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for s in samples { sum += s * s }
        return (sum / Float(samples.count)).squareRoot()
    }

    /// Inside Clon the auto-pause is OFF: every take is kept whole and in sync (the engine trims silences later).
    static let autoPauseInsideClone = false
    /// Takes allowed (Clone 1 ... Clone 4).
    static let maxTakes = 4
}
