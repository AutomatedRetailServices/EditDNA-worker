#if DEBUG
import AVFoundation
import CoreImage
import SwiftUI

/// Debug builds only (`-CutSellScreen split_selftest`): checks the Clon compositor on synthetic videos
/// made here (nothing from the repo, nothing of a user):
/// 0. Silhouette (default): part 1's "person" (a red shape, cut out by a stand-in for Vision) is pasted over the
///    WHOLE of part 2 -- also where it crosses the middle --, part 2's own person and room stay, no line in the
///    middle, part 1 is matched to part 2's brightness, length = part 2, both sounds; Vision on a video with no
///    person -> the silhouette reports it failed and the simpler halves blend still works (the app's fallback);
///    and how long a 30 s 720x1280 video takes through the real Vision pass.
/// 1. Seam: part 1 red (changing every frame), part 2 blue → the middle blends smoothly (no line), the
///    video lasts as long as part 2, and part 1 holds its last frame after it ends.
/// 2. Sound: part 1 a quiet 440 Hz tone, part 2 a loud 660 Hz tone → both heard at an even level while
///    part 1 lasts, only 660 Hz after.
/// 3. Shift: a white square moved 18 px between the parts → the aligner measures it and the joined
///    video moves part 1's square onto part 2's place.
@MainActor
final class SplitSelfTest: ObservableObject {
    @Published var lines: [String] = []
    @Published var finished = false

    private func pass(_ ok: Bool, _ name: String, _ detail: String = "") {
        lines.append("\(ok ? "PASS" : "FAIL") \(name)\(detail.isEmpty ? "" : " · \(detail)")")
    }

    func run() async {
        do {
            try await silhouette()
            await fallbackAndTiming()
            try await seamAndSound()
            try await shift()
        } catch {
            pass(false, "self-test crashed", "\(error)")
        }
        finished = true
    }

    // MARK: 1 + 2

    private func seamAndSound() async throws {
        let size = CGSize(width: 360, height: 640)
        let red = try await Self.movie(name: "split-st-red", size: size, seconds: 5, tone: 440, amplitude: 4_000) { frame, _, _ in
            (UInt8(min(255, 80 + frame * 3)), 0, 0)
        }
        let blue = try await Self.movie(name: "split-st-blue", size: size, seconds: 7, tone: 660, amplitude: 16_000) { _, _, _ in
            (0, 0, 255)
        }
        let joined = try await SplitCompositor.join(part1: red, part2: blue, offset: .zero, blend: .halves)
        let duration = try await AVURLAsset(url: joined).load(.duration).seconds
        pass(abs(duration - 7) < 0.15, "length = part 2", String(format: "%.2f s", duration))

        // A row in the middle of the picture at 1 s.
        guard let row = await Self.row(of: joined, at: 1, y: 320) else { pass(false, "read frame 1 s"); return }
        let width = row.count
        let leftR = Double(row[20].r), rightB = Double(row[width - 20].b)
        pass(abs(leftR - 110) < 25 && row[20].b < 60, "left is part 1 at 1 s", "R \(Int(leftR)) B \(row[20].b)")
        pass(rightB > 200 && row[width - 20].r < 60, "right is part 2", "B \(Int(rightB)) R \(row[width - 20].r)")
        // No line: across the middle red only goes down and blue only goes up, and their mix stays whole.
        var monotonic = true, whole = true, blended = 0
        for x in 1..<width {
            if Int(row[x].r) > Int(row[x - 1].r) + 6 || Int(row[x].b) + 6 < Int(row[x - 1].b) { monotonic = false }
            // The blend is done in linear light (Core Image), so compare in linear light.
            func lin(_ v: Double) -> Double { let c = v / 255; return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
            let mixture = lin(Double(row[x].r)) / max(1e-4, lin(leftR)) + lin(Double(row[x].b)) / max(1e-4, lin(rightB))
            if mixture < 0.8 || mixture > 1.25 { whole = false }
            if row[x].b > 40 && row[x].b < 215 { blended += 1 }
        }
        pass(monotonic, "no line at the seam (smooth change)")
        pass(whole, "no dark or bright stripe at the seam")
        pass(blended >= 8 && blended <= 40, "soft edge ≈ 60 px at 1080", "\(blended) px of \(width)")

        // After part 1 ended (5 s): its last frame stays.
        if let late = await Self.row(of: joined, at: 6.2, y: 320) {
            let expected = Double(min(255, 80 + 49 * 3))
            pass(abs(Double(late[20].r) - expected) < 25, "part 1 holds its last frame", "R \(late[20].r) ≈ \(Int(expected))")
        } else { pass(false, "read frame 6.2 s") }

        // Sound.
        let samples = await Self.mono(of: joined)
        let rate = 44_100.0
        func tone(_ hz: Double, from: Double, to: Double) -> Double {
            let a = Int(from * rate), b = min(samples.count, Int(to * rate))
            guard b > a else { return 0 }
            return Self.goertzel(Array(samples[a..<b]), hz: hz, rate: rate)
        }
        let f440 = tone(440, from: 1, to: 4), f660 = tone(660, from: 1, to: 4)
        let ratio = f440 / max(1e-9, f660)
        pass(f440 > 0.005 && f660 > 0.005, "both voices heard", String(format: "440 Hz %.3f · 660 Hz %.3f", f440, f660))
        pass(ratio > 0.6 && ratio < 1.6, "voices at an even level", String(format: "ratio %.2f (before: 0.25)", ratio))
        let after440 = tone(440, from: 5.5, to: 6.8), after660 = tone(660, from: 5.5, to: 6.8)
        pass(after440 < after660 * 0.1, "only part 2's sound after part 1 ends", String(format: "440 %.4f · 660 %.4f", after440, after660))
        try? FileManager.default.removeItem(at: joined)
    }

    // MARK: 3

    private func shift() async throws {
        let size = CGSize(width: 360, height: 640)
        // A textured "room" (like a real one) with a white square; in part 2 the whole picture is 18 px to
        // the right, as if the phone had been nudged.
        func room(shift: Int) -> (Int, Int, Int) -> (UInt8, UInt8, UInt8) {
            { _, x0, y in
                let x = x0 - shift
                if x >= 60 && x < 100 && y >= 300 && y < 340 { return (255, 255, 255) }
                // Non-repeating blotches (like furniture and walls), not a wave pattern: a repeating
                // texture matches itself at several places and no aligner can tell which is right.
                func noise(_ cx: Int, _ cy: Int) -> Double {
                    var hash = UInt32(truncatingIfNeeded: (cx &* 73_856_093) ^ (cy &* 19_349_663) ^ 0x5bd1e995)
                    hash = (hash ^ (hash >> 13)) &* 1_274_126_177
                    hash ^= hash >> 16
                    return Double(hash % 1_000) / 1_000
                }
                let coarse = noise(Int((Double(x) / 37).rounded(.down)), Int((Double(y) / 29).rounded(.down)))
                let fine = noise(Int((Double(x) / 7).rounded(.down)) + 9_000, Int((Double(y) / 7).rounded(.down)))
                let v = UInt8(max(30, min(180, 40 + 100 * coarse + 35 * fine)))
                return (v, UInt8(max(0, Int(v) - 12)), UInt8(min(255, Int(v) + 8)))
            }
        }
        let part1 = try await Self.movie(name: "split-st-blot1", size: size, seconds: 2, tone: 440, amplitude: 8_000, paint: room(shift: 0))
        let part2 = try await Self.movie(name: "split-st-blot2", size: size, seconds: 2, tone: 660, amplitude: 8_000, paint: room(shift: 18))
        if let a = await SplitAligner.firstFrame(of: part1), let b = await SplitAligner.firstFrame(of: part2) {
            lines.append("INFO vision 270 · \(SplitAligner.raw(part1: a, live: b, width: 270))")
            lines.append("INFO vision full · \(SplitAligner.raw(part1: a, live: b, width: nil))")
            lines.append("INFO vision same · \(SplitAligner.raw(part1: a, live: a, width: 270))")
            lines.append("INFO extents \(a.extent) \(b.extent)")
        }
        guard let first1 = await SplitAligner.firstFrame(of: part1), let first2 = await SplitAligner.firstFrame(of: part2),
              let offset = SplitAligner.offset(part1: first1, live: first2) else {
            pass(false, "aligner measured the shift"); return
        }
        let expected = 18.0 / 360
        pass(abs(offset.x - expected) < 0.012 && abs(offset.y) < 0.012, "aligner measured the shift",
             String(format: "x %.3f (expected %.3f) y %.3f", offset.x, expected, offset.y))
        pass(offset.magnitude > 0.02, "a 5% move is flagged (> 2%)")
        let joined = try await SplitCompositor.join(part1: part1, part2: part2, offset: offset, blend: .halves)
        let still = try await SplitCompositor.join(part1: part1, part2: part2, offset: .zero, blend: .halves)
        func centre(_ url: URL) async -> Double? {
            guard let row = await Self.row(of: url, at: 0.5, y: 320) else { return nil }
            let bright = (0..<row.count / 2).filter { row[$0].r > 215 && row[$0].g > 215 && row[$0].b > 215 }
            return bright.isEmpty ? nil : Double(bright.reduce(0, +)) / Double(bright.count)
        }
        let moved = await centre(joined), unmoved = await centre(still)
        pass(moved.map { abs($0 - 97.5) < 4 } ?? false, "shift applied: part 1 lines up with part 2",
             "square centre \(moved.map { String(format: "%.1f", $0) } ?? "-") (part 2: 97.5; without the fix: \(unmoved.map { String(format: "%.1f", $0) } ?? "-"))")
        try? FileManager.default.removeItem(at: joined)
        try? FileManager.default.removeItem(at: still)
    }

    // MARK: 0 · Silhouette

    private func silhouette() async throws {
        let size = CGSize(width: 360, height: 640)
        // Part 1: a brighter grey room (140) and a red "person" from x 150 to 230 -- it crosses the middle (180).
        let part1 = try await Self.movie(name: "clone-st-p1", size: size, seconds: 5, tone: 440, amplitude: 6_000) { _, x, y in
            (x >= 150 && x < 230 && y >= 200 && y < 600) ? (220, 30, 30) : (140, 140, 140)
        }
        // Part 2: the room (120) and a blue person on the right, from x 260 to 330.
        let part2 = try await Self.movie(name: "clone-st-p2", size: size, seconds: 7, tone: 660, amplitude: 6_000) { _, x, y in
            (x >= 260 && x < 330 && y >= 200 && y < 600) ? (30, 30, 220) : (120, 120, 120)
        }
        // A stand-in for Vision: the red shape is "the person".
        SplitVideoCompositor.testMask = { image in
            image.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 4, y: -4, z: -4, w: 0), "inputGVector": CIVector(x: 4, y: -4, z: -4, w: 0),
                "inputBVector": CIVector(x: 4, y: -4, z: -4, w: 0), "inputBiasVector": CIVector(x: -0.6, y: -0.6, z: -0.6, w: 0),
            ]).applyingFilter("CIColorClamp")
        }
        defer { SplitVideoCompositor.testMask = nil }
        let gains = await SplitCompositor.colourGains(part1: part1, part2: part2)
        let stats = SplitStats()
        let joined = try await SplitCompositor.join(part1: part1, part2: part2, offset: .zero, blend: .silhouette,
                                                    gains: gains, stats: stats)
        lines.append("INFO silhouette · \(stats.summary) · gains \(String(format: "%.2f %.2f %.2f", gains.0, gains.1, gains.2))")
        guard let row = await Self.row(of: joined, at: 1, y: 400) else { pass(false, "read the clone"); return }
        func isRed(_ p: Pixel) -> Bool { p.r > 120 && p.b < 90 && p.g < 90 }
        func isBlue(_ p: Pixel) -> Bool { p.b > 150 && p.r < 90 }
        pass(isRed(row[170]) && isRed(row[210]), "part 1's person over part 2 (also past the middle)",
             "x170 \(row[170].r),\(row[170].b) · x210 \(row[210].r),\(row[210].b)")
        pass(isBlue(row[295]), "part 2's person stays", "x295 \(row[295].r),\(row[295].b)")
        let room = [row[40], row[100], row[245]]
        pass(room.allSatisfy { abs(Int($0.r) - 120) < 14 && abs(Int($0.b) - 120) < 14 }, "the room is part 2's (whole background)",
             room.map { "\($0.r)" }.joined(separator: ","))
        // No line in the middle: across the room (x 235..255, away from both people) and on part 1's own side
        // (x 20..140) nothing jumps.
        var jump = 0
        for x in Array(21..<140) + Array(236..<255) where abs(Int(row[x].r) - Int(row[x - 1].r)) > 8 { jump += 1 }
        pass(jump == 0, "no line in the middle of the clone", "\(jump) jumps")
        pass(gains.0 < 0.95 && gains.0 > 0.6, "part 1 matched to part 2's brightness", String(format: "gain %.2f", gains.0))
        pass(row[190].r < 215, "the cut-out is toned to part 2", "R \(row[190].r) (unmatched 220)")
        // Soft edge: the cut-out's border goes through some in-between values (not a hard cut).
        let edge = (140..<156).filter { row[$0].r > 128 && row[$0].r < 210 }.count
        pass(edge >= 1, "soft edge on the cut-out", "\(edge) px in between")
        let duration = try await AVURLAsset(url: joined).load(.duration).seconds
        pass(abs(duration - 7) < 0.15, "clone length = part 2", String(format: "%.2f s", duration))
        let samples = await Self.mono(of: joined)
        func tone(_ hz: Double) -> Double {
            let a = Int(1.0 * 44_100), b = min(samples.count, Int(4.0 * 44_100))
            return b > a ? Self.goertzel(Array(samples[a..<b]), hz: hz, rate: 44_100) : 0
        }
        let a440 = tone(440), a660 = tone(660)
        pass(a440 > 0.005 && a660 > 0.005 && a440 / max(1e-9, a660) > 0.6 && a440 / max(1e-9, a660) < 1.6,
             "both voices, even level", String(format: "440 %.3f · 660 %.3f", a440, a660))
        try? FileManager.default.removeItem(at: joined)
    }

    // MARK: 0b · Fallback + timing (the real Vision pass)

    private func fallbackAndTiming() async {
        let size = CGSize(width: 360, height: 640)
        do {
            let a = try await Self.movie(name: "split-st-red", size: size, seconds: 5, tone: 440, amplitude: 4_000) { f, _, _ in
                (UInt8(min(255, 80 + f * 3)), 0, 0)
            }
            let b = try await Self.movie(name: "split-st-blue", size: size, seconds: 7, tone: 660, amplitude: 16_000) { _, _, _ in (0, 0, 255) }
            // No person in these videos: the silhouette must say so (the app then uses the halves blend).
            let stats = SplitStats()
            var failed = false
            do {
                let url = try await SplitCompositor.join(part1: a, part2: b, offset: .zero, blend: .silhouette, stats: stats)
                try? FileManager.default.removeItem(at: url)
            } catch { failed = true }
            pass(failed, "no person found → the silhouette reports it (fallback)", stats.summary)
            let halves = try await SplitCompositor.join(part1: a, part2: b, offset: .zero, blend: .halves)
            let d = (try? await AVURLAsset(url: halves).load(.duration).seconds) ?? 0
            pass(abs(d - 7) < 0.15, "the fallback blend still makes the video", String(format: "%.2f s", d))
            try? FileManager.default.removeItem(at: halves)

            // Timing: a 30 s 720x1280 30 fps video through the real Vision pass.
            let p1 = try await Self.fastMovie(name: "clone-st-30a", width: 720, height: 1280, seconds: 30, fps: 30, shade: 90)
            let p2 = try await Self.fastMovie(name: "clone-st-30b", width: 720, height: 1280, seconds: 30, fps: 30, shade: 110)
            let timing = SplitStats()
            let started = Date()
            do {
                let url = try await SplitCompositor.join(part1: p1, part2: p2, offset: .zero, blend: .silhouette, stats: timing)
                try? FileManager.default.removeItem(at: url)
            } catch {}
            lines.append(String(format: "INFO timing · 30 s 720x1280 video through the silhouette pass: %.1f s (%@)",
                                Date().timeIntervalSince(started), timing.summary))
            let h0 = Date()
            if let url = try? await SplitCompositor.join(part1: p1, part2: p2, offset: .zero, blend: .halves) {
                try? FileManager.default.removeItem(at: url)
            }
            lines.append(String(format: "INFO timing · same video, halves blend: %.1f s", Date().timeIntervalSince(h0)))
        } catch {
            pass(false, "fallback test crashed", "\(error)")
        }
    }

    // MARK: Synthetic media

    struct Pixel { let r: UInt8, g: UInt8, b: UInt8 }

    /// A video made here: `paint(frame, x, y)` gives each pixel (top-left origin); a steady tone.
    static func movie(name: String, size: CGSize, seconds: Double, tone: Double, amplitude: Double,
                      paint: @escaping (Int, Int, Int) -> (UInt8, UInt8, UInt8)) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(name).mov")
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let w = Int(size.width), h = Int(size.height)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 4_000_000],
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44_100, AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 96_000,
        ])
        video.expectsMediaDataInRealTime = false
        audio.expectsMediaDataInRealTime = false
        writer.add(video); writer.add(audio)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        let fps = 10, rate = 44_100.0, chunk = 4_410
        var format = AudioStreamBasicDescription(mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
                                                 mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
                                                 mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2,
                                                 mChannelsPerFrame: 1, mBitsPerChannel: 16, mReserved: 0)
        var description: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &format, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                       magicCookie: nil, extensions: nil, formatDescriptionOut: &description)
        guard let description else { throw CocoaError(.fileWriteUnknown) }
        func ready(_ input: AVAssetWriterInput) async throws {
            var waited = 0
            while !input.isReadyForMoreMediaData {
                if writer.status == .failed { throw writer.error ?? CocoaError(.fileWriteUnknown) }
                waited += 1
                if waited > 2_000 { throw CocoaError(.fileWriteUnknown) }
                try await Task.sleep(for: .milliseconds(5))
            }
        }
        for frame in 0..<Int(seconds * Double(fps)) {
            try await ready(video)
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, nil, &buffer)
            if let buffer {
                CVPixelBufferLockBaseAddress(buffer, [])
                if let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) {
                    let stride = CVPixelBufferGetBytesPerRow(buffer)
                    for y in 0..<h {
                        let row = base + y * stride
                        for x in 0..<w {
                            let (r, g, b) = paint(frame, x, y)
                            row[x * 4] = b; row[x * 4 + 1] = g; row[x * 4 + 2] = r; row[x * 4 + 3] = 255
                        }
                    }
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(fps)))
            }
            try await ready(audio)
            let start = frame * chunk
            var samples = [Int16](repeating: 0, count: chunk)
            for i in 0..<chunk { samples[i] = Int16(amplitude * sin(2 * .pi * tone * Double(start + i) / rate)) }
            var block: CMBlockBuffer?
            CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: chunk * 2, blockAllocator: nil,
                                               customBlockSource: nil, offsetToData: 0, dataLength: chunk * 2, flags: 0,
                                               blockBufferOut: &block)
            guard let block else { continue }
            samples.withUnsafeBytes { raw in
                _ = CMBlockBufferReplaceDataBytes(with: raw.baseAddress!, blockBuffer: block, offsetIntoDestination: 0, dataLength: chunk * 2)
            }
            var sample: CMSampleBuffer?
            CMAudioSampleBufferCreateReadyWithPacketDescriptions(
                allocator: nil, dataBuffer: block, formatDescription: description, sampleCount: chunk,
                presentationTimeStamp: CMTime(value: CMTimeValue(start), timescale: 44_100),
                packetDescriptions: nil, sampleBufferOut: &sample)
            if let sample { audio.append(sample) }
        }
        video.markAsFinished()
        audio.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        return url
    }

    /// A plain video made fast (grey with a moving white bar), for timing only.
    static func fastMovie(name: String, width w: Int, height h: Int, seconds: Double, fps: Int, shade: UInt8) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(name).mov")
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
        ])
        video.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
        writer.add(video)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        for frame in 0..<Int(seconds * Double(fps)) {
            var waited = 0
            while !video.isReadyForMoreMediaData {
                waited += 1
                if waited > 4_000 { throw CocoaError(.fileWriteUnknown) }
                try await Task.sleep(for: .milliseconds(5))
            }
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, nil, &buffer)
            guard let buffer else { continue }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                let stride = CVPixelBufferGetBytesPerRow(buffer)
                memset(base, Int32(shade), stride * h)
                let bar = (frame * 8) % max(1, h - 40)
                memset(base + bar * stride, 255, stride * 40)
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(fps)))
        }
        video.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        return url
    }

    /// One row of pixels (top-left origin) of the frame at `time`.
    static func row(of url: URL, at time: Double, y: Int) async -> [Pixel]? {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        guard let image = try? await generator.image(at: CMTime(seconds: time, preferredTimescale: 600)).image else { return nil }
        let w = image.width, h = image.height
        var data = [UInt8](repeating: 0, count: w * h * 4)
        guard let context = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let line = min(h - 1, max(0, y))
        return (0..<w).map { x in
            let i = (line * w + x) * 4
            return Pixel(r: data[i], g: data[i + 1], b: data[i + 2])
        }
    }

    /// The whole sound, mono float at 44.1 kHz.
    static func mono(of url: URL) async -> [Float] {
        let asset = AVURLAsset(url: url)
        guard let tracks = try? await asset.loadTracks(withMediaType: .audio), !tracks.isEmpty,
              let reader = try? AVAssetReader(asset: asset) else { return [] }
        let output = AVAssetReaderAudioMixOutput(audioTracks: tracks, audioSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false, AVLinearPCMIsBigEndianKey: false, AVSampleRateKey: 44_100, AVNumberOfChannelsKey: 1,
        ])
        reader.add(output)
        guard reader.startReading() else { return [] }
        var all: [Float] = []
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var data = [Float](repeating: 0, count: length / 4)
            data.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            all.append(contentsOf: data)
        }
        return all
    }

    /// Amplitude of one frequency in `samples` (Goertzel).
    static func goertzel(_ samples: [Float], hz: Double, rate: Double) -> Double {
        let k = 2 * cos(2 * .pi * hz / rate)
        var s1 = 0.0, s2 = 0.0
        for v in samples {
            let s0 = Double(v) + k * s1 - s2
            s2 = s1; s1 = s0
        }
        let power = s1 * s1 + s2 * s2 - k * s1 * s2
        return 2 * power.squareRoot() / Double(max(1, samples.count))
    }
}

/// The self-test screen: one line per check (the probe reads `splitSelfTest`).
struct SplitSelfTestView: View {
    @StateObject private var test = SplitSelfTest()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 6) {
                Text(verbatim: test.finished ? "Clon self-test · done" : "Clon self-test · running…")
                    .font(CS.inter(17, .bold)).foregroundStyle(CS.wiInk)
                ForEach(Array(test.lines.enumerated()), id: \.offset) { _, line in
                    Text(verbatim: line).font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(line.hasPrefix("FAIL") ? CS.wiError : CS.wiInk)
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(CS.wiPage.ignoresSafeArea())
        .overlay(alignment: .bottom) {
            Text(verbatim: (test.finished ? "DONE " : "RUNNING ") + test.lines.joined(separator: " | "))
                .font(.system(size: 2)).opacity(0.05)
                .accessibilityIdentifier("splitSelfTest")
        }
        .task { await test.run() }
    }
}
#endif
