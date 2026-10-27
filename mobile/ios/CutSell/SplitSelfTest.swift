#if DEBUG
import AVFoundation
import CoreImage
import SwiftUI

/// Debug builds only (`-CutSellScreen split_selftest`): checks the Split compositor on synthetic videos
/// made here (nothing from the repo, nothing of a user):
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
            try await seamAndSound()
            try await shift()
            await pro()
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
        let joined = try await SplitCompositor.join(part1: red, part2: blue, offset: .zero, blend: .basic)
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
                let wave = 35 * sin(Double(x) / 9) * cos(Double(y) / 13) + 20 * sin(Double(x + 2 * y) / 23)
                var hash = UInt32(truncatingIfNeeded: (x &* 73_856_093) ^ (y &* 19_349_663))
                hash = (hash ^ (hash >> 13)) &* 1_274_126_177
                let grain = Double(hash % 31) - 15
                let v = UInt8(max(30, min(180, 105 + wave + grain)))
                return (v, UInt8(max(0, Int(v) - 12)), UInt8(min(255, Int(v) + 8)))
            }
        }
        let part1 = try await Self.movie(name: "split-st-room1", size: size, seconds: 2, tone: 440, amplitude: 8_000, paint: room(shift: 0))
        let part2 = try await Self.movie(name: "split-st-room2", size: size, seconds: 2, tone: 660, amplitude: 8_000, paint: room(shift: 18))
        guard let first1 = await SplitAligner.firstFrame(of: part1), let first2 = await SplitAligner.firstFrame(of: part2),
              let offset = SplitAligner.offset(part1: first1, live: first2) else {
            pass(false, "aligner measured the shift"); return
        }
        let expected = 18.0 / 360
        pass(abs(offset.x - expected) < 0.012 && abs(offset.y) < 0.012, "aligner measured the shift",
             String(format: "x %.3f (expected %.3f) y %.3f", offset.x, expected, offset.y))
        pass(offset.magnitude > 0.02, "a 5% move is flagged (> 2%)")
        let joined = try await SplitCompositor.join(part1: part1, part2: part2, offset: offset, blend: .basic)
        let still = try await SplitCompositor.join(part1: part1, part2: part2, offset: .zero, blend: .basic)
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

    // MARK: Pro (no person in a synthetic video: only that it runs or says it can't)

    private func pro() async {
        let size = CGSize(width: 360, height: 640)
        do {
            let a = try await Self.movie(name: "split-st-red", size: size, seconds: 5, tone: 440, amplitude: 4_000) { f, _, _ in
                (UInt8(min(255, 80 + f * 3)), 0, 0)
            }
            let b = try await Self.movie(name: "split-st-blue", size: size, seconds: 7, tone: 660, amplitude: 16_000) { _, _, _ in (0, 0, 255) }
            let started = Date()
            let url = try await SplitCompositor.join(part1: a, part2: b, offset: .zero, blend: .pro)
            let seconds = Date().timeIntervalSince(started)
            let duration = (try? await AVURLAsset(url: url).load(.duration).seconds) ?? 0
            pass(abs(duration - 7) < 0.15, "Pro blend runs", String(format: "%.1f s video in %.1f s", duration, seconds))
            try? FileManager.default.removeItem(at: url)
        } catch {
            lines.append("INFO Pro blend not available here · \(error.localizedDescription)")
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
                Text(verbatim: test.finished ? "Split self-test · done" : "Split self-test · running…")
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
