#if DEBUG
import AVFoundation
import UIKit

/// Build 18 Simulator probe only (Debug builds): media made on the spot, so the overlay timeline can
/// be tested with a photo AND a video without any file in the repository.
enum ProbeMedia {
    /// The middle of the first Auto Zoom AI push-in (between its 2nd and 3rd ◇: fully zoomed).
    static func firstAIPlateau(_ marks: [KeyframeMark]) -> Double? {
        let times = marks.filter(\.isAI).map(\.time).sorted()
        guard times.count >= 4 else { return nil }
        return (times[1] + times[2]) / 2
    }

    /// A 600 x 600 orange photo with a dark cross (easy to see over the video).
    static func makePhoto() -> URL? {
        let size = CGSize(width: 600, height: 600)
        let image = UIGraphicsImageRenderer(size: size).image { context in
            UIColor(red: 1, green: 0.55, blue: 0.1, alpha: 1).setFill()
            context.fill(CGRect(origin: .zero, size: size))
            UIColor(white: 0.1, alpha: 1).setFill()
            context.fill(CGRect(x: 280, y: 60, width: 40, height: 480))
            context.fill(CGRect(x: 60, y: 280, width: 480, height: 40))
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("probe-photo-\(UUID().uuidString).jpg")
        guard let data = image.jpegData(compressionQuality: 0.9), (try? data.write(to: url)) != nil else { return nil }
        return url
    }

    /// A short green video (H.264, 30 fps) with a white bar moving across.
    static func makeVideo(seconds: Double) async -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("probe-video-\(UUID().uuidString).mp4")
        let width = 360, height = 640, fps: Int32 = 30
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return nil }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        guard writer.canAdd(input) else { return nil }
        writer.add(input)
        guard writer.startWriting() else { return nil }
        writer.startSession(atSourceTime: .zero)
        let frames = Int(seconds * Double(fps))
        for index in 0..<frames {
            while !input.isReadyForMoreMediaData { try? await Task.sleep(nanoseconds: 5_000_000) }
            guard let pool = adaptor.pixelBufferPool else { break }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let buffer else { break }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                let row = CVPixelBufferGetBytesPerRow(buffer)
                let pixels = base.assumingMemoryBound(to: UInt8.self)
                let bar = (index * 8) % width
                for y in 0..<height {
                    for x in 0..<width {
                        let offset = y * row + x * 4
                        let white = abs(x - bar) < 12
                        pixels[offset] = white ? 255 : 60        // B
                        pixels[offset + 1] = white ? 255 : 180   // G
                        pixels[offset + 2] = white ? 255 : 40    // R
                        pixels[offset + 3] = 255
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(index), timescale: fps))
        }
        input.markAsFinished()
        await writer.finishWriting()
        return writer.status == .completed ? url : nil
    }
}
#endif
