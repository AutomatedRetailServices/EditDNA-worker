import AVFoundation
import Photos
import SwiftUI
import UIKit

/// Build 18: what the iPhone itself saw, for problems that only happen on a real phone (the
/// Simulator passed build 17's checks while sounds stayed silent on Swanny's iPhone).
///
/// - `DiagnosticsEventLog`: the last 50 events (sounds, mute, keyframes, uploads, preview), kept
///   on this iPhone only. Field values are bounded scalars; anything that could be personal
///   (names, titles, text, links, paths, account or project ids, tokens) is dropped before it is
///   stored, so "Send report" never carries it.
/// - `EditorMixReport`: after every preview rebuild, each sound / voice on the timeline: whether
///   its file is on the iPhone, its length, whether it went into the mix, its volume and the
///   level measured in the mix itself (read back with AVAssetReader, the same mix the player
///   plays).
/// - `DiagnosticsView`: Settings (You) → five taps on the version number.

// MARK: - Events

final class DiagnosticsEventLog: @unchecked Sendable {
    static let shared = DiagnosticsEventLog()
    static let limit = 50

    struct Entry: Codable, Equatable {
        let at: Date
        let event: String
        let fields: [String: String]

        var line: String {
            let time = DiagnosticsEventLog.timeFormatter.string(from: at)
            let pairs = fields.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " ")
            return pairs.isEmpty ? "\(time) \(event)" : "\(time) \(event) \(pairs)"
        }
    }

    static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "MM-dd HH:mm:ss"
        return f
    }()

    private let lock = NSLock()
    private var entries: [Entry]
    private let queue = DispatchQueue(label: "ai.cutsell.diagnostics.events", qos: .utility)

    private static var fileURL: URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("diagnostics_events.json")
    }

    private init() {
        if let data = try? Data(contentsOf: Self.fileURL),
           let saved = try? JSONDecoder().decode([Entry].self, from: data) {
            entries = Array(saved.suffix(Self.limit))
        } else {
            entries = []
        }
    }

    /// Words in a field name that may carry something personal or secret: never stored.
    static let blockedKeyParts = ["user", "token", "email", "url", "uri", "path", "name", "title", "text",
                                  "caption", "project", "account", "session", "key", "secret", "auth", "phone", "location"]

    /// Only bounded technical values (event names, numbers, states, error types) are kept.
    static func sanitize(_ fields: [String: String]) -> [String: String] {
        var kept: [String: String] = [:]
        for (key, value) in fields {
            let lower = key.lowercased()
            if blockedKeyParts.contains(where: { lower.contains($0) }) { continue }
            let lowerValue = value.lowercased()
            if lowerValue.contains("http") || lowerValue.contains("/") || lowerValue.contains("@") { continue }
            kept[String(key.prefix(32))] = String(value.prefix(60))
        }
        return kept
    }

    func add(_ event: String, _ fields: [String: String]) {
        let entry = Entry(at: Date(), event: String(event.prefix(48)), fields: Self.sanitize(fields))
        lock.lock()
        entries.append(entry)
        if entries.count > Self.limit { entries.removeFirst(entries.count - Self.limit) }
        let snapshot = entries
        lock.unlock()
        queue.async {
            if let data = try? JSONEncoder().encode(snapshot) { try? data.write(to: Self.fileURL, options: .atomic) }
        }
        DispatchQueue.main.async { NotificationCenter.default.post(name: .cutsellDiagnosticsChanged, object: nil) }
    }

    var recent: [Entry] {
        lock.lock(); defer { lock.unlock() }
        return entries
    }

    func clear() {
        lock.lock(); entries = []; lock.unlock()
        queue.async { try? FileManager.default.removeItem(at: Self.fileURL) }
    }
}

extension Notification.Name {
    static let cutsellDiagnosticsChanged = Notification.Name("cutsell.diagnostics.changed")
}

// MARK: - What the preview really mixes

struct EditorMixEntry: Equatable, Identifiable {
    enum Kind: String { case sound, voice, music, overlay }
    let id: String
    let kind: Kind
    /// "Typing", "Good", "Voice-over 1"… (CutSell's own names, never a file name of the creator).
    let label: String
    let start: Double
    let length: Double
    let fileOnPhone: Bool
    /// Length of the file's sound (nil: the file could not be read).
    var fileSeconds: Double?
    /// Seconds put into the preview's composition (0: it is not in the mix).
    var insertedSeconds: Double
    /// Volume the mix gives it (0...2).
    let volume: Double
    /// Loudness measured in the mix over its window (dBFS); nil until measured.
    var measuredDB: Double?
    /// Loudest level that really went through the player while it played (nil: not played yet).
    var heardDB: Double?
    var problem: String?

    var inMix: Bool { insertedSeconds > 0.01 && volume > 0.001 }
}

struct EditorMixReport: Equatable {
    var builtAt = Date()
    var entries: [EditorMixEntry] = []
    /// The video's own sound per clip: start, end, level.
    var videoLevels: [(start: Double, end: Double, level: Double)] = []
    var videoVolume: Double = 1
    var duration: Double = 0

    static func == (a: EditorMixReport, b: EditorMixReport) -> Bool {
        a.builtAt == b.builtAt && a.entries == b.entries && a.videoVolume == b.videoVolume && a.duration == b.duration
            && a.videoLevels.map(\.level) == b.videoLevels.map(\.level)
    }

    /// One short line per sound, for the probe's hidden label and the report.
    var summary: String {
        let sounds = entries.map { entry in
            String(format: "%@@%.2f+%.2f file=%@ dur=%@ in=%.2f vol=%.2f db=%@ heard=%@%@", entry.label.replacingOccurrences(of: " ", with: "_"),
                   entry.start, entry.length, entry.fileOnPhone ? "yes" : "no",
                   entry.fileSeconds.map { String(format: "%.2f", $0) } ?? "?", entry.insertedSeconds, entry.volume,
                   entry.measuredDB.map { String(format: "%.1f", $0) } ?? "?",
                   entry.heardDB.map { String(format: "%.1f", $0) } ?? "?", entry.problem.map { " problem=\($0)" } ?? "")
        }
        let base = videoLevels.map { String(format: "%.1f-%.1f:%.2f", $0.start, $0.end, $0.level) }.joined(separator: ",")
        return "mix sounds=[\(sounds.joined(separator: ";"))] video=[\(base)] video_volume=\(String(format: "%.2f", videoVolume))"
    }
}

@MainActor
final class EditorDiagnostics: ObservableObject {
    static let shared = EditorDiagnostics()
    @Published var mix = EditorMixReport()
    @Published var keyframeCount = 0
    @Published var aiKeyframeCount = 0

    /// Loudness (dBFS, RMS) of what `tracks` of `asset` really carry between `start` and `end` on the
    /// preview's timeline (the track content itself; the mix's constant volume is added by the caller).
    nonisolated static func measure(asset: AVAsset, tracks: [AVAssetTrack], start: Double, end: Double) async -> Double? {
        guard end - start > 0.02, !tracks.isEmpty, let reader = try? AVAssetReader(asset: asset) else { return nil }
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false,
            AVSampleRateKey: 44_100, AVNumberOfChannelsKey: 1,
        ]
        let output = AVAssetReaderAudioMixOutput(audioTracks: tracks, audioSettings: settings)
        guard reader.canAdd(output) else { return nil }
        reader.add(output)
        reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600),
                                       end: CMTime(seconds: end, preferredTimescale: 600))
        guard reader.startReading() else { return nil }
        var sum = 0.0
        var count = 0
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            let samples = length / MemoryLayout<Float>.size
            guard samples > 0 else { continue }
            var floats = [Float](repeating: 0, count: samples)
            let status = floats.withUnsafeMutableBytes { raw in
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: samples * MemoryLayout<Float>.size,
                                           destination: raw.baseAddress!)
            }
            guard status == noErr else { continue }
            for value in floats { sum += Double(value * value) }
            count += samples
        }
        guard count > 0 else { return -120 }
        let rms = (sum / Double(count)).squareRoot()
        return rms > 0 ? max(-120, 20 * log10(rms)) : -120
    }
}

// MARK: - Device facts for the report

enum DiagnosticsFacts {
    static var appVersion: String {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "\(version) (\(build))"
    }

    static var buildNumber: String { Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?" }

    static var audioLines: [String] {
        let session = AVAudioSession.sharedInstance()
        let outputs = session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ", ")
        return [
            "Category: \(session.category.rawValue.replacingOccurrences(of: "AVAudioSessionCategory", with: ""))",
            "Mode: \(session.mode.rawValue.replacingOccurrences(of: "AVAudioSessionMode", with: ""))",
            "Output: \(outputs.isEmpty ? "none" : outputs)",
            String(format: "Output volume: %.0f%%", session.outputVolume * 100),
            "Other audio playing: \(session.isOtherAudioPlaying ? "yes" : "no")",
            "Silenced by another app: \(session.secondaryAudioShouldBeSilencedHint ? "yes" : "no")",
        ]
    }

    static var photosPermission: String {
        switch PHPhotoLibrary.authorizationStatus(for: .readWrite) {
        case .authorized: return "Full access"
        case .limited: return "Limited (only the photos you picked)"
        case .denied: return "Denied"
        case .restricted: return "Restricted"
        case .notDetermined: return "Not asked yet"
        @unknown default: return "Unknown"
        }
    }

    @MainActor
    static func soundLines(_ report: EditorMixReport) -> [String] {
        guard !report.entries.isEmpty else { return ["No sounds or voices on the last timeline played."] }
        return report.entries.map { entry in
            var parts = [String(format: "%@ at %@ (%.2f s)", entry.label, EditorTime.short(entry.start), entry.length)]
            parts.append("file \(entry.fileOnPhone ? "on iPhone" : "MISSING")")
            parts.append(entry.fileSeconds.map { String(format: "length %.2f s", $0) } ?? "length unknown")
            parts.append(entry.inMix ? String(format: "in mix %.2f s", entry.insertedSeconds) : "NOT in mix")
            parts.append(String(format: "volume %.0f%%", entry.volume * 100))
            if let db = entry.measuredDB { parts.append(String(format: "level %.1f dB", db)) }
            parts.append(entry.heardDB.map { String(format: "heard while playing %.1f dB", $0) } ?? "not played yet")
            if let problem = entry.problem { parts.append("problem: \(problem)") }
            return parts.joined(separator: " · ")
        }
    }

    @MainActor
    static func report() -> String {
        let mix = EditorDiagnostics.shared.mix
        var lines: [String] = ["CutSell diagnostics report", "App: \(appVersion)", "iPhone: \(CutSellDiagnostics.deviceModel) · iOS \(CutSellDiagnostics.iosVersion)", ""]
        lines.append("Audio")
        lines += audioLines.map { "  \($0)" }
        lines.append("")
        lines.append("Photos permission: \(photosPermission)")
        lines.append("")
        lines.append("Sounds in the last preview (built \(DiagnosticsEventLog.timeFormatter.string(from: mix.builtAt)))")
        lines += soundLines(mix).map { "  \($0)" }
        lines.append(String(format: "  Video sound: %.0f%%", mix.videoVolume * 100))
        for level in mix.videoLevels {
            lines.append(String(format: "  Clip %@-%@: %.0f%%", EditorTime.short(level.start), EditorTime.short(level.end), level.level * 100))
        }
        lines.append("")
        lines.append("Keyframes: \(EditorDiagnostics.shared.keyframeCount) (\(EditorDiagnostics.shared.aiKeyframeCount) from Auto Zoom AI)")
        let sounds = (SfxLibrary.buttons.map(\.id) + ["typing_long"]).map { id in "\(id)=\(SfxLibrary.localURL(id) != nil ? "on iPhone" : "missing")" }
        lines.append("Sound files: \(sounds.joined(separator: ", "))")
        lines.append("")
        lines.append("Last \(DiagnosticsEventLog.limit) events")
        lines += DiagnosticsEventLog.shared.recent.reversed().map { "  \($0.line)" }
        return lines.joined(separator: "\n")
    }

    /// The report as a text file to share (no personal data, no keys).
    @MainActor
    static func reportFile() -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("CutSell-diagnostics-build\(buildNumber).txt")
        do {
            try report().data(using: .utf8)?.write(to: url, options: .atomic)
            return url
        } catch {
            return nil
        }
    }
}

// MARK: - The screen

/// Settings (You) → five taps on the version number.
struct DiagnosticsView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var editor = EditorDiagnostics.shared
    @State private var events: [DiagnosticsEventLog.Entry] = Array(DiagnosticsEventLog.shared.recent.reversed())
    @State private var reportURL: URL?
    @State private var refresh = 0

    var body: some View {
        NavigationStack {
            List {
                Section("App") {
                    row("Build", DiagnosticsFacts.buildNumber, id: "diagBuild")
                    row("Version", DiagnosticsFacts.appVersion)
                    row("iPhone", "\(CutSellDiagnostics.deviceModel) · iOS \(CutSellDiagnostics.iosVersion)")
                }
                Section("Audio") {
                    ForEach(DiagnosticsFacts.audioLines, id: \.self) { Text($0).font(.footnote.monospaced()) }
                }
                .accessibilityIdentifier("diagAudio")
                Section("Photos") {
                    row("Permission", DiagnosticsFacts.photosPermission, id: "diagPhotos")
                }
                Section("Sounds on the timeline (last preview)") {
                    ForEach(DiagnosticsFacts.soundLines(editor.mix), id: \.self) { Text($0).font(.footnote) }
                    Text(String(format: "Video sound: %.0f%%", editor.mix.videoVolume * 100)).font(.footnote)
                }
                .accessibilityIdentifier("diagSounds")
                Section("Last \(DiagnosticsEventLog.limit) events") {
                    if events.isEmpty { Text("No events yet.").foregroundStyle(.secondary) }
                    ForEach(Array(events.enumerated()), id: \.offset) { _, entry in
                        Text(entry.line).font(.caption.monospaced())
                    }
                }
                .accessibilityIdentifier("diagEvents")
            }
            .id(refresh)
            .navigationTitle("Diagnostics")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    if let reportURL {
                        ShareLink(item: reportURL) { Text("Send report") }
                            .accessibilityIdentifier("diagSendReport")
                    }
                }
            }
            .onAppear { reload() }
            .onReceive(NotificationCenter.default.publisher(for: .cutsellDiagnosticsChanged)) { _ in
                events = Array(DiagnosticsEventLog.shared.recent.reversed())
            }
        }
        .preferredColorScheme(.dark)
        .accessibilityIdentifier("diagnosticsScreen")
    }

    private func reload() {
        events = Array(DiagnosticsEventLog.shared.recent.reversed())
        reportURL = DiagnosticsFacts.reportFile()
        refresh += 1
    }

    private func row(_ title: String, _ value: String, id: String? = nil) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value).foregroundStyle(.secondary).multilineTextAlignment(.trailing)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(id ?? "diag_\(title)")
    }
}
