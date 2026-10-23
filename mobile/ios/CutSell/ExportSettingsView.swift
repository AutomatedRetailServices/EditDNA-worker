import SwiftUI
import UIKit

/// Export settings (Figma 950:8730 Original · 950:8790 4K). Exported on this iPhone from the original in
/// Photos; never upscaled (a choice above the original's resolution is greyed out).
struct ExportSettingsView: View {
    @ObservedObject var model: DraftEditorViewModel
    @ObservedObject var exporter: DeviceExport
    @Binding var settings: ExportSettings
    let projectID: String
    let session: CutSellSession?
    /// The server export (fallback when this iPhone can't).
    let onServerExport: () -> Void
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL

    private static let blue = Color(hex: 0x1769FF)
    private static let control = Color(hex: 0x0A1326)
    private static let elevated = Color(hex: 0x0B1730)
    private static let secondaryBG = Color(hex: 0x081226)
    private static let primaryGradient = LinearGradient(
        stops: [.init(color: Color(hex: 0x19E6FF), location: 0.097), .init(color: Color(hex: 0x1769FF), location: 0.516),
                .init(color: Color(hex: 0xFF21D7), location: 0.903)],
        startPoint: UnitPoint(x: 0.25, y: 0.07), endPoint: UnitPoint(x: 0.75, y: 0.93))

    private var info: ExportSourceInfo { exporter.info }
    private var cutSeconds: Double {
        model.selectedClips.reduce(0.0) { sum, clip in
            let start = clip["start"]?.doubleValue ?? 0, end = clip["end"]?.doubleValue ?? start
            return sum + max(0, end - start) / ClipSpeed.of(clip)
        }
    }
    private var maxShort: Int { ExportSourceInfo.standardShortSide(info.shortSide) }
    private var is4K: Bool { settings.canvas(for: info).width >= 2160 }

    var body: some View {
        ZStack(alignment: .top) {
            Color(hex: 0x050814).ignoresSafeArea()
            preview
            LinearGradient(stops: [.init(color: Color(hex: 0x050814, opacity: 0.04), location: 0),
                                   .init(color: Color(hex: 0x050814, opacity: 0.18), location: 0.55),
                                   .init(color: Color(hex: 0x050814, opacity: 0.92), location: 1)],
                           startPoint: .top, endPoint: .bottom)
                .frame(height: 300).frame(maxWidth: .infinity).ignoresSafeArea(edges: .top).allowsHitTesting(false)
            header
            VStack(spacing: 0) {
                Spacer(minLength: 0)
                sheet
            }
            .ignoresSafeArea(edges: .bottom)
        }
        .task { await exporter.detect(model: model) }
        .accessibilityIdentifier("exportSettings")
    }

    // MARK: Top (Figma: back · file name · more, the cut's picture)

    private var header: some View {
        HStack {
            Button { if !exporter.running { exporter.reset(); dismiss() } } label: {
                Image("ExportBack").resizable().frame(width: 44, height: 44)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(L10n.tr("Back"))
            .accessibilityIdentifier("exportBack")
            Text(verbatim: model.project.title.isEmpty ? "CutSell" : model.project.title)
                .font(CS.inter(14, .semibold)).foregroundStyle(CS.textPrimaryF)
                .lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: .infinity)
            Menu {
                Button(L10n.tr("Export on our server (1080p)")) { dismiss(); onServerExport() }
            } label: {
                Image("ExportMore").resizable().frame(width: 44, height: 44)
            }
            .accessibilityIdentifier("exportMore")
        }
        .frame(height: 44)
        .padding(.horizontal, 24)
        .padding(.top, 4)
    }

    private var preview: some View {
        Group {
            if let still = ThumbnailStore.shared.image(for: projectID) {
                Image(uiImage: still).resizable().scaledToFill()
            } else {
                Color(hex: 0x0C1730)
            }
        }
        .frame(width: 118, height: 210)
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(Color(hex: 0x19E6FF, opacity: 0.8), lineWidth: 1.5))
        .shadow(color: Color(hex: 0x7C3AED, opacity: 0.22), radius: 12, y: 5)
        .shadow(color: Color(hex: 0x19E6FF, opacity: 0.28), radius: 9)
        .padding(.top, 34)
    }

    // MARK: Bottom sheet

    private var sheet: some View {
        VStack(alignment: .leading, spacing: 14) {
            Capsule().fill(Color(hex: 0x6F86AE, opacity: 0.72)).frame(width: 44, height: 5).frame(maxWidth: .infinity, alignment: .leading)
            VStack(alignment: .leading, spacing: 4) {
                Text(L10n.tr("Export")).font(CS.inter(26, .semibold)).foregroundStyle(CS.textPrimaryF)
                Text(verbatim: "9:16  •  \(L10n.tr("Final cut")) \(Self.clock(cutSeconds))")
                    .font(CS.inter(13)).foregroundStyle(CS.textSecondaryF)
            }
            switch exporter.phase {
            case .idle, .failed, .noSpace:
                choices
            case .original(let value):
                progress(L10n.tr("Getting your original from iCloud…"), value)
            case .exporting(let value):
                progress(is4K ? L10n.tr("Exporting 4K…") : L10n.tr("Exporting…"), value)
            case .saving:
                progress(L10n.tr("Saving to Photos…"), 1)
            case .done(let url):
                done(url)
            }
        }
        .padding(.horizontal, 24).padding(.top, 14).padding(.bottom, 34)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Self.elevated, in: UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28))
        .overlay(UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28).stroke(CS.borderDefault, lineWidth: 1))
        .shadow(color: .black.opacity(0.5), radius: 14, y: -8)
    }

    @ViewBuilder
    private var choices: some View {
        detected
        segment(L10n.tr("Resolution"), ExportResolution.allCases, selected: settings.resolution, id: "exportResolution",
                label: { $0.label }, enabled: { r in (r.shortSide ?? 0) <= maxShort }) { settings.resolution = $0 }
        Text(L10n.tr(resolutionHint)).font(CS.inter(11)).foregroundStyle(CS.textSecondaryF).padding(.top, -7)
            .accessibilityIdentifier("exportResolutionHint")
        segment(L10n.tr("Frame rate"), ExportFrameRate.allCases, selected: settings.frameRate, id: "exportFrameRate",
                label: { $0.label }, enabled: { ($0.value ?? 0) <= max(30, info.fps) }) { settings.frameRate = $0 }
        segment(L10n.tr("Quality"), ExportQuality.allCases, selected: settings.quality, id: "exportQualityChoice",
                label: { $0.label }, enabled: { _ in true }) { settings.quality = $0 }
        HStack {
            Text(L10n.tr("Estimated file size")).font(CS.inter(13)).foregroundStyle(CS.textSecondaryF)
            Spacer()
            Text(verbatim: "≈ \(Self.size(settings.estimatedBytes(for: info, seconds: cutSeconds)))")
                .font(CS.inter(18, .semibold)).foregroundStyle(CS.textPrimaryF)
                .accessibilityIdentifier("exportEstimate")
        }
        .padding(.horizontal, 16).frame(height: 56)
        .background(Self.secondaryBG, in: RoundedRectangle(cornerRadius: 16))
        if case .noSpace(let bytes) = exporter.phase {
            noSpace(bytes)
        } else if case .failed(let why) = exporter.phase {
            VStack(alignment: .leading, spacing: 6) {
                Text(L10n.tr(why)).font(CS.inter(12)).foregroundStyle(Color(hex: 0xFF8A8A))
                Button(L10n.tr("Export on our server (1080p)")) { dismiss(); onServerExport() }
                    .font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                    .accessibilityIdentifier("exportServerFallback")
            }
        }
        Button { exporter.start(model: model, settings: settings) } label: {
            Text(is4K ? L10n.tr("Export 4K to Photos") : L10n.tr("Export to Photos"))
                .font(CS.inter(16, .semibold)).foregroundStyle(CS.textPrimaryF)
                .frame(maxWidth: .infinity).frame(height: 56)
                .background(Self.primaryGradient, in: RoundedRectangle(cornerRadius: 18))
                .shadow(color: Color(hex: 0x1769FF, opacity: 0.28), radius: 8, y: 6)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("exportToPhotos")
        Button { dismiss() } label: {
            Text(L10n.tr("Cancel")).font(CS.inter(14, .medium)).foregroundStyle(CS.textMutedF).frame(maxWidth: .infinity)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("exportCancel")
    }

    private var resolutionHint: String {
        if !info.originals { return "Your original isn't on this iPhone: exporting from the 1080p copy." }
        if maxShort < 1440 { return "Your video was recorded in \(maxShort)p." }
        if is4K { return "4K from your original, with its own detail." }
        return "Match keeps the detected source resolution."
    }

    private var detected: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(verbatim: "✦  " + (is4K ? L10n.tr("4K export selected")
                                         : info.originals ? L10n.tr("Original quality detected") : L10n.tr("1080p copy (original not found)")))
                .font(CS.inter(13, .semibold)).foregroundStyle(Color(hex: 0x19E6FF))
            Text(verbatim: "\(info.width) × \(info.height)  •  \(info.fps) fps  •  \(info.codec)  •  \(info.hdr ? "HDR" : "SDR")")
                .font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(CS.borderDefault, lineWidth: 1))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: "\(info.width)x\(info.height) \(info.fps)fps \(info.codec) originals:\(info.originals)"))
        .accessibilityIdentifier("exportDetected")
    }

    private func segment<T: Identifiable & Equatable>(_ title: String, _ items: [T], selected: T, id: String,
                                                       label: @escaping (T) -> String, enabled: @escaping (T) -> Bool,
                                                       pick: @escaping (T) -> Void) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(CS.inter(13, .semibold)).foregroundStyle(CS.textSecondaryF)
            HStack(spacing: 2) {
                ForEach(items) { item in
                    let on = item == selected
                    let ok = enabled(item)
                    Button { if ok { pick(item) } } label: {
                        Text(L10n.tr(label(item)))
                            .font(CS.inter(11, on ? .semibold : .medium))
                            .foregroundStyle(on ? CS.textPrimaryF : ok ? CS.textSecondaryF : CS.textSecondaryF.opacity(0.35))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .background(on ? Self.blue : .clear, in: RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
                    .disabled(!ok)
                    .accessibilityIdentifier("\(id)_\(item.id)")
                }
            }
            .padding(2)
            .frame(height: 40)
            .background(Self.control, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(CS.borderDefault, lineWidth: 1))
        }
    }

    private func progress(_ title: String, _ value: Double) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(title).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
                Spacer()
                Text(verbatim: "\(Int((value * 100).rounded()))%").font(CS.inter(18, .semibold)).monospacedDigit()
                    .foregroundStyle(CS.textPrimaryF).accessibilityIdentifier("exportPercent")
            }
            GeometryReader { box in
                ZStack(alignment: .leading) {
                    Capsule().fill(Self.control)
                    Capsule().fill(Self.primaryGradient).frame(width: max(6, box.size.width * value))
                }
            }
            .frame(height: 8)
            Text(L10n.tr("Keep CutSell open until it finishes.")).font(CS.inter(12)).foregroundStyle(CS.textMutedF)
            Button { exporter.cancel() } label: {
                Text(L10n.tr("Cancel")).font(CS.inter(14, .medium)).foregroundStyle(CS.textMutedF).frame(maxWidth: .infinity)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("exportCancelRunning")
        }
        .padding(.vertical, 8)
    }

    private func done(_ url: URL) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(L10n.tr("Saved to Photos"), systemImage: "checkmark.circle.fill")
                .font(CS.inter(16, .semibold)).foregroundStyle(Color(hex: 0x19E6FF))
                .accessibilityIdentifier("exportSaved")
            if exporter.usedCopy {
                Text(L10n.tr("Your original wasn't on this iPhone, so this export used the 1080p copy."))
                    .font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
            }
            FinishedExportActionsView(remoteURL: url, projectID: projectID, session: session, alreadySaved: true)
            Button { exporter.reset(); dismiss() } label: {
                Text(L10n.tr("Done")).font(CS.inter(16, .semibold)).foregroundStyle(CS.textPrimaryF)
                    .frame(maxWidth: .infinity).frame(height: 50)
                    .background(Self.primaryGradient, in: RoundedRectangle(cornerRadius: 18))
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("exportDone")
        }
    }

    private func noSpace(_ bytes: Int64) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L10n.tr("Not enough space on your iPhone. Free up \(String(format: "%.1f", Double(bytes) / 1e9)) GB"))
                .font(CS.inter(13, .semibold)).foregroundStyle(Color(hex: 0xFF8A8A))
                .accessibilityIdentifier("exportNoSpace")
            HStack(spacing: 16) {
                Button(L10n.tr("Open Settings")) {
                    if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
                }
                .accessibilityIdentifier("exportOpenSettings")
                Button(L10n.tr("Export on our server (1080p)")) { dismiss(); onServerExport() }
            }
            .font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
        }
    }

    static func clock(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    static func size(_ bytes: Int64) -> String {
        bytes >= 1_000_000_000 ? String(format: "%.1f GB", Double(bytes) / 1e9) : "\(max(1, bytes / 1_000_000)) MB"
    }
}
