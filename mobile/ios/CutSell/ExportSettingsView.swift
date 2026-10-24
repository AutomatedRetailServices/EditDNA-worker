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
    private static let primaryGradient = CS.accentGradient   // White Indigo: #1769FF → #4A5BFF

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
        if case .failed = exporter.phase {
            ExportFailedView(exporter: exporter,
                             onBack: { exporter.reset(); dismiss() },
                             onTryAgain: { exporter.start(model: model, settings: settings) },
                             onChangeSettings: { exporter.reset() },
                             onServerExport: { dismiss(); onServerExport() })
        } else {
            settingsScreen
        }
    }

    private var settingsScreen: some View {
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
        .accessibilityElement(children: .contain)
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
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(Color(fixed: 0xFFFFFF), lineWidth: 3))
        .shadow(color: Color(fixed: 0x1769FF, opacity: 0.28), radius: 16, y: 12)       // White Indigo video frame
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
                .font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
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
            // White Indigo: the same turning brand ring as Single · Processing (Export · Rendering).
            BrandRing(size: 72).frame(maxWidth: .infinity)
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
                .font(CS.inter(16, .semibold)).foregroundStyle(CS.wiSuccess)
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

/// White Indigo · ERROR · Export/Rendering · Failed (Figma 1233:3199): the still ring with a red "!", the % it reached,
/// the steps (done ✓, the one that stopped !, the rest —), "Nothing was lost", blue "Try Again", outlined
/// "Change export settings".
struct ExportFailedView: View {
    @ObservedObject var exporter: DeviceExport
    let onBack: () -> Void
    let onTryAgain: () -> Void
    let onChangeSettings: () -> Void
    let onServerExport: () -> Void

    private static let red = Color(fixed: 0xE5484D)
    private static let steps = ["Preparing video", "Rendering frames", "Encoding video", "Saving to Photos"]

    /// The Figma step that stopped: preparing → 0, encoding → 2, saving → 3.
    private var failedIndex: Int { exporter.lastStep == 0 ? 0 : exporter.lastStep == 1 ? 2 : 3 }

    var body: some View {
        ZStack {
            CS.wiPage.ignoresSafeArea()
            ScrollView {
                VStack(spacing: 0) {
                    WIBackBar(title: "Export failed", onBack: onBack)
                    BrandRing(animating: false, size: 110)
                        .overlay(alignment: .bottomTrailing) {
                            Text(verbatim: "!").font(CS.inter(13, .heavy)).foregroundStyle(Color(fixed: 0xFFFFFF))
                                .frame(width: 22, height: 22)
                                .background(Self.red, in: Circle())
                                .overlay(Circle().stroke(Color(fixed: 0xFFFFFF), lineWidth: 2))
                                .padding(.trailing, 24).padding(.bottom, 24)
                        }
                        .padding(.top, 6)
                    Text(verbatim: "\(Int((exporter.lastProgress * 100).rounded()))%")
                        .font(CS.inter(40, .heavy)).foregroundStyle(CS.wiInk)
                        .accessibilityIdentifier("exportFailedPercent")
                    Text(L10n.tr("Export stopped")).font(CS.inter(15, .medium)).foregroundStyle(CS.wiSecondary)
                    Text(L10n.tr("Your video was not saved")).font(CS.inter(12, .medium)).foregroundStyle(CS.accent)
                        .padding(.top, 4)
                    VStack(spacing: 4) {
                        ForEach(Array(Self.steps.enumerated()), id: \.offset) { index, title in step(index, title) }
                    }
                    .padding(12)
                    .wiCard(radius: 18, shadow: false)
                    .padding(.top, 22)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(L10n.tr("Nothing was lost")).font(CS.inter(14, .semibold)).foregroundStyle(CS.wiInk)
                        Text(L10n.tr("Your cut and export settings are still saved.")).font(CS.inter(12, .medium))
                            .foregroundStyle(CS.wiSecondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(14)
                    .wiCard(radius: 16, shadow: false)
                    .padding(.top, 14)
                    Button(action: onTryAgain) {
                        Text(L10n.tr("Try Again")).font(CS.inter(16, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                            .frame(maxWidth: .infinity).frame(height: 52)
                            .background(CS.accent, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("exportTryAgain")
                        .padding(.top, 20)
                    Button(action: onChangeSettings) {
                        Text(L10n.tr("Change export settings")).font(CS.inter(16, .semibold)).foregroundStyle(CS.wiInk)
                            .frame(maxWidth: .infinity).frame(height: 52)
                            .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(CS.accent, lineWidth: 1.5))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("exportChangeSettings")
                    .padding(.top, 12)
                    Button(L10n.tr("Export on our server (1080p)"), action: onServerExport)
                        .font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
                        .accessibilityIdentifier("exportServerFallback")
                        .padding(.top, 14)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
        }
        .wiStatusBarCover()
        .accessibilityIdentifier("exportFailed")
    }

    private func step(_ index: Int, _ title: String) -> some View {
        let failed = index == failedIndex
        let done = index < failedIndex
        return HStack(spacing: 10) {
            Group {
                if done { Image(systemName: "checkmark").font(.system(size: 13, weight: .bold)).foregroundStyle(CS.wiReady) }
                else if failed { Text(verbatim: "!").font(CS.inter(15, .heavy)).foregroundStyle(Self.red) }
                else { Circle().stroke(CS.wiSecondary, lineWidth: 1.5).frame(width: 11, height: 11) }
            }
            .frame(width: 16)
            Text(L10n.tr(title)).font(CS.inter(14, failed ? .semibold : .medium))
                .foregroundStyle(failed ? CS.wiInk : CS.wiSecondary)
            Spacer()
            if failed { Text(L10n.tr("Failed")).font(CS.inter(12, .medium)).foregroundStyle(CS.accent) }
            else if !done { Text(verbatim: "—").foregroundStyle(CS.wiSecondary) }
        }
        .padding(.horizontal, 10)
        .frame(height: 44)
        .background(failed ? Color(fixed: 0xFDEBEC) : .clear, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
