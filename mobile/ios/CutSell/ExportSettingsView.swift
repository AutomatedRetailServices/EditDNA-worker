import SwiftUI
import UserNotifications
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
    /// Design screenshots only: the destination already chosen.
    var previewDestination: Destination? = nil
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL
    /// WHERE TO? (Figma 1259:3101): Photos only, or Photos then the Plan's scheduling sheet.
    enum Destination { case photos, plan }
    @State private var destination: Destination = .photos
    @State private var sharing: URL?

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
            Color(fixed: 0xE9ECF1).ignoresSafeArea()
            preview
            header
            VStack(spacing: 0) {
                Spacer(minLength: 0)
                sheet
            }
            .ignoresSafeArea(edges: .bottom)
        }
        .task { await exporter.detect(model: model) }
        .onAppear { if let previewDestination { destination = previewDestination } }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("exportSettings")
    }

    // MARK: Top (Figma: back · file name · more, the cut's picture)

    private var header: some View {
        // Figma 1259:3101: only the round light-grey back button (#F3F5F9, black arrow).
        HStack {
            Button { if !exporter.running { exporter.reset(); dismiss() } } label: { WIHeaderIcon(image: "WIBack") }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.tr("Back"))
                .accessibilityIdentifier("exportBack")
            Spacer()
        }
        .frame(height: 44)
        .padding(.horizontal, 20)
        .padding(.top, 4)
    }

    private var preview: some View {
        Group {
            if let still = ThumbnailStore.shared.image(for: projectID) {
                Image(uiImage: still).resizable().scaledToFill()
            } else {
                Color(fixed: 0xD5DAE3)
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
            Capsule().fill(Color(fixed: 0xD5D9E2)).frame(width: 40, height: 5).frame(maxWidth: .infinity)
            if case .done = exporter.phase, destination == .plan {} else {
            VStack(alignment: .leading, spacing: 4) {
                Text(L10n.tr("Export")).font(CS.inter(28, .bold)).foregroundStyle(CS.wiInk)
                Text(verbatim: "9:16 · \(L10n.tr("Final cut")) \(Self.clock(cutSeconds))")
                    .font(CS.inter(14)).foregroundStyle(CS.wiSecondary)
            }
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
                if destination == .plan {
                    ExportScheduleContent(projectID: projectID, seconds: cutSeconds, onDone: { exporter.reset(); dismiss() })
                } else {
                    done(url)
                }
            }
        }
        .padding(.horizontal, 20).padding(.top, 10).padding(.bottom, 34)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CS.wiPage, in: UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28))
        .environment(\.colorScheme, .light)
        .sheet(item: Binding(get: { sharing.map { ShareURL(url: $0) } }, set: { sharing = $0?.url })) { item in
            ShareSheet(items: [item.url]).ignoresSafeArea()
        }
    }

    @ViewBuilder
    private var choices: some View {
        detected
        segment(L10n.tr("Resolution"), ExportResolution.allCases, selected: settings.resolution, id: "exportResolution",
                label: { $0.label }, enabled: { r in (r.shortSide ?? 0) <= maxShort }) { settings.resolution = $0 }
        Text(L10n.tr(resolutionHint)).font(CS.inter(11)).foregroundStyle(CS.wiSecondary).padding(.top, -7)
            .accessibilityIdentifier("exportResolutionHint")
        segment(L10n.tr("Frame rate"), ExportFrameRate.allCases, selected: settings.frameRate, id: "exportFrameRate",
                label: { $0.label }, enabled: { ($0.value ?? 0) <= max(30, info.fps) }) { settings.frameRate = $0 }
        segment(L10n.tr("Quality"), ExportQuality.allCases, selected: settings.quality, id: "exportQualityChoice",
                label: { $0.label }, enabled: { _ in true }) { settings.quality = $0 }
        HStack {
            Text(L10n.tr("Estimated file size")).font(CS.inter(13)).foregroundStyle(CS.wiSecondary)
            Spacer()
            Text(verbatim: "≈ \(Self.size(settings.estimatedBytes(for: info, seconds: cutSeconds)))")
                .font(CS.inter(17, .bold)).foregroundStyle(CS.wiInk)
                .accessibilityIdentifier("exportEstimate")
        }
        .padding(.horizontal, 14).frame(height: 44)
        .background(CS.wiSurface2, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(CS.wiBorder, lineWidth: 1))
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
        Text(L10n.tr("WHERE TO?")).font(CS.inter(12, .bold)).foregroundStyle(CS.accent).padding(.top, 2)
        whereTo(icon: "square.and.arrow.down", title: "Save to Photos", detail: "Export and keep it on this iPhone",
                blue: false, id: "exportToPhotos") { destination = .photos; exporter.start(model: model, settings: settings) }
        whereTo(icon: "calendar", title: "Add to Plan to post", detail: "TikTok · Instagram · Facebook",
                blue: true, id: "exportToPlan") { destination = .plan; exporter.start(model: model, settings: settings) }
        Button { dismiss() } label: {
            Text(L10n.tr("Cancel")).font(CS.inter(15, .medium)).foregroundStyle(CS.wiSecondary).frame(maxWidth: .infinity)
                .padding(.top, 4)
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
            Text(verbatim: "✦ " + (is4K ? L10n.tr("4K export selected")
                                         : info.originals ? L10n.tr("Original quality detected") : L10n.tr("1080p copy (original not found)")))
                .font(CS.inter(15, .bold)).foregroundStyle(CS.accent)
            Text(verbatim: "\(info.width) × \(info.height) · \(info.fps) fps · \(info.codec)")
                .font(CS.inter(12)).foregroundStyle(CS.wiSecondary)
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(fixed: 0xEEF3FF), in: RoundedRectangle(cornerRadius: 14))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: "\(info.width)x\(info.height) \(info.fps)fps \(info.codec) originals:\(info.originals)"))
        .accessibilityIdentifier("exportDetected")
    }

    private func segment<T: Identifiable & Equatable>(_ title: String, _ items: [T], selected: T, id: String,
                                                       label: @escaping (T) -> String, enabled: @escaping (T) -> Bool,
                                                       pick: @escaping (T) -> Void) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(CS.inter(14, .medium)).foregroundStyle(CS.wiInk)
            HStack(spacing: 2) {
                ForEach(items) { item in
                    let on = item == selected
                    let ok = enabled(item)
                    Button { if ok { pick(item) } } label: {
                        Text(L10n.tr(label(item)))
                            .font(CS.inter(14, on ? .semibold : .medium))
                            // Figma 1259:3101: chosen = #1769FF with WHITE text; not available = light grey, still legible.
                            .foregroundStyle(on ? Color(fixed: 0xFFFFFF) : ok ? CS.wiInk : Color(fixed: 0xB5BBC7))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .background(on ? CS.accent : .clear, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .disabled(!ok)
                    .accessibilityIdentifier("\(id)_\(item.id)")
                }
            }
            .padding(3)
            .frame(height: 44)
            .background(CS.wiSurface2, in: Capsule())
            .overlay(Capsule().stroke(CS.wiBorder, lineWidth: 1))
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

    /// "Save to Photos" finished (White Indigo, like Figma 1241:3561): check, "Saved to Photos", the file, Share and Done.
    private func done(_ url: URL) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "checkmark").font(.system(size: 40, weight: .bold)).foregroundStyle(CS.wiSuccess)
                .frame(width: 96, height: 96)
                .background(CS.wiSuccessBg, in: Circle())
                .overlay(Circle().stroke(CS.wiSuccess, lineWidth: 2))
            Text(L10n.tr("Saved to Photos")).font(CS.inter(26, .bold)).foregroundStyle(CS.wiInk)
                .accessibilityIdentifier("exportSaved")
            Text(verbatim: "\(info.width) × \(info.height) · \(Self.clock(cutSeconds))").font(CS.inter(13)).foregroundStyle(CS.wiSecondary)
            if exporter.usedCopy {
                Text(L10n.tr("Your original wasn't on this iPhone, so this export used the 1080p copy."))
                    .font(CS.inter(12)).foregroundStyle(CS.wiSecondary).multilineTextAlignment(.center)
            }
            Button { sharing = url } label: {
                Text(L10n.tr("Share")).font(CS.inter(16, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(maxWidth: .infinity).frame(height: 52).background(CS.accent, in: Capsule())
            }
            .buttonStyle(.plain)
            .padding(.top, 8)
            .accessibilityIdentifier("exportShare")
            Button { exporter.reset(); dismiss() } label: {
                Text(L10n.tr("Done")).font(CS.inter(16, .semibold)).foregroundStyle(CS.wiInk)
                    .frame(maxWidth: .infinity).frame(height: 52)
                    .background(CS.wiPage, in: Capsule())
                    .overlay(Capsule().stroke(CS.accent, lineWidth: 1.5))
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("exportDone")
        }
        .frame(maxWidth: .infinity)
    }

    private func whereTo(icon: String, title: String, detail: String, blue: Bool, id: String,
                         action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: icon).font(.system(size: 18, weight: .medium))
                    .foregroundStyle(blue ? Color(fixed: 0xFFFFFF) : CS.accent)
                    .frame(width: 40, height: 40)
                    .background(blue ? Color(fixed: 0xFFFFFF, opacity: 0.18) : Color(fixed: 0xEEF3FF),
                                in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    Text(L10n.tr(title)).font(CS.inter(17, .bold))
                    Text(L10n.tr(detail)).font(CS.inter(12, .medium)).opacity(blue ? 0.9 : 1)
                        .foregroundStyle(blue ? Color(fixed: 0xFFFFFF) : CS.wiSecondary)
                }
                .foregroundStyle(blue ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                Spacer()
                Image(systemName: "chevron.right").font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(blue ? Color(fixed: 0xFFFFFF) : CS.wiSecondary)
            }
            .padding(.horizontal, 14).frame(height: 72)
            .background(blue ? CS.accent : CS.wiPage, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(blue ? .clear : CS.wiBorder, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
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

struct ShareURL: Identifiable { let url: URL; var id: String { url.absoluteString } }

/// Figma 1259:3199 "Programar publicación": after the export, where and when to post it. The video goes to This week on
/// that day and time with its suggested caption; at that time a reminder opens it ready to share. "Decide later" keeps it
/// ready to plan (first in Plan my week).
struct ExportScheduleContent: View {
    let projectID: String
    let seconds: Double
    let onDone: () -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var store = PlanWeekStore.shared
    @ObservedObject private var catalog = CatalogStore.shared
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @State private var network: PostNetwork = .tiktok
    @State private var day = PlanGoalView.tomorrow()
    @State private var time = "19:00"
    @State private var pickingTime = false
    @State private var saving = false
    @State private var errorMessage: String?

    static let suggested = ["09:00", "13:00", "19:00", "21:00"]

    private var pick: PlanPick? {
        store.plannable(projects: appState.projects, catalog: catalog).first { $0.projectID == projectID }
    }
    private var days: [String] {
        let calendar = Calendar(identifier: .gregorian)
        return (0..<5).compactMap { calendar.date(byAdding: .day, value: $0, to: Date()) }.map { PlanDay.iso.string(from: $0) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Color(fixed: 0xD5DAE3).frame(width: 40, height: 52)
                    .overlay { if let image = thumbnails.image(for: projectID) { Image(uiImage: image).resizable().scaledToFill() } }
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                VStack(alignment: .leading, spacing: 3) {
                    Text(L10n.tr("Ready to post")).font(CS.inter(17, .bold)).foregroundStyle(CS.wiInk)
                    Text([pick?.brand, pick?.product, seconds > 0 ? ExportSettingsView.clock(seconds) : nil].compactMap { $0 }.joined(separator: " · "))
                        .font(CS.inter(12, .medium)).foregroundStyle(CS.wiSuccess).lineLimit(1)
                }
                Spacer()
            }
            .padding(12)
            .background(CS.wiSuccessBg, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .accessibilityIdentifier("scheduleReady")

            Text(L10n.tr("Post on")).font(CS.inter(14, .medium)).foregroundStyle(CS.wiInk)
            HStack(spacing: 8) {
                ForEach(PostNetwork.allCases) { n in
                    choice(n.rawValue, on: network == n, id: "scheduleNetwork_\(n.rawValue)") { network = n }
                }
            }
            Text(L10n.tr("Day")).font(CS.inter(14, .medium)).foregroundStyle(CS.wiInk)
            HStack(spacing: 6) {
                ForEach(days, id: \.self) { d in
                    let on = d == day
                    Button { day = d } label: {
                        VStack(spacing: 2) {
                            Text(L10n.tr(d == days.first ? "Today" : PlanDay.weekdayName(d))).font(CS.inter(11, .medium))
                            Text(verbatim: d == days.first ? PlanDay.short(d) : String(Int(d.suffix(2)) ?? 0))
                                .font(CS.inter(d == days.first ? 12 : 17, .bold))
                        }
                        .foregroundStyle(on ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                        .frame(maxWidth: .infinity).frame(height: 56)
                        .background(on ? CS.accent : CS.wiSurface2, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(on ? .clear : CS.wiBorder, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("scheduleDay_\(d)")
                }
            }
            Text(L10n.tr("Time")).font(CS.inter(14, .medium)).foregroundStyle(CS.wiInk)
            HStack(spacing: 6) {
                ForEach(Self.suggested, id: \.self) { t in
                    choice(PlanDay.clock(t), on: t == time, id: "scheduleTime_\(t)", small: true) { time = t }
                }
            }
            Button(Self.suggested.contains(time) ? L10n.tr("Other time ›") : "\(PlanDay.clock(time)) ›") { pickingTime = true }
                .font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
                .accessibilityIdentifier("scheduleOtherTime")
            HStack(spacing: 10) {
                Image(systemName: "bell").font(.system(size: 15, weight: .medium)).foregroundStyle(CS.wiInk)
                Text(L10n.tr("We'll remind you at that time to post it in one tap.")).font(CS.inter(12)).foregroundStyle(CS.wiSecondary)
            }
            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(CS.wiSurface2, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
            if let errorMessage { Text(L10n.tr(errorMessage)).font(CS.inter(12)).foregroundStyle(CS.wiError) }
            Button { Task { await schedule() } } label: {
                Group {
                    if saving { BrandRing(size: 22) } else {
                        Text(L10n.tr("Schedule · \(PlanDay.short(day)) · \(PlanDay.clock(time))")).font(CS.inter(17, .bold))
                    }
                }
                .foregroundStyle(Color(fixed: 0xFFFFFF))
                .frame(maxWidth: .infinity).frame(height: 54).background(CS.accent, in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(saving)
            .accessibilityIdentifier("scheduleSave")
            Button { PlanLater.add(projectID); onDone() } label: {
                Text(L10n.tr("Decide later")).font(CS.inter(15, .medium)).foregroundStyle(CS.wiSecondary).frame(maxWidth: .infinity)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("scheduleLater")
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("scheduleSheet")
        .task { await catalog.refresh(session: appState.session, projects: appState.projects) }
        .sheet(isPresented: $pickingTime) {
            TimePickerSheet(time: time, title: "Other time") { time = $0 }
                .presentationDetents([.height(330)])
                .presentationBackground(CS.wiPage)
        }
    }

    private func choice(_ title: String, on: Bool, id: String, small: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(L10n.tr(title)).font(CS.inter(small ? 13 : 15, .semibold)).lineLimit(1).minimumScaleFactor(0.8)
                .foregroundStyle(on ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                .frame(maxWidth: .infinity).frame(height: small ? 40 : 44)
                .background(on ? CS.accent : CS.wiSurface2, in: Capsule())
                .overlay(Capsule().stroke(on ? .clear : CS.wiBorder, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }

    private func schedule() async {
        saving = true
        defer { saving = false }
        do {
            PostNetwork.set(network, for: projectID)
            try await store.add(projectID: projectID, date: day, time: time, session: appState.session)
            PlanLater.remove(projectID)
            if store.state?.prefs.remind ?? true {
                _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])
            }
            PostReminders.shared.schedule(store.state)
            AppRouter.shared.pending = .thisWeek          // This week shows it on that day, at that time
            onDone()
        } catch {
            errorMessage = "Couldn’t schedule it. Check your connection and try again."
        }
    }
}
