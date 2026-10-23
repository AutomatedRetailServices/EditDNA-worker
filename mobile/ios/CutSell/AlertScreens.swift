import SwiftUI
import UIKit

/// Permission screens (Figma 03 Create: Camera 76:72, Microphone 36:236, Gallery).
struct PermissionScreen: View {
    enum Kind {
        case camera, microphone, photos

        var screenTitle: String {
            switch self {
            case .camera: return "Camera"
            case .microphone: return "Microphone"
            case .photos: return "Gallery"
            }
        }

        var heading: String {
            switch self {
            case .camera: return "Allow camera access"
            case .microphone: return "Allow microphone access"
            case .photos: return "Allow gallery access"
            }
        }

        var message: String {
            switch self {
            case .camera: return "CutSell.ai needs camera access so you can record video."
            case .microphone: return "CutSell.ai needs microphone access when you record video or voice-over."
            case .photos: return "Choose videos from your library to create a CutSell edit."
            }
        }

        var buttonTitle: String {
            switch self {
            case .camera: return "Allow camera"
            case .microphone: return "Allow microphone"
            case .photos: return "Allow gallery"
            }
        }
    }

    let kind: Kind
    /// true when iOS already said no: the button then opens Settings.
    var deniedBefore = false
    var onClose: (() -> Void)?
    let onAllow: () -> Void

    /// White Indigo: white screen, the icon in a soft blue circle, Bold title, grey text, blue button.
    var body: some View {
        ZStack {
            CS.wiPage.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 0) {
                WIHeader {
                    if let onClose, deniedBefore {
                        Button(action: onClose) {
                            Image(systemName: "xmark").font(.system(size: 15, weight: .semibold)).foregroundStyle(CS.wiInk)
                                .frame(width: 40, height: 40)
                                .background(CS.wiSurface2, in: Circle())
                                .overlay(Circle().stroke(CS.wiBorder, lineWidth: 1))
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(L10n.tr("Close"))
                    }
                }
                WIScreenTitle(title: kind.screenTitle).padding(.top, 2)

                Spacer()

                VStack(spacing: 0) {
                    icon
                        .foregroundStyle(CS.accent)
                        .frame(width: 72, height: 72)
                        .background(CS.accentSoft, in: Circle())
                    Text(L10n.tr(kind.heading))
                        .font(CS.inter(22, .heavy))
                        .foregroundStyle(CS.wiInk)
                        .multilineTextAlignment(.center)
                        .padding(.top, 18)
                    Text(L10n.tr(kind.message))
                        .font(CS.inter(14, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 8)
                    WIPillButton(title: deniedBefore ? "Open Settings" : "Continue",
                                 action: deniedBefore ? Self.openSettings : onAllow)
                        .padding(.top, 28)
                    Text(L10n.tr("You can change this later in Settings."))
                        .font(CS.inter(12, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .padding(.top, 14)
                }
                .padding(20)
                .frame(maxWidth: .infinity)
                .wiCard(shadow: false)

                Spacer()
                Spacer()
            }
            .padding(.horizontal, 20)
        }
    }

    @ViewBuilder private var icon: some View {
        switch kind {
        case .camera: Image("WIRecord").renderingMode(.template).resizable().frame(width: 32, height: 32)
        case .microphone: Image(systemName: "mic").font(.system(size: 28, weight: .regular))
        case .photos: Image("WIUpload").renderingMode(.template).resizable().frame(width: 30, height: 30)
        }
    }

    static func openSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }
}

/// Full-screen problem screens (Figma 03 Create: 36:247, 76:93, 76:103, 952:42).
struct ProblemScreen: View {
    enum Tone { case error, info }

    let screenTitle: String
    let heading: String
    let message: String
    var tone: Tone = .error
    let primaryTitle: String
    let primaryAction: () -> Void
    var secondaryTitle: String?
    var secondaryAction: (() -> Void)?

    /// White Indigo error style (Figma 1176:2858): icon in a coloured circle, Bold title, grey explanation, blue button.
    var body: some View {
        ZStack {
            CS.wiPage.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 0) {
                WIHeader { EmptyView() }
                WIScreenTitle(title: screenTitle).padding(.top, 2)

                Spacer()

                VStack(spacing: 0) {
                    WIStatusBadge(kind: tone == .error ? .error : .info)
                    Text(L10n.tr(heading))
                        .font(CS.inter(22, .heavy))
                        .foregroundStyle(CS.wiInk)
                        .multilineTextAlignment(.center)
                        .padding(.top, 20)
                    Text(L10n.tr(message))
                        .font(CS.inter(14, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .multilineTextAlignment(.center)
                        .lineSpacing(4)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 8)
                        .padding(.horizontal, 10)
                    VStack(spacing: 10) {
                        WIPillButton(title: primaryTitle, action: primaryAction)
                        if let secondaryTitle, let secondaryAction {
                            WIPillButton(title: secondaryTitle, style: .outline, action: secondaryAction)
                        }
                    }
                    .padding(.top, 32)
                }
                .frame(maxWidth: .infinity)

                Spacer()
                Spacer()
            }
            .padding(.horizontal, 20)
        }
    }
}

/// "Upload limit exceeded" / "This video is too long" (Figma 69:231, 76:82): the chosen video
/// in the back, a card at the bottom.
struct LimitScreen: View {
    let detail: String
    let heading: String
    let message: String
    let onChooseAnother: () -> Void
    let onClose: () -> Void

    /// White Indigo sheet (Figma 1176:2888 "Video too long"): 45 % dim, white sheet with handle, warning mark,
    /// title, grey text, the selected video with its length in red, and the blue button.
    var body: some View {
        ZStack(alignment: .bottom) {
            Color(fixed: 0x0B1020, opacity: 0.45).ignoresSafeArea()
                .onTapGesture(perform: onClose)
                .accessibilityLabel(L10n.tr("Close"))
                .accessibilityAddTraits(.isButton)
            WISheetCard {
                VStack(spacing: 0) {
                    WIStatusBadge(kind: .warning, size: 56).padding(.top, 21)
                    Text(L10n.tr(heading))
                        .font(CS.inter(22, .heavy))
                        .foregroundStyle(CS.wiInk)
                        .multilineTextAlignment(.center)
                        .padding(.top, 12)
                    Text(L10n.tr(message))
                        .font(CS.inter(14, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .multilineTextAlignment(.center)
                        .lineSpacing(4)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 8)
                        .padding(.horizontal, 20)
                    HStack(spacing: 12) {
                        RoundedRectangle(cornerRadius: 8, style: .continuous).fill(CS.wiSurface2).frame(width: 36, height: 48)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(L10n.tr("Selected video")).font(CS.inter(14, .bold)).foregroundStyle(CS.wiInk)
                            Text(L10n.tr(detail)).font(CS.inter(12, .medium)).foregroundStyle(CS.wiError).lineLimit(1)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 11)
                    .frame(height: 64)
                    .wiCard(radius: 18, shadow: false)
                    .padding(.top, 18)
                    WIPillButton(title: "Choose another video", action: onChooseAnother)
                        .padding(.top, 20)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 16)
            }
        }
    }
}

/// Limited photo access card (Figma 35:110 "Gallery access").
struct GalleryAccessCard: View {
    let onSelectMore: () -> Void
    let onKeep: () -> Void

    var body: some View {
        ZStack {
            Color(fixed: 0x0B1020, opacity: 0.45).ignoresSafeArea()
            VStack(spacing: 0) {
                Text(L10n.tr("Gallery  access"))
                    .font(CS.inter(22, .heavy))
                    .foregroundStyle(CS.wiInk)
                Text(L10n.tr("Choose videos from your library to create a CutSell edit."))
                    .font(CS.inter(14, .medium))
                    .foregroundStyle(CS.wiSecondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 10)
                WIPillButton(title: "Select More Photos…", action: onSelectMore)
                    .padding(.top, 24)
                WIPillButton(title: "Keep Current Selection", style: .outline, action: onKeep)
                    .padding(.top, 10)
            }
            .padding(.horizontal, 26)
            .padding(.vertical, 30)
            .frame(width: 322)
            .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        }
    }
}

/// Problems the Create flow can show, in the words of the Figma screens.
enum CreateProblem: Identifiable, Equatable {
    case unsupportedMedia
    case supportedFormats
    case tooLarge(detail: String, sizeText: String)
    case tooLong(detail: String)
    case clipLimit
    case couldNotAdd

    var id: String {
        switch self {
        case .unsupportedMedia: return "unsupported"
        case .supportedFormats: return "formats"
        case .tooLarge: return "large"
        case .tooLong: return "long"
        case .clipLimit: return "limit"
        case .couldNotAdd: return "couldNotAdd"
        }
    }
}

/// The limits shown on Figma's "Supported video files" screen (952:42).
enum CreateLimits {
    /// The 10-minute and 1 GB limits from Figma are not applied yet: the app accepted longer and
    /// bigger videos before (it shrinks big ones itself), so turning them on is the Product
    /// Owner's decision. Formats and the 10-clip limit always apply.
    static let sizeAndLengthLimitsOn = false
    static let maxDurationSeconds: Double = 10 * 60
    static let maxFileBytes: Int64 = 1_000_000_000
    static let maxClips = 10
    static let supportedExtensions: Set<String> = ["mp4", "mov", "m4v"]

    static func sizeText(_ bytes: Int64) -> String {
        if bytes >= 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1_000_000_000) }
        return "\(bytes / 1_000_000) MB"
    }

    static func durationText(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

/// Shows the right Figma screen for a problem.
struct CreateProblemView: View {
    let problem: CreateProblem
    /// Back to the gallery / choose another video.
    let onBack: () -> Void
    var onShowFormats: (() -> Void)?
    var onRemoveClip: (() -> Void)?
    var onRetry: (() -> Void)?

    var body: some View {
        switch problem {
        case .unsupportedMedia:
            ProblemScreen(
                screenTitle: "Can’t add this video",
                heading: "Unsupported media",
                message: "This video format isn’t supported yet. Choose another video or review the formats CutSell accepts.",
                primaryTitle: "Back to library", primaryAction: onBack,
                secondaryTitle: "View supported formats", secondaryAction: onShowFormats
            )
        case .supportedFormats:
            ProblemScreen(
                screenTitle: "Supported video files",
                heading: "Formats and limits",
                message: "MP4, MOV and M4V\nH.264, HEVC and ProRes\nUp to 10 minutes and 1 GB",
                tone: .info,
                primaryTitle: "Choose a video", primaryAction: onBack,
                secondaryTitle: "Back", secondaryAction: onBack
            )
        case .tooLarge(let detail, let sizeText):
            LimitScreen(
                detail: detail,
                heading: "Upload limit exceeded",
                message: "This file is \(sizeText). CutSell accepts files up to 1 GB. Your original file is unchanged.",
                onChooseAnother: onBack, onClose: onBack
            )
        case .tooLong(let detail):
            LimitScreen(
                detail: detail,
                heading: "This video is too long",
                message: "Choose a video up to 10 minutes. Your original file remains unchanged.",
                onChooseAnother: onBack, onClose: onBack
            )
        case .clipLimit:
            ProblemScreen(
                screenTitle: "Clip limit reached",
                heading: "10 clips maximum",
                message: "Process these clips or remove one before adding another.",
                primaryTitle: "Review clips", primaryAction: onBack,
                secondaryTitle: "Remove a clip", secondaryAction: onRemoveClip ?? onBack
            )
        case .couldNotAdd:
            ProblemScreen(
                screenTitle: "Couldn’t add this clip",
                heading: "Something went wrong",
                message: "Try recording the clip again. Your saved clips are still here.",
                primaryTitle: "Back to clips", primaryAction: onBack,
                secondaryTitle: "Try again", secondaryAction: onRetry ?? onBack
            )
        }
    }
}
