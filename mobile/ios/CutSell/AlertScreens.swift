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

    var body: some View {
        ZStack {
            ScreenBackground()
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    BrandLogo()
                    Spacer()
                    if let onClose, deniedBefore {
                        Button(action: onClose) {
                            Image(systemName: "xmark")
                                .font(.system(size: 17, weight: .semibold))
                                .foregroundStyle(.white)
                                .frame(width: 44, height: 44)
                        }
                        .accessibilityLabel("Close")
                    }
                }
                .padding(.top, 8)
                Text(L10n.tr(kind.screenTitle))
                    .font(CS.inter(28, .bold))
                    .foregroundStyle(.white)
                    .padding(.top, 14)

                Spacer()

                VStack(spacing: 0) {
                    icon
                        .frame(height: 64)
                        .padding(.top, 34)
                    Text(L10n.tr(kind.heading))
                        .font(CS.inter(23, .bold))
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                        .padding(.top, 18)
                    Text(L10n.tr(kind.message))
                        .font(CS.inter(15))
                        .foregroundStyle(CS.textSecondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 12)
                    PrimaryButton(
                        title: deniedBefore ? "Open Settings" : "Continue",
                        height: 54,
                        cornerRadius: 16,
                        fontSize: 16,
                        gradient: CS.alertButtonGradient,
                        action: deniedBefore ? Self.openSettings : onAllow
                    )
                    .padding(.top, 44)
                    .padding(.horizontal, 6)
                    Text("You can change this later in Settings.")
                        .font(CS.inter(13))
                        .foregroundStyle(CS.textMuted)
                        .padding(.top, 22)
                        .padding(.bottom, 18)
                }
                .padding(.horizontal, 22)
                .background(Color(hex: 0x0B0F1C), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .stroke(LinearGradient(colors: [Color(hex: 0x7C3AED), Color(hex: 0xF01ADB)], startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1.5)
                )
                .padding(.horizontal, 20)

                Spacer()
                Spacer()
            }
            .padding(.horizontal, 16)
        }
    }

    @ViewBuilder private var icon: some View {
        switch kind {
        case .camera: Image("TabCreate").resizable().scaledToFit().frame(width: 64)
        case .microphone: Text("🎙️").font(.system(size: 50))
        case .photos: Image("IconUpload").resizable().scaledToFit().frame(width: 54)
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

    var body: some View {
        ZStack {
            ScreenBackground()
            VStack(alignment: .leading, spacing: 0) {
                BrandLogo()
                    .padding(.top, 8)
                Text(L10n.tr(screenTitle))
                    .font(CS.inter(28, .bold))
                    .foregroundStyle(.white)
                    .padding(.top, 14)

                Spacer()

                VStack(spacing: 0) {
                    Text("!")
                        .font(CS.inter(54, .bold))
                        .foregroundStyle(tone == .error ? CS.alert : CS.cyan)
                    Text(L10n.tr(heading))
                        .font(CS.inter(25, .bold))
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                        .padding(.top, 22)
                    Text(L10n.tr(message))
                        .font(CS.inter(15))
                        .foregroundStyle(CS.textSecondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 12)
                        .padding(.horizontal, 20)
                    VStack(spacing: 12) {
                        PrimaryButton(
                            title: primaryTitle, height: 56, cornerRadius: 16, fontSize: 16,
                            gradient: CS.alertButtonGradient, action: primaryAction
                        )
                        if let secondaryTitle, let secondaryAction {
                            SecondaryButton(title: secondaryTitle, height: 52, cornerRadius: 16, fontSize: 15, action: secondaryAction)
                        }
                    }
                    .frame(width: 270)
                    .padding(.top, 76)
                }
                .frame(maxWidth: .infinity)

                Spacer()
                Spacer()
            }
            .padding(.horizontal, 16)
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

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(hex: 0x0A1230), Color(hex: 0x1A0E2E)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            VStack(spacing: 0) {
                HStack {
                    Button(action: onClose) {
                        Image(systemName: "xmark")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundStyle(.white)
                            .frame(width: 44, height: 44)
                    }
                    .accessibilityLabel("Close")
                    Spacer()
                    Text(L10n.tr(detail))
                        .font(CS.inter(14, .medium))
                        .foregroundStyle(.white)
                }
                .padding(.horizontal, 12)
                Spacer()
                Text("Selected video")
                    .font(CS.inter(23, .semibold))
                    .foregroundStyle(Color(hex: 0x8FB0E8))
                Spacer()
                Spacer()
                VStack(alignment: .leading, spacing: 0) {
                    Text(L10n.tr(heading))
                        .font(CS.inter(26, .bold))
                        .foregroundStyle(.white)
                    Text(L10n.tr(message))
                        .font(CS.inter(15))
                        .foregroundStyle(CS.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 12)
                    PrimaryButton(
                        title: "Choose another video", height: 54, cornerRadius: 16, fontSize: 16,
                        gradient: CS.alertButtonGradient, textColor: Color(hex: 0x0B1A40), action: onChooseAnother
                    )
                    .padding(.top, 16)
                    Spacer(minLength: 0)
                }
                .padding(22)
                .frame(maxWidth: .infinity, alignment: .leading)
                .frame(height: 282)
                .background(Color(hex: 0x070C1D, opacity: 0.96), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Color(hex: 0x1A6BFF), lineWidth: 1))
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
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
            Color.black.opacity(0.55).ignoresSafeArea()
            VStack(spacing: 0) {
                Text("Gallery  access")
                    .font(CS.inter(25, .bold))
                    .foregroundStyle(.white)
                Text("Choose videos from your library to create a CutSell edit.")
                    .font(CS.inter(16))
                    .foregroundStyle(CS.textSecondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 14)
                PrimaryButton(
                    title: "Select More Photos…", height: 54, cornerRadius: 18, fontSize: 17,
                    gradient: CS.alertButtonGradient, action: onSelectMore
                )
                .padding(.top, 30)
                SecondaryButton(title: "Keep Current Selection", height: 54, cornerRadius: 18, fontSize: 17, action: onKeep)
                    .padding(.top, 10)
            }
            .padding(.horizontal, 26)
            .padding(.vertical, 30)
            .frame(width: 322)
            .background(Color(hex: 0x131826), in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 28, style: .continuous).stroke(Color(hex: 0x33BFFF), lineWidth: 1.5))
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
