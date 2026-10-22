import SwiftUI
import UIKit

/// Before the first cut is processed, CutSell says which AI services receive the video and asks
/// for "Agree". Without it nothing is uploaded or processed (App Review 5.1.2).
@MainActor
final class AIConsentCenter: ObservableObject {
    static let shared = AIConsentCenter()
    static let key = "cutsell.aiConsent.v1"

    @Published var asking = false
    private var waiting: [CheckedContinuation<Bool, Never>] = []
    private weak var presented: UIViewController?

    var agreed: Bool { UserDefaults.standard.bool(forKey: Self.key) }

    /// true when the creator agreed (now or before); false if they closed the notice.
    ///
    /// Build 19: the notice is presented by UIKit on top of WHATEVER is on screen. It used to be a
    /// SwiftUI `.sheet` on the app's root, and the upload screen (a full-screen cover) hid it: the
    /// first cut waited for an "Agree" that never appeared, at "Uploading 0 %", without ever
    /// calling the server (builds 16-18 on a real iPhone).
    func ensure() async -> Bool {
        if agreed { return true }
        CutSellDiagnostics.log("consent_asked", [:])
        return await withCheckedContinuation { continuation in
            waiting.append(continuation)
            guard !asking else { return }
            asking = true
            present()
        }
    }

    /// Presents the notice and makes sure it is really on screen: while iOS is still closing the
    /// gallery (or any sheet), a presentation is silently refused, so it tries again every 0.5 s.
    private func present(attempt: Int = 1) {
        guard asking else { return }
        if let presented, presented.viewIfLoaded?.window != nil { return }
        if let top = Self.topViewController(), top.transitionCoordinator == nil, !top.isBeingPresented, !top.isBeingDismissed {
            let host = presented ?? {
                let host = UIHostingController(rootView: AIConsentSheet())
                host.modalPresentationStyle = .pageSheet
                host.isModalInPresentation = true
                return host
            }()
            presented = host
            if host.presentingViewController == nil { top.present(host, animated: true) }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in
            guard let self, self.asking else { return }
            if let shown = self.presented, shown.viewIfLoaded?.window != nil {
                CutSellDiagnostics.log("consent_shown", ["attempt": String(attempt)])
                return
            }
            if attempt % 10 == 0 { CutSellDiagnostics.log("consent_not_shown_yet", ["attempt": String(attempt)]) }
            self.present(attempt: attempt + 1)
        }
    }

    func answer(_ agree: Bool) {
        if agree { UserDefaults.standard.set(true, forKey: Self.key) }
        CutSellDiagnostics.log(agree ? "consent_accepted" : "consent_declined", [:])
        asking = false
        presented?.dismiss(animated: true)
        presented = nil
        let all = waiting
        waiting = []
        for continuation in all { continuation.resume(returning: agree) }
    }

    /// The view controller on top of everything (sheets and full-screen covers included).
    static func topViewController() -> UIViewController? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let window = scenes.flatMap(\.windows).first { $0.isKeyWindow } ?? scenes.first?.windows.first
        var top = window?.rootViewController
        while let next = top?.presentedViewController, !next.isBeingDismissed { top = next }
        return top
    }
}

enum AIConsentError: LocalizedError {
    case declined
    var errorDescription: String? { "To make a cut, agree to the AI processing notice." }
}

/// The notice itself (shown once, before the first cut).
struct AIConsentSheet: View {
    @ObservedObject var center = AIConsentCenter.shared
    @State private var legal: LegalDocument?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(L10n.tr("Before your first cut")).font(CS.inter(22, .bold)).foregroundStyle(Color.ink)
            Text(L10n.tr("To make your cut, CutSell uses these two AI services:"))
                .font(CS.inter(15)).foregroundStyle(CS.textSecondaryF)
            VStack(alignment: .leading, spacing: 10) {
                row("Deepgram", "turns the speech in your video's audio into text.")
                row("Anthropic (Claude)", "reads that text to pick your best takes. It does not receive the video or the audio.")
            }
            Text(L10n.tr("No other AI service receives your content. Cloudflare and Amazon Web Services store your videos, and Modal renders your cut."))
                .font(CS.inter(13)).foregroundStyle(CS.textSecondaryF).fixedSize(horizontal: false, vertical: true)
            Text(L10n.tr("They receive it only so CutSell can make your cut. CutSell never sells your content. You can delete your cuts or your account at any time."))
                .font(CS.inter(13)).foregroundStyle(CS.textSecondaryF).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 16) {
                Button(L10n.tr("Privacy Policy")) { legal = .privacy }.font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                Button(L10n.tr("Terms of Use")) { legal = .terms }.font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
            }
            Spacer(minLength: 0)
            PrimaryButton(title: "Agree", height: 52, cornerRadius: 16, fontSize: 16) { center.answer(true) }
                .accessibilityIdentifier("aiConsentAgree")
            Button(L10n.tr("Not now")) { center.answer(false) }
                .font(CS.inter(14, .semibold)).foregroundStyle(CS.textMutedF)
                .frame(maxWidth: .infinity)
                .accessibilityIdentifier("aiConsentNotNow")
        }
        .padding(24)
        .background(Color(hex: 0x0A1326).ignoresSafeArea())
        .presentationDetents([.large])
        .interactiveDismissDisabled()
        .appColorScheme()
        .sheet(item: $legal) { LegalView(document: $0) }
        .accessibilityIdentifier("aiConsentSheet")
    }

    private func row(_ name: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Circle().fill(CS.cyan).frame(width: 6, height: 6).padding(.top, 7)
            (Text(L10n.tr(name)).bold().foregroundColor(Color.ink) + Text(L10n.tr(" — \(text)")).foregroundColor(CS.textSecondaryF))
                .font(CS.inter(14)).fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Privacy Policy and Terms of Use inside the app

enum LegalDocument: String, Identifiable {
    case privacy = "privacy_policy"
    case terms = "terms_of_use"
    var id: String { rawValue }
    var title: String { self == .privacy ? "Privacy Policy" : "Terms of Use" }

    /// The text shipped with the app (Legal/*.md).
    var text: String {
        guard let url = Bundle.main.url(forResource: rawValue, withExtension: "md"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return "" }
        return text
    }
}

struct LegalView: View {
    let document: LegalDocument
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(Array(document.text.components(separatedBy: "\n").enumerated()), id: \.offset) { _, line in
                        lineView(line)
                    }
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(hex: 0x050814).ignoresSafeArea())
            .navigationTitle(L10n.tr(document.title))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button(L10n.tr("Done")) { dismiss() } } }
        }
        .appColorScheme()
        .accessibilityIdentifier("legal_\(document.rawValue)")
    }

    @ViewBuilder
    private func lineView(_ line: String) -> some View {
        if line.hasPrefix("# ") {
            EmptyView()
        } else if line.hasPrefix("## ") {
            Text(line.dropFirst(3)).font(CS.inter(17, .semibold)).foregroundStyle(Color.ink).padding(.top, 8)
        } else if line.hasPrefix("- ") {
            Text(.init("•  " + line.dropFirst(2))).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
        } else if !line.trimmingCharacters(in: .whitespaces).isEmpty {
            Text(.init(line)).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
        }
    }
}
