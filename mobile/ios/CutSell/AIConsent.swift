import SwiftUI

/// Before the first cut is processed, CutSell says which AI services receive the video and asks
/// for "Agree". Without it nothing is uploaded or processed (App Review 5.1.2).
@MainActor
final class AIConsentCenter: ObservableObject {
    static let shared = AIConsentCenter()
    static let key = "cutsell.aiConsent.v1"

    @Published var asking = false
    private var waiting: CheckedContinuation<Bool, Never>?

    var agreed: Bool { UserDefaults.standard.bool(forKey: Self.key) }

    /// true when the creator agreed (now or before); false if they closed the notice.
    func ensure() async -> Bool {
        if agreed { return true }
        if let waiting { waiting.resume(returning: false) }
        return await withCheckedContinuation { continuation in
            waiting = continuation
            asking = true
        }
    }

    func answer(_ agree: Bool) {
        if agree { UserDefaults.standard.set(true, forKey: Self.key) }
        asking = false
        waiting?.resume(returning: agree)
        waiting = nil
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
            Text("Before your first cut").font(CS.inter(22, .bold)).foregroundStyle(.white)
            Text("To make your cut, CutSell sends your content to these AI services:")
                .font(CS.inter(15)).foregroundStyle(CS.textSecondaryF)
            VStack(alignment: .leading, spacing: 10) {
                row("Deepgram", "turns the speech in your video's audio into text.")
                row("Anthropic (Claude)", "reads that text to pick your best takes. It does not receive the video.")
                row("Amazon Web Services and Modal", "store and process your videos to render the cut.")
            }
            Text("They receive it only so CutSell can make your cut. CutSell never sells your content. You can delete your cuts or your account at any time.")
                .font(CS.inter(13)).foregroundStyle(CS.textSecondaryF).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 16) {
                Button("Privacy Policy") { legal = .privacy }.font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                Button("Terms of Use") { legal = .terms }.font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
            }
            Spacer(minLength: 0)
            PrimaryButton(title: "Agree", height: 52, cornerRadius: 16, fontSize: 16) { center.answer(true) }
                .accessibilityIdentifier("aiConsentAgree")
            Button("Not now") { center.answer(false) }
                .font(CS.inter(14, .semibold)).foregroundStyle(CS.textMutedF)
                .frame(maxWidth: .infinity)
                .accessibilityIdentifier("aiConsentNotNow")
        }
        .padding(24)
        .background(Color(hex: 0x0A1326).ignoresSafeArea())
        .presentationDetents([.large])
        .interactiveDismissDisabled()
        .preferredColorScheme(.dark)
        .sheet(item: $legal) { LegalView(document: $0) }
        .accessibilityIdentifier("aiConsentSheet")
    }

    private func row(_ name: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Circle().fill(CS.cyan).frame(width: 6, height: 6).padding(.top, 7)
            (Text(name).bold().foregroundColor(.white) + Text(" — \(text)").foregroundColor(CS.textSecondaryF))
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
            .navigationTitle(document.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } } }
        }
        .preferredColorScheme(.dark)
        .accessibilityIdentifier("legal_\(document.rawValue)")
    }

    @ViewBuilder
    private func lineView(_ line: String) -> some View {
        if line.hasPrefix("# ") {
            EmptyView()
        } else if line.hasPrefix("## ") {
            Text(line.dropFirst(3)).font(CS.inter(17, .semibold)).foregroundStyle(.white).padding(.top, 8)
        } else if line.hasPrefix("- ") {
            Text(.init("•  " + line.dropFirst(2))).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
        } else if !line.trimmingCharacters(in: .whitespaces).isEmpty {
            Text(.init(line)).font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
        }
    }
}
