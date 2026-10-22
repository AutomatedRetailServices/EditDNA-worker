import Photos
import SwiftUI
import UIKit
#if canImport(TikTokOpenShareSDK)
import TikTokOpenSDKCore
import TikTokOpenShareSDK
#endif

/// "Post to TikTok" (Export screen and Plan · Day).
/// Official path (TikTok Share Kit): when the app has a TikTok client key and a redirect link, TikTok opens with the
/// video already loaded. Until TikTok approves CutSell, the path that works without approval: the video is saved to
/// Photos, the caption is copied ("Caption copied, paste it in TikTok") and the TikTok app opens (or its App Store
/// page). Back in CutSell: "Did you post it?" → Yes, posted / Not yet.
@MainActor
final class TikTokPoster: ObservableObject {
    static let shared = TikTokPoster()
    static let appStore = URL(string: "https://apps.apple.com/app/id835599320")!     // TikTok on the App Store
    static let appSchemes = ["snssdk1233://", "snssdk1180://", "tiktok://"]       // the TikTok app (global / US)

    /// "Caption copied, paste it in TikTok" (the screen that started the post shows it).
    @Published var notice: String?
    /// Back from TikTok: ask about this video.
    @Published var askPosted: PendingPost?
    @Published private(set) var busy = false

    struct PendingPost: Identifiable, Equatable { let id: String; let onPosted: (() async -> Void)? 
        static func == (a: PendingPost, b: PendingPost) -> Bool { a.id == b.id } }

    private var pending: PendingPost?
    private var leftApp = false
    #if canImport(TikTokOpenShareSDK)
    private var request: TikTokShareRequest?          // kept until TikTok answers (Share Kit asks for it)
    #endif

    /// The TikTok client key and the redirect link the app was built with (TestFlight secrets), if any.
    var clientKey: String? { Self.configured("TikTokClientKey") }
    var redirectURI: String? { Self.configured("CutSellTikTokRedirectURI") }
    var usesShareKit: Bool {
        #if canImport(TikTokOpenShareSDK)
        return clientKey != nil && redirectURI != nil
        #else
        return false
        #endif
    }

    private static func configured(_ key: String) -> String? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: key) as? String else { return nil }
        let clean = value.trimmingCharacters(in: .whitespaces)
        return clean.isEmpty || clean.hasPrefix("$(") || clean == "unset" ? nil : clean
    }

    /// The whole post: save the video, copy the caption, open TikTok (Share Kit when configured).
    func post(video: URL, caption: String?, projectID: String, onNotice: ((String) -> Void)? = nil,
              onPosted: (() async -> Void)? = nil) async {
        guard !busy else { return }
        busy = true
        defer { busy = false }
        let identifier: String
        do {
            identifier = try await Self.saveToPhotos(video)
        } catch {
            notice = "Allow CutSell to add videos to Photos, then try again."
            onNotice?(notice ?? "")
            return
        }
        if let caption, !caption.isEmpty {
            UIPasteboard.general.string = caption
            notice = "Caption copied, paste it in TikTok"
        } else {
            notice = "Video saved to Photos"
        }
        if let notice { onNotice?(notice) }
        pending = PendingPost(id: projectID, onPosted: onPosted)
        leftApp = false
        CutSellDiagnostics.log("tiktok_post_started", ["share_kit": usesShareKit ? "1" : "0", "caption": caption == nil ? "0" : "1"])
        try? await Task.sleep(nanoseconds: 1_200_000_000)       // she sees the notice before TikTok opens
        if usesShareKit, shareKit(identifier) { return }
        openTikTok()
    }

    func openTikTok() {
        for scheme in Self.appSchemes {
            if let url = URL(string: scheme), UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url)
                CutSellDiagnostics.log("tiktok_opened", ["how": "app"])
                return
            }
        }
        UIApplication.shared.open(Self.appStore)                 // TikTok not installed: its App Store page
        CutSellDiagnostics.log("tiktok_opened", ["how": "app_store"])
    }

    #if canImport(TikTokOpenShareSDK)
    private func shareKit(_ identifier: String) -> Bool {
        guard let redirectURI else { return false }
        let request = TikTokShareRequest(localIdentifiers: [identifier], mediaType: .video, redirectURI: redirectURI)
        self.request = request
        let sent = request.send { [weak self] response in
            Task { @MainActor in
                self?.request = nil
                if let share = response as? TikTokShareResponse, share.errorCode != .noError {
                    CutSellDiagnostics.log("tiktok_share_kit_failed", ["code": "\(share.errorCode.rawValue)"])
                }
            }
        }
        CutSellDiagnostics.log("tiktok_opened", ["how": sent ? "share_kit" : "share_kit_unavailable"])
        return sent
    }
    #else
    private func shareKit(_ identifier: String) -> Bool { false }
    #endif

    /// Scene phases: leaving for TikTok, then coming back → "Did you post it?"
    func sceneChanged(_ phase: ScenePhase) {
        switch phase {
        case .background: if pending != nil { leftApp = true }
        case .active:
            if leftApp, let pending {
                askPosted = pending
                self.pending = nil
                leftApp = false
            }
        default: break
        }
    }

    /// TikTok Share Kit's answer (its URL scheme / redirect link). Anything else is not TikTok's.
    static func handle(_ url: URL) {
        #if canImport(TikTokOpenShareSDK)
        _ = TikTokURLHandler.handleOpenURL(url)
        #endif
    }

    /// Probe only: as if she came back from TikTok.
    func simulateReturn() {
        if let pending { askPosted = pending; self.pending = nil }
    }

    static func saveToPhotos(_ file: URL) async throws -> String {
        let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        guard status == .authorized || status == .limited else { throw CocoaError(.userCancelled) }
        var identifier: String?
        try await PHPhotoLibrary.shared().performChanges {
            identifier = PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: file)?.placeholderForCreatedAsset?.localIdentifier
        }
        return identifier ?? ""
    }
}

/// The big "Post to TikTok" button. The official TikTok logo shows when the "TikTokLogo" asset from TikTok's brand
/// kit is in the app (never redrawn here); until then the button is text only.
struct PostToTikTokButton: View {
    let title: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if UIImage(named: "TikTokLogo") != nil {
                    Image("TikTokLogo").resizable().scaledToFit().frame(height: 22)
                }
                Text(L10n.tr(title)).font(CS.sf(16, .semibold))
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity).frame(height: 54)
            .background(Capsule().fill(Color.black))
            .overlay(Capsule().stroke(Color.white.opacity(0.22), lineWidth: 1))
        }
        .accessibilityIdentifier("postToTikTok")
    }
}

/// "Did you post it?" after coming back from TikTok (any screen).
struct TikTokReturnPrompt: ViewModifier {
    @ObservedObject private var poster = TikTokPoster.shared
    @Environment(\.scenePhase) private var phase

    func body(content: Content) -> some View {
        content
            .onChange(of: phase) { _, value in poster.sceneChanged(value) }
            .alert("Did you post it?", isPresented: Binding(get: { poster.askPosted != nil }, set: { if !$0 { poster.askPosted = nil } }),
                   presenting: poster.askPosted) { post in
                Button("Yes, posted") {
                    Task { await post.onPosted?() }
                    CutSellDiagnostics.log("tiktok_post_answer", ["posted": "1"])
                }
                Button("Not yet", role: .cancel) { CutSellDiagnostics.log("tiktok_post_answer", ["posted": "0"]) }
            } message: { _ in
                Text("We'll mark the video as posted in your plan.")
            }
    }
}

extension View {
    func tiktokReturnPrompt() -> some View { modifier(TikTokReturnPrompt()) }
}
