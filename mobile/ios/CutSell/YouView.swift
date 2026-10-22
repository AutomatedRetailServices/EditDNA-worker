import AVFoundation
import Photos
import SwiftUI
import UserNotifications

// MARK: - You (Figma 07_You · You/Overview v2 1144:114) and its pages
//
// Overview: account card (real name / email), this month's edits, Account & Security, Plan & Usage, Privacy & Data,
// Preferences, Connections, Help & Legal. Pages: Preferences v2 (1144:198), App Language (592:57), Appearance
// (1144:240), Help & Support v2 (1144:263), Connections (1144:298), Account & Security (586:63), Privacy & Data
// (586:67), Plan & Usage (586:64) -- each page shows only what CutSell really has (no invented plans or devices).

enum YouPage: String, Identifiable, Hashable {
    case account, plan, privacy, preferences, language, appearance, connections, help
    var id: String { rawValue }
}

/// Settings kept on this iPhone (Preferences).
@MainActor
final class AppPrefs: ObservableObject {
    static let shared = AppPrefs()
    enum Appearance: String, CaseIterable, Identifiable { case automatic, dark, light; var id: String { rawValue } }

    private static func bool(_ key: String, _ fallback: Bool) -> Bool { UserDefaults.standard.object(forKey: key) as? Bool ?? fallback }
    @Published var appearanceRaw = UserDefaults.standard.string(forKey: "cutsell.pref.appearance") ?? Appearance.automatic.rawValue {
        didSet { UserDefaults.standard.set(appearanceRaw, forKey: "cutsell.pref.appearance") }
    }
    @Published var haptics = bool("cutsell.pref.haptics", true) { didSet { UserDefaults.standard.set(haptics, forKey: "cutsell.pref.haptics") } }
    @Published var cameraDefaults = bool("cutsell.pref.cameraDefaults", true) {
        didSet { UserDefaults.standard.set(cameraDefaults, forKey: "cutsell.pref.cameraDefaults") }
    }
    @Published var autoSaveExports = bool("cutsell.pref.autoSaveExports", true) {
        didSet { UserDefaults.standard.set(autoSaveExports, forKey: "cutsell.pref.autoSaveExports") }
    }
    @Published var notifyProcessing = bool("cutsell.pref.notifyProcessing", true) {
        didSet { UserDefaults.standard.set(notifyProcessing, forKey: "cutsell.pref.notifyProcessing") }
    }
    @Published var notifyReminders = bool("cutsell.pref.notifyReminders", true) {
        didSet { UserDefaults.standard.set(notifyReminders, forKey: "cutsell.pref.notifyReminders") }
    }
    @Published var notifyUpdates = bool("cutsell.pref.notifyUpdates", false) {
        didSet { UserDefaults.standard.set(notifyUpdates, forKey: "cutsell.pref.notifyUpdates") }
    }

    var appearance: Appearance {
        get { Appearance(rawValue: appearanceRaw) ?? .automatic }
        set { appearanceRaw = newValue.rawValue }
    }

    /// The app's colour scheme. The light palette (point 10) is not built yet, so every choice draws the dark
    /// CutSell look for now; the editor is always dark.
    var colorScheme: ColorScheme { .dark }

    nonisolated static var remindersOn: Bool { UserDefaults.standard.object(forKey: "cutsell.pref.notifyReminders") as? Bool ?? true }
    nonisolated static var processingAlertsOn: Bool { UserDefaults.standard.object(forKey: "cutsell.pref.notifyProcessing") as? Bool ?? true }
}

/// The signed-in account's name and email (Sign in with Apple; the server keeps both for the account).
@MainActor
final class AccountProfile: ObservableObject {
    static let shared = AccountProfile()
    @Published var name: String = UserDefaults.standard.string(forKey: "cutsell.account.name") ?? ""
    @Published var email: String = UserDefaults.standard.string(forKey: "cutsell.account.email") ?? ""

    func remember(name: String?, email: String?) {
        if let name, !name.trimmingCharacters(in: .whitespaces).isEmpty {
            self.name = name.trimmingCharacters(in: .whitespaces)
            UserDefaults.standard.set(self.name, forKey: "cutsell.account.name")
        }
        if let email, !email.isEmpty {
            self.email = email
            UserDefaults.standard.set(email, forKey: "cutsell.account.email")
        }
    }

    func clear() {
        name = ""; email = ""
        UserDefaults.standard.removeObject(forKey: "cutsell.account.name")
        UserDefaults.standard.removeObject(forKey: "cutsell.account.email")
    }

    /// Name / email the server has for this account (silently nothing if it can't answer).
    func refresh(session: CutSellSession?) async {
        guard let session, session.isAppleAccount else { return }
        struct Me: Decodable { let email: String?; let name: String? }
        if let me: Me = try? await APIClient.shared.request("/v1/account/me",
                                                             query: [URLQueryItem(name: "user_id", value: session.userID)]) {
            remember(name: me.name, email: me.email)
        }
    }

    func setName(_ value: String, session: CutSellSession?) async {
        remember(name: value, email: nil)
        guard let session, session.isAppleAccount else { return }
        struct Body: Encodable { let user_id: String; let name: String }
        struct Me: Decodable { let email: String?; let name: String? }
        let _: Me? = try? await APIClient.shared.request("/v1/account/me", method: "PUT",
                                                          body: Body(user_id: session.userID, name: value))
    }

    /// The email inside an Apple identity token (a JWT): shown at once, before the server answers.
    nonisolated static func email(fromIdentityToken token: String) -> String? {
        let parts = token.split(separator: ".")
        guard parts.count > 1 else { return nil }
        var base = String(parts[1]).replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        while base.count % 4 != 0 { base += "=" }
        guard let data = Data(base64Encoded: base),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return json["email"] as? String
    }
}

// MARK: Overview

struct YouView: View {
    let selectTab: (MainTab) -> Void

    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var language: L10n
    @ObservedObject private var profile = AccountProfile.shared
    @State private var page: YouPage?

    private var isApple: Bool { appState.session?.isAppleAccount == true }

    var body: some View {
        ZStack {
            YouStyle.base.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    BrandLogo().padding(.top, 4)
                    Text(L10n.tr("You")).font(.system(size: 30, weight: .semibold)).foregroundStyle(YouStyle.primary).padding(.top, 11)
                    accountCard.padding(.top, 9)
                    usageCard.padding(.top, 16)
                    VStack(spacing: 2) {
                        row("YouGlyphAccount", "Account & Security", "Profile, sign-in and devices", .account)
                        row("YouGlyphPlan", "Plan & Usage", "Subscription, usage and history", .plan)
                        row("YouGlyphPrivacy", "Privacy & Data", "Permissions, consent and deletion", .privacy)
                        row("YouGlyphPrefs", "Preferences", "Language, camera and notifications", .preferences, tile: "YouRowIconPrefs")
                        row(nil, "Connections", "TikTok · coming soon", .connections)
                        row("YouGlyphHelp", "Help & Legal", "Guides, support and policies", .help)
                    }
                    .padding(.top, 16)
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 24)
            }
        }
        .safeAreaInset(edge: .bottom) { CSTabBar(current: .you, select: selectTab) }
        .fullScreenCover(item: $page) { start in
            YouPagesView(start: start)
                .environmentObject(appState)
                .environmentObject(language)
        }
        .task { await profile.refresh(session: appState.session) }
        .onAppear {
            // After a language change the app re-draws: back on the page she was using.
            if let again = YouReopen.page { YouReopen.page = nil; page = again }
        }
        .accessibilityIdentifier("youOverview")
    }

    private var displayName: String {
        if !profile.name.isEmpty { return profile.name }
        if !profile.email.isEmpty, let local = profile.email.split(separator: "@").first, !profile.email.contains("privaterelay") {
            return String(local).capitalized
        }
        return isApple ? L10n.tr("Your account") : L10n.tr("Guest")
    }

    private var accountCard: some View {
        Button { page = .account } label: {
            HStack(spacing: 14) {
                ZStack {
                    Image("YouAvatar").resizable().frame(width: 50, height: 50)
                    Text(verbatim: String(displayName.prefix(1)).uppercased())
                        .font(.system(size: 20, weight: .semibold)).foregroundStyle(YouStyle.primary)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text(verbatim: displayName).font(.system(size: 17, weight: .semibold)).foregroundStyle(YouStyle.primary).lineLimit(1)
                    Text(verbatim: isApple ? (profile.email.isEmpty ? L10n.tr("Signed in with Apple") : profile.email)
                                           : L10n.tr("Using CutSell without an account."))
                        .font(.system(size: 12)).foregroundStyle(YouStyle.secondary).lineLimit(1)
                }
                Spacer()
                Text(verbatim: "›").font(.system(size: 24)).foregroundStyle(YouStyle.muted)
            }
            .padding(.horizontal, 15)
            .frame(height: 86)
            .youCard(radius: 18)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("youAccountCard")
    }

    /// This month's edits (real count; 30 = the monthly number Create shows).
    private var usageCard: some View {
        let used = PlanMeter.cutsThisMonth(appState.projects)
        let limit = PlanMeter.figmaMonthlyCuts
        return Button { page = .plan } label: {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text(L10n.tr("THIS MONTH")).font(.system(size: 12, weight: .semibold)).foregroundStyle(CS.cyan)
                    Spacer()
                    Text(verbatim: L10n.tr("Details") + "  ›").font(.system(size: 12, weight: .semibold)).foregroundStyle(YouStyle.secondary)
                }
                Text(L10n.tr("\(used) of \(limit) edits used")).font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(YouStyle.primary).padding(.top, 15)
                HStack(alignment: .center) {
                    Text(L10n.tr("Resets \(YouStyle.nextMonthStart())")).font(.system(size: 11)).foregroundStyle(YouStyle.muted)
                    Spacer()
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3).fill(YouStyle.control).frame(width: 142, height: 6)
                        RoundedRectangle(cornerRadius: 3).fill(Color(hex: 0x0E46D9))
                            .frame(width: max(4, 142 * min(1, CGFloat(used) / CGFloat(max(1, limit)))), height: 6)
                    }
                }
                .padding(.top, 12)
            }
            .padding(.horizontal, 15).padding(.top, 13).padding(.bottom, 26)
            .youCard(radius: 18, border: Color(hex: 0x0E46D9))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("youUsageCard")
    }

    private func row(_ glyph: String?, _ title: String, _ detail: String, _ target: YouPage, tile: String = "YouRowIcon") -> some View {
        Button { page = target } label: {
            HStack(spacing: 12) {
                ZStack {
                    Image(tile).resizable().frame(width: 32, height: 32)
                    if let glyph { Image(glyph).resizable().frame(width: 16, height: 16) }
                    else { Text(verbatim: "↗").font(.system(size: 13, weight: .bold)).foregroundStyle(CS.cyan) }
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text(LocalizedStringKey(title)).font(.system(size: 15, weight: .semibold)).foregroundStyle(YouStyle.primary)
                    Text(LocalizedStringKey(detail)).font(.system(size: 11)).foregroundStyle(YouStyle.muted)
                }
                Spacer()
                Text(verbatim: "›").font(.system(size: 22)).foregroundStyle(YouStyle.muted)
            }
            .padding(.horizontal, 11)
            .frame(height: 56)
            .youCard(radius: 14)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("youRow_\(target.rawValue)")
    }
}

// MARK: Pages

struct YouPagesView: View {
    let start: YouPage
    @Environment(\.dismiss) private var dismiss
    @State private var stack: [YouPage] = []

    var body: some View {
        ZStack {
            YouStyle.base.ignoresSafeArea()
            let current = stack.last ?? start
            Group {
                switch current {
                case .account: AccountPage(back: back)
                case .plan: PlanUsagePage(back: back)
                case .privacy: PrivacyPage(back: back)
                case .preferences: PreferencesPage(back: back, open: { stack.append($0) })
                case .language: LanguagePage(back: back)
                case .appearance: AppearancePage(back: back)
                case .connections: ConnectionsPage(back: back)
                case .help: HelpPage(back: back)
                }
            }
            .id(current)
            .transition(.move(edge: .trailing))
        }
        .animation(.easeOut(duration: 0.2), value: stack)
        .preferredColorScheme(.dark)
    }

    private func back() {
        if stack.isEmpty { dismiss() } else { stack.removeLast() }
    }
}

/// Page frame (Figma: back control at 8,44 · title 22 pt at 64,55 · subtitle 13 pt at 24,103).
struct YouScreen<Content: View>: View {
    let title: String
    let subtitle: String
    let back: () -> Void
    @ViewBuilder var content: () -> Content

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 8) {
                    Button(action: back) {
                        Image("ControlBack").resizable().frame(width: 44, height: 44).frame(width: 48, height: 48)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(L10n.tr("Back"))
                    .accessibilityIdentifier("youBack")
                    Text(LocalizedStringKey(title)).font(.system(size: 22, weight: .semibold)).foregroundStyle(YouStyle.primary)
                }
                .padding(.leading, -16)
                Text(LocalizedStringKey(subtitle)).font(.system(size: 13)).foregroundStyle(YouStyle.secondary).padding(.top, 11)
                content()
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 30)
        }
        .scrollBounceBehavior(.basedOnSize)
    }
}

struct YouSection<Content: View>: View {
    let label: String
    @ViewBuilder var content: () -> Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(LocalizedStringKey(label)).font(.system(size: 12, weight: .semibold)).foregroundStyle(CS.cyan)
            VStack(spacing: 0) { content() }
                .padding(.horizontal, 15)
                .youCard(radius: 18)
        }
        .padding(.top, 25)
    }
}

/// One row of a card: title (+ detail under it), value / chevron, or a switch.
struct YouRow: View {
    let title: String
    var detail: String? = nil
    var value: String? = nil
    var titleColor: Color = YouStyle.primary
    var chevron = true
    var divider = true
    var badge: String? = nil
    var bold = false
    /// Figma Connections: 17 pt like the bold rows, regular weight.
    var weightRegular = false
    var id: String? = nil
    var action: (() -> Void)? = nil

    var body: some View {
        VStack(spacing: 0) {
            Button { action?() } label: {
                HStack(alignment: .center, spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(L10n.tr(title)).font(.system(size: bold ? 17 : 14, weight: bold && !weightRegular ? .semibold : .regular)).foregroundStyle(titleColor)
                        if let detail { Text(L10n.tr(detail)).font(.system(size: 11)).foregroundStyle(YouStyle.muted) }
                    }
                    Spacer(minLength: 8)
                    if let badge {
                        Text(L10n.tr(badge)).font(.system(size: 9, weight: .bold)).foregroundStyle(Color(hex: 0xFFB020))
                            .padding(.horizontal, 9).frame(height: 18)
                            .background(Capsule().fill(Color(hex: 0xFFB020, opacity: 0.12)))
                            .overlay(Capsule().stroke(Color(hex: 0xFFB020, opacity: 0.8), lineWidth: 1))
                    }
                    if let value {
                        Text(verbatim: L10n.tr(value) + (chevron ? "  ›" : "")).font(.system(size: 12)).foregroundStyle(YouStyle.secondary)
                    } else if chevron {
                        Text(verbatim: "›").font(.system(size: 20)).foregroundStyle(YouStyle.muted)
                    }
                }
                .frame(minHeight: 50)
                .padding(.vertical, detail == nil ? 0 : 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .allowsHitTesting(action != nil)          // a row without an action looks the same, it just does nothing
            .accessibilityIdentifier(id ?? "youRow")
            if divider { Rectangle().fill(YouStyle.border).frame(height: 1) }
        }
    }
}

struct YouToggleRow: View {
    let title: String
    var note: String? = nil
    @Binding var isOn: Bool
    var divider = true
    var id: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(L10n.tr(title)).font(.system(size: 14)).foregroundStyle(YouStyle.primary)
                Spacer()
                Toggle("", isOn: $isOn).labelsHidden().tint(Color(hex: 0x1769FF)).accessibilityIdentifier(id)
            }
            .frame(minHeight: 50)
            if let note {
                Text(L10n.tr(note)).font(.system(size: 11)).foregroundStyle(YouStyle.muted)
                    .fixedSize(horizontal: false, vertical: true).padding(.bottom, 14).padding(.trailing, 60)
            }
            if divider { Rectangle().fill(YouStyle.border).frame(height: 1) }
        }
    }
}

/// Radio list rows (Language, Appearance): title + detail, cyan ring when chosen.
struct YouChoiceRow: View {
    let title: String
    let detail: String
    let chosen: Bool
    var divider = true
    var id: String
    let action: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Button(action: action) {
                HStack {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(verbatim: title).font(.system(size: 17, weight: chosen ? .semibold : .regular)).foregroundStyle(YouStyle.primary)
                        Text(verbatim: detail).font(.system(size: 11)).foregroundStyle(YouStyle.muted)
                    }
                    Spacer()
                    ZStack {
                        Circle().stroke(chosen ? CS.cyan : YouStyle.muted, lineWidth: 2).frame(width: 22, height: 22)
                        if chosen { Circle().fill(CS.cyan).frame(width: 11, height: 11) }
                    }
                }
                .frame(height: 61)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(chosen ? .isSelected : [])
            .accessibilityIdentifier(id)
            if divider { Rectangle().fill(YouStyle.border).frame(height: 1) }
        }
    }
}

struct YouNoteCard: View {
    let title: String
    let lines: [String]
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L10n.tr(title)).font(.system(size: 15, weight: .semibold)).foregroundStyle(YouStyle.primary)
            ForEach(lines, id: \.self) { line in
                Text(L10n.tr(line)).font(.system(size: 11.5)).foregroundStyle(YouStyle.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(15)
        .frame(maxWidth: .infinity, minHeight: 116, alignment: .topLeading)
        .youCard(radius: 18)
        .padding(.top, 32)
    }
}

// MARK: Preferences v2 (1144:198)

struct PreferencesPage: View {
    let back: () -> Void
    let open: (YouPage) -> Void
    @ObservedObject private var prefs = AppPrefs.shared
    @EnvironmentObject private var language: L10n

    var body: some View {
        YouScreen(title: "Preferences", subtitle: "Make CutSell work the way you do", back: back) {
            YouSection(label: "GENERAL") {
                YouRow(title: "App language", value: language.choice == .automatic ? "System Default" : language.choice.title, id: "prefLanguage") { open(.language) }
                YouRow(title: "Appearance", value: YouStyle.title(prefs.appearance), id: "prefAppearance") { open(.appearance) }
                YouToggleRow(title: "Haptic feedback", isOn: $prefs.haptics, divider: false, id: "prefHaptics")
            }
            YouSection(label: "CAMERA & EXPORT") {
                YouToggleRow(title: "Use CutSell camera defaults",
                             note: "Uses the resolution, frame rate and HDR saved inside CutSell when supported by this iPhone.",
                             isOn: $prefs.cameraDefaults, id: "prefCamera")
                YouToggleRow(title: "Auto-save exports to Photos", isOn: $prefs.autoSaveExports, divider: false, id: "prefAutoSave")
                YouRow(title: "Default export quality", value: "1080p HD", chevron: false, divider: false)
            }
            YouSection(label: "NOTIFICATIONS") {
                YouToggleRow(title: "Processing complete", isOn: $prefs.notifyProcessing, id: "prefProcessing")
                YouToggleRow(title: "Time to post reminders", isOn: Binding(get: { prefs.notifyReminders }, set: { on in
                    prefs.notifyReminders = on
                    PostReminders.shared.remindersChanged()
                }), id: "prefReminders")
                YouToggleRow(title: "Product updates", isOn: $prefs.notifyUpdates, divider: false, id: "prefUpdates")
            }
        }
        .accessibilityIdentifier("youPreferences")
    }
}

// MARK: App Language (592:57)

struct LanguagePage: View {
    let back: () -> Void
    @EnvironmentObject private var language: L10n

    var body: some View {
        YouScreen(title: "App Language", subtitle: "Choose the language used throughout CutSell", back: back) {
            VStack(spacing: 0) {
                YouChoiceRow(title: L10n.tr("System Default"), detail: L10n.tr("Matches iPhone language"),
                             chosen: language.choice == .automatic, id: "language_automatic") { pick(.automatic) }
                YouChoiceRow(title: "English", detail: "United States", chosen: language.choice == .english, id: "language_en") { pick(.english) }
                YouChoiceRow(title: "Español", detail: "Estados Unidos / Latinoamérica", chosen: language.choice == .spanish,
                             divider: false, id: "language_es") { pick(.spanish) }
            }
            .padding(.horizontal, 17)
            .youCard(radius: 18)
            .padding(.top, 32)
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("languageCard")
            YouNoteCard(title: "Changes apply immediately", lines: [
                "Navigation, processing states, export screens and account settings use the selected language.",
                "Privacy and legal pages open in the selected language when available.",
            ])
            Text(L10n.tr("Video filenames and technical metadata remain unchanged."))
                .font(.system(size: 11)).foregroundStyle(YouStyle.muted).padding(.top, 34)
        }
    }

    private func pick(_ choice: AppLanguage) {
        guard choice != language.choice else { return }
        language.set(choice, reopen: .you)
        YouReopen.page = .language
    }
}

/// After a language change the app re-draws; You opens its Language page again.
enum YouReopen {
    @MainActor static var page: YouPage?
}

// MARK: Appearance (1144:240)

struct AppearancePage: View {
    let back: () -> Void
    @ObservedObject private var prefs = AppPrefs.shared

    var body: some View {
        YouScreen(title: "Appearance", subtitle: "Choose how CutSell looks", back: back) {
            VStack(spacing: 0) {
                choice(.automatic, "Follows your iPhone: light by day, dark at night")
                choice(.dark, "The CutSell look you know")
                choice(.light, "White background, easier in daylight", divider: false)
            }
            .padding(.horizontal, 17)
            .youCard(radius: 18)
            .padding(.top, 32)
            YouNoteCard(title: "The video editor stays dark", lines: [
                "Videos are easier to judge on a dark background, like in CapCut. Everything else follows your choice.",
            ])
        }
    }

    private func choice(_ value: AppPrefs.Appearance, _ detail: String, divider: Bool = true) -> some View {
        YouChoiceRow(title: YouStyle.title(value), detail: L10n.tr(detail), chosen: prefs.appearance == value, divider: divider,
                     id: "appearance_\(value.rawValue)") { prefs.appearance = value }
    }
}

// MARK: Connections (1144:298)

struct ConnectionsPage: View {
    let back: () -> Void
    var body: some View {
        YouScreen(title: "Connections", subtitle: "Link your accounts to post from CutSell", back: back) {
            VStack(spacing: 0) {
                YouRow(title: "TikTok", detail: "Post and track your videos", chevron: false, badge: "SOON", bold: true, weightRegular: true)
                YouRow(title: "Instagram Reels", detail: "Later", chevron: false, badge: "SOON", bold: true)
                YouRow(title: "YouTube Shorts", detail: "Later", chevron: false, divider: false, badge: "SOON", bold: true, weightRegular: true)
            }
            .padding(.horizontal, 17)
            .youCard(radius: 18)
            .padding(.top, 32)
            YouNoteCard(title: "Until then: Post to TikTok", lines: [
                "We save the video, copy your caption and open TikTok. You just paste and post.",
            ])
        }
    }
}

// MARK: Help & Support v2 (1144:263)

struct HelpPage: View {
    let back: () -> Void
    @EnvironmentObject private var appState: AppState
    @State private var showingDiagnostics = false
    @State private var legal: LegalDocument?
    @State private var notice: String?
    @State private var cacheBytes: Int64 = 0
    @State private var versionTaps: [Date] = []

    var body: some View {
        YouScreen(title: "Help & Support", subtitle: "Get help, recover the app and report an issue", back: back) {
            YouSection(label: "HELP TOOLS") {
                YouRow(title: "Report a Problem", detail: "Send feedback tied to the edit") { showingDiagnostics = true }
                YouRow(title: "Contact Support", detail: "support@cutsell.ai") {
                    if let url = URL(string: "mailto:support@cutsell.ai") { UIApplication.shared.open(url) }
                }
                YouRow(title: "Refresh App Data", detail: "Fix stuck screens · keeps your projects") { Task { await refresh() } }
                YouRow(title: "Storage & Originals", detail: L10n.tr("Manage local cache and 4K source files") + cacheText,
                       divider: false) { clearCache() }
            }
            YouSection(label: "LEARN") {
                YouRow(title: "How to use CutSell", detail: "Step-by-step guides for every feature")
                YouRow(title: "Replay intro", detail: "See the 4 welcome screens again", divider: false)
            }
            YouNoteCard(title: "You control what is shared", lines: [
                "No video, audio or transcript is sent unless you choose it in the report.",
            ])
            if let notice {
                Text(L10n.tr(notice)).font(.system(size: 12, weight: .semibold)).foregroundStyle(CS.cyan).padding(.top, 14)
                    .accessibilityIdentifier("helpNotice")
            }
            Text(verbatim: YouStyle.versionLine)
                .font(.system(size: 12)).foregroundStyle(YouStyle.muted).padding(.top, 34)
                .contentShape(Rectangle())
                .onTapGesture { versionTapped() }
                .accessibilityIdentifier("appVersion")
        }
        .sheet(isPresented: $showingDiagnostics) { DiagnosticsView() }
        .sheet(item: $legal) { LegalView(document: $0) }
        .task { cacheBytes = YouStyle.cacheSize() }
    }

    private var cacheText: String {
        cacheBytes > 0 ? " · " + ByteCountFormatter.string(fromByteCount: cacheBytes, countStyle: .file) : ""
    }

    private func refresh() async {
        try? await appState.refreshProjects()
        notice = "App data refreshed. Your projects are safe."
    }

    private func clearCache() {
        for name in ["Sources", "Drafts"] { try? FileManager.default.removeItem(at: LocalCaches.folder(name)) }
        cacheBytes = YouStyle.cacheSize()
        notice = "Cache cleared on this iPhone. Your projects are safe."
    }

    /// Five quick taps on the version open Diagnostics (hidden, for real-iPhone reports).
    private func versionTapped() {
        let now = Date()
        versionTaps = versionTaps.filter { now.timeIntervalSince($0) < 3 } + [now]
        if versionTaps.count >= 5 { versionTaps = []; showingDiagnostics = true }
    }
}

// MARK: Account & Security (586:63)

struct AccountPage: View {
    let back: () -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var profile = AccountProfile.shared
    @State private var editingName = false
    @State private var nameDraft = ""
    @State private var showingLogout = false
    @State private var showingDelete = false
    @State private var errorMessage: String?

    private var session: CutSellSession? { appState.session }
    private var isApple: Bool { session?.isAppleAccount == true }

    var body: some View {
        YouScreen(title: "Account & Security", subtitle: "Your CutSell identity across iPhone and web", back: back) {
            YouSection(label: "PROFILE") {
                YouRow(title: "Display name", detail: profile.name.isEmpty ? "Add your name" : profile.name, divider: true) {
                    nameDraft = profile.name; editingName = true
                }
                if isApple {
                    YouRow(title: "Email", detail: profile.email.isEmpty ? "Hidden by Apple" : profile.email, chevron: false)
                }
                YouRow(title: "Sign-in method", detail: isApple ? "Sign in with Apple" : "No account (this iPhone only)",
                       chevron: false, divider: false)
            }
            YouSection(label: "SECURITY") {
                if isApple {
                    YouRow(title: "Apple account", detail: "Connected with Apple", chevron: false)
                    YouRow(title: "Sign out", detail: "This iPhone only", titleColor: CS.error, divider: false) { showingLogout = true }
                } else {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(L10n.tr("Sign in with Apple to keep your projects on every iPhone you use."))
                            .font(.system(size: 12)).foregroundStyle(YouStyle.secondary).fixedSize(horizontal: false, vertical: true)
                        AppleSignInButton()
                    }
                    .padding(.vertical, 14)
                }
            }
            if let userID = session?.userID {
                Text(verbatim: L10n.tr("Account ID") + "  •  " + YouStyle.masked(userID))
                    .font(.system(size: 11)).foregroundStyle(YouStyle.muted).padding(.top, 24)
                Text(L10n.tr("This ID links your projects across iPhone and web."))
                    .font(.system(size: 11.5)).foregroundStyle(YouStyle.secondary).padding(.top, 10)
            }
            VStack(spacing: 0) {
                YouRow(title: "Delete account", detail: "Permanently removes account and eligible data", titleColor: CS.error,
                       divider: false) { showingDelete = true }
            }
            .padding(.horizontal, 15)
            .youCard(radius: 18)
            .padding(.top, 46)
        }
        .alert(L10n.tr("Display name"), isPresented: $editingName) {
            TextField(L10n.tr("Your name"), text: $nameDraft)
            Button(L10n.tr("Save")) { Task { await profile.setName(nameDraft, session: session) } }
            Button(L10n.tr("Cancel"), role: .cancel) {}
        }
        .confirmationDialog(L10n.tr("Log out of CutSell?"), isPresented: $showingLogout, titleVisibility: .visible) {
            Button(L10n.tr("Log Out")) { Task { profile.clear(); await appState.logout() } }
            Button(L10n.tr("Cancel"), role: .cancel) {}
        } message: {
            Text(L10n.tr("Your projects stay in your account. Sign in with Apple again to see them."))
        }
        .confirmationDialog(L10n.tr("Delete CutSell account?"), isPresented: $showingDelete, titleVisibility: .visible) {
            Button(L10n.tr("Delete My Account"), role: .destructive) {
                Task {
                    do { try await appState.deleteAccount(); profile.clear() } catch { errorMessage = error.localizedDescription }
                }
            }
            Button(L10n.tr("Cancel"), role: .cancel) {}
        } message: {
            Text(L10n.tr("This permanently deletes your CutSell projects and account data and signs out all indexed sessions."))
        }
        .alert(L10n.tr("CutSell"), isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button(L10n.tr("OK"), role: .cancel) {}
        } message: { Text(L10n.tr(errorMessage ?? "")) }
    }
}

// MARK: Privacy & Data (586:67)

struct PrivacyPage: View {
    let back: () -> Void
    @State private var legal: LegalDocument?
    @State private var cleared = false

    var body: some View {
        YouScreen(title: "Privacy & Data", subtitle: "Control access, storage and consent", back: back) {
            YouSection(label: "IPHONE PERMISSIONS") {
                YouRow(title: "Photos", value: YouStyle.photosStatus()) { YouStyle.openSettings() }
                YouRow(title: "Camera", value: YouStyle.avStatus(.video)) { YouStyle.openSettings() }
                YouRow(title: "Microphone", value: YouStyle.avStatus(.audio), divider: false) { YouStyle.openSettings() }
            }
            YouSection(label: "AI & CONTENT") {
                YouRow(title: "Cloud & AI processing", value: AIConsentCenter.shared.agreed ? "Allowed" : "Not asked yet", chevron: false,
                       divider: false)
            }
            YouSection(label: "DATA & RETENTION") {
                YouRow(title: "Clear local cache", value: cleared ? "Cleared" : "On this iPhone", divider: false) {
                    for name in ["Sources", "Drafts"] { try? FileManager.default.removeItem(at: LocalCaches.folder(name)) }
                    cleared = true
                }
            }
            Button { legal = .privacy } label: {
                Text(verbatim: L10n.tr("Privacy Policy") + "  ›").font(.system(size: 13, weight: .semibold)).foregroundStyle(CS.cyan)
            }
            .buttonStyle(.plain)
            .padding(.top, 30)
            .accessibilityIdentifier("privacyPolicyLink")
            Button { legal = .terms } label: {
                Text(verbatim: L10n.tr("Terms of Use") + "  ›").font(.system(size: 13, weight: .semibold)).foregroundStyle(CS.cyan)
            }
            .buttonStyle(.plain)
            .padding(.top, 14)
            .accessibilityIdentifier("termsLink")
        }
        .sheet(item: $legal) { LegalView(document: $0) }
    }
}

// MARK: Plan & Usage (586:64) -- real numbers only; there is no paid plan yet

struct PlanUsagePage: View {
    let back: () -> Void
    @EnvironmentObject private var appState: AppState

    var body: some View {
        let used = PlanMeter.cutsThisMonth(appState.projects)
        let limit = PlanMeter.figmaMonthlyCuts
        YouScreen(title: "Plan & Usage", subtitle: "Your edits this month", back: back) {
            YouSection(label: "THIS MONTH") {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text(L10n.tr("Edits processed")).font(.system(size: 14, weight: .semibold)).foregroundStyle(YouStyle.primary)
                        Spacer()
                        Text(verbatim: "\(used) / \(limit)").font(.system(size: 14, weight: .semibold)).foregroundStyle(YouStyle.primary)
                    }
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3).fill(YouStyle.control).frame(height: 6)
                        GeometryReader { geo in
                            RoundedRectangle(cornerRadius: 3).fill(Color(hex: 0x0E46D9))
                                .frame(width: max(4, geo.size.width * min(1, CGFloat(used) / CGFloat(max(1, limit)))), height: 6)
                        }
                        .frame(height: 6)
                    }
                }
                .padding(.vertical, 16)
                Rectangle().fill(YouStyle.border).frame(height: 1)
                YouRow(title: "Editing time saved", value: TimeSavedLedger.hoursMinutes(TimeSavedLedger.total(userID: appState.session?.userID ?? "")), chevron: false,
                       divider: false)
            }
            Text(L10n.tr("Resets \(YouStyle.nextMonthStart())"))
                .font(.system(size: 11)).foregroundStyle(YouStyle.muted).padding(.top, 20)
        }
    }
}

// MARK: Style

enum YouStyle {
    static let base = Color(hex: 0x050814)
    static let card = Color(hex: 0x0C1730)
    static let border = Color(hex: 0x19345F)
    static let control = Color(hex: 0x0A1326)
    static let primary = Color(hex: 0xF7FAFF)
    static let secondary = Color(hex: 0xAFC4E8)
    static let muted = Color(hex: 0x6F86AE)

    static func title(_ appearance: AppPrefs.Appearance) -> String {
        switch appearance {
        case .automatic: return L10n.tr("Automatic")
        case .dark: return L10n.tr("Dark")
        case .light: return L10n.tr("Light")
        }
    }

    /// "Nov 1" / "1 nov": the first day of next month in the app's language.
    static func nextMonthStart(now: Date = Date()) -> String {
        let calendar = Calendar(identifier: .gregorian)
        let start = calendar.date(from: calendar.dateComponents([.year, .month], from: now)) ?? now
        let next = calendar.date(byAdding: .month, value: 1, to: start) ?? now
        let f = DateFormatter(); f.locale = L10n.locale; f.setLocalizedDateFormatFromTemplate("MMM d")
        return f.string(from: next).replacingOccurrences(of: ".", with: "")
    }

    /// "CutSell 1.0 • Build 32 • iOS 17+" (the real version of this app).
    static var versionLine: String {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "1.0"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "CutSell \(version)  •  Build \(build)  •  iOS 17+"
    }

    static func masked(_ id: String) -> String {
        guard id.count > 10 else { return id }
        return String(id.prefix(6)) + "••••" + String(id.suffix(4))
    }

    static func cacheSize() -> Int64 {
        let root = LocalCaches.root
        guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in walker {
            total += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    static func photosStatus() -> String {
        switch PHPhotoLibrary.authorizationStatus(for: .readWrite) {
        case .authorized: return "Full access"
        case .limited: return "Selected Photos"
        case .denied, .restricted: return "Denied"
        default: return "Not asked yet"
        }
    }

    static func avStatus(_ media: AVMediaType) -> String {
        switch AVCaptureDevice.authorizationStatus(for: media) {
        case .authorized: return "Allowed"
        case .denied, .restricted: return "Denied"
        default: return "Not asked yet"
        }
    }

    static func openSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
    }
}

extension View {
    func youCard(radius: CGFloat, border: Color = YouStyle.border) -> some View {
        background(YouStyle.card, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).stroke(border, lineWidth: 1))
    }
}
