import SwiftUI

/// You tab: the notification bell and the account (sign in with Apple, log out, delete account).
struct YouView: View {
    let selectTab: (MainTab) -> Void

    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var notificationCenter: NotificationCenterModel
    @State private var showingNotifications = false
    @State private var showingLogout = false
    @State private var showingDeleteAccount = false
    @State private var legal: LegalDocument?
    @State private var errorMessage: String?

    private var isApple: Bool { appState.session?.isAppleAccount == true }

    var body: some View {
        ZStack {
            ScreenBackground()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        BrandLogo()
                        Spacer()
                        CircleIconButton(size: 40, action: { showingNotifications = true }) {
                            Image(systemName: "bell")
                                .font(.system(size: 17, weight: .semibold))
                                .foregroundStyle(.white)
                        }
                        .overlay(alignment: .topTrailing) {
                            if notificationCenter.unreadCount > 0 {
                                Text("\(min(notificationCenter.unreadCount, 9))")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.white)
                                    .frame(width: 17, height: 17)
                                    .background(CS.alert, in: Circle())
                                    .offset(x: 4, y: -4)
                            }
                        }
                        .accessibilityLabel("Notifications")
                        .accessibilityIdentifier("notificationsBell")
                    }
                    .padding(.top, 8)

                    Text("You")
                        .font(CS.inter(34, .semibold))
                        .foregroundStyle(.white)
                        .padding(.top, 14)
                    Text(isApple ? "Signed in with Apple." : "Using CutSell without an account.")
                        .font(CS.inter(14))
                        .foregroundStyle(CS.textSecondary)
                        .padding(.top, 4)

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Account")
                            .font(CS.inter(13, .semibold))
                            .foregroundStyle(CS.textMuted)
                        if isApple {
                            SecondaryButton(title: "Log Out", height: 50, cornerRadius: 16) { showingLogout = true }
                        } else {
                            Text("Sign in with Apple to keep your projects on every iPhone you use.")
                                .font(CS.inter(13))
                                .foregroundStyle(CS.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                            AppleSignInButton()
                        }
                        Button("Delete Account", role: .destructive) { showingDeleteAccount = true }
                            .font(CS.inter(14, .semibold))
                            .foregroundStyle(CS.alert)
                            .padding(.top, 6)
                    }
                    .padding(18)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(hex: 0x040812, opacity: 0.96), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous).stroke(Color(hex: 0x14478C), lineWidth: 1))
                    .padding(.top, 28)

                    // Privacy Policy and Terms of Use, inside the app.
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Legal").font(CS.inter(13, .semibold)).foregroundStyle(CS.textMuted)
                        Button("Privacy Policy") { legal = .privacy }
                            .font(CS.inter(15, .medium)).foregroundStyle(CS.cyan)
                            .accessibilityIdentifier("privacyPolicyLink")
                        Button("Terms of Use") { legal = .terms }
                            .font(CS.inter(15, .medium)).foregroundStyle(CS.cyan)
                            .accessibilityIdentifier("termsLink")
                    }
                    .padding(18)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(hex: 0x040812, opacity: 0.96), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous).stroke(Color(hex: 0x14478C), lineWidth: 1))
                    .padding(.top, 14)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
        }
        .safeAreaInset(edge: .bottom) { CSTabBar(current: .you, select: selectTab) }
        .sheet(item: $legal) { LegalView(document: $0) }
        .sheet(isPresented: $showingNotifications) {
            NotificationInboxView()
                .environmentObject(notificationCenter)
        }
        .confirmationDialog("Log out of CutSell?", isPresented: $showingLogout, titleVisibility: .visible) {
            Button("Log Out") { Task { await appState.logout() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Your projects stay in your account. Sign in with Apple again to see them.")
        }
        .confirmationDialog("Delete CutSell account?", isPresented: $showingDeleteAccount, titleVisibility: .visible) {
            Button("Delete My Account", role: .destructive) {
                Task {
                    do { try await appState.deleteAccount() } catch { errorMessage = error.localizedDescription }
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This permanently deletes your CutSell projects and account data and signs out all indexed sessions.")
        }
        .alert("CutSell", isPresented: Binding(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )) { Button("OK", role: .cancel) {} } message: { Text(errorMessage ?? "") }
    }
}

struct NotificationInboxView: View {
    @EnvironmentObject private var notificationCenter: NotificationCenterModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if notificationCenter.items.isEmpty {
                    ContentUnavailableView("No updates", systemImage: "bell", description: Text("Draft and export updates will appear here."))
                } else {
                    List(notificationCenter.items) { item in
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: icon(item.kind))
                                .foregroundStyle(item.kind.contains("failed") ? CS.alert : CS.cyan)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(title(item.kind)).font(.headline)
                                Text(item.projectID).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .scrollContentBackground(.hidden)
                }
            }
            .background(CS.background.ignoresSafeArea())
            .navigationTitle("Updates")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Mark read") { notificationCenter.markAllRead() }
                        .disabled(notificationCenter.unreadCount == 0)
                }
            }
            .onAppear { notificationCenter.markAllRead() }
        }
        .preferredColorScheme(.dark)
    }

    private func title(_ kind: String) -> String {
        switch kind {
        case "draft_ready": return "Draft ready"
        case "render_finished": return "Export ready"
        case "processing_failed": return "Cut needs another try"
        case "render_failed": return "Export failed"
        default: return kind.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    private func icon(_ kind: String) -> String {
        switch kind {
        case "draft_ready": return "wand.and.stars"
        case "render_finished": return "checkmark.circle"
        case "processing_failed", "render_failed": return "exclamationmark.triangle"
        default: return "bell"
        }
    }
}
