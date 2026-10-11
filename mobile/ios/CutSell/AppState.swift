import Foundation

@MainActor
final class AppState: ObservableObject {
    @Published var session: CutSellSession?
    @Published var projects: [Project] = []
    @Published var isBootstrapping = true
    @Published var bootstrapError: String?
    /// Shows the welcome screen (Sign in with Apple / continue without an account): after logging
    /// out, after deleting the account, or when an Apple session is no longer valid.
    @Published var needsSignIn = false
    /// A one-time message after signing in (e.g. guest projects that could not be moved).
    @Published var accountNotice: String?

    private let api = APIClient.shared

    func bootstrap() async {
        isBootstrapping = true
        bootstrapError = nil
        defer { isBootstrapping = false }
        do {
            if let saved = try KeychainStore.load() {
                do {
                    try await start(with: saved)
                    return
                } catch APIError.http(let code, _) where code == 401 {
                    // The saved session expired or was revoked on the server. Sessions in use no
                    // longer expire (the server extends them on every request), so this is rare.
                    KeychainStore.clear()
                    if saved.isAppleAccount {
                        // An Apple account is never lost: signing in again returns the same projects.
                        await signOutLocally()
                        needsSignIn = true
                        return
                    }
                }
            }
            if needsSignIn { return }
            // No saved session: start without an account (guest). Signing in with Apple later keeps
            // the guest's projects.
            let fresh = try await api.createSession()
            try KeychainStore.save(fresh)
            try await start(with: fresh)
        } catch {
            bootstrapError = error.localizedDescription
        }
    }

    private func start(with current: CutSellSession) async throws {
        session = current
        await api.setSession(current)
        try await refreshProjects()
    }

    /// Call when a request made outside bootstrap fails: a 401 means the session is gone, so start over.
    func recoverIfUnauthorized(_ error: Error) async {
        guard let apiError = error as? APIError, case .http(let code, _) = apiError, code == 401 else { return }
        await bootstrap()
    }

    /// Sign in with Apple. If this phone was using a guest account, the server keeps its projects
    /// on the Apple account (the current guest session goes with the request).
    func signInWithApple(identityToken: String, nonce: String?) async throws {
        let value = try await api.createAppleSession(identityToken: identityToken, nonce: nonce)
        let current = CutSellSession(userID: value.userID, accessToken: value.accessToken, accountProvider: "apple")
        try KeychainStore.save(current)
        needsSignIn = false
        bootstrapError = nil
        try await start(with: current)
        if value.accountOutcome == "existing_account_guest_not_merged",
           let count = value.guestProjectsNotMerged, count > 0 {
            accountNotice = "This Apple ID already had a CutSell account, so you're now in that account. "
                + "The \(count) project\(count == 1 ? "" : "s") made without an account on this iPhone "
                + "could not be moved to it."
        }
    }

    /// From the welcome screen: use CutSell without an account.
    func continueWithoutAccount() async {
        needsSignIn = false
        await bootstrap()
    }

    /// Signs out this phone only. The account and its projects stay on the server; signing in
    /// with Apple again brings them back.
    func logout() async {
        try? await api.logout()
        KeychainStore.clear()
        await signOutLocally()
        needsSignIn = true
    }

    private func signOutLocally() async {
        session = nil
        projects = []
        await api.setSession(nil)
    }

    func refreshProjects() async throws {
        guard let session else { return }
        let response: ProjectListResponse = try await api.request(
            "/v1/projects",
            query: [URLQueryItem(name: "user_id", value: session.userID)]
        )
        projects = response.projects
    }

    func createProject(title: String?) async throws -> Project {
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String; let title: String? }
        let project: Project = try await api.request(
            "/v1/projects",
            method: "POST",
            body: Body(user_id: session.userID, title: title)
        )
        projects.insert(project, at: 0)
        return project
    }

    func deleteProject(_ project: Project) async throws {
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String; let confirmation: String }
        struct Result: Decodable { let status: String; let project_id: String }
        let _: Result = try await api.request(
            "/v1/projects/\(project.id)",
            method: "DELETE",
            body: Body(user_id: session.userID, confirmation: "DELETE")
        )
        projects.removeAll { $0.id == project.id }
    }

    func deleteAccount() async throws {
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String; let confirmation: String }
        struct Result: Decodable { let status: String; let user_id: String }
        let _: Result = try await api.request(
            "/v1/auth/account",
            method: "DELETE",
            body: Body(user_id: session.userID, confirmation: "DELETE MY ACCOUNT")
        )
        // Clear in order (not via clearSession's detached task), then show the welcome screen:
        // the person chooses to sign in with Apple or continue without an account.
        KeychainStore.clear()
        await signOutLocally()
        needsSignIn = true
    }

    func clearSession() {
        KeychainStore.clear()
        session = nil
        projects = []
        Task { await api.setSession(nil) }
    }
}
