import Foundation

@MainActor
final class AppState: ObservableObject {
    @Published var session: CutSellSession?
    @Published var projects: [Project] = []
    @Published var isBootstrapping = true
    @Published var bootstrapError: String?

    private let api = APIClient.shared

    var persistentAppleAuthEnabled: Bool {
        if let value = UserDefaults.standard.object(forKey: "cutsell.auth.apple.enabled") as? Bool {
            return value
        }
        return false
    }

    func bootstrap() async {
        defer { isBootstrapping = false }
        do {
            let current: CutSellSession
            if let saved = try KeychainStore.load() {
                current = saved
            } else {
                // Commercial Apple auth remains gated until Apple Developer App ID /
                // entitlement setup is explicitly activated. Closed-beta staging keeps
                // using the anonymous bootstrap so current recovery is not disrupted.
                current = try await api.createSession()
                try KeychainStore.save(current)
            }
            session = current
            await api.setSession(current)
            try await refreshProjects()
        } catch {
            bootstrapError = error.localizedDescription
        }
    }

    func establishAppleSession(identityToken: String, nonce: String?) async throws {
        guard persistentAppleAuthEnabled else {
            throw APIError.http(409, "Sign in with Apple is not enabled for this build")
        }
        let current = try await api.createAppleSession(identityToken: identityToken, nonce: nonce)
        try KeychainStore.save(current)
        session = current
        await api.setSession(current)
        try await refreshProjects()
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
        clearSession()
    }

    func clearSession() {
        KeychainStore.clear()
        session = nil
        projects = []
        Task { await api.setSession(nil) }
    }
}
