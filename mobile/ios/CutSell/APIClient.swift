import Foundation

actor APIClient {
    static let shared = APIClient()

    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    private(set) var session: CutSellSession?

    var baseURL: URL {
        if let raw = UserDefaults.standard.string(forKey: "cutsell.api.baseURL"),
           let url = Self.validBaseURL(raw) {
            return url
        }
        if let raw = Bundle.main.object(forInfoDictionaryKey: "CutSellAPIBaseURL") as? String,
           let url = Self.validBaseURL(raw) {
            return url
        }
        return URL(string: "http://127.0.0.1:8000")!
    }

    private static func validBaseURL(_ raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let url = URL(string: trimmed),
              let scheme = url.scheme?.lowercased(),
              scheme == "https" || scheme == "http",
              url.host != nil else {
            return nil
        }
        return url
    }

    func setSession(_ session: CutSellSession?) {
        self.session = session
    }

    func createSession() async throws -> CutSellSession {
        var request = URLRequest(url: baseURL.appending(path: "/v1/auth/session"))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        try validate(response: response, data: data)
        let value = try decoder.decode(SessionResponse.self, from: data)
        let session = CutSellSession(userID: value.userID, accessToken: value.accessToken)
        self.session = session
        return session
    }

    func request<T: Decodable>(
        _ path: String,
        method: String = "GET",
        body: (any Encodable)? = nil,
        query: [URLQueryItem] = []
    ) async throws -> T {
        var components = URLComponents(url: baseURL.appending(path: path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { components.queryItems = query }
        var request = URLRequest(url: components.url!)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let token = session?.accessToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try encoder.encode(AnyEncodable(body))
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    func upload(_ data: Data, to url: URL, headers: [String: String] = [:]) async throws -> HTTPURLResponse {
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        headers.forEach { request.setValue($1, forHTTPHeaderField: $0) }
        let (_, response) = try await URLSession.shared.upload(for: request, from: data)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw APIError.http((response as? HTTPURLResponse)?.statusCode ?? -1, "upload failed")
        }
        return http
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            let message = (try? JSONDecoder().decode(APIErrorBody.self, from: data).detail)
                ?? String(data: data, encoding: .utf8)
                ?? "request failed"
            throw APIError.http(http.statusCode, message)
        }
    }
}

private struct APIErrorBody: Decodable { let detail: String }

enum APIError: LocalizedError {
    case invalidResponse
    case http(Int, String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse: "Invalid server response"
        case .http(let code, let message): "\(message) (\(code))"
        }
    }
}

private struct AnyEncodable: Encodable {
    private let encodeClosure: (Encoder) throws -> Void
    init(_ wrapped: any Encodable) {
        self.encodeClosure = { encoder in try wrapped.encode(to: encoder) }
    }
    func encode(to encoder: Encoder) throws { try encodeClosure(encoder) }
}
