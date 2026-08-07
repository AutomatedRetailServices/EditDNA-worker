import Foundation

struct SessionResponse: Codable {
    let userID: String
    let accessToken: String
    let expiresIn: Int

    enum CodingKeys: String, CodingKey {
        case userID = "user_id"
        case accessToken = "access_token"
        case expiresIn = "expires_in"
    }
}

struct CutSellSession: Codable {
    let userID: String
    let accessToken: String
}

struct Project: Codable, Identifiable, Hashable {
    let projectID: String
    let userID: String
    var title: String
    var state: String
    var latestJobID: String?
    var sourceCount: Int?
    var updatedAt: String?

    var id: String { projectID }

    enum CodingKeys: String, CodingKey {
        case projectID = "project_id"
        case userID = "user_id"
        case title, state
        case latestJobID = "latest_job_id"
        case sourceCount = "source_count"
        case updatedAt = "updated_at"
    }
}

struct ProjectListResponse: Codable {
    let projects: [Project]
}

struct SourceInput: Codable, Hashable {
    let originalName: String
    let uri: String
    let sourceOrder: Int
    let durationSec: Double

    enum CodingKeys: String, CodingKey {
        case originalName = "original_name"
        case uri
        case sourceOrder = "source_order"
        case durationSec = "duration_sec"
    }
}

struct FlowBSubmitResponse: Codable {
    let jobID: String
    let queue: String
    let state: String

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case queue, state
    }
}

struct JobStatus: Codable {
    let jobID: String
    let state: String
    let progress: Int?
    let result: JSONValue?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case state, progress, result, error
    }
}

struct DraftSnapshot: Codable {
    let projectID: String
    let userID: String
    let revision: Int
    let draft: JSONValue
    let sources: [JSONValue]

    enum CodingKeys: String, CodingKey {
        case projectID = "project_id"
        case userID = "user_id"
        case revision, draft, sources
    }
}

enum JSONValue: Codable, Hashable {
    case string(String), number(Double), bool(Bool), object([String: JSONValue]), array([JSONValue]), null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else { self = .array(try container.decode([JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}
