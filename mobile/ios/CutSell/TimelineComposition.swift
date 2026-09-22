import Foundation

/// D-282A-backed timeline composition: the SAME `broll_placements`/
/// `voice_over_placements` contract `cutsell_app/timeline_routes.py`
/// already serves. This is the Overlay/Voice-over layer only -- Main
/// Video stays on the pre-existing `draft`/`draft-edits` system
/// (`DraftEditorViewModel`'s own `selectedClips`), per this gate's own
/// scope: "Main Video usa las fuentes/timeline canónicos ya existentes."
enum TimelineAudioMode: String, Codable, CaseIterable {
    case keepPrimaryVoice = "KEEP_PRIMARY_VOICE"
    case useBrollAudio = "USE_BROLL_AUDIO"
    case muteBrollAudio = "MUTE_BROLL_AUDIO"
}

struct BrollPlacement: Codable, Identifiable, Hashable {
    var placementID: String
    var assetID: String
    var timelineStartSec: Double
    var timelineEndSec: Double
    var sourceInSec: Double
    var sourceOutSec: Double
    var audioMode: TimelineAudioMode

    var id: String { placementID }

    enum CodingKeys: String, CodingKey {
        case placementID = "placement_id"
        case assetID = "asset_id"
        case timelineStartSec = "timeline_start_sec"
        case timelineEndSec = "timeline_end_sec"
        case sourceInSec = "source_in_sec"
        case sourceOutSec = "source_out_sec"
        case audioMode = "audio_mode"
    }
}

struct VoiceOverPlacement: Codable, Identifiable, Hashable {
    var placementID: String
    var assetID: String
    var timelineStartSec: Double
    var timelineEndSec: Double
    var sourceInSec: Double
    var sourceOutSec: Double
    var transcriptReference: String?

    var id: String { placementID }

    enum CodingKeys: String, CodingKey {
        case placementID = "placement_id"
        case assetID = "asset_id"
        case timelineStartSec = "timeline_start_sec"
        case timelineEndSec = "timeline_end_sec"
        case sourceInSec = "source_in_sec"
        case sourceOutSec = "source_out_sec"
        case transcriptReference = "transcript_reference"
    }
}

/// Mirrors `TimelineGetResponse` exactly (`cutsell_app/timeline_routes.py`).
struct TimelineComposition: Codable, Hashable {
    var contractVersion: Int
    var baseEditIdentity: String
    var timelineDurationSec: Double
    var timelineRevisionIdentity: String
    var brollPlacements: [BrollPlacement]
    var voiceOverPlacements: [VoiceOverPlacement]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case baseEditIdentity = "base_edit_identity"
        case timelineDurationSec = "timeline_duration_sec"
        case timelineRevisionIdentity = "timeline_revision_identity"
        case brollPlacements = "broll_placements"
        case voiceOverPlacements = "voice_over_placements"
    }
}

struct TimelineSaveResponse: Codable {
    let outcome: String
    let timelineRevisionIdentity: String?
    let reasons: [String]

    enum CodingKeys: String, CodingKey {
        case outcome
        case timelineRevisionIdentity = "timeline_revision_identity"
        case reasons
    }
}

enum TimelineCompositionError: LocalizedError {
    case noTimelineSaved
    case saveRejected(outcome: String, reasons: [String])

    var errorDescription: String? {
        switch self {
        case .noTimelineSaved:
            return "This project has no saved overlay/voice-over timeline yet."
        case .saveRejected(let outcome, let reasons):
            let detail = reasons.isEmpty ? "" : " (\(reasons.joined(separator: ", ")))"
            return "CutSell could not save the timeline: \(outcome)\(detail)"
        }
    }
}

/// D-282A Stage: the client only ever names an asset_id/placement_id and
/// timing -- never a storage reference. `save` always echoes back the
/// server's own newly-minted `timeline_revision_identity`, which the
/// caller must keep for the NEXT save's own `expectedRevisionIdentity`
/// (optimistic concurrency, the same real authority D-281/D-282 built).
enum TimelineCompositionClient {
    static func get(
        projectID: String,
        userID: String,
        api: APIClient = .shared
    ) async throws -> TimelineComposition {
        do {
            return try await api.request(
                "/v1/projects/\(projectID)/timeline",
                query: [URLQueryItem(name: "user_id", value: userID)]
            )
        } catch let error as APIError {
            if case .http(404, _) = error { throw TimelineCompositionError.noTimelineSaved }
            throw error
        }
    }

    static func save(
        projectID: String,
        userID: String,
        baseEditAssetID: String,
        timelineDurationSec: Double,
        brollPlacements: [BrollPlacement],
        voiceOverPlacements: [VoiceOverPlacement],
        expectedRevisionIdentity: String?,
        api: APIClient = .shared
    ) async throws -> TimelineSaveResponse {
        struct Body: Encodable {
            let user_id: String
            let base_edit_asset_id: String
            let timeline_duration_sec: Double
            let broll_placements: [BrollPlacement]
            let voice_over_placements: [VoiceOverPlacement]
            let expected_revision_identity: String?
        }
        let response: TimelineSaveResponse = try await api.request(
            "/v1/projects/\(projectID)/timeline",
            method: "PUT",
            body: Body(
                user_id: userID,
                base_edit_asset_id: baseEditAssetID,
                timeline_duration_sec: timelineDurationSec,
                broll_placements: brollPlacements,
                voice_over_placements: voiceOverPlacements,
                expected_revision_identity: expectedRevisionIdentity
            )
        )
        guard response.outcome == "TIMELINE_SAVE_SUCCEEDED" else {
            throw TimelineCompositionError.saveRejected(outcome: response.outcome, reasons: response.reasons)
        }
        return response
    }
}
