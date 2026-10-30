import SwiftUI

/// 30 Oct (PO): Fonts / Style sheet -- reading and saving the caption look, each text's font and look,
/// and duplicating a clip (Edit menu).
extension DraftEditorViewModel {
    static let captionsLookKey = "captions"

    /// The captions' look: what the sheet set (caption_style), else nil = the preset's own look.
    var captionStyle: TextLook? {
        if let live = liveLooks[Self.captionsLookKey] { return live }
        guard let json = displayDraft?["caption_style"], json.objectValue != nil else { return nil }
        return TextLook(json: json)
    }

    /// What the sheet starts from for the captions (the preset's look until she styles them).
    var captionLookForSheet: TextLook { captionStyle ?? TextLook.legacyCaption(preset: captionPreset) }

    func setCaptionLook(_ look: TextLook, commit: Bool) {
        if !commit { liveLooks[Self.captionsLookKey] = look; return }
        liveLooks[Self.captionsLookKey] = nil
        Task { await setDraftValue("caption_style", look.json) }
    }

    // MARK: Text overlays

    func textOverlay(_ id: String) -> [String: JSONValue]? {
        textOverlays.first { $0["overlay_id"]?.stringValue == id }
    }

    /// nil = the text was never styled (drawn the old way: Inter Bold, white, soft shadow).
    func textLook(_ item: [String: JSONValue]) -> TextLook? {
        let id = item["overlay_id"]?.stringValue ?? ""
        if let live = liveLooks[id] { return live }
        guard let json = item["style"], json.objectValue != nil else { return nil }
        return TextLook(json: json)
    }

    func textLookForSheet(_ id: String) -> TextLook {
        if let item = textOverlay(id), let look = textLook(item) { return look }
        var look = TextLook()
        look.shadowColor = 0x000000; look.shadowOpacity = 0.6; look.shadowBlur = 2
        return look
    }

    func textFont(_ item: [String: JSONValue]) -> String? { item["font"]?.stringValue }

    func setTextLook(id: String, _ look: TextLook, commit: Bool) {
        if !commit { liveLooks[id] = look; return }
        liveLooks[id] = nil
        Task { await setTextOverlayFields(id: id, ["style": look.json]) }
    }

    func setTextFont(id: String, _ font: String) {
        Task { await setTextOverlayFields(id: id, ["font": .string(font)]) }
    }

    func setTextOverlayFields(id: String, _ fields: [String: JSONValue]) async {
        await setDraftValue("text_overlays", .array(textOverlays.map { item in
            guard item["overlay_id"]?.stringValue == id else { return .object(item) }
            var next = item
            for (key, value) in fields { next[key] = value }
            return .object(next)
        }))
    }

    // MARK: Duplicate (Edit menu)

    /// A copy of the clip right after it (same source window, its own id). Returns the new id.
    @discardableResult
    func duplicateClip(_ clipID: String) async -> String? {
        guard let clip = selectedClips.first(where: { $0["clip_id"]?.stringValue == clipID }) else { return nil }
        let taken = Set(selectedClips.compactMap { $0["clip_id"]?.stringValue }
                        + alternateClips.compactMap { $0["clip_id"]?.stringValue })
        var n = 2
        while taken.contains("\(clipID)-copy\(n)") { n += 1 }
        let newID = "\(clipID)-copy\(n)"
        var copy = clip
        copy["clip_id"] = .string(newID)
        await setDraftValue("selected", .array(selectedClips.flatMap { item -> [JSONValue] in
            item["clip_id"]?.stringValue == clipID ? [.object(item), .object(copy)] : [.object(item)]
        }))
        return newID
    }
}

/// A text overlay drawn on the video: the old look when it was never styled, else its font and look.
struct TextOverlayLabel: View {
    let item: [String: JSONValue]
    let look: TextLook?
    let font: String?

    var body: some View {
        let scale = CGFloat(item["scale"]?.doubleValue ?? 1)
        let text = item["text"]?.stringValue ?? ""
        if look == nil && font == nil {
            Text(verbatim: text)
                .font(CS.inter(18 * scale, .bold))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.6), radius: 2)
        } else {
            let spec = CaptionFontCatalog.spec(for: font ?? "inter")
            // Every face at about the same visual size as the old 18 pt Inter Bold.
            let size = 18 * scale * CGFloat(spec.renderSize) / 84
            LookedText(text: text, look: look ?? TextLook(), font: .custom(spec.postScriptName, fixedSize: size),
                       unit: size / 84)
        }
    }
}
