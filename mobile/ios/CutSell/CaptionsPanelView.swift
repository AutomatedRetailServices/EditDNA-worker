import SwiftUI

/// Editor v2 Captions panel (Figma `04_Editor`, rows "2 · Captions" and
/// "2b · Captions"), shown right under the video so every change is seen on
/// the preview at once. It only sets the whole-video caption settings the
/// server already supports (`/v1/draft-edits/caption-settings`):
/// on/off, look (Classic / Highlight / Box / Yellow), the colour of the
/// spoken word for Highlight, black or white box for Box, and the typeface.
struct CaptionsPanelView: View {
    @ObservedObject var model: DraftEditorViewModel
    let onDone: () -> Void

    /// The colour last chosen for Highlight / the box last chosen for Box,
    /// so switching looks back and forth keeps the creator's choice.
    @State private var lastHighlight = "highlight_green"
    @State private var lastBox = "box"

    private enum Look: String, CaseIterable { case classic, highlight, box, yellow }

    private struct WordColour: Identifiable {
        let preset: String
        let name: String
        let hex: UInt32
        var id: String { preset }
    }

    private static let wordColours: [WordColour] = [
        WordColour(preset: "highlight_green", name: "Green", hex: CaptionLook.highlightGreen),
        WordColour(preset: "highlight_red", name: "Red", hex: CaptionLook.highlightRed),
        WordColour(preset: "highlight_blue", name: "Blue", hex: CaptionLook.highlightBlue),
    ]

    /// Size of each typeface's name on its chip (Figma).
    private static let chipSizes: [String: CGFloat] = [
        "oswald": 15, "anton": 15, "bebas_neue": 16, "bangers": 16,
    ]

    private var enabled: Bool { model.captionsEnabled }

    private var currentLook: Look {
        let preset = model.captionPreset
        if preset.hasPrefix("highlight") { return .highlight }
        if preset == "box" || preset == "box_light" || preset == "clean" { return .box }
        if preset == "yellow" { return .yellow }
        return .classic
    }

    /// The exact Highlight preset in use ("highlight" alone is the green one).
    private var currentHighlight: String {
        let preset = model.captionPreset
        return preset == "highlight" ? "highlight_green" : preset
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            styleRow
                .disabled(!enabled)
                .opacity(enabled ? 1 : 0.35)
            fontRow
                .disabled(!enabled)
                .opacity(enabled ? 1 : 0.35)
            if enabled && currentLook == .highlight {
                wordColourRow
            } else if enabled && currentLook == .box {
                boxVariantRow
            }
            Text(enabled ? "Tap the caption on the video to fix words." : "Captions are off")
                .font(.system(size: 10))
                .foregroundStyle(CaptionPanelColours.hint)
                .frame(maxWidth: .infinity)
                .padding(.top, 2)
                .accessibilityIdentifier("captions.panel.hint")
        }
        .padding(10)
        .background(CaptionPanelColours.background, in: RoundedRectangle(cornerRadius: 16))
        .environment(\.colorScheme, .dark)
        .onAppear {
            if currentLook == .highlight { lastHighlight = currentHighlight }
            if model.captionPreset == "box_light" { lastBox = "box_light" }
        }
        .accessibilityIdentifier("captions.panel")
    }

    // MARK: Header: title, On/Off, Done

    private var header: some View {
        HStack(spacing: 8) {
            Text("Captions")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(CaptionPanelColours.title)
            Spacer()
            Text(enabled ? "On" : "Off")
                .font(.system(size: 12))
                .foregroundStyle(CaptionPanelColours.label)
            Toggle("Captions", isOn: Binding(
                get: { model.captionsEnabled },
                set: { newValue in Task { await model.setCaptionSettings(enabled: newValue) } }
            ))
            .labelsHidden()
            .tint(CaptionPanelColours.selected)
            .accessibilityIdentifier("captions.enableToggle")
            Button(action: onDone) {
                Text("Done")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(CaptionPanelColours.strong)
                    .padding(.horizontal, 14)
                    .frame(height: 28)
                    .background(CaptionPanelColours.button, in: Capsule())
                    .overlay(Capsule().stroke(CaptionPanelColours.accent, lineWidth: 1.5))
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("captions.doneButton")
        }
        .frame(minHeight: 32)
    }

    // MARK: Looks

    private var styleRow: some View {
        HStack(spacing: 6) {
            ForEach(Look.allCases, id: \.self) { look in
                Button {
                    choose(look)
                } label: {
                    styleTile(look)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(title(of: look))
                .accessibilityAddTraits(currentLook == look ? [.isSelected] : [])
                .accessibilityIdentifier("captions.style.\(look.rawValue)")
            }
        }
    }

    private func styleTile(_ look: Look) -> some View {
        let selected = enabled && currentLook == look
        return VStack(spacing: 4) {
            sample(look)
                .frame(height: 24)
            Text(title(of: look))
                .font(.system(size: 10))
                .foregroundStyle(CaptionPanelColours.label)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 54)
        .background(CaptionPanelColours.tile, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(selected ? CaptionPanelColours.selected : CaptionPanelColours.border,
                        lineWidth: selected ? 2 : 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private func sample(_ look: Look) -> some View {
        switch look {
        case .classic:
            Text("Aa").font(.system(size: 17, weight: .semibold)).foregroundStyle(Color.white)
        case .highlight:
            Text("Aa").font(.system(size: 17, weight: .semibold))
                .foregroundStyle(Color(captionHex: highlightHex(lastHighlight)))
        case .box:
            Text("Aa").font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Color(captionHex: CaptionLook.ink))
                .frame(width: 34, height: 22)
                .background(Color.white, in: RoundedRectangle(cornerRadius: 4))
        case .yellow:
            Text("Aa").font(.system(size: 17, weight: .semibold))
                .foregroundStyle(Color(captionHex: CaptionLook.yellow))
        }
    }

    private func title(of look: Look) -> String {
        switch look {
        case .classic: return "Classic"
        case .highlight: return "Highlight"
        case .box: return "Box"
        case .yellow: return "Yellow"
        }
    }

    private func choose(_ look: Look) {
        let preset: String
        switch look {
        case .classic: preset = "classic"
        case .highlight: preset = lastHighlight
        case .box: preset = lastBox
        case .yellow: preset = "yellow"
        }
        guard preset != model.captionPreset else { return }
        Task { await model.setCaptionSettings(preset: preset) }
    }

    private func highlightHex(_ preset: String) -> UInt32 {
        Self.wordColours.first { $0.preset == preset }?.hex ?? CaptionLook.highlightGreen
    }

    // MARK: Typefaces (scrolls sideways)

    private var fontRow: some View {
        ScrollViewReader { reader in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(CaptionFontCatalog.all) { spec in
                        fontChip(spec)
                            .id(spec.key)
                    }
                }
                .padding(.trailing, 40)
            }
            .overlay(alignment: .trailing) {
                LinearGradient(
                    colors: [CaptionPanelColours.background.opacity(0), CaptionPanelColours.background],
                    startPoint: .leading, endPoint: .trailing
                )
                .frame(width: 40)
                .allowsHitTesting(false)
            }
            .onAppear {
                // After the first layout, so the chosen typeface is in view.
                DispatchQueue.main.async { reader.scrollTo(model.captionFont, anchor: .center) }
            }
        }
        .frame(height: 34)
    }

    private func fontChip(_ spec: CaptionFontSpec) -> some View {
        let selected = enabled && model.captionFont == spec.key
        return Button {
            guard spec.key != model.captionFont else { return }
            Task { await model.setCaptionSettings(font: spec.key) }
        } label: {
            Text(spec.displayName)
                .font(.custom(spec.postScriptName, fixedSize: Self.chipSizes[spec.key] ?? 13))
                .foregroundStyle(selected ? Color.white : CaptionPanelColours.chipText)
                .lineLimit(1)
                .padding(.horizontal, 12)
                .frame(height: 34)
                .background(CaptionPanelColours.tile, in: RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(selected ? CaptionPanelColours.selected : CaptionPanelColours.border,
                                lineWidth: selected ? 2 : 1)
                )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Font \(spec.displayName)")
        .accessibilityAddTraits(selected ? [.isSelected] : [])
        .accessibilityIdentifier("captions.font.\(spec.key)")
    }

    // MARK: Highlight: colour of the spoken word

    private var wordColourRow: some View {
        HStack(spacing: 12) {
            Text("Word color")
                .font(.system(size: 11))
                .foregroundStyle(CaptionPanelColours.label)
            ForEach(Self.wordColours) { colour in
                let selected = currentHighlight == colour.preset
                Button {
                    lastHighlight = colour.preset
                    guard colour.preset != model.captionPreset else { return }
                    Task { await model.setCaptionSettings(preset: colour.preset) }
                } label: {
                    Circle()
                        .fill(Color(captionHex: colour.hex))
                        .frame(width: 22, height: 22)
                        .padding(4)
                        .overlay(Circle().stroke(Color.white, lineWidth: selected ? 2.5 : 0))
                        .frame(width: 30, height: 30)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Word color \(colour.name)")
                .accessibilityAddTraits(selected ? [.isSelected] : [])
                .accessibilityIdentifier("captions.wordColor.\(colour.name.lowercased())")
            }
            Spacer()
        }
        .padding(.leading, 4)
        .frame(minHeight: 30)
    }

    // MARK: Box: black box or white box

    private var boxVariantRow: some View {
        HStack(spacing: 6) {
            boxVariant(preset: "box", title: "Black box, white words",
                       box: Color(captionHex: CaptionLook.ink), words: Color.white)
            boxVariant(preset: "box_light", title: "White box, black words",
                       box: Color.white, words: Color(captionHex: CaptionLook.ink))
        }
    }

    private func boxVariant(preset: String, title: String, box: Color, words: Color) -> some View {
        let current = model.captionPreset == "clean" ? "box" : model.captionPreset
        let selected = current == preset
        return Button {
            lastBox = preset
            guard preset != model.captionPreset else { return }
            Task { await model.setCaptionSettings(preset: preset) }
        } label: {
            HStack(spacing: 8) {
                Text("Aa")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(words)
                    .frame(width: 24, height: 18)
                    .background(box)
                    .overlay(Rectangle().stroke(CaptionPanelColours.sampleEdge, lineWidth: 0.5))
                Text(title)
                    .font(.system(size: 10))
                    .foregroundStyle(CaptionPanelColours.chipText)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 6)
            .frame(maxWidth: .infinity)
            .frame(height: 30)
            .background(CaptionPanelColours.tile, in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(selected ? CaptionPanelColours.selected : CaptionPanelColours.border,
                            lineWidth: selected ? 2 : 1)
            )
            .opacity(selected ? 1 : 0.6)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? [.isSelected] : [])
        .accessibilityIdentifier("captions.box.\(preset)")
    }
}

/// Editor v2 colours (Figma tokens) used by the Captions panel.
enum CaptionPanelColours {
    static let background = Color(captionHex: 0x02040A)
    static let tile = Color(captionHex: 0x040812)
    static let border = Color(captionHex: 0x14336B)
    static let selected = Color(captionHex: 0x1AA6FF)
    static let accent = Color(captionHex: 0x1AE5FF)
    static let title = Color(captionHex: 0xF2F7FF)
    static let strong = Color(captionHex: 0xFAFCFF)
    static let label = Color(captionHex: 0xADC7F0)
    static let chipText = Color(captionHex: 0xC7D6F5)
    static let hint = Color(captionHex: 0x6B8CBD)
    static let sampleEdge = Color(captionHex: 0x5C7AAD)
    static let button = Color(red: 4 / 255, green: 8 / 255, blue: 19 / 255, opacity: 0.95)
}
