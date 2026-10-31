import SwiftUI

/// 30 Oct (PO, reference: a sheet with "Fonts" and "Style"): the white 3D sheet that rises for the
/// captions and for a text. Fonts: every face, in a grid, each name written in its own face. Style: the
/// "Aa" looks, then Text / Highlight / Stroke / Background / Shadow / Spacing with every colour (swatches
/// plus the colour wheel) and sliders. Changes show at once on the video; sliders save when the finger lifts.
/// 30 Oct (PO, build 68 feedback): the sheet can be dragged taller / shorter (or down to close), and while
/// typing it shrinks to the text field sitting right on the keyboard (the video stays in view).
@MainActor
final class TextSheetLayout: ObservableObject {
    @Published var expanded = false
    @Published var keyboard: CGFloat = 0
    private var observers: [NSObjectProtocol] = []

    init() {
        let center = NotificationCenter.default
        observers.append(center.addObserver(forName: UIResponder.keyboardWillChangeFrameNotification, object: nil, queue: .main) { [weak self] note in
            guard let frame = note.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect else { return }
            let height = max(0, UIScreen.main.bounds.maxY - frame.minY)
            MainActor.assumeIsolated { withAnimation(.easeOut(duration: 0.25)) { self?.keyboard = height } }
        })
        observers.append(center.addObserver(forName: UIResponder.keyboardWillHideNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { withAnimation(.easeOut(duration: 0.25)) { self?.keyboard = 0 } }
        })
    }

    deinit { observers.forEach(NotificationCenter.default.removeObserver) }

    /// Height the sheet takes at the bottom of the editor.
    var height: CGFloat {
        if keyboard > 0 { return TextStyleSheet.typingBar + keyboard }
        return expanded ? TextStyleSheet.tallHeight : TextStyleSheet.height
    }
}

struct TextStyleSheet: View {
    static var height: CGFloat { 452 + max(0, ToolbarV4.bottomInset - 8) }
    static var tallHeight: CGFloat { max(height, UIScreen.main.bounds.height * 0.74) }
    /// Handle + the text field and ✓, above the keyboard.
    static let typingBar: CGFloat = 92

    @ObservedObject var model: DraftEditorViewModel
    let target: TextStyleTarget
    let playhead: Double
    @ObservedObject var layout: TextSheetLayout
    /// Closed with ✓ (the id of a text that was just added, if any).
    let onDone: (String?) -> Void

    enum Tab: String, CaseIterable { case fonts = "Fonts", style = "Style" }
    enum Part: String, CaseIterable {
        case text = "Text", highlight = "Highlight", stroke = "Stroke", background = "Background"
        case shadow = "Shadow", spacing = "Spacing"
    }

    @State private var tab: Tab = .fonts
    @State private var part: Part = .text
    @State private var look = TextLook()
    @State private var font = CaptionFontCatalog.defaultKey
    @State private var text = ""
    @State private var loaded = false
    @State private var saving = false
    @FocusState private var typing: Bool

    private var isCaptions: Bool { target == .captions }
    private var parts: [Part] { isCaptions ? Part.allCases : Part.allCases.filter { $0 != .highlight } }
    private var textID: String? { if case .text(let id) = target { return id } else { return nil } }

    @State private var dragY: CGFloat = 0

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 0) {
                Capsule().fill(CS.wiBorder).frame(width: 40, height: 5).padding(.top, 8)
                header.padding(.horizontal, 16).padding(.top, 10)
                if layout.keyboard == 0 {
                    tabs.padding(.horizontal, 16).padding(.top, 10)
                }
            }
            .contentShape(Rectangle())
            .gesture(sheetDrag)
            if layout.keyboard == 0 {
                Group {
                    switch tab {
                    case .fonts: fontsGrid
                    case .style: styleTab
                    }
                }
                .frame(maxHeight: .infinity, alignment: .top)
                .clipped()
            } else {
                Spacer(minLength: 0)
            }
        }
        .frame(height: max(Self.typingBar, layout.height - max(0, dragY)), alignment: .top)
        .background(
            UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28, style: .continuous)
                .fill(Color(fixed: 0xFFFFFF))
                .shadow(color: Color(fixed: 0x0B1020, opacity: 0.12), radius: 18, y: -4)
                .ignoresSafeArea(edges: .bottom)
        )
        .onAppear(perform: load)
        .onDisappear { layout.expanded = false; typing = false }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("textStyleSheet")
    }

    /// Up: taller (more fonts in view). Down: back to the normal height, then closed.
    private var sheetDrag: some Gesture {
        DragGesture(minimumDistance: 8)
            .onChanged { value in dragY = value.translation.height }
            .onEnded { value in
                let dy = value.translation.height
                withAnimation(.spring(response: 0.34, dampingFraction: 0.86)) {
                    dragY = 0
                    if layout.keyboard > 0 {
                        if dy > 30 { typing = false }
                    } else if dy < -40 {
                        layout.expanded = true
                    } else if dy > 60 {
                        if layout.expanded { layout.expanded = false } else { Task { await done() } }
                    }
                }
            }
    }

    // MARK: Header: the text being written (or Captions On), and ✓

    private var header: some View {
        HStack(spacing: 12) {
            if isCaptions {
                Text(L10n.tr("Captions")).font(CS.inter(18, .heavy)).foregroundStyle(CS.wiInk)
                Spacer()
                Text(L10n.tr("On")).font(CS.inter(13, .semibold)).foregroundStyle(CS.wiSecondary)
                Toggle("", isOn: Binding(get: { model.captionsEnabled },
                                         set: { model.applyCaptionSettings(enabled: $0) }))
                    .labelsHidden().tint(CS.accent)
                    .accessibilityIdentifier("captionsOnToggle")
            } else {
                TextField("", text: $text, prompt: Text(L10n.tr("Enter text")).foregroundStyle(CS.wiSecondary))
                    .font(CS.inter(16, .semibold))
                    .foregroundStyle(CS.wiInk)
                    .tint(CS.accent)
                    .submitLabel(.done)
                    .onSubmit { typing = false }       // "Done" on the keyboard: back to Fonts / Style
                    .focused($typing)
                    .padding(.horizontal, 16).padding(.vertical, 12)
                    .frame(minHeight: 50)
                    .puff3D(.white, in: Capsule(), depth: 3)
                    .accessibilityIdentifier("textStyleField")
            }
            Button { Task { await done() } } label: {
                Group {
                    if saving { ProgressView().tint(.white) } else {
                        Image(systemName: "checkmark").font(.system(size: 18, weight: .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                    }
                }
                .frame(width: 48, height: 48)
                .puff3D(.blue, in: Circle(), depth: 4)
            }
            .buttonStyle(BubblePress())
            .disabled(saving)
            .accessibilityIdentifier("textStyleDone")
        }
    }

    private var tabs: some View {
        HStack(spacing: 22) {
            ForEach(Tab.allCases, id: \.self) { item in
                Button { typing = false; withAnimation(.easeOut(duration: 0.15)) { tab = item } } label: {
                    VStack(spacing: 5) {
                        Text(L10n.tr(item.rawValue)).font(CS.inter(16, tab == item ? .bold : .semibold))
                            .foregroundStyle(tab == item ? CS.wiInk : CS.wiSecondary)
                        Capsule().fill(tab == item ? CS.accent : .clear).frame(height: 3)
                    }
                    .fixedSize()
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("textStyleTab_\(item.rawValue)")
            }
            Spacer()
        }
    }

    // MARK: Fonts

    private var fontsGrid: some View {
        ScrollView(showsIndicators: true) {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 3), spacing: 10) {
                ForEach(CaptionFontCatalog.all, id: \.key) { spec in
                    let on = font == spec.key
                    Button { pickFont(spec.key) } label: {
                        Text(verbatim: spec.displayName)
                            .font(.custom(spec.postScriptName, size: 19))
                            .lineLimit(1).minimumScaleFactor(0.55)
                            .foregroundStyle(on ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                            .padding(.horizontal, 8)
                            .frame(maxWidth: .infinity).frame(height: 52)
                            .puff3D(on ? .blue : .white, in: RoundedRectangle(cornerRadius: 16, style: .continuous), depth: 3)
                    }
                    .buttonStyle(BubblePress())
                    .accessibilityIdentifier("textFont_\(spec.key)")
                }
            }
            .padding(.horizontal, 16).padding(.top, 14).padding(.bottom, 24)
        }
    }

    // MARK: Style

    private var styleTab: some View {
        VStack(alignment: .leading, spacing: 12) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(TextLook.presets) { preset in presetChip(preset) }
                }
                .padding(.horizontal, 16).padding(.vertical, 4)
            }
            .padding(.top, 12)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 18) {
                    ForEach(parts, id: \.self) { item in
                        Button { part = item } label: {
                            VStack(spacing: 4) {
                                Text(L10n.tr(item.rawValue)).font(CS.inter(14, part == item ? .bold : .semibold))
                                    .foregroundStyle(part == item ? CS.wiInk : CS.wiSecondary)
                                Capsule().fill(part == item ? CS.accent : .clear).frame(height: 3)
                            }
                            .fixedSize()
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("textStylePart_\(item.rawValue)")
                    }
                }
                .padding(.horizontal, 16)
            }
            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 14) { partControls }
                    .padding(.horizontal, 16).padding(.bottom, 24)
            }
        }
    }

    private func presetChip(_ preset: TextLook.Preset) -> some View {
        let on = look.preset == preset.key
        let spec = CaptionFontCatalog.spec(for: font)
        return Button { change { $0 = $0.applying(preset: preset.key) } } label: {
            LookedText(text: "Aa", look: preset.look, font: .custom(spec.postScriptName, size: 22), unit: 0.32)
                .frame(width: 58, height: 58)
                .background(
                    LinearGradient(colors: [Color(fixed: 0x8E97AA), Color(fixed: 0x5F687C)], startPoint: .top, endPoint: .bottom),
                    in: RoundedRectangle(cornerRadius: 16, style: .continuous)
                )
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(on ? CS.accent : Color(fixed: 0xFFFFFF, opacity: 0.6), lineWidth: on ? 3 : 1))
                .shadow(color: Color(fixed: 0x0B1020, opacity: 0.15), radius: 4, y: 3)
        }
        .buttonStyle(BubblePress())
        .accessibilityIdentifier("textPreset_\(preset.key)")
    }

    @ViewBuilder
    private var partControls: some View {
        switch part {
        case .text:
            ColorRow(selected: look.color, allowsNone: false) { hex in change { $0.color = hex ?? 0xFFFFFF; $0.preset = nil } }
            LookSlider(title: "Size", value: sizeBinding, range: 0.5...2.0, shown: { "\(Int(($0 * 100).rounded()))%" },
                       onCommit: commitSize)
            LookSlider(title: "Opacity", value: binding(\.opacity), range: 0.1...1, shown: { "\(Int(($0 * 100).rounded()))%" },
                       onCommit: commit)
        case .highlight:
            Text(L10n.tr("The word being said changes colour.")).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
            ColorRow(selected: look.highlightColor, allowsNone: true) { hex in change { $0.highlightColor = hex } }
        case .stroke:
            ColorRow(selected: look.strokeColor, allowsNone: true) { hex in
                change { $0.strokeColor = hex; if hex != nil && $0.strokeWidth == 0 { $0.strokeWidth = 4 }; $0.preset = nil }
            }
            LookSlider(title: "Width", value: binding(\.strokeWidth), range: 0...12, shown: { String(format: "%.0f", $0) },
                       onCommit: commit)
        case .background:
            ColorRow(selected: look.bgOpacity > 0 ? look.bgColor : nil, allowsNone: true) { hex in
                change { $0.bgColor = hex; $0.bgOpacity = hex == nil ? 0 : max($0.bgOpacity, 0.8); $0.preset = nil }
            }
            LookSlider(title: "Opacity", value: binding(\.bgOpacity), range: 0...1, shown: { "\(Int(($0 * 100).rounded()))%" },
                       onCommit: commit)
            LookSlider(title: "Corners", value: binding(\.bgRadius), range: 0...40, shown: { String(format: "%.0f", $0) },
                       onCommit: commit)
        case .shadow:
            ColorRow(selected: look.shadowOpacity > 0 ? look.shadowColor : nil, allowsNone: true) { hex in
                change { $0.shadowColor = hex; $0.shadowOpacity = hex == nil ? 0 : max($0.shadowOpacity, 0.7)
                    if hex != nil && $0.shadowBlur == 0 && $0.shadowDistance == 0 { $0.shadowBlur = 3; $0.shadowDistance = 3 }
                    $0.preset = nil }
            }
            LookSlider(title: "Opacity", value: binding(\.shadowOpacity), range: 0...1, shown: { "\(Int(($0 * 100).rounded()))%" },
                       onCommit: commit)
            LookSlider(title: "Blur", value: binding(\.shadowBlur), range: 0...20, shown: { String(format: "%.0f", $0) },
                       onCommit: commit)
            LookSlider(title: "Distance", value: binding(\.shadowDistance), range: 0...20, shown: { String(format: "%.0f", $0) },
                       onCommit: commit)
        case .spacing:
            LookSlider(title: "Letters", value: binding(\.letterSpacing), range: -5...30, shown: { String(format: "%.0f", $0) },
                       onCommit: commit)
            LookSlider(title: "Lines", value: binding(\.lineSpacing), range: -20...60, shown: { String(format: "%.0f", $0) },
                       onCommit: commit)
            HStack(spacing: 10) {
                ForEach([("left", "text.alignleft"), ("center", "text.aligncenter"), ("right", "text.alignright")], id: \.0) { key, icon in
                    let on = look.align == key
                    Button { change { $0.align = key } } label: {
                        Image(systemName: icon).font(.system(size: 17, weight: .semibold))
                            .foregroundStyle(on ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                            .frame(width: 52, height: 44)
                            .puff3D(on ? .blue : .white, in: RoundedRectangle(cornerRadius: 14, style: .continuous), depth: 3)
                    }
                    .buttonStyle(BubblePress())
                    .accessibilityIdentifier("textAlign_\(key)")
                }
                Spacer()
                Button { change { $0.uppercase.toggle() } } label: {
                    Text(verbatim: "AA").font(CS.inter(15, .heavy))
                        .foregroundStyle(look.uppercase ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                        .frame(width: 60, height: 44)
                        .puff3D(look.uppercase ? .blue : .white, in: RoundedRectangle(cornerRadius: 14, style: .continuous), depth: 3)
                }
                .buttonStyle(BubblePress())
                .accessibilityIdentifier("textUppercase")
            }
        }
    }

    // MARK: Reading / writing

    private func load() {
        guard !loaded else { return }
        loaded = true
        CaptionFontLoader.registerIfNeeded()
        switch target {
        case .captions:
            look = model.captionLookForSheet
            font = model.captionFont
            tab = .style
        case .text(let id):
            look = model.textLookForSheet(id)
            font = model.textOverlay(id).flatMap { model.textFont($0) } ?? "inter"
            text = model.textOverlay(id)?["text"]?.stringValue ?? ""
        case .newText:
            var start = TextLook(); start.strokeColor = 0x000000; start.strokeWidth = 4; start.preset = "outline"
            look = start
            font = "montserrat"
            typing = true
        }
    }

    private func binding(_ path: WritableKeyPath<TextLook, Double>) -> Binding<Double> {
        Binding(get: { look[keyPath: path] }, set: { value in look[keyPath: path] = value; push(commit: false) })
    }

    private var sizeBinding: Binding<Double> {
        Binding(get: { currentScale }, set: { liveScale = $0 })
    }
    @State private var liveScale: Double?
    private var currentScale: Double {
        if let liveScale { return liveScale }
        switch target {
        case .captions: return model.captionScale
        case .text(let id): return model.textOverlay(id)?["scale"]?.doubleValue ?? 1
        case .newText: return newScale
        }
    }
    @State private var newScale: Double = 1.2

    private func commitSize() {
        guard let value = liveScale else { return }
        liveScale = nil
        switch target {
        case .captions: model.applyCaptionSettings(scale: min(2, max(0.5, value)))
        case .text(let id): Task { await model.updateTextOverlay(id: id, scale: min(3, max(0.5, value))) }
        case .newText: newScale = value
        }
    }

    private func change(_ edit: (inout TextLook) -> Void) {
        edit(&look)
        push(commit: true)
    }

    private func commit() { push(commit: true) }

    private func push(commit: Bool) {
        switch target {
        case .captions: model.setCaptionLook(look, commit: commit)
        case .text(let id): model.setTextLook(id: id, look, commit: commit)
        case .newText: break     // saved with the text on ✓
        }
    }

    private func pickFont(_ key: String) {
        font = key
        switch target {
        case .captions: model.applyCaptionSettings(font: key)
        case .text(let id): model.setTextFont(id: id, key)
        case .newText: break
        }
    }

    @MainActor
    private func done() async {
        typing = false
        switch target {
        case .captions:
            onDone(nil)
        case .text(let id):
            let typed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !typed.isEmpty, typed != model.textOverlay(id)?["text"]?.stringValue {
                await model.updateTextOverlay(id: id, text: typed)
            }
            onDone(nil)
        case .newText:
            let typed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !typed.isEmpty else { onDone(nil); return }
            saving = true
            let before = Set(model.textOverlays.compactMap { $0["overlay_id"]?.stringValue })
            let end = min(max(1, model.timelineDuration), playhead + 3)
            await model.addTextOverlay(text: typed, start: max(0, min(playhead, end - 0.2)), end: end)
            let added = model.textOverlays.compactMap { $0["overlay_id"]?.stringValue }.first { !before.contains($0) }
            if let added {
                var fields: [String: JSONValue] = ["font": .string(font), "style": look.json]
                if abs(newScale - 1) > 0.01 { fields["scale"] = .number(min(3, max(0.5, newScale))) }
                await model.setTextOverlayFields(id: added, fields)
            }
            saving = false
            onDone(added)
        }
    }
}

/// Every colour: "none" (when it can be off), the colour wheel, then the swatches.
struct ColorRow: View {
    let selected: UInt32?
    let allowsNone: Bool
    let onPick: (UInt32?) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                if allowsNone {
                    Button { onPick(nil) } label: {
                        ZStack {
                            Circle().fill(Color(fixed: 0xFFFFFF))
                            Rectangle().fill(Color(fixed: 0xFF3B30)).frame(width: 2.5, height: 34).rotationEffect(.degrees(45))
                        }
                        .frame(width: 38, height: 38).clipShape(Circle())
                        .overlay(Circle().stroke(selected == nil ? CS.accent : CS.wiBorder, lineWidth: selected == nil ? 3 : 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("colorNone")
                }
                ColorPicker("", selection: Binding(get: { Color(lookHex: selected ?? 0xFFFFFF) },
                                                   set: { onPick($0.lookHex) }), supportsOpacity: false)
                    .labelsHidden()
                    .frame(width: 38, height: 38)
                    .accessibilityIdentifier("colorWheel")
                ForEach(TextLook.palette, id: \.self) { hex in
                    let on = selected == hex
                    Button { onPick(hex) } label: {
                        Circle().fill(Color(lookHex: hex)).frame(width: 38, height: 38)
                            .overlay(Circle().stroke(CS.wiBorder, lineWidth: hex == 0xFFFFFF ? 1 : 0))
                            .padding(3)
                            .overlay(Circle().stroke(on ? CS.accent : .clear, lineWidth: 3))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier(String(format: "color_%06X", hex))
                }
            }
            .padding(.vertical, 4)
        }
    }
}

/// "Size  ———●———  120%": saves when the finger lifts.
struct LookSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let shown: (Double) -> String
    let onCommit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(L10n.tr(title)).font(CS.inter(14, .semibold)).foregroundStyle(CS.wiInk)
                Spacer()
                Text(verbatim: shown(value)).font(CS.inter(13, .semibold)).foregroundStyle(CS.wiSecondary).monospacedDigit()
            }
            Slider(value: $value, in: range) { editing in if !editing { onCommit() } }
                .tint(CS.accent)
                .accessibilityIdentifier("lookSlider_\(title)")
        }
    }
}

/// A text on the video in the editor: drawn like the export; selected = dashed box, drag / pinch.
struct TextOverlayOnVideo: View {
    let item: [String: JSONValue]
    let look: TextLook?
    let font: String?
    let frame: CGSize
    let isSelected: Bool
    let onTap: () -> Void
    let onMoved: (Double, Double, Double) -> Void

    @State private var drag: CGSize = .zero
    @State private var pinch: Double = 1

    var body: some View {
        let x = item["x"]?.doubleValue ?? 0.5
        let y = item["y"]?.doubleValue ?? 0.2
        let scale = item["scale"]?.doubleValue ?? 1
        var shown = item
        shown["scale"] = .number(min(3, max(0.5, scale * pinch)))
        return TextOverlayLabel(item: shown, look: look, font: font)
            .overlay {
                if isSelected {
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color.white.opacity(0.9), style: StrokeStyle(lineWidth: 1.2, dash: [5, 3]))
                        .padding(-6)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture(perform: onTap)
            .gesture(DragGesture(minimumDistance: 4).simultaneously(with: MagnifyGesture())
                .onChanged { value in
                    if let d = value.first { drag = d.translation }
                    if let m = value.second { pinch = Double(m.magnification) }
                }
                .onEnded { _ in
                    let nx = min(1, max(0, x + Double(drag.width / max(1, frame.width))))
                    let ny = min(1, max(0, y + Double(drag.height / max(1, frame.height))))
                    let ns = min(3, max(0.5, scale * pinch))
                    drag = .zero; pinch = 1
                    onMoved(nx, ny, ns)
                }, including: isSelected ? .all : .subviews)
            .position(x: frame.width * CGFloat(x) + drag.width, y: frame.height * CGFloat(y) + drag.height)
            .accessibilityIdentifier("textOverlay_\(item["overlay_id"]?.stringValue ?? "")")
    }
}
