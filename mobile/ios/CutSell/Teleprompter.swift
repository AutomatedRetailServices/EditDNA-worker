import SwiftUI

// MARK: - Teleprompter (point 7)
//
// Camera ("Record with CutSell") side rail → Script. The script sits at the top of the preview, close to the
// lens, and scrolls by itself while recording (paused between clips). Drag it to move back or forward. Speed,
// text size, edit and restart are under it. The script stays on this iPhone. No Figma frame exists for it: it
// uses the camera's own look (black glass, white SF text, cyan when on).

@MainActor
final class TeleprompterStore: ObservableObject {
    static let shared = TeleprompterStore()
    private enum Key { static let script = "cutsell.prompter.script", on = "cutsell.prompter.on",
                                  speed = "cutsell.prompter.speed", size = "cutsell.prompter.size" }

    @Published var script: String { didSet { UserDefaults.standard.set(script, forKey: Key.script) } }
    @Published var isOn: Bool { didSet { UserDefaults.standard.set(isOn, forKey: Key.on) } }
    /// 1 (slow) … 5 (fast).
    @Published var speed: Int { didSet { UserDefaults.standard.set(speed, forKey: Key.speed) } }
    /// 0 small · 1 medium · 2 large.
    @Published var size: Int { didSet { UserDefaults.standard.set(size, forKey: Key.size) } }

    private init() {
        let d = UserDefaults.standard
        script = d.string(forKey: Key.script) ?? ""
        isOn = d.bool(forKey: Key.on)
        speed = (d.object(forKey: Key.speed) as? Int).map { min(max($0, 1), 5) } ?? 3
        size = (d.object(forKey: Key.size) as? Int).map { min(max($0, 0), 2) } ?? 1
    }

    var hasScript: Bool { !script.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    /// Points per second at each speed (about 1.5 to 3.5 words a second at the medium size).
    var pointsPerSecond: CGFloat { [14, 22, 32, 44, 58][speed - 1] }
    var fontSize: CGFloat { [22, 28, 34][size] }
}

struct TeleprompterOverlay: View {
    @ObservedObject var store = TeleprompterStore.shared
    /// Scrolls only while this is true (the camera is recording).
    let running: Bool
    let onEdit: () -> Void

    @State private var offset: CGFloat = 0
    @State private var lastTick: Date?
    @State private var dragBase: CGFloat?
    @State private var textHeight: CGFloat = 0

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { geo in
                TimelineView(.animation(minimumInterval: 1 / 60, paused: !running || dragBase != nil)) { context in
                    Text(store.script)
                        .font(.system(size: store.fontSize, weight: .semibold))
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                        .lineSpacing(store.fontSize * 0.25)
                        .frame(width: max(0, geo.size.width - 28))
                        .fixedSize(horizontal: false, vertical: true)
                        .background(GeometryReader { text in
                            Color.clear.onAppear { textHeight = text.size.height }
                                .onChange(of: text.size.height) { _, h in textHeight = h }
                        })
                        .padding(.top, geo.size.height * 0.3)
                        .frame(width: geo.size.width, alignment: .top)
                        .offset(y: -offset)
                        .onChange(of: context.date) { _, now in tick(now, visible: geo.size.height) }
                }
            }
            .clipped()
            .mask(LinearGradient(stops: [.init(color: .clear, location: 0), .init(color: .black, location: 0.16),
                                         .init(color: .black, location: 0.84), .init(color: .clear, location: 1)],
                                 startPoint: .top, endPoint: .bottom))
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 4)
                .onChanged { value in
                    if dragBase == nil { dragBase = offset }
                    offset = clamp((dragBase ?? 0) - value.translation.height)
                }
                .onEnded { _ in dragBase = nil; lastTick = nil })
            .accessibilityIdentifier("prompterText")

            controls
        }
        .background(Color.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Color.white.opacity(0.14), lineWidth: 1))
        .onChange(of: running) { _, _ in lastTick = nil }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("prompterOverlay")
    }

    private var controls: some View {
        HStack(spacing: 6) {
            iconButton("arrow.counterclockwise", id: "prompterRestart", label: "Start over") {
                offset = 0; lastTick = nil
            }
            Spacer(minLength: 0)
            iconButton("minus", id: "prompterSlower", label: "Slower") { store.speed = max(1, store.speed - 1) }
            Text(L10n.tr("Speed \(store.speed)"))
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white)
                .frame(minWidth: 72)
                .accessibilityIdentifier("prompterSpeed")
            iconButton("plus", id: "prompterFaster", label: "Faster") { store.speed = min(5, store.speed + 1) }
            Spacer(minLength: 0)
            iconButton("textformat.size", id: "prompterSize", label: "Text size") { store.size = (store.size + 1) % 3 }
            iconButton("pencil", id: "prompterEdit", label: "Edit script", action: onEdit)
        }
        .padding(.horizontal, 8)
        .frame(height: 44)
    }

    private func iconButton(_ symbol: String, id: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 40, height: 40)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(L10n.tr(label))
        .accessibilityIdentifier(id)
    }

    private func tick(_ now: Date, visible: CGFloat) {
        guard running, dragBase == nil else { lastTick = nil; return }
        defer { lastTick = now }
        guard let last = lastTick else { return }
        let dt = min(0.1, now.timeIntervalSince(last))
        offset = clamp(offset + store.pointsPerSecond * CGFloat(dt))
    }

    /// From the first line to the last line reaching the top third.
    private func clamp(_ value: CGFloat) -> CGFloat { min(max(0, value), max(0, textHeight)) }
}

/// Write or paste the script.
struct TeleprompterEditor: View {
    @ObservedObject var store = TeleprompterStore.shared
    @Environment(\.dismiss) private var dismiss
    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text(L10n.tr("Write what you'll say. It scrolls by itself while you record, next to the camera."))
                    .font(.system(size: 13))
                    .foregroundStyle(CS.textSecondary)
                ZStack(alignment: .topLeading) {
                    if draft.isEmpty {
                        Text(L10n.tr("Hi! Today I'm showing you…"))
                            .font(.system(size: 17))
                            .foregroundStyle(CS.textMuted)
                            .padding(.horizontal, 5).padding(.vertical, 8)
                    }
                    TextEditor(text: $draft)
                        .font(.system(size: 17))
                        .foregroundStyle(.white)
                        .scrollContentBackground(.hidden)
                        .focused($focused)
                        .accessibilityIdentifier("prompterEditorText")
                }
                .padding(10)
                .background(CS.fieldBackground, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.fieldBorder, lineWidth: 1))
                HStack {
                    Button {
                        if let text = UIPasteboard.general.string { draft = text }
                    } label: {
                        Label(L10n.tr("Paste"), systemImage: "doc.on.clipboard")
                            .font(.system(size: 14, weight: .semibold))
                    }
                    .accessibilityIdentifier("prompterPaste")
                    Spacer()
                    Text(L10n.tr("\(wordCount) words · about \(minutesText)"))
                        .font(.system(size: 12))
                        .foregroundStyle(CS.textMuted)
                }
            }
            .padding(16)
            .background(CS.sheet.ignoresSafeArea())
            .navigationTitle(L10n.tr("Script"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(L10n.tr("Cancel")) { dismiss() }.accessibilityIdentifier("prompterCancel")
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(L10n.tr("Done")) {
                        store.script = draft
                        store.isOn = store.hasScript
                        dismiss()
                    }
                    .fontWeight(.semibold)
                    .accessibilityIdentifier("prompterDone")
                }
            }
        }
        .preferredColorScheme(.dark)
        .onAppear { draft = store.script; focused = draft.isEmpty }
    }

    private var wordCount: Int { draft.split { $0.isWhitespace || $0.isNewline }.count }
    /// Spoken at about 150 words a minute.
    private var minutesText: String {
        let seconds = Int((Double(wordCount) / 150 * 60).rounded())
        return seconds < 60 ? "\(seconds) s" : "\(seconds / 60) min \(seconds % 60) s"
    }
}
