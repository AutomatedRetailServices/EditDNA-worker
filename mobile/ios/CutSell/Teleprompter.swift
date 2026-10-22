import SwiftUI

// MARK: - Teleprompter (Figma 12_Prototype: script sheet 1145:2988 · recording 1151:2824 over camera 645:23)
//
// Camera side rail: Script (show / hide) and Speed (0.5× – 2.0×). The script is saved per product on this
// iPhone. While recording, the text sits near the lens on a fading shadow (no box): the current line large and
// white, the others smaller and see-through, a gradient progress bar under them. It moves only while the
// camera records (after 3·2·1), so it stops between clips. Tap it to show the dashed frame (drag to move, pull a
// corner to resize), ✎ edit, Auto-scroll (pause / go on) and ⚙ (speed, size, color); they hide after 2 s.
// Frame position and size are kept for next time.

/// Who the script is for: the product (and, inside a Mix, the section being recorded).
struct TeleprompterContext: Equatable {
    var productID: String?
    var productLabel: String?
    var mixSection: String?
}

@MainActor
final class TeleprompterStore: ObservableObject {
    static let shared = TeleprompterStore()
    private enum Key {
        static let scripts = "cutsell.prompter.scripts", parts = "cutsell.prompter.parts", on = "cutsell.prompter.on"
        static let speed = "cutsell.prompter.speed2", size = "cutsell.prompter.size2", color = "cutsell.prompter.color"
        static let countdown = "cutsell.prompter.countdown", frame = "cutsell.prompter.frame"
        static let product = "cutsell.prompter.product", productLabel = "cutsell.prompter.productLabel"
    }
    static let general = "_general"
    static let speeds: [Double] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    static let colors: [(id: String, name: String, hex: UInt32)] = [("white", "White", 0xFFFFFF),
                                                                     ("yellow", "Yellow", 0xFFE14D),
                                                                     ("cyan", "Cyan", 0x19E6FF)]
    /// Figma 1151:2860: x 22, y 86, 346 × 206 on a 390 × 844 screen.
    static let defaultFrame = CGRect(x: 22, y: 86, width: 346, height: 206)

    /// Script per product id (`general` when no product is chosen).
    @Published private(set) var scripts: [String: String]
    /// "✦ Split for Mix": per product, the text of each Mix section (hook, body, benefit, cta).
    @Published private(set) var parts: [String: [String: String]]
    @Published var isOn: Bool { didSet { d.set(isOn, forKey: Key.on) } }
    @Published var speed: Double { didSet { d.set(speed, forKey: Key.speed) } }
    /// Size of the current line in the sheet's terms (Figma 28 = current line 23 pt, others 17 pt).
    @Published var size: Int { didSet { d.set(size, forKey: Key.size) } }
    @Published var colorID: String { didSet { d.set(colorID, forKey: Key.color) } }
    @Published var countdown: Bool { didSet { d.set(countdown, forKey: Key.countdown) } }
    @Published var frame: CGRect { didSet { d.set([frame.minX, frame.minY, frame.width, frame.height], forKey: Key.frame) } }
    /// The product of the camera outside a Mix (last one picked in the sheet).
    @Published var productID: String? { didSet { d.set(productID, forKey: Key.product) } }
    @Published var productLabel: String? { didSet { d.set(productLabel, forKey: Key.productLabel) } }
    /// Inside a Mix: the section being recorded (its split part is shown when there is one).
    @Published var context = TeleprompterContext()
    /// "Try the teleprompter now": the camera opens with the sheet.
    @Published var openSheetNext = false

    private let d = UserDefaults.standard

    private init() {
        let d = UserDefaults.standard
        scripts = (d.data(forKey: Key.scripts)).flatMap { try? JSONDecoder().decode([String: String].self, from: $0) } ?? [:]
        parts = (d.data(forKey: Key.parts)).flatMap { try? JSONDecoder().decode([String: [String: String]].self, from: $0) } ?? [:]
        isOn = d.bool(forKey: Key.on)
        let s = d.object(forKey: Key.speed) as? Double ?? 1.0
        speed = min(max(s, 0.5), 2.0)
        size = min(max(d.object(forKey: Key.size) as? Int ?? 28, 18), 40)
        colorID = d.string(forKey: Key.color) ?? "white"
        countdown = d.object(forKey: Key.countdown) as? Bool ?? true
        if let f = d.array(forKey: Key.frame) as? [Double], f.count == 4 {
            frame = CGRect(x: f[0], y: f[1], width: f[2], height: f[3])
        } else {
            frame = Self.defaultFrame
        }
        productID = d.string(forKey: Key.product)
        productLabel = d.string(forKey: Key.productLabel)
    }

    var key: String { (context.productID ?? productID) ?? Self.general }
    var label: String? { context.productLabel ?? productLabel }

    func script(for key: String) -> String { scripts[key] ?? "" }
    func setScript(_ text: String, for key: String) {
        scripts[key] = text
        if let data = try? JSONEncoder().encode(scripts) { d.set(data, forKey: Key.scripts) }
    }
    func setParts(_ value: [String: String], for key: String) {
        parts[key] = value
        if let data = try? JSONEncoder().encode(parts) { d.set(data, forKey: Key.parts) }
    }

    /// What the camera shows: inside a Mix, that section's split part when there is one; else the product's script.
    var shownScript: String {
        if let section = context.mixSection?.lowercased(), let part = parts[key]?[Self.partKey(section)],
           !part.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return part
        }
        return script(for: key)
    }
    var hasScript: Bool { !shownScript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    /// Mix section names (also in Spanish) → the split part.
    static func partKey(_ section: String) -> String {
        switch section {
        case "hook", "gancho": return "hook"
        case "body", "cuerpo": return "body"
        case "benefit", "beneficio": return "benefit"
        case "cta": return "cta"
        default: return section
        }
    }

    var textColor: Color { Color(fixed: Self.colors.first { $0.id == colorID }?.hex ?? 0xFFFFFF) }
    /// "1.0×", "1.25×", "0.75×"
    var speedText: String {
        let hundredths = Int((speed * 100).rounded())
        return hundredths % 10 == 0 ? String(format: "%.1f×", speed) : String(format: "%.2f×", speed)
    }

    func nextSpeed() {
        speed = Self.speeds.first { $0 > speed + 0.01 } ?? Self.speeds[0]
    }

    /// Sentences, in order (the current one is drawn large).
    static func lines(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in text.replacingOccurrences(of: "\n", with: " ") {
            current.append(ch)
            if ".!?…".contains(ch) {
                let line = current.trimmingCharacters(in: .whitespaces)
                if !line.isEmpty { out.append(line) }
                current = ""
            }
        }
        let rest = current.trimmingCharacters(in: .whitespaces)
        if !rest.isEmpty { out.append(rest) }
        return out
    }

    /// Seconds a line stays current at 1.0× (about 2.6 words a second, at least 1.2 s).
    static func seconds(for line: String, speed: Double) -> Double {
        let words = max(1, line.split(whereSeparator: { $0.isWhitespace }).count)
        return max(1.2, Double(words) / 2.6) / max(0.5, speed)
    }
}

// MARK: - Recording overlay (1151:2824)

struct TeleprompterOverlay: View {
    @ObservedObject var store = TeleprompterStore.shared
    /// The camera is recording (after 3·2·1).
    let running: Bool
    let onEdit: () -> Void

    @State private var position: Double = 0          // line index + fraction
    @State private var lastTick: Date?
    @State private var paused = false                // Auto-scroll off
    @State private var showControls = false
    @State private var lastTouch = Date()
    @State private var dragStart: CGRect?
    @State private var showingSettings = false

    private var lines: [String] { TeleprompterStore.lines(store.shownScript) }
    private var current: Int { min(max(0, Int(position)), max(0, lines.count - 1)) }
    private var progress: CGFloat {
        guard lines.count > 0 else { return 0 }
        return min(1, CGFloat(position) / CGFloat(lines.count))
    }
    private var big: CGFloat { CGFloat(store.size) * 23 / 28 }
    private var small: CGFloat { CGFloat(store.size) * 17 / 28 }

    var body: some View {
        GeometryReader { geo in
            let scale = geo.size.width / 390
            let f = CGRect(x: store.frame.minX * scale, y: store.frame.minY, width: store.frame.width * scale,
                           height: store.frame.height)
            ZStack(alignment: .topLeading) {
                // Script/Fade: black 82 % → 55 % (70 %) → clear, 300 high, behind the text
                LinearGradient(stops: [.init(color: .black.opacity(0.82), location: 0),
                                       .init(color: .black.opacity(0.55), location: 0.7),
                                       .init(color: .clear, location: 1)],
                               startPoint: .top, endPoint: .bottom)
                    .frame(height: max(300, f.maxY + 20))
                    .ignoresSafeArea(edges: .top)
                    .allowsHitTesting(false)

                TimelineView(.animation(minimumInterval: 1 / 30, paused: !running || paused)) { context in
                    linesView(width: f.width)
                        .onChange(of: context.date) { _, now in tick(now) }
                }
                .frame(width: f.width, height: f.height - 14, alignment: .center)
                .clipped()
                .offset(x: f.minX, y: f.minY)
                .contentShape(Rectangle())
                .onTapGesture { touch() }
                .gesture(showControls ? moveGesture(scale: scale) : nil)
                .accessibilityIdentifier("prompterText")

                // Progress (Figma: 2 high, white 20 %, gradient fill)
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.2))
                    Capsule().fill(CS.brandStroke).frame(width: max(2, (f.width - 16) * progress))
                }
                .frame(width: f.width - 16, height: 2)
                .offset(x: f.minX + 8, y: f.maxY - 12)
                .allowsHitTesting(false)

                if showControls { controls(f, scale: scale) }
            }
        }
        .onChange(of: running) { _, _ in lastTick = nil }
        .onChange(of: store.shownScript) { _, _ in position = 0 }
        .task(id: lastTouch) {
            guard showControls else { return }
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            if Date().timeIntervalSince(lastTouch) >= 1.9 { withAnimation(.easeOut(duration: 0.2)) { showControls = false } }
        }
        .sheet(isPresented: $showingSettings) { TeleprompterSettings().presentationDetents([.height(330)]) }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("prompterOverlay")
    }

    private func linesView(width: CGFloat) -> some View {
        VStack(spacing: 6) {
            ForEach(Array(lines.enumerated()), id: \.offset) { i, line in
                if i >= current - 1 && i <= current + 2 {
                    Text(line)
                        .font(i == current ? CS.inter(big, .bold) : CS.inter(small, .medium))
                        .foregroundStyle(store.textColor.opacity(opacity(i)))
                        .multilineTextAlignment(.center)
                        .lineSpacing(i == current ? big * 0.12 : small * 0.12)
                        .frame(width: max(0, width - 16))
                        .fixedSize(horizontal: false, vertical: true)
                        .id(i)
                }
            }
        }
        .animation(.easeOut(duration: 0.25), value: current)
    }

    private func opacity(_ i: Int) -> Double {
        switch i - current {
        case -1: return 0.4
        case 0: return 1
        case 1: return 0.6
        default: return 0.3
        }
    }

    // Tap: dashed frame + 4 corners + grabber, ✎ · Auto-scroll · ⚙ above it, the hint under it
    @ViewBuilder
    private func controls(_ f: CGRect, scale: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 16)
            .stroke(Color.white.opacity(0.28), style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
            .frame(width: f.width, height: f.height)
            .offset(x: f.minX, y: f.minY)
            .allowsHitTesting(false)
        Capsule().fill(Color.white.opacity(0.5)).frame(width: 36, height: 4)
            .offset(x: f.midX - 18, y: f.minY - 6)
            .allowsHitTesting(false)
        ForEach(0..<4, id: \.self) { corner in
            let x = corner % 2 == 0 ? f.minX : f.maxX
            let y = corner < 2 ? f.minY : f.maxY
            Circle().fill(.white).overlay(Circle().stroke(Color(fixed: 0x19E6FF), lineWidth: 2))
                .frame(width: 8, height: 8)
                .frame(width: 34, height: 34)
                .contentShape(Rectangle())
                .offset(x: x - 17, y: y - 17)
                .gesture(resizeGesture(corner: corner, scale: scale))
                .accessibilityIdentifier("prompterCorner\(corner)")
        }
        pill(width: 34) { Text(verbatim: "✎").font(CS.inter(12, .semibold)).foregroundStyle(.white) }
            .onTapGesture { touch(); onEdit() }
            .offset(x: f.minX, y: f.minY - 34)
            .accessibilityIdentifier("prompterEdit")
        pill(width: 120) {
            Text(verbatim: (paused ? "▶  " : "❚❚  ") + L10n.tr("Auto-scroll"))
                .font(CS.inter(12, .semibold)).foregroundStyle(paused ? Color.white : Color(fixed: 0x19E6FF))
                .lineLimit(1).minimumScaleFactor(0.8)
        }
        .onTapGesture { touch(); paused.toggle(); lastTick = nil }
        .offset(x: f.midX - 60, y: f.minY - 34)
        .accessibilityIdentifier("prompterAutoScroll")
        pill(width: 34) { Text(verbatim: "⚙").font(CS.inter(12, .semibold)).foregroundStyle(.white) }
            .onTapGesture { touch(); showingSettings = true }
            .offset(x: f.maxX - 34, y: f.minY - 34)
            .accessibilityIdentifier("prompterSettings")
        Text(L10n.tr("Drag to move · pull a corner to resize"))
            .font(CS.inter(11, .medium)).foregroundStyle(Color.white.opacity(0.8))
            .frame(width: f.width)
            .offset(x: f.minX, y: f.maxY + 10)
            .allowsHitTesting(false)
    }

    private func pill<Content: View>(width: CGFloat, @ViewBuilder content: () -> Content) -> some View {
        content()
            .frame(width: width, height: 26)
            .background(Color.black.opacity(0.5), in: Capsule())
            .overlay(Capsule().stroke(Color.white.opacity(0.2), lineWidth: 1))
            .contentShape(Capsule())
    }

    private func touch() {
        lastTouch = Date()
        if !showControls { withAnimation(.easeOut(duration: 0.15)) { showControls = true } }
    }

    private func moveGesture(scale: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 3)
            .onChanged { value in
                if dragStart == nil { dragStart = store.frame }
                guard let start = dragStart else { return }
                var next = start
                next.origin.x = min(max(0, start.minX + value.translation.width / scale), 390 - start.width)
                next.origin.y = min(max(40, start.minY + value.translation.height), 700 - start.height)
                store.frame = next
                lastTouch = Date()
            }
            .onEnded { _ in dragStart = nil }
    }

    private func resizeGesture(corner: Int, scale: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 1)
            .onChanged { value in
                if dragStart == nil { dragStart = store.frame }
                guard let s = dragStart else { return }
                let dx = value.translation.width / scale, dy = value.translation.height
                var minX = s.minX, minY = s.minY, maxX = s.maxX, maxY = s.maxY
                if corner % 2 == 0 { minX = min(s.maxX - 180, max(0, s.minX + dx)) } else { maxX = max(s.minX + 180, min(390, s.maxX + dx)) }
                if corner < 2 { minY = min(s.maxY - 110, max(40, s.minY + dy)) } else { maxY = max(s.minY + 110, min(700, s.maxY + dy)) }
                store.frame = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
                lastTouch = Date()
            }
            .onEnded { _ in dragStart = nil }
    }

    private func tick(_ now: Date) {
        guard running, !paused, !lines.isEmpty else { lastTick = nil; return }
        defer { lastTick = now }
        guard let last = lastTick else { return }
        let dt = min(0.1, now.timeIntervalSince(last))
        let line = lines[current]
        let step = dt / TeleprompterStore.seconds(for: line, speed: store.speed)
        position = min(Double(lines.count), position + step)
    }
}

/// ⚙: speed, text size and text color.
struct TeleprompterSettings: View {
    @ObservedObject var store = TeleprompterStore.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Color(fixed: 0x4D5770)).frame(width: 40, height: 5).frame(maxWidth: .infinity).padding(.top, 9)
            Text(L10n.tr("Script settings")).font(CS.inter(22, .bold)).foregroundStyle(.white).padding(.top, 15)
            ScriptStepperRow(title: "Scroll speed", value: store.speedText, id: "prompterSettingsSpeed",
                             minus: { store.speed = max(0.5, ((store.speed - 0.1) * 10).rounded() / 10) },
                             plus: { store.speed = min(2.0, ((store.speed + 0.1) * 10).rounded() / 10) })
                .padding(.top, 22)
            ScriptStepperRow(title: "Text size", value: "\(store.size)", id: "prompterSettingsSize",
                             minus: { store.size = max(18, store.size - 2) }, plus: { store.size = min(40, store.size + 2) })
                .padding(.top, 14)
            HStack {
                Text(L10n.tr("Text color")).font(CS.inter(14, .semibold)).foregroundStyle(.white)
                Spacer()
                ForEach(TeleprompterStore.colors, id: \.id) { color in
                    Button { store.colorID = color.id } label: {
                        Circle().fill(Color(fixed: color.hex)).frame(width: 26, height: 26)
                            .overlay(Circle().stroke(store.colorID == color.id ? AnyShapeStyle(CS.brandStroke)
                                                                                  : AnyShapeStyle(Color(fixed: 0x33405C)),
                                                     lineWidth: store.colorID == color.id ? 3 : 1).frame(width: 34, height: 34))
                            .frame(width: 40, height: 40)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(L10n.tr(color.name))
                    .accessibilityIdentifier("prompterColor_\(color.id)")
                }
            }
            .padding(.top, 14)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 23)
        .presentationBackground(Color(fixed: 0x0D121F))
        .presentationCornerRadius(28)
        .presentationDragIndicator(.hidden)
        .preferredColorScheme(.dark)
    }
}

/// "Scroll speed  −  1.0×  +" (Figma 1145:3003-3008): − on #1F2638, + on the gradient, value cyan bold 14.
struct ScriptStepperRow: View {
    let title: String
    let value: String
    let id: String
    let minus: () -> Void
    let plus: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            Text(L10n.tr(title)).font(CS.inter(14, .semibold)).foregroundStyle(.white)
            Spacer()
            Button(action: minus) {
                Text(verbatim: "−").font(CS.inter(18, .bold)).foregroundStyle(.white)
                    .frame(width: 36, height: 36).background(Color(fixed: 0x1F2638), in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(L10n.tr("Less"))
            .accessibilityIdentifier(id + "Minus")
            Text(verbatim: value).font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0x19E6FF))
                .frame(width: 62)
                .accessibilityIdentifier(id + "Value")
            Button(action: plus) {
                Text(verbatim: "+").font(CS.inter(18, .bold)).foregroundStyle(.white)
                    .frame(width: 36, height: 36).background(CS.brandStroke, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(L10n.tr("More"))
            .accessibilityIdentifier(id + "Plus")
        }
        .frame(height: 36)
    }
}

/// The gradient switch of 1145:3017 (44 × 26), grey when off.
struct GradientSwitch: View {
    @Binding var isOn: Bool
    var body: some View {
        Button { withAnimation(.easeInOut(duration: 0.15)) { isOn.toggle() } } label: {
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule().fill(isOn ? AnyShapeStyle(CS.brandStroke) : AnyShapeStyle(Color(fixed: 0x33405C)))
                Circle().fill(.white).frame(width: 20, height: 20).padding(3)
            }
            .frame(width: 44, height: 26)
        }
        .buttonStyle(.plain)
        .accessibilityValue(isOn ? "On" : "Off")
    }
}

// MARK: - Script sheet (1145:2988)

struct TeleprompterSheet: View {
    @ObservedObject var store = TeleprompterStore.shared
    @ObservedObject private var catalog = CatalogStore.shared
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    /// "Record with script": the camera starts (after 3·2·1 when that is on).
    let onRecord: () -> Void

    @State private var draft = ""
    @State private var splitting = false
    @State private var notice: String?
    @FocusState private var focused: Bool

    private var inMix: Bool { store.context.productID != nil }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                Capsule().fill(Color(fixed: 0x4D5770)).frame(width: 40, height: 5).frame(maxWidth: .infinity).padding(.top, 9)
                Text(L10n.tr("Your script")).font(CS.inter(22, .bold)).foregroundStyle(.white).padding(.top, 15)
                Text(L10n.tr("It scrolls near the lens while you record.")).font(CS.inter(13)).foregroundStyle(Color(fixed: 0x99A3BD))
                    .padding(.top, 5)
                productChip.padding(.top, 14)
                ZStack(alignment: .topLeading) {
                    if draft.isEmpty {
                        Text(L10n.tr("Paste or write what you'll say…"))
                            .font(CS.inter(14)).foregroundStyle(Color(fixed: 0x99A3BD))
                            .padding(.horizontal, 5).padding(.vertical, 8)
                    }
                    TextEditor(text: $draft)
                        .font(CS.inter(14))
                        .foregroundStyle(.white)
                        .scrollContentBackground(.hidden)
                        .focused($focused)
                        .accessibilityIdentifier("prompterEditorText")
                }
                .padding(.horizontal, 9.5)
                .padding(.vertical, 4.5)
                .frame(height: 200)
                .background(Color(fixed: 0x0F1726), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.brandStroke, lineWidth: 1.5))
                .padding(.top, 12)
                HStack(spacing: 8) {
                    Button {
                        if let text = UIPasteboard.general.string { draft = text }
                    } label: {
                        Text(L10n.tr("Paste")).font(CS.inter(13, .semibold)).foregroundStyle(.white)
                            .frame(width: 96, height: 32)
                            .overlay(Capsule().stroke(Color(fixed: 0x596685), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("prompterPaste")
                    Button { Task { await split() } } label: {
                        Group {
                            if splitting { ProgressView().tint(.white) }
                            else { Text(L10n.tr("✦ Split for Mix")).font(CS.inter(13, .semibold)).foregroundStyle(.white) }
                        }
                        .frame(maxWidth: .infinity).frame(height: 32)
                        .background(Color(fixed: 0x121A2B), in: Capsule())
                        .overlay(Capsule().stroke(CS.brandStroke, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .disabled(splitting || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("prompterSplit")
                }
                .padding(.top, 12)
                if let notice {
                    Text(L10n.tr(notice)).font(CS.inter(12, .semibold)).foregroundStyle(Color(fixed: 0x19E6FF))
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 8)
                        .accessibilityIdentifier("prompterNotice")
                }
                ScriptStepperRow(title: "Scroll speed", value: store.speedText, id: "prompterSpeed",
                                 minus: { store.speed = max(0.5, ((store.speed - 0.1) * 10).rounded() / 10) },
                                 plus: { store.speed = min(2.0, ((store.speed + 0.1) * 10).rounded() / 10) })
                    .padding(.top, 18)
                ScriptStepperRow(title: "Text size", value: "\(store.size)", id: "prompterSize",
                                 minus: { store.size = max(18, store.size - 2) }, plus: { store.size = min(40, store.size + 2) })
                    .padding(.top, 14)
                HStack {
                    Text(L10n.tr("Start after 3·2·1")).font(CS.inter(14, .semibold)).foregroundStyle(.white)
                    Spacer()
                    GradientSwitch(isOn: $store.countdown).accessibilityIdentifier("prompterCountdown")
                }
                .padding(.leading, 13).padding(.trailing, 16)
                .frame(height: 44)
                .background(Color(fixed: 0x0F1726), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(fixed: 0x33405C), lineWidth: 1))
                .padding(.top, 14)
                Button {
                    save()
                    store.isOn = store.hasScript
                    dismiss()
                    if store.hasScript { onRecord() }
                } label: {
                    Text(L10n.tr("Record with script"))
                        .font(.system(size: 15, weight: .semibold)).foregroundStyle(.white)
                        .frame(maxWidth: .infinity).frame(height: 50)
                        .background(IntroView.buttonGradient, in: RoundedRectangle(cornerRadius: 25, style: .continuous))
                        .shadow(color: Color(fixed: 0x1969FF, opacity: 0.25), radius: 9, x: 0, y: 6)
                }
                .buttonStyle(.plain)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .accessibilityIdentifier("prompterRecord")
                .padding(.top, 26)
            }
            .padding(.horizontal, 23)
            .padding(.bottom, 24)
        }
        .scrollDismissesKeyboard(.interactively)
        .presentationDetents([.height(694), .large])
        .presentationBackground(Color(fixed: 0x0D121F))
        .presentationCornerRadius(28)
        .presentationDragIndicator(.hidden)
        .preferredColorScheme(.dark)
        .onAppear { draft = store.script(for: store.key) }
        .onDisappear { save() }
        .task { await catalog.refresh(session: appState.session, projects: appState.projects) }
    }

    /// "Product: Rhino · Retractable Straps" (#1F2638, 30 high). Outside a Mix it picks the product of the script.
    private var productChip: some View {
        Menu {
            Button(L10n.tr("No product")) { pick(nil, nil) }
            ForEach(catalog.allProducts, id: \.product.productID) { item in
                Button {
                    pick(item.product.productID, "\(item.brand.name) · \(item.product.name)")
                } label: { Text(verbatim: "\(item.brand.name) · \(item.product.name)") }
            }
        } label: {
            HStack {
                Text(verbatim: L10n.tr("Product:") + " " + (store.label ?? L10n.tr("No product")))
                    .font(CS.inter(12, .semibold)).foregroundStyle(.white).lineLimit(1)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .frame(height: 30)
            .background(Color(fixed: 0x1F2638), in: Capsule())
        }
        .disabled(inMix)
        .accessibilityIdentifier("prompterProduct")
    }

    private func pick(_ id: String?, _ label: String?) {
        save()
        store.productID = id
        store.productLabel = label
        draft = store.script(for: store.key)
    }

    private func save() { store.setScript(draft, for: store.key) }

    private struct SplitBody: Encodable { let user_id: String; let script: String; let product: String? }
    private struct SplitAnswer: Decodable { let parts: [String: String] }

    private func split() async {
        save()
        guard let session = appState.session else { notice = "Sign in or continue to use Split for Mix."; return }
        splitting = true
        defer { splitting = false }
        do {
            let answer: SplitAnswer = try await APIClient.shared.request(
                "/v1/scripts/split", method: "POST",
                body: SplitBody(user_id: session.userID, script: draft, product: store.label), timeout: 40)
            store.setParts(answer.parts, for: store.key)
            notice = "Split into Hook, Body, Benefit and CTA. Each part shows when you record that Mix section."
            CutSellDiagnostics.log("prompter_split", ["ok": "1"])
        } catch {
            notice = "Split for Mix isn't available right now. Your script is saved."
            CutSellDiagnostics.log("prompter_split", ["ok": "0", "error": String(describing: error).prefix(80).description])
        }
    }
}

// MARK: - 3·2·1 before recording

struct ScriptCountdown: View {
    let onDone: () -> Void
    @State private var n = 3

    var body: some View {
        ZStack {
            Color.black.opacity(0.35).ignoresSafeArea()
            Text(verbatim: "\(n)")
                .font(CS.inter(96, .bold))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.4), radius: 10)
                .id(n)
                .transition(.scale.combined(with: .opacity))
        }
        .allowsHitTesting(true)
        .accessibilityIdentifier("prompterCountdownOverlay")
        .task {
            for value in [3, 2, 1] {
                withAnimation(.easeOut(duration: 0.2)) { n = value }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            onDone()
        }
    }
}
