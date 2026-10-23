import SwiftUI

/// "Choose your style" before the cut (Figma 03 Create, node 1056:22).
/// Two switches, both on by default: Overlap audio (smooth audio at every join) and
/// Auto Zoom AI (zooms in on key moments). Both are sent to the server with the cut
/// (`dialogue_overlap_enabled` / `audio_overlap` and `auto_zoom_enabled`) and remembered for the
/// next videos when the box is ticked.
struct StyleChoice: Equatable {
    var overlapAudio: Bool = true
    var autoZoom: Bool = true

    private static let overlapKey = "cutsell.style.overlap_audio"
    private static let autoZoomKey = "cutsell.style.auto_zoom"
    private static let rememberKey = "cutsell.style.remember"

    static var remember: Bool {
        get { UserDefaults.standard.object(forKey: rememberKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: rememberKey) }
    }

    static func saved() -> StyleChoice {
        let defaults = UserDefaults.standard
        return StyleChoice(
            overlapAudio: defaults.object(forKey: overlapKey) as? Bool ?? true,
            autoZoom: defaults.object(forKey: autoZoomKey) as? Bool ?? true
        )
    }

    func save() {
        UserDefaults.standard.set(overlapAudio, forKey: Self.overlapKey)
        UserDefaults.standard.set(autoZoom, forKey: Self.autoZoomKey)
    }
}

/// The cyan switch of the Figma (51×31, track #19E6FF when on).
struct CSSwitch: View {
    @Binding var isOn: Bool

    var body: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) { isOn.toggle() }
        } label: {
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule().fill(isOn ? CS.accent : Color(fixed: 0xD5DAE3))
                Circle().fill(.white).frame(width: 27, height: 27).padding(2)
            }
            .frame(width: 51, height: 31)
        }
        .buttonStyle(.plain)
        .accessibilityValue(isOn ? "On" : "Off")
    }
}

/// One row of a toggle list: title, one-line explanation, switch on the right.
struct StyleToggleRow: View {
    let title: String
    let subtitle: String
    @Binding var isOn: Bool
    let identifier: String

    var body: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 5) {
                Text(L10n.tr(title)).font(CS.cardTitle).foregroundStyle(CS.wiInk)
                Text(L10n.tr(subtitle)).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
                    .lineLimit(1).minimumScaleFactor(0.85)
            }
            Spacer(minLength: 0)
            CSSwitch(isOn: $isOn).accessibilityIdentifier(identifier)
        }
        .padding(.horizontal, 15)
        .frame(height: 65)
    }
}

struct StyleSheetView: View {
    let ctaTitle: String
    let onCreate: (StyleChoice) -> Void

    @State private var choice = StyleChoice.saved()
    @State private var remember = StyleChoice.remember

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(L10n.tr("Choose your style"))
                .font(CS.inter(22, .heavy))
                .foregroundStyle(CS.wiInk)
            Text(L10n.tr("You can change this later in the editor."))
                .font(CS.inter(13, .medium))
                .foregroundStyle(CS.wiSecondary)
                .padding(.top, 4)

            VStack(spacing: 0) {
                StyleToggleRow(title: "Overlap audio", subtitle: "Smooth audio between cuts, no abrupt jumps.",
                               isOn: $choice.overlapAudio, identifier: "styleOverlapToggle")
                Rectangle().fill(CS.wiBorder).frame(height: 1).padding(.horizontal, 15)
                StyleToggleRow(title: "Auto Zoom AI", subtitle: "Zooms in on key moments automatically.",
                               isOn: $choice.autoZoom, identifier: "styleAutoZoomToggle")
            }
            .wiCard(radius: 16, shadow: false)
            .padding(.top, 18)

            CheckboxRow(title: "Use this style for my next videos", isOn: $remember)
                .padding(.top, 16)

            Spacer().frame(height: 12)

            PrimaryButton(title: ctaTitle) {
                StyleChoice.remember = remember
                if remember { choice.save() }
                onCreate(choice)
            }
            .accessibilityIdentifier("styleCreateButton")
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        // White Indigo: always the light look (also when shown over the dark camera).
        .environment(\.colorScheme, .light)
    }
}
