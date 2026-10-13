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
                Capsule().fill(isOn ? CS.cyan : Color(hex: 0x2A3A5C))
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
                Text(title).font(.system(size: 15, weight: .semibold)).foregroundStyle(CS.textPrimaryF)
                Text(subtitle).font(.system(size: 12)).foregroundStyle(CS.textMutedF)
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
            Text("Choose your style")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(Color(hex: 0xF2F7FF))
            Text("You can change this later in the editor.")
                .font(.system(size: 13))
                .foregroundStyle(CS.textSecondaryF)
                .padding(.top, 6)

            VStack(spacing: 0) {
                StyleToggleRow(title: "Overlap audio", subtitle: "Smooth audio between cuts, no abrupt jumps.",
                               isOn: $choice.overlapAudio, identifier: "styleOverlapToggle")
                Rectangle().fill(CS.borderDefault).frame(height: 1).padding(.horizontal, 15)
                StyleToggleRow(title: "Auto Zoom AI", subtitle: "Zooms in on key moments automatically.",
                               isOn: $choice.autoZoom, identifier: "styleAutoZoomToggle")
            }
            .background(CS.card, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.cardBorder, lineWidth: 1))
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
        .padding(.horizontal, 19)
        .padding(.bottom, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
