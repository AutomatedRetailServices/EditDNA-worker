import SwiftUI

/// "Choose your style" before the cut (Figma 03 Create, node 1056:22).
/// The server does not use Pace / Zoom yet, so the choice is kept on this iPhone only
/// (and remembered for the next videos when the box is ticked).
struct StyleChoice: Equatable {
    enum Pace: String, CaseIterable { case normal, fast }
    enum Zoom: String, CaseIterable { case off, subtle, dynamic }

    var pace: Pace = .normal
    var zoom: Zoom = .subtle

    private static let paceKey = "cutsell.style.pace"
    private static let zoomKey = "cutsell.style.zoom"
    private static let rememberKey = "cutsell.style.remember"

    static var remember: Bool {
        get { UserDefaults.standard.object(forKey: rememberKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: rememberKey) }
    }

    static func saved() -> StyleChoice {
        let defaults = UserDefaults.standard
        return StyleChoice(
            pace: Pace(rawValue: defaults.string(forKey: paceKey) ?? "") ?? .normal,
            zoom: Zoom(rawValue: defaults.string(forKey: zoomKey) ?? "") ?? .subtle
        )
    }

    func save() {
        UserDefaults.standard.set(pace.rawValue, forKey: Self.paceKey)
        UserDefaults.standard.set(zoom.rawValue, forKey: Self.zoomKey)
    }
}

struct StyleSheetView: View {
    let ctaTitle: String
    let onCreate: (StyleChoice) -> Void

    @State private var choice = StyleChoice.saved()
    @State private var remember = StyleChoice.remember

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule()
                .fill(Color(hex: 0x3A5280))
                .frame(width: 40, height: 4)
                .frame(maxWidth: .infinity)
                .padding(.top, 10)
                .padding(.bottom, 18)

            Text("Choose your style")
                .font(CS.inter(18, .semibold))
                .foregroundStyle(.white)
            Text("We edit your video with this style. You can change it later in the editor.")
                .font(CS.inter(12))
                .foregroundStyle(CS.textMuted)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 6)

            Text("Pace")
                .font(CS.inter(12, .medium))
                .foregroundStyle(CS.textSecondary)
                .padding(.top, 18)
                .padding(.bottom, 8)
            HStack(spacing: 6) {
                OptionCard(title: "Normal", subtitle: "Natural pauses", isSelected: choice.pace == .normal) {
                    choice.pace = .normal
                }
                OptionCard(title: "Fast", subtitle: "Cuts the pauses", isSelected: choice.pace == .fast) {
                    choice.pace = .fast
                }
            }

            Text("Zoom")
                .font(CS.inter(12, .medium))
                .foregroundStyle(CS.textSecondary)
                .padding(.top, 14)
                .padding(.bottom, 8)
            HStack(spacing: 6) {
                OptionCard(title: "Off", subtitle: "No zoom", isSelected: choice.zoom == .off) { choice.zoom = .off }
                OptionCard(title: "Subtle", subtitle: "2–5%", isSelected: choice.zoom == .subtle) { choice.zoom = .subtle }
                OptionCard(title: "Dynamic", subtitle: "2–12%", isSelected: choice.zoom == .dynamic) { choice.zoom = .dynamic }
            }

            CheckboxRow(title: "Use this style for my next videos", isOn: $remember)
                .padding(.top, 18)

            Spacer(minLength: 24)

            PrimaryButton(title: ctaTitle) {
                StyleChoice.remember = remember
                if remember { choice.save() }
                onCreate(choice)
            }
            .accessibilityIdentifier("styleCreateButton")
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CS.sheet.ignoresSafeArea())
        .presentationDetents([.height(430)])
        .presentationDragIndicator(.hidden)
        .presentationCornerRadius(24)
        .presentationBackground(CS.sheet)
    }
}
