import SwiftUI

/// 23 Oct (PO, Figma 1215:50 "Opción C"): before uploading or recording, the creator says what she is promoting.
/// The choice travels with the videos (source metadata product_id / brand / product) and each new project is placed
/// in that product right away (POST /v1/catalog/assign), so it lands in Edits › Products without "need a product",
/// Plan my week spreads it by brand / product, and its caption uses them. Bulk can still say "Detect for me".
enum PromotingChoice {
    struct Pick: Equatable {
        var productID: String?
        var brand: String
        var product: String

        var metadata: [String: String] {
            var out = ["brand": brand, "product": product]
            if let productID { out["product_id"] = productID }
            return out
        }
    }

    /// The pick of the flow being opened now (nil = "Detect for me" / not asked). Captured by each job when it is made.
    nonisolated(unsafe) static var current: Pick?

    private static let recentKey = "cutsell.promoting.recent"

    /// Product ids, last used first.
    static var recent: [String] { UserDefaults.standard.stringArray(forKey: recentKey) ?? [] }

    static func remember(_ pick: Pick) {
        guard let id = pick.productID else { return }
        var list = recent.filter { $0 != id }
        list.insert(id, at: 0)
        UserDefaults.standard.set(Array(list.prefix(30)), forKey: recentKey)
    }

    /// Puts a just-created project in the chosen product (never fails or slows the upload).
    @MainActor
    static func place(projectID: String, pick: Pick?, session: CutSellSession?) {
        guard let pick else { return }
        Task { @MainActor in
            do {
                try await CatalogStore.shared.assign([projectID], productID: pick.productID,
                                                     brand: pick.productID == nil ? pick.brand : nil,
                                                     product: pick.productID == nil ? pick.product : nil,
                                                     session: session, projects: [])
            } catch {
                CutSellDiagnostics.log("promoting_assign_failed", ["error": String(describing: error).prefix(60).description])
            }
        }
    }
}

/// What the sheet was opened for.
enum PromotingPurpose: Equatable {
    case upload(CreateMode), record, faceless, salesMix

    var allowsDetect: Bool { self == .upload(.bulk) }
}

/// The sheet (white, radius 28, 45 % dim, grey handle): title, blue-bordered search, "Brand · Product" rows (last used
/// first), "Add “…” as a new product", the blue Continue (only with a product chosen) and, in Bulk only,
/// "Mixed products? Detect for me".
struct PromotingSheet: View {
    let purpose: PromotingPurpose
    let onContinue: (PromotingChoice.Pick?) -> Void
    let onClose: () -> Void

    @EnvironmentObject private var appState: AppState
    @ObservedObject private var catalog = CatalogStore.shared
    @State private var query = ""
    @State private var selected: PromotingChoice.Pick?
    @State private var addingNew = false
    @State private var newBrand = ""
    @State private var newProduct = ""
    @State private var saving = false
    @FocusState private var focus: Field?

    enum Field { case search, brand, product }

    private struct Row: Identifiable { let id: String; let brand: String; let product: String }

    private var rows: [Row] {
        let order = Dictionary(uniqueKeysWithValues: PromotingChoice.recent.enumerated().map { ($1, $0) })
        let all = catalog.allProducts.map { Row(id: $0.product.productID, brand: $0.brand.name, product: $0.product.name) }
            .sorted { (order[$0.id] ?? Int.max, $0.brand + $0.product) < (order[$1.id] ?? Int.max, $1.brand + $1.product) }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return all }
        return all.filter { "\($0.brand) \($0.product)".localizedCaseInsensitiveContains(q) }
    }

    private var typed: String { query.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        ZStack(alignment: .bottom) {
            Color(fixed: 0x0B1020, opacity: 0.45).ignoresSafeArea()
                .onTapGesture(perform: onClose)
                .accessibilityLabel(L10n.tr("Close"))
                .accessibilityAddTraits(.isButton)
            WISheetCard {
                VStack(alignment: .leading, spacing: 0) {
                    Text(L10n.tr("What are you promoting?"))
                        .font(CS.inter(22, .heavy))
                        .foregroundStyle(CS.wiInk)
                        .padding(.top, 17)
                    if addingNew { newProductForm } else { searchAndResults }
                    WIPillButtonGradient(title: "Continue", enabled: selected != nil && !addingNew) {
                        if let selected { PromotingChoice.remember(selected) }
                        onContinue(selected)
                    }
                    .padding(.top, 24)
                    .accessibilityIdentifier("promotingContinue")
                    if purpose.allowsDetect {
                        Button { onContinue(nil) } label: {
                            Text(L10n.tr("Mixed products? Detect for me"))
                                .font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
                                .frame(maxWidth: .infinity).frame(height: 36)
                        }
                        .buttonStyle(.plain)
                        .padding(.top, 4)
                        .accessibilityIdentifier("promotingDetect")
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, purpose.allowsDetect ? 8 : 20)
            }
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("promotingSheet")
        }
        .task { await catalog.refresh(session: appState.session, projects: appState.projects) }
    }

    /// Figma 1215:159 "Search" + 1215:165 "Results".
    private var searchAndResults: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                Image("WISearch").renderingMode(.template).resizable().frame(width: 18, height: 18)
                    .foregroundStyle(CS.wiInk)
                TextField("", text: $query, prompt: Text(L10n.tr("Search brand or product")).foregroundStyle(CS.wiSecondary))
                    .font(CS.inter(16, .semibold))
                    .foregroundStyle(CS.wiInk)
                    .tint(CS.accent)
                    .focused($focus, equals: .search)
                    .submitLabel(.done)
                    .autocorrectionDisabled()
                    .accessibilityIdentifier("promotingSearch")
            }
            .padding(.horizontal, 13)
            .frame(height: 52)
            .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.accent, lineWidth: 1.5))
            .padding(.top, 16)

            VStack(spacing: 0) {
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(rows) { row in
                            let isOn = selected?.productID == row.id
                            Button { selected = .init(productID: row.id, brand: row.brand, product: row.product) } label: {
                                HStack(spacing: 0) {
                                    Text(verbatim: row.brand + " · ").font(CS.inter(15, .medium)).foregroundStyle(CS.wiSecondary)
                                    Text(verbatim: row.product).font(CS.inter(15, .bold)).foregroundStyle(CS.wiInk)
                                    Spacer(minLength: 6)
                                    Image(isOn ? "WICheck" : "WIChevronSmall").renderingMode(.template).resizable()
                                        .frame(width: 16, height: 16)
                                        .foregroundStyle(isOn ? CS.accent : CS.wiSecondary)
                                }
                                .lineLimit(1)
                                .padding(.horizontal, 16)
                                .frame(height: 52)
                                .background(isOn ? CS.accentSoft : .clear)
                                .overlay(alignment: .bottom) {
                                    Rectangle().fill(CS.wiBorder).frame(height: 1).padding(.horizontal, 16)
                                }
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("promotingRow")
                        }
                    }
                }
                .frame(height: min(CGFloat(rows.count), 3) * 52)
                .scrollBounceBehavior(.basedOnSize)
                // "Add “Cera…” as a new product" (or, with nothing typed, "+ New brand or product").
                Button {
                    newProduct = typed
                    newBrand = ""
                    addingNew = true
                    focus = .brand
                } label: {
                    HStack(spacing: 8) {
                        Image("WIPlus").renderingMode(.template).resizable().frame(width: 18, height: 18)
                        Text(typed.isEmpty ? L10n.tr("New brand or product") : L10n.tr("Add “\(typed)” as a new product"))
                            .font(CS.inter(14, .bold)).lineLimit(1).truncationMode(.middle)
                        Spacer(minLength: 0)
                    }
                    .foregroundStyle(CS.accent)
                    .padding(.horizontal, 16)
                    .frame(height: 52)
                    .background(CS.accentSoft)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("promotingAddNew")
            }
            .background(CS.wiCard)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
        }
    }

    /// "Add as a new product": asks the brand and the product, saves it in her list and chooses it.
    private var newProductForm: some View {
        VStack(alignment: .leading, spacing: 8) {
            field("Brand", text: $newBrand, which: .brand)
                .padding(.top, 16)
            field("Product", text: $newProduct, which: .product)
            HStack(spacing: 10) {
                Button { addingNew = false } label: {
                    Text(L10n.tr("Cancel")).font(CS.inter(14, .bold)).foregroundStyle(CS.wiInk)
                        .frame(maxWidth: .infinity).frame(height: 44)
                        .overlay(Capsule().stroke(CS.wiBorder, lineWidth: 1))
                }
                .buttonStyle(.plain)
                Button { Task { await saveNew() } } label: {
                    Group {
                        if saving { ProgressView().tint(.white) } else { Text(L10n.tr("Save")).font(CS.inter(14, .bold)) }
                    }
                    .foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(maxWidth: .infinity).frame(height: 44)
                    .background(CS.accent, in: Capsule())
                    .opacity(canSave ? 1 : 0.4)
                }
                .buttonStyle(.plain)
                .disabled(!canSave || saving)
                .accessibilityIdentifier("promotingSaveNew")
            }
            .padding(.top, 4)
        }
    }

    private var canSave: Bool {
        !newBrand.trimmingCharacters(in: .whitespaces).isEmpty && !newProduct.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private func field(_ title: String, text: Binding<String>, which: Field) -> some View {
        TextField("", text: text, prompt: Text(L10n.tr(title)).foregroundStyle(CS.wiSecondary))
            .font(CS.inter(16, .semibold))
            .foregroundStyle(CS.wiInk)
            .tint(CS.accent)
            .focused($focus, equals: which)
            .padding(.horizontal, 14)
            .frame(height: 50)
            .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(focus == which ? CS.accent : CS.wiBorder, lineWidth: focus == which ? 1.5 : 1))
            .accessibilityIdentifier("promotingNew\(title)")
    }

    @MainActor
    private func saveNew() async {
        let brand = newBrand.trimmingCharacters(in: .whitespacesAndNewlines)
        let product = newProduct.trimmingCharacters(in: .whitespacesAndNewlines)
        saving = true
        defer { saving = false }
        // Saved in her list now; if the server cannot be reached, the names still travel with the videos.
        let id = try? await catalog.createProduct(brand: brand, product: product, session: appState.session,
                                                  projects: appState.projects)
        selected = .init(productID: id, brand: brand, product: product)
        query = ""
        addingNew = false
        focus = nil
    }
}

/// 52 pt pill with the blue gradient (Figma 1215:182); 40 % when disabled.
struct WIPillButtonGradient: View {
    let title: String
    var enabled = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(L10n.tr(title)).font(CS.inter(16, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                .frame(maxWidth: .infinity).frame(height: 52)
                .background(CS.accentGradient, in: Capsule())
                .opacity(enabled ? 1 : 0.4)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }
}
