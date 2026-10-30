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

/// The sheet (white, radius 28, 45 % dim, grey handle): title, blue-bordered search, one row per brand A→Z that opens
/// its products A→Z (30 Oct, PO), "Add product" under a brand, "Add “…” as a new product" (one step: saves, chooses
/// and goes on), the blue Continue (only with a product chosen) and, in Bulk only, "Mixed products? Detect for me".
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
    @State private var expanded: Set<String> = []
    /// The brand is already known (from "Add product" under it): only the product is asked.
    @State private var brandLocked = false
    @FocusState private var focus: Field?

    enum Field { case search, brand, product }

    /// 30 Oct (PO): one row per brand (A→Z); tapping a brand shows its products A→Z under it.
    private struct BrandGroup: Identifiable {
        let name: String
        let products: [(id: String, name: String)]
        var id: String { name.lowercased() }
    }

    private var groups: [BrandGroup] {
        var byBrand: [String: (name: String, products: [(id: String, name: String)])] = [:]
        for item in catalog.allProducts {
            let key = item.brand.name.trimmingCharacters(in: .whitespaces).lowercased()
            var entry = byBrand[key] ?? (item.brand.name, [])
            if !entry.products.contains(where: { $0.id == item.product.productID }) {
                entry.products.append((item.product.productID, item.product.name))
            }
            byBrand[key] = entry
        }
        let all = byBrand.values.map { entry in
            BrandGroup(name: entry.name, products: entry.products.sorted {
                $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending })
        }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        let q = typed
        guard !q.isEmpty else { return all }
        return all.compactMap { group in
            if group.name.localizedCaseInsensitiveContains(q) { return group }
            let hits = group.products.filter { $0.name.localizedCaseInsensitiveContains(q) }
            return hits.isEmpty ? nil : BrandGroup(name: group.name, products: hits)
        }
    }

    /// While searching every matching brand is open; otherwise only the ones she tapped.
    private func isOpen(_ group: BrandGroup) -> Bool { !typed.isEmpty || expanded.contains(group.id) }

    /// Rows on screen (brands + open products + "add product"), for the list height.
    private var visibleRows: Int {
        groups.reduce(0) { $0 + 1 + (isOpen($1) ? $1.products.count + 1 : 0) }
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
                    if !addingNew {
                        WIPillButtonGradient(title: "Continue", enabled: selected != nil) {
                            if let selected { PromotingChoice.remember(selected) }
                            onContinue(selected)
                        }
                        .padding(.top, 24)
                        .accessibilityIdentifier("promotingContinue")
                    }
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
                        ForEach(groups) { group in
                            brandRow(group)
                            if isOpen(group) {
                                ForEach(group.products, id: \.id) { product in
                                    productRow(brand: group.name, id: product.id, name: product.name)
                                }
                                addProductRow(brand: group.name)
                            }
                        }
                    }
                }
                .frame(height: min(CGFloat(visibleRows), 4) * 52)
                .scrollBounceBehavior(.basedOnSize)
                // "Add “Cera…” as a new product" (or, with nothing typed, "+ New brand or product").
                Button {
                    newProduct = typed
                    newBrand = ""
                    brandLocked = false
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

    private func brandRow(_ group: BrandGroup) -> some View {
        let open = isOpen(group)
        let hasPick = group.products.contains { $0.id == selected?.productID }
        return Button {
            withAnimation(.easeOut(duration: 0.18)) {
                if expanded.contains(group.id) { expanded.remove(group.id) } else { expanded.insert(group.id) }
            }
        } label: {
            HStack(spacing: 8) {
                Text(verbatim: group.name).font(CS.inter(15, .bold)).foregroundStyle(CS.wiInk)
                Text(verbatim: "\(group.products.count)").font(CS.inter(12, .semibold)).foregroundStyle(CS.wiSecondary)
                if hasPick && !open {
                    Text(verbatim: "· " + (selected?.product ?? "")).font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
                }
                Spacer(minLength: 6)
                Image("WIChevronSmall").renderingMode(.template).resizable()
                    .frame(width: 16, height: 16)
                    .rotationEffect(.degrees(open ? 90 : 0))
                    .foregroundStyle(CS.wiSecondary)
            }
            .lineLimit(1)
            .padding(.horizontal, 16)
            .frame(height: 52)
            .overlay(alignment: .bottom) { Rectangle().fill(CS.wiBorder).frame(height: 1).padding(.horizontal, 16) }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("promotingBrand_\(group.name)")
    }

    private func productRow(brand: String, id: String, name: String) -> some View {
        let isOn = selected?.productID == id
        return Button { selected = .init(productID: id, brand: brand, product: name) } label: {
            HStack(spacing: 0) {
                Text(verbatim: name).font(CS.inter(15, isOn ? .bold : .medium)).foregroundStyle(CS.wiInk)
                Spacer(minLength: 6)
                if isOn {
                    Image("WICheck").renderingMode(.template).resizable().frame(width: 16, height: 16).foregroundStyle(CS.accent)
                }
            }
            .lineLimit(1)
            .padding(.leading, 32).padding(.trailing, 16)
            .frame(height: 52)
            .background(isOn ? CS.accentSoft : CS.wiPage.opacity(0.6))
            .overlay(alignment: .bottom) { Rectangle().fill(CS.wiBorder).frame(height: 1).padding(.leading, 32) }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("promotingRow")
    }

    /// "+ Add product" under an open brand: only the product name is asked.
    private func addProductRow(brand: String) -> some View {
        Button {
            newBrand = brand
            newProduct = ""
            brandLocked = true
            addingNew = true
            focus = .product
        } label: {
            HStack(spacing: 6) {
                Image("WIPlus").renderingMode(.template).resizable().frame(width: 14, height: 14)
                Text(L10n.tr("Add product")).font(CS.inter(13, .bold))
                Spacer(minLength: 0)
            }
            .foregroundStyle(CS.accent)
            .padding(.leading, 32).padding(.trailing, 16)
            .frame(height: 52)
            .background(CS.wiPage.opacity(0.6))
            .overlay(alignment: .bottom) { Rectangle().fill(CS.wiBorder).frame(height: 1).padding(.horizontal, 16) }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("promotingAddProduct_\(brand)")
    }

    /// "Add as a new product": asks the brand and the product, saves it in her list and chooses it.
    private var newProductForm: some View {
        VStack(alignment: .leading, spacing: 8) {
            if brandLocked {
                Text(verbatim: newBrand).font(CS.inter(15, .bold)).foregroundStyle(CS.wiSecondary).padding(.top, 16)
            } else {
                field("Brand", text: $newBrand, which: .brand)
                    .padding(.top, 16)
            }
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
                        if saving { ProgressView().tint(.white) } else { Text(L10n.tr("Save & continue")).font(CS.inter(14, .bold)) }
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
        var brand = newBrand.trimmingCharacters(in: .whitespacesAndNewlines)
        let product = newProduct.trimmingCharacters(in: .whitespacesAndNewlines)
        // A brand she already has goes under that brand (same spelling), never a second "Nivea".
        if let known = catalog.allProducts.first(where: {
            $0.brand.name.trimmingCharacters(in: .whitespaces).caseInsensitiveCompare(brand) == .orderedSame }) {
            brand = known.brand.name
            // The same product again: choose it instead of making a copy.
            if let same = catalog.allProducts.first(where: {
                $0.brand.name == known.brand.name && $0.product.name.caseInsensitiveCompare(product) == .orderedSame }) {
                let pick = PromotingChoice.Pick(productID: same.product.productID, brand: brand, product: same.product.name)
                PromotingChoice.remember(pick)
                addingNew = false
                onContinue(pick)
                return
            }
        }
        saving = true
        defer { saving = false }
        // Saved in her list now; if the server cannot be reached, the names still travel with the videos.
        let id = try? await catalog.createProduct(brand: brand, product: product, session: appState.session,
                                                  projects: appState.projects)
        let pick = PromotingChoice.Pick(productID: id, brand: brand, product: product)
        selected = pick
        query = ""
        addingNew = false
        focus = nil
        // One step (30 Oct, PO): saving the new brand / product also chooses it and goes on.
        PromotingChoice.remember(pick)
        onContinue(pick)
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
