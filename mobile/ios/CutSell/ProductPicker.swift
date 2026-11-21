import SwiftUI

/// 23 Oct (PO, Figma 1215:50 "Opción C"): before uploading or recording, the creator says what she is promoting.
/// The choice travels with the videos (source metadata product_id / brand / product) and each new project is placed
/// in that product right away (POST /v1/catalog/assign), so it lands in Edits › Products without "need a product",
/// Plan my week spreads it by brand / product, and its caption uses them. Bulk can still say "Detect for me".
/// PO 21 Nov (option A): the last used products first (most recent first), then the rest by brand and product A→Z.
enum PromotingOrder {
    static func sorted(_ rows: [(id: String, brand: String, product: String)], recent: [String])
        -> [(id: String, brand: String, product: String)] {
        let rank = Dictionary(recent.enumerated().map { ($1, $0) }, uniquingKeysWith: { a, _ in a })
        return rows.sorted { a, b in
            switch (rank[a.id], rank[b.id]) {
            case let (x?, y?): return x < y
            case (_?, nil): return true
            case (nil, _?): return false
            default:
                let byBrand = a.brand.localizedCaseInsensitiveCompare(b.brand)
                if byBrand != .orderedSame { return byBrand == .orderedAscending }
                return a.product.localizedCaseInsensitiveCompare(b.product) == .orderedAscending
            }
        }
    }
}

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

            // PO 21 Nov (Figma 3094:357, option A "un toque"): one flat list, product big + brand under it, the last
            // used first; one tap chooses and goes on (no Continue). Brands as chips to filter.
            Text(L10n.tr(listTitle))
                .font(CS.inter(11, .bold)).foregroundStyle(CS.wiSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, 4).padding(.top, 8)
            ScrollView {
                VStack(spacing: 0) {
                    ForEach(Array(flatRows.enumerated()), id: \.element.id) { index, row in
                        flatRow(row, last: index == flatRows.count - 1)
                    }
                    if flatRows.isEmpty {
                        Text(L10n.tr("Nothing found"))
                            .font(CS.inter(14, .medium)).foregroundStyle(CS.wiSecondary)
                            .frame(maxWidth: .infinity).frame(height: 60)
                    }
                }
            }
            .frame(height: min(CGFloat(max(1, flatRows.count)), 4) * 60)
            .scrollBounceBehavior(.basedOnSize)
            .background(Color(fixed: 0xFFFFFF), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

            if brandNames.count > 1 {
                Text(L10n.tr("ALL BRANDS"))
                    .font(CS.inter(11, .bold)).foregroundStyle(CS.wiSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.leading, 4).padding(.top, 10)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(brandNames, id: \.self) { name in brandChip(name) }
                    }
                    .padding(.horizontal, 6)
                }
                .frame(height: 44)
                .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
            }

            // "+ New brand or product" (or "Add “Cera…” as a new product"; with a brand chip on, its brand is kept).
            Button {
                newProduct = typed
                newBrand = brandFilter ?? ""
                brandLocked = brandFilter != nil
                addingNew = true
                focus = brandFilter != nil ? .product : .brand
            } label: {
                HStack(spacing: 8) {
                    Image("WIPlus").renderingMode(.template).resizable().frame(width: 14, height: 14)
                    Text(typed.isEmpty ? L10n.tr("New brand or product") : L10n.tr("Add “\(typed)” as a new product"))
                        .font(CS.inter(14, .bold)).lineLimit(1).truncationMode(.middle)
                    Spacer(minLength: 0)
                }
                .foregroundStyle(CS.accent)
                .padding(.horizontal, 4)
                .frame(height: 40)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.top, 8)
            .accessibilityIdentifier("promotingAddNew")
        }
    }

    // MARK: Option A rows

    struct FlatRow: Identifiable, Equatable {
        let id: String
        let brand: String
        let product: String
    }

    @State private var brandFilter: String?

    private var listTitle: String {
        if !typed.isEmpty || brandFilter != nil { return "PRODUCTS" }
        return PromotingChoice.recent.isEmpty ? "PRODUCTS" : "RECENT"
    }

    private var brandNames: [String] { groups.map(\.name) }

    /// Every product (filtered by the search and the brand chip), the last used first, then A→Z.
    private var flatRows: [FlatRow] {
        let rows = groups
            .filter { brandFilter == nil || $0.name.caseInsensitiveCompare(brandFilter ?? "") == .orderedSame }
            .flatMap { group in group.products.map { FlatRow(id: $0.id, brand: group.name, product: $0.name) } }
        return PromotingOrder.sorted(rows.map { ($0.id, $0.brand, $0.product) }, recent: PromotingChoice.recent)
            .map { FlatRow(id: $0.id, brand: $0.brand, product: $0.product) }
    }

    private func flatRow(_ row: FlatRow, last: Bool) -> some View {
        let isLastUsed = PromotingChoice.recent.first == row.id
        return Button {
            let pick = PromotingChoice.Pick(productID: row.id, brand: row.brand, product: row.product)
            selected = pick
            PromotingChoice.remember(pick)
            onContinue(pick)
        } label: {
            HStack(spacing: 12) {
                Text(verbatim: String(row.brand.prefix(1)).uppercased())
                    .font(CS.inter(15, .heavy)).foregroundStyle(CS.accent)
                    .frame(width: 36, height: 36)
                    .background(CS.accentSoft, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    Text(verbatim: row.product).font(CS.inter(15, .bold)).foregroundStyle(CS.wiInk)
                    Text(verbatim: row.brand).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
                }
                .lineLimit(1)
                Spacer(minLength: 6)
                if isLastUsed {
                    Text(L10n.tr("Last used")).font(CS.inter(11, .bold)).foregroundStyle(CS.accent)
                        .padding(.horizontal, 8).frame(height: 22)
                        .background(CS.accentSoft, in: Capsule())
                } else {
                    Image("WIChevronSmall").renderingMode(.template).resizable()
                        .frame(width: 16, height: 16).foregroundStyle(CS.wiSecondary)
                }
            }
            .padding(.horizontal, 16)
            .frame(height: 60)
            .overlay(alignment: .bottom) {
                if !last { Rectangle().fill(CS.wiBorder).frame(height: 1).padding(.leading, 64).padding(.trailing, 16) }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("promotingRow")
    }

    private func brandChip(_ name: String) -> some View {
        let on = brandFilter?.caseInsensitiveCompare(name) == .orderedSame
        return Button {
            withAnimation(.easeOut(duration: 0.15)) { brandFilter = on ? nil : name }
        } label: {
            Text(verbatim: name).font(CS.inter(13, .semibold))
                .foregroundStyle(on ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                .padding(.horizontal, 12).frame(height: 32)
                .background(on ? CS.accent : CS.accentSoft, in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("promotingBrand_\(name)")
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
                        .softPuff(.white, in: Capsule())
                }
                .buttonStyle(.plain)
                Button { Task { await saveNew() } } label: {
                    Group {
                        if saving { ProgressView().tint(.white) } else { Text(L10n.tr("Save & continue")).font(CS.inter(14, .bold)) }
                    }
                    .foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(maxWidth: .infinity).frame(height: 44)
                    .softPuff(.blue, in: Capsule())
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
                .softPuff(.blue, in: Capsule())
                .opacity(enabled ? 1 : 0.4)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }
}
