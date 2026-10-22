import SwiftUI

// MARK: - Model (GET /v1/catalog …)

struct CatalogProduct: Codable, Identifiable, Hashable {
    let productID: String
    var name: String
    let category: String?
    let hasThumbnail: Bool?
    let ready: Int
    let drafts: Int
    let processing: Int
    let videos: Int
    var id: String { productID }
    enum CodingKeys: String, CodingKey {
        case productID = "product_id", name, category, hasThumbnail = "has_thumbnail", ready, drafts, processing, videos
    }
}

struct CatalogBrand: Codable, Identifiable, Hashable {
    let brandID: String
    var name: String
    var products: [CatalogProduct]
    let videos: Int
    var id: String { brandID }
    enum CodingKeys: String, CodingKey { case brandID = "brand_id", name, products, videos }
}

struct CatalogSuggestion: Codable, Hashable {
    let brand: String?
    let product: String?
    let category: String?
    let confidence: Double?
    var label: String? {
        guard let brand, let product, !brand.isEmpty, !product.isEmpty else { return nil }
        return "\(brand) · \(product)"
    }
}

struct CatalogUncategorized: Codable, Hashable {
    let projectID: String
    let title: String?
    let suggestion: CatalogSuggestion?
    enum CodingKeys: String, CodingKey { case projectID = "project_id", title, suggestion }
}

struct CatalogVideoLink: Codable, Hashable {
    let productID: String?
    let source: String?
    enum CodingKeys: String, CodingKey { case productID = "product_id", source }
}

struct CatalogOverview: Codable {
    var brands: [CatalogBrand]
    var uncategorized: [CatalogUncategorized]
    var videos: [String: CatalogVideoLink]?
}

struct CatalogProductVideo: Codable, Identifiable, Hashable {
    let projectID: String
    let title: String?
    let state: String?
    let bucket: String
    let durationSec: Double?
    /// Mix: the Mix it was made in and its code ("H1 · B2 · Be1 · C2"); nil for a normal cut.
    var mixID: String? = nil
    var mixCode: String? = nil
    var id: String { projectID }
    enum CodingKeys: String, CodingKey {
        case projectID = "project_id", title, state, bucket, durationSec = "duration_sec", mixID = "mix_id", mixCode = "mix_code"
    }
}

struct CatalogProductDetail: Codable {
    let productID: String
    var name: String
    let category: String?
    let brandID: String?
    let brand: String?
    var videos: [CatalogProductVideo]
    enum CodingKeys: String, CodingKey {
        case productID = "product_id", name, category, brandID = "brand_id", brand, videos
    }
}

/// The product a video screen opens (NavigationLink value).
struct ProductRoute: Hashable {
    let productID: String
    /// Mix · Results (Figma 1136:2916): only the videos of this Mix, with "Export all".
    var mixID: String? = nil
}

// MARK: - Store

/// Brands and products of this creator (virtual folders: the server links each cut to a product; no file moves).
/// `-CutSellCatalogMock` (simulator probe) answers from memory, so the screens can be tested before the server
/// that has the catalog is deployed.
@MainActor
final class CatalogStore: ObservableObject {
    static let shared = CatalogStore()

    @Published private(set) var overview = CatalogOverview(brands: [], uncategorized: [], videos: [:])
    @Published private(set) var available = true
    @Published var lastError: String?

    private var mock: MockCatalog? = ProcessInfo.processInfo.arguments.contains("-CutSellCatalogMock") ? MockCatalog() : nil

    var allProducts: [(brand: CatalogBrand, product: CatalogProduct)] {
        overview.brands.flatMap { brand in brand.products.map { (brand, $0) } }
    }

    func refresh(session: CutSellSession?, projects: [Project]) async {
        if let mock {
            overview = mock.overview(projects: projects)
            return
        }
        guard let session else { return }
        do {
            overview = try await APIClient.shared.request("/v1/catalog",
                                                          query: [URLQueryItem(name: "user_id", value: session.userID)])
            available = true
        } catch APIError.http(404, _) {
            available = false                      // a server without the catalog yet
        } catch {
            lastError = error.localizedDescription
        }
    }

    func detail(_ productID: String, session: CutSellSession?, projects: [Project]) async throws -> CatalogProductDetail {
        if let mock { return try mock.detail(productID, projects: projects) }
        guard let session else { throw APIError.invalidResponse }
        return try await APIClient.shared.request("/v1/catalog/products/\(productID)",
                                                  query: [URLQueryItem(name: "user_id", value: session.userID)])
    }

    struct AssignResult: Decodable { let productID: String; let moved: Int
        enum CodingKeys: String, CodingKey { case productID = "product_id", moved } }

    @discardableResult
    func assign(_ projectIDs: [String], productID: String? = nil, brand: String? = nil, product: String? = nil,
                session: CutSellSession?, projects: [Project]) async throws -> AssignResult {
        struct Body: Encodable {
            let user_id: String; let project_ids: [String]; let product_id: String?; let brand: String?; let product: String?
        }
        let result: AssignResult
        if let mock {
            result = try mock.assign(projectIDs, productID: productID, brand: brand, product: product)
        } else {
            guard let session else { throw APIError.invalidResponse }
            result = try await APIClient.shared.request("/v1/catalog/assign", method: "POST",
                                                        body: Body(user_id: session.userID, project_ids: projectIDs,
                                                                   product_id: productID, brand: brand, product: product))
        }
        CutSellDiagnostics.log("catalog_assigned", ["videos": String(projectIDs.count), "new": String(productID == nil)])
        await refresh(session: session, projects: projects)
        return result
    }

    func rename(kind: String, id: String, name: String, session: CutSellSession?, projects: [Project]) async throws {
        struct Body: Encodable { let user_id: String; let kind: String; let id: String; let name: String }
        struct Done: Decodable {}
        if let mock {
            try mock.rename(kind: kind, id: id, name: name)
        } else {
            guard let session else { throw APIError.invalidResponse }
            let _: Done = try await APIClient.shared.request("/v1/catalog/rename", method: "POST",
                                                             body: Body(user_id: session.userID, kind: kind, id: id, name: name))
        }
        CutSellDiagnostics.log("catalog_renamed", ["kind": kind])
        await refresh(session: session, projects: projects)
    }

    func productID(of projectID: String) -> String? { overview.videos?[projectID]?.productID }

    /// A new brand + product (Mix: "+ New brand or product"). Returns its product id.
    func createProduct(brand: String, product: String, session: CutSellSession?, projects: [Project]) async throws -> String {
        struct Body: Encodable { let user_id: String; let brand: String; let product: String }
        struct Answer: Decodable { let product_id: String? }
        let id: String?
        if let mock {
            id = mock.create(brand: brand, product: product)
        } else {
            guard let session else { throw APIError.invalidResponse }
            let answer: Answer = try await APIClient.shared.request("/v1/catalog/products", method: "POST",
                                                                    body: Body(user_id: session.userID, brand: brand, product: product))
            id = answer.product_id
        }
        await refresh(session: session, projects: projects)
        guard let id else { throw APIError.invalidResponse }
        return id
    }

    /// Probe only: the videos of a mock Mix land in their product with the MIX tag and code.
    func mockAddMix(productID: String, videos: [(String, String)]) { mock?.addMix(productID: productID, videos: videos) }
}

// MARK: - Edits · Products (Figma 1135:2686)

struct ProductsList: View {
    @ObservedObject var store: CatalogStore
    let projects: [Project]
    let onAssign: ([String], CatalogSuggestion?) -> Void
    @State private var search = ""
    @ObservedObject private var thumbnails = ThumbnailStore.shared

    private var brands: [CatalogBrand] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return store.overview.brands }
        return store.overview.brands.compactMap { brand in
            if brand.name.localizedCaseInsensitiveContains(query) { return brand }
            var copy = brand
            copy.products = brand.products.filter { $0.name.localizedCaseInsensitiveContains(query) }
            return copy.products.isEmpty ? nil : copy
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text(L10n.tr("⌕")).font(CS.inter(16, .medium)).foregroundStyle(Color(hex: 0x99A3BD))
                TextField("", text: $search, prompt: Text(L10n.tr("Search brand or product")).foregroundStyle(Color(hex: 0x99A3BD)))
                    .font(CS.inter(13))
                    .foregroundStyle(.white)
                    .accessibilityIdentifier("productsSearch")
            }
            .padding(.horizontal, 13)
            .frame(height: 40)
            .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
            .padding(.top, 12)

            if !store.overview.uncategorized.isEmpty {
                needBanner.padding(.top, 12)
            }

            if !store.available {
                Text(L10n.tr("Your products will appear here as your cuts are organized."))
                    .font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 24)
            } else if brands.isEmpty && store.overview.uncategorized.isEmpty {
                Text(L10n.tr("No products yet. New cuts are grouped by the brand and product you talk about."))
                    .font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 24)
                    .accessibilityIdentifier("productsEmpty")
            }

            ForEach(brands) { brand in
                HStack(alignment: .firstTextBaseline) {
                    Text(brand.name).font(CS.inter(17, .bold)).foregroundStyle(Color.ink)
                    Spacer()
                    Text(L10n.tr("\(brand.products.count) product\(brand.products.count == 1 ? "" : "s") · \(brand.videos) video\(brand.videos == 1 ? "" : "s")"))
                        .font(CS.inter(11, .medium)).foregroundStyle(Color(hex: 0x99A3BD))
                }
                .padding(.top, 14)
                VStack(spacing: 8) {
                    ForEach(brand.products) { product in
                        NavigationLink(value: ProductRoute(productID: product.productID)) {
                            ProductRow(product: product, cover: cover(for: product))
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("productRow_\(product.name)")
                    }
                }
                .padding(.top, 7)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("productsList")
    }

    /// Figma 1135:2834: "X videos need a product · We couldn’t tell from the video · Assign ›".
    private var needBanner: some View {
        let count = store.overview.uncategorized.count
        return Button {
            let ids = store.overview.uncategorized.map(\.projectID)
            let suggestions = store.overview.uncategorized.compactMap(\.suggestion).filter { $0.label != nil }
            let common = Dictionary(grouping: suggestions, by: { $0.label ?? "" }).max { $0.value.count < $1.value.count }
            onAssign(ids, common?.value.first)
        } label: {
            HStack(spacing: 10) {
                Image("ProductsNeedDot").resizable().frame(width: 10, height: 10)
                VStack(alignment: .leading, spacing: 3) {
                    Text(L10n.tr("\(count) video\(count == 1 ? "" : "s") need\(count == 1 ? "s" : "") a product"))
                        .font(CS.inter(14, .semibold)).foregroundStyle(Color.ink)
                    Text(L10n.tr("We couldn’t tell from the video")).font(CS.inter(11)).foregroundStyle(Color(hex: 0x99A3BD))
                }
                Spacer()
                Text(L10n.tr("Assign ›")).font(CS.inter(13, .semibold)).foregroundStyle(Color(hex: 0xFFB840))
            }
            .padding(.leading, 15).padding(.trailing, 16)
            .frame(height: 52)
            // Day mode (Figma 1146:91): a white card with the soft shadow.
            .background(Color(dark: 0x1F170D, light: 0xFFFFFF), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(dark: 0x73521F, light: 0xE2E6EF), lineWidth: 1))
            .dayCardShadow()
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("productsAssignBanner")
    }

    /// The newest video of the product with a thumbnail on this phone; else the server's product photo.
    private func cover(for product: CatalogProduct) -> UIImage? {
        let ids = projects.filter { store.productID(of: $0.projectID) == product.productID }.map(\.projectID)
        return ids.lazy.compactMap { thumbnails.image(for: $0) }.first
    }
}

/// Figma 1135:2841 (350 × 76).
struct ProductRow: View {
    let product: CatalogProduct
    let cover: UIImage?

    var body: some View {
        HStack(spacing: 0) {
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Color(hex: 0x1F2638))
                    .frame(width: 44, height: 52).offset(x: 7, y: -5)
                Group {
                    if let cover { Image(uiImage: cover).resizable().scaledToFill() } else { Color(hex: 0x1F2638) }
                }
                .frame(width: 44, height: 54)
                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            }
            .frame(width: 58, alignment: .leading)
            .padding(.leading, 11)
            .padding(.top, 5)
            VStack(alignment: .leading, spacing: 0) {
                Text(product.name).font(CS.inter(14, .semibold)).foregroundStyle(Color.ink).lineLimit(1)
                Text(L10n.tr("\(product.videos) video\(product.videos == 1 ? "" : "s")"))
                    .font(CS.inter(11)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 3)
                HStack(spacing: 10) {
                    if product.ready > 0 { Text(L10n.tr("\(product.ready) ready")).foregroundStyle(Color(hex: 0x33D98C)) }
                    if product.drafts > 0 { Text(L10n.tr("\(product.drafts) draft\(product.drafts == 1 ? "" : "s")")).foregroundStyle(Color(hex: 0x99A3BD)) }
                    if product.processing > 0 { Text(L10n.tr("\(product.processing) processing")).foregroundStyle(CS.cyan) }
                }
                .font(CS.inter(11, .semibold))
                .padding(.top, 5)
            }
            Spacer(minLength: 8)
            Text(L10n.tr("›")).font(CS.inter(20, .medium)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.trailing, 13)
        }
        .frame(height: 76)
        .frame(maxWidth: .infinity)
        .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
        .contentShape(Rectangle())
    }
}

// MARK: - Product detail (Figma 1135:2876)

struct ProductDetailView: View {
    let route: ProductRoute
    @ObservedObject var store = CatalogStore.shared
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @State private var detail: CatalogProductDetail?
    @State private var filter = "all"
    @State private var selecting = false
    @State private var selected: Set<String> = []
    @State private var renaming = false
    @State private var newName = ""
    @State private var assigning: AssignRequest?
    @State private var openProject: Project?
    @State private var errorMessage: String?
    @ObservedObject private var mixes = MixCenter.shared
    @StateObject private var exporter = MixExporter()

    private var mix: MixSummary? { route.mixID.flatMap { id in mixes.mixes.first { $0.mixID == id } } }
    private var mixVideos: [CatalogProductVideo] {
        guard let id = route.mixID else { return [] }
        let mine = (detail?.videos ?? []).filter { $0.mixID == id }
        guard let mix else { return mine }
        let order = Dictionary(uniqueKeysWithValues: mix.videos.map { ($0.projectID, $0.n) })
        return mine.sorted { (order[$0.projectID] ?? 0) < (order[$1.projectID] ?? 0) }
    }

    private var videos: [CatalogProductVideo] {
        guard let detail else { return [] }
        if route.mixID != nil { return mixVideos }
        return filter == "all" ? detail.videos : detail.videos.filter { $0.bucket == filter }
    }

    private func count(_ bucket: String) -> Int {
        guard let detail else { return 0 }
        return bucket == "all" ? detail.videos.count : detail.videos.filter { $0.bucket == bucket }.count
    }

    private var drafts: [CatalogProductVideo] { detail?.videos.filter { $0.bucket == "drafts" } ?? [] }

    var body: some View {
        ZStack {
            ScreenBackground()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        Button { dismiss() } label: {
                            Text(L10n.tr("‹ \(detail?.brand ?? "Products")")).font(CS.inter(16, .semibold)).foregroundStyle(CS.cyan)
                        }
                        .accessibilityIdentifier("productBack")
                        Spacer()
                        Menu {
                            Button(L10n.tr("Rename product")) { newName = detail?.name ?? ""; renaming = true }
                            Button(selecting ? "Done selecting" : "Select videos") { selecting.toggle(); selected = [] }
                        } label: {
                            Text(L10n.tr("•••")).font(CS.sf(16, .semibold)).foregroundStyle(Color(hex: 0xF2F7FF))
                                .frame(width: 34, height: 34)
                                .background(Color(hex: 0x040813, opacity: 0.9), in: Circle())
                                .overlay(Circle().stroke(Color(hex: 0x144080), lineWidth: 1))
                        }
                    }
                    .padding(.top, 8)
                    HStack(alignment: .firstTextBaseline) {
                        Text(L10n.tr(detail?.name ?? " ")).font(CS.inter(26, .bold)).foregroundStyle(Color.ink).lineLimit(2)
                            .accessibilityIdentifier("productTitle")
                        Spacer(minLength: 8)
                        Button { newName = detail?.name ?? ""; renaming = true } label: {
                            Text(L10n.tr("✎ Rename")).font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                        }
                        .accessibilityIdentifier("productRename")
                    }
                    .padding(.top, 12)
                    Text(L10n.tr(subtitle)).font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 4)
                        .accessibilityIdentifier("productSubtitle")

                    if route.mixID != nil {
                        mixBanner.padding(.top, 18)
                    } else {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            chip("all", "All \(count("all"))", width: 64)
                            chip("ready", "Ready \(count("ready"))", width: 76)
                            chip("drafts", "Drafts \(count("drafts"))", width: 78)
                            chip("processing", "Processing \(count("processing"))", width: 100)
                        }
                    }
                    .padding(.top, 14)

                    Text(L10n.tr("✦  Grouped automatically from what you said in each video."))
                        .font(CS.inter(11, .medium)).foregroundStyle(Color(hex: 0x99A3BD))
                        .padding(.horizontal, 13)
                        .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
                        .background(Color(hex: 0x121A2B), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color(hex: 0x19E5FF), lineWidth: 1))
                        .padding(.top, 12)
                    }

                    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 3), spacing: 8) {
                        ForEach(videos) { video in tile(video) }
                    }
                    .padding(.top, 14)

                    HStack(spacing: 10) {
                        Button {
                            if selecting && !selected.isEmpty {
                                assigning = AssignRequest(projectIDs: Array(selected), suggestion: nil)
                            } else {
                                selecting.toggle(); selected = []
                            }
                        } label: {
                            Text(L10n.tr(selecting ? (selected.isEmpty ? "Cancel" : "Move \(selected.count)") : "Select"))
                                .font(CS.inter(15, .semibold)).foregroundStyle(.white)
                                .frame(width: 110, height: 50)
                                .overlay(Capsule().stroke(Color(hex: 0x596685), lineWidth: 1))
                        }
                        .accessibilityIdentifier("productSelect")
                        if route.mixID != nil {
                            Button { Task { await exportAll() } } label: {
                                Text(L10n.tr(exporter.running ? "Exporting \(exporter.done + 1) of \(exporter.total)…"
                                     : "Export all \(selecting && !selected.isEmpty ? selected.count : readyMixVideos.count)"))
                                    .font(CS.sf(15, .semibold)).foregroundStyle(.white)
                                    .frame(maxWidth: .infinity).frame(height: 50)
                                    .background(Capsule().fill(MixStyle.cta))
                                    .shadow(color: Color(hex: 0x1969FF).opacity(0.25), radius: 9, y: 6)
                            }
                            .disabled(readyMixVideos.isEmpty || exporter.running)
                            .opacity(readyMixVideos.isEmpty ? 0.5 : 1)
                            .accessibilityIdentifier("mixExportAll")
                        } else {
                        Button { openFirstDraft() } label: {
                            Text(L10n.tr("Edit \(drafts.count) draft\(drafts.count == 1 ? "" : "s") in bulk"))
                                .font(CS.sf(15, .semibold)).foregroundStyle(.white)
                                .frame(maxWidth: .infinity).frame(height: 50)
                                .background(Capsule().fill(LinearGradient(stops: [
                                    .init(color: Color(hex: 0x1969FF), location: 0), .init(color: Color(hex: 0x7C3AED), location: 0.55),
                                    .init(color: Color(hex: 0x19E6FF), location: 1)], startPoint: .leading, endPoint: .trailing)))
                                .shadow(color: Color(hex: 0x1969FF).opacity(0.25), radius: 9, y: 6)
                        }
                        .disabled(drafts.isEmpty)
                        .opacity(drafts.isEmpty ? 0.5 : 1)
                        .accessibilityIdentifier("productBulkEdit")
                        }
                    }
                    .padding(.top, 22)
                    if let message = exporter.message {
                        Text(L10n.tr(message)).font(CS.inter(12, .semibold)).foregroundStyle(CS.cyan).padding(.top, 10)
                            .accessibilityIdentifier("mixExportMessage")
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .task { await load() }
        .refreshable { await load() }
        .onReceive(NotificationCenter.default.publisher(for: .mixProgress)) { _ in Task { await load() } }
        .navigationDestination(item: $openProject) { project in ProjectDetailView(project: project) }
        .alert(L10n.tr("Rename product"), isPresented: $renaming) {
            TextField(L10n.tr("Product name"), text: $newName).accessibilityIdentifier("productRenameField")
            Button(L10n.tr("Save")) { Task { await rename() } }
            Button(L10n.tr("Cancel"), role: .cancel) {}
        }
        .sheet(item: $assigning, onDismiss: { selecting = false; selected = []; Task { await load() } }) { request in
            AssignProductSheet(request: request)
                .environmentObject(appState)
                .presentationDetents([.height(560), .large])
                .presentationBackground(Color(hex: 0x0D121F))
        }
        .alert(L10n.tr("CutSell"), isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button(L10n.tr("OK"), role: .cancel) {}
        } message: { Text(L10n.tr(errorMessage ?? "")) }
    }

    private var readyMixVideos: [CatalogProductVideo] {
        let ready = mixVideos.filter { $0.bucket != "processing" }
        return selecting && !selected.isEmpty ? ready.filter { selected.contains($0.projectID) } : ready
    }

    /// "✦  Mix ready: 12 videos from 9 clips" (Figma 1136:2920); while it is being made, how many are ready.
    private var mixBanner: some View {
        let total = mix?.total ?? mixVideos.count
        let ready = mix?.ready ?? mixVideos.filter { $0.bucket != "processing" }.count
        let text = ready >= total ? "✦  Mix ready: \(total) videos from \(mix?.clipCount ?? 0) clips"
            : "✦  Making your Mix: \(ready) of \(total) ready"
        return Text(L10n.tr(text)).font(CS.inter(13, .semibold)).foregroundStyle(.white)
            .padding(.horizontal, 12.5)
            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            .background(Color(hex: 0x121A2B), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color(hex: 0x19E5FF), lineWidth: 1.5))
            .accessibilityIdentifier("mixBanner")
    }

    private func exportAll() async {
        await exporter.exportAll(readyMixVideos.map(\.projectID), session: appState.session)
        selecting = false; selected = []
    }

    private var subtitle: String {
        guard let detail else { return "" }
        if route.mixID != nil {
            let n = mix?.total ?? mixVideos.count
            return [detail.brand, "Mix", "\(n) video\(n == 1 ? "" : "s")"].compactMap { $0 }.joined(separator: " · ")
        }
        let parts = [detail.brand, detail.category, "\(detail.videos.count) video\(detail.videos.count == 1 ? "" : "s")"]
        return parts.compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
    }

    private func chip(_ key: String, _ title: String, width: CGFloat) -> some View {
        EditsFilterChip(title: title, isSelected: filter == key) { filter = key }
            .frame(width: width)
            .accessibilityIdentifier("productFilter_\(key)")
    }

    /// Figma 1135:3031 (110 × 160): photo, then status + length on a dark strip.
    private func tile(_ video: CatalogProductVideo) -> some View {
        let project = appState.projects.first { $0.projectID == video.projectID }
        let isSelected = selected.contains(video.projectID)
        return Button {
            if selecting {
                if isSelected { selected.remove(video.projectID) } else { selected.insert(video.projectID) }
            } else if let project {
                openProject = project
            }
        } label: {
            Color(hex: 0x050A17)
                .aspectRatio(110.0 / 160.0, contentMode: .fit)
                .overlay {
                    if let image = thumbnails.image(for: video.projectID) {
                        Image(uiImage: image).resizable().scaledToFill()
                    }
                }
                .overlay(alignment: .topLeading) {
                    if video.mixCode != nil {
                        Text(L10n.tr("MIX")).font(CS.inter(9, .bold)).foregroundStyle(.white)
                            .frame(width: 34, height: 18).background(CS.brandFull, in: Capsule()).padding(7)
                            .accessibilityIdentifier("mixBadge")
                    }
                }
                .overlay(alignment: .bottom) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(video.mixCode.map { video.bucket == "processing" ? "\($0) · MAKING" : $0 }
                             ?? (video.bucket == "ready" ? "READY" : video.bucket == "processing" ? "PROCESSING" : "DRAFT"))
                            .font(CS.inter(9, .bold))
                            .foregroundStyle(video.mixCode != nil ? CS.cyan : video.bucket == "ready" ? Color(hex: 0x33D98C)
                                             : video.bucket == "processing" ? CS.cyan : Color(hex: 0x99A3BD))
                            .lineLimit(1).minimumScaleFactor(0.8)
                        Text(Self.length(video.durationSec)).font(CS.inter(11, .semibold)).foregroundStyle(.white)
                    }
                    .padding(.horizontal, 8)
                    .frame(maxWidth: .infinity, minHeight: video.mixCode != nil ? 42 : 36, alignment: .leading)
                    .background(Color.black.opacity(video.mixCode != nil ? 0.6 : 0.55))
                }
                .overlay(alignment: .topTrailing) {
                    if selecting {
                        Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                            .font(.system(size: 20)).foregroundStyle(isSelected ? CS.cyan : .white).padding(7)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(isSelected ? CS.cyan : Color(hex: 0x33405C), lineWidth: isSelected ? 2 : 1))
        }
        .buttonStyle(.plain)
        .task { if let project { await thumbnails.load(project: project, session: appState.session) } }
        .accessibilityIdentifier("productVideo")
    }

    static func length(_ seconds: Double?) -> String {
        guard let seconds, seconds > 0 else { return "–" }
        let total = Int(seconds.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    private func load() async {
        if route.mixID != nil { await MixCenter.shared.refresh(session: appState.session) }
        do {
            detail = try await store.detail(route.productID, session: appState.session, projects: appState.projects)
        } catch {
            errorMessage = "Couldn’t open this product. Please try again."
        }
    }

    private func rename() async {
        let name = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        do {
            try await store.rename(kind: "product", id: route.productID, name: name,
                                   session: appState.session, projects: appState.projects)
            await load()
        } catch {
            errorMessage = "Couldn’t rename it. Please try again."
        }
    }

    /// "Edit N drafts in bulk": opens the drafts one after another, starting with the first.
    private func openFirstDraft() {
        guard let first = drafts.first, let project = appState.projects.first(where: { $0.projectID == first.projectID })
        else { return }
        CutSellDiagnostics.log("catalog_bulk_edit", ["drafts": String(drafts.count)])
        openProject = project
    }
}

// MARK: - "Which product is this?" (Figma 1135:3059)

struct AssignRequest: Identifiable {
    let id = UUID()
    let projectIDs: [String]
    let suggestion: CatalogSuggestion?
}

struct AssignProductSheet: View {
    let request: AssignRequest
    @ObservedObject var store = CatalogStore.shared
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var chosen: String?
    @State private var addingNew = false
    @State private var newBrand = ""
    @State private var newProduct = ""
    @State private var working = false
    @State private var errorMessage: String?

    private var count: Int { request.projectIDs.count }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Color(hex: 0x4D5770)).frame(width: 40, height: 5).frame(maxWidth: .infinity).padding(.top, 9)
            Text(L10n.tr("Which product is this?")).font(CS.inter(22, .bold)).foregroundStyle(.white).padding(.top, 17)
            Text(L10n.tr("\(count) video\(count == 1 ? "" : "s") selected")).font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD))
                .padding(.top, 5)

            if let suggestion = request.suggestion, let label = suggestion.label {
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(L10n.tr("✦ SUGGESTED FROM THE VIDEO")).font(CS.inter(10, .bold)).foregroundStyle(CS.cyan)
                        Text(L10n.tr(label)).font(CS.inter(15, .semibold)).foregroundStyle(.white).lineLimit(1)
                    }
                    Spacer()
                    Button { Task { await move(brand: suggestion.brand, product: suggestion.product) } } label: {
                        Text(L10n.tr("Confirm")).font(CS.inter(13, .semibold)).foregroundStyle(.white)
                            .frame(width: 78, height: 30)
                            .background(Capsule().fill(LinearGradient(stops: [
                                .init(color: Color(hex: 0x19E5FF), location: 0), .init(color: Color(hex: 0x1769FF), location: 0.35),
                                .init(color: Color(hex: 0x7D3BED), location: 0.70), .init(color: Color(hex: 0xFF21D6), location: 1)],
                                startPoint: .leading, endPoint: .trailing)))
                    }
                    .accessibilityIdentifier("assignConfirmSuggestion")
                }
                .padding(.horizontal, 14.5)
                .frame(height: 64)
                .background(Color(hex: 0x141C2E), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x19E5FF), lineWidth: 1.5))
                .padding(.top, 18)
            }

            Text(L10n.tr(store.allProducts.isEmpty ? "You have no products yet" : "Or choose one of your products"))
                .font(CS.inter(12, .semibold)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 16)

            ScrollView {
                VStack(spacing: 0) {
                    ForEach(store.allProducts, id: \.product.productID) { item in
                        Button { chosen = item.product.productID } label: {
                            HStack(spacing: 12) {
                                ZStack {
                                    Image("ProductsRadio").resizable().frame(width: 20, height: 20)
                                    if chosen == item.product.productID { Circle().fill(CS.cyan).frame(width: 10, height: 10) }
                                }
                                Text(L10n.tr("\(item.brand.name) · \(item.product.name)"))
                                    .font(CS.inter(14, .medium)).foregroundStyle(.white).lineLimit(1)
                                Spacer()
                            }
                            .frame(height: 45)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("assignOption_\(item.brand.name) · \(item.product.name)")
                        Rectangle().fill(Color(hex: 0x21293B)).frame(height: 1)
                    }
                    Button { addingNew = true } label: {
                        Text(L10n.tr("+  New brand or product")).font(CS.inter(14, .semibold)).foregroundStyle(CS.cyan)
                            .frame(maxWidth: .infinity, alignment: .leading).frame(height: 40)
                    }
                    .accessibilityIdentifier("assignNew")
                }
            }
            .frame(maxHeight: 260)

            Button { Task { await move(productID: chosen) } } label: {
                Group {
                    if working { ProgressView().tint(.white) } else {
                        Text(L10n.tr("Move \(count) video\(count == 1 ? "" : "s")")).font(CS.sf(15, .semibold)).foregroundStyle(.white)
                    }
                }
                .frame(maxWidth: .infinity).frame(height: 50)
                .background(Capsule().fill(LinearGradient(stops: [
                    .init(color: Color(hex: 0x1969FF), location: 0), .init(color: Color(hex: 0x7C3AED), location: 0.55),
                    .init(color: Color(hex: 0x19E6FF), location: 1)], startPoint: .leading, endPoint: .trailing)))
                .shadow(color: Color(hex: 0x1969FF).opacity(0.25), radius: 9, y: 6)
            }
            .disabled(chosen == nil || working)
            .opacity(chosen == nil ? 0.55 : 1)
            .padding(.top, 12)
            .accessibilityIdentifier("assignMove")
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 23)
        .background(Color(hex: 0x0D121F).ignoresSafeArea())
        .appColorScheme()
        .alert(L10n.tr("New brand or product"), isPresented: $addingNew) {
            TextField(L10n.tr("Brand (e.g. CeraVe)"), text: $newBrand).accessibilityIdentifier("assignNewBrand")
            TextField(L10n.tr("Product (e.g. Hydrating Cleanser)"), text: $newProduct).accessibilityIdentifier("assignNewProduct")
            Button(L10n.tr("Move here")) { Task { await move(brand: newBrand, product: newProduct) } }
            Button(L10n.tr("Cancel"), role: .cancel) {}
        }
        .alert(L10n.tr("CutSell"), isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button(L10n.tr("OK"), role: .cancel) {}
        } message: { Text(L10n.tr(errorMessage ?? "")) }
        .task { await store.refresh(session: appState.session, projects: appState.projects) }
    }

    private func move(productID: String? = nil, brand: String? = nil, product: String? = nil) async {
        if productID == nil && (brand ?? "").trimmingCharacters(in: .whitespaces).isEmpty { return }
        working = true
        defer { working = false }
        do {
            try await store.assign(request.projectIDs, productID: productID, brand: brand,
                                   product: (product ?? "").isEmpty ? brand : product,
                                   session: appState.session, projects: appState.projects)
            dismiss()
        } catch {
            errorMessage = "Couldn’t move the videos. Please try again."
        }
    }
}

// MARK: - Probe-only in-memory catalog (-CutSellCatalogMock)

@MainActor
final class MockCatalog {
    private var brands: [String: String] = [:]                     // id -> name
    private var products: [String: (brand: String, name: String)] = [:]
    private var links: [String: String] = [:]                      // project -> product
    private var seeded = false
    private var demo: [Project] = []

    /// The probe account may have no cuts yet: the mock then brings its own 12 sample videos.
    private func effective(_ projects: [Project]) -> [Project] {
        if !projects.isEmpty { return projects }
        if demo.isEmpty {
            let states = ["draft_ready", "draft_ready", "needs_review", "processing"]
            demo = (0..<12).map { index in
                Project(projectID: "demo_\(index)", userID: "probe", title: "Sample cut \(index + 1)",
                        state: states[index % states.count], latestJobID: nil, sourceCount: 1, updatedAt: nil)
            }
        }
        return demo
    }

    private func seed(_ projects: [Project]) {
        guard !seeded, !projects.isEmpty else { return }
        seeded = true
        let b1 = "brd_cerave", b2 = "brd_dior"
        brands = [b1: "CeraVe", b2: "Dior"]
        products = ["prd_hc": (b1, "Hydrating Cleanser"), "prd_mc": (b1, "Moisturizing Cream"), "prd_md": (b2, "Miss Dior")]
        for (index, project) in projects.enumerated() where index < projects.count - 3 {
            links[project.projectID] = ["prd_hc", "prd_mc", "prd_md"][index % 3]
        }
    }

    private func bucket(_ state: String) -> String {
        state == "draft_ready" ? "ready" : Project.processingStates.contains(state) ? "processing" : "drafts"
    }

    func overview(projects given: [Project]) -> CatalogOverview {
        let projects = effective(given)
        seed(projects)
        var out: [CatalogBrand] = []
        for (brandID, name) in brands.sorted(by: { $0.value < $1.value }) {
            let items = products.filter { $0.value.brand == brandID }.sorted { $0.value.name < $1.value.name }.map { id, value -> CatalogProduct in
                let mine = projects.filter { links[$0.projectID] == id }
                return CatalogProduct(productID: id, name: value.name, category: "Skincare", hasThumbnail: false,
                                      ready: mine.filter { bucket($0.state) == "ready" }.count,
                                      drafts: mine.filter { bucket($0.state) == "drafts" }.count,
                                      processing: mine.filter { bucket($0.state) == "processing" }.count, videos: mine.count)
            }
            if !items.isEmpty { out.append(CatalogBrand(brandID: brandID, name: name, products: items, videos: items.reduce(0) { $0 + $1.videos })) }
        }
        let loose = projects.filter { links[$0.projectID] == nil && bucket($0.state) != "processing" }.map {
            CatalogUncategorized(projectID: $0.projectID, title: $0.title,
                                 suggestion: CatalogSuggestion(brand: "CeraVe", product: "Foaming Cleanser", category: "Skincare", confidence: 0.6))
        }
        return CatalogOverview(brands: out, uncategorized: loose,
                               videos: links.mapValues { CatalogVideoLink(productID: $0, source: "ai") })
    }

    func detail(_ productID: String, projects given: [Project]) throws -> CatalogProductDetail {
        let projects = effective(given)
        guard let product = products[productID] else { throw APIError.http(404, "product not found") }
        var videos = projects.filter { links[$0.projectID] == productID }.map {
            CatalogProductVideo(projectID: $0.projectID, title: $0.title, state: $0.state, bucket: bucket($0.state), durationSec: 30)
        }
        for (id, video) in mixVideos.sorted(by: { Int($0.key.split(separator: "_").last ?? "") ?? 0 < Int($1.key.split(separator: "_").last ?? "") ?? 0 }) where video.productID == productID {
            let live = MixCenter.shared.mixes.flatMap(\.videos).first { $0.projectID == id }
            let ready = live?.state == "ready"
            videos.append(CatalogProductVideo(projectID: id, title: video.code, state: ready ? "draft_ready" : "processing",
                                              bucket: ready ? "ready" : "processing", durationSec: live?.durationSec,
                                              mixID: String(id.split(separator: "_").prefix(2).joined(separator: "_")),
                                              mixCode: video.code))
        }
        return CatalogProductDetail(productID: productID, name: product.name, category: "Skincare", brandID: product.brand,
                                    brand: brands[product.brand], videos: videos)
    }

    func assign(_ ids: [String], productID: String?, brand: String?, product: String?) throws -> CatalogStore.AssignResult {
        var target = productID
        if target == nil {
            let brandName = (brand ?? "").trimmingCharacters(in: .whitespaces)
            let brandID = brands.first { $0.value.lowercased() == brandName.lowercased() }?.key ?? "brd_\(UUID().uuidString.prefix(6))"
            brands[brandID] = brands[brandID] ?? brandName
            let name = (product ?? brandName).trimmingCharacters(in: .whitespaces)
            target = products.first { $0.value.brand == brandID && $0.value.name.lowercased() == name.lowercased() }?.key
                ?? "prd_\(UUID().uuidString.prefix(6))"
            products[target!] = products[target!] ?? (brandID, name)
        }
        for id in ids { links[id] = target }
        return try JSONDecoder().decode(CatalogStore.AssignResult.self,
                                        from: JSONSerialization.data(withJSONObject: ["product_id": target!, "moved": ids.count]))
    }

    func rename(kind: String, id: String, name: String) throws {
        if kind == "product", let value = products[id] { products[id] = (value.brand, name) }
        if kind == "brand" { brands[id] = name }
    }

    private var mixVideos: [String: (productID: String, code: String)] = [:]

    func create(brand: String, product: String) -> String {
        let brandID = brands.first { $0.value.lowercased() == brand.lowercased() }?.key ?? "brd_\(UUID().uuidString.prefix(6))"
        brands[brandID] = brands[brandID] ?? brand
        let id = products.first { $0.value.brand == brandID && $0.value.name.lowercased() == product.lowercased() }?.key
            ?? "prd_\(UUID().uuidString.prefix(6))"
        products[id] = products[id] ?? (brandID, product)
        return id
    }

    func addMix(productID: String, videos: [(String, String)]) {
        for (id, code) in videos { mixVideos[id] = (productID, code) }
    }
}
