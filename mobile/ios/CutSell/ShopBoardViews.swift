import Combine
import PhotosUI
import SwiftUI
import UIKit

// Shop board screens (PO 11 Nov 2026), Figma page "💡 10 · Shop board (propuesta)":
// 1 Home card (3032:3) · 2 Add products (3032:73) · 2b Shape it (3033:77) · 3 Choose a style (3032:143) ·
// 4 Your board (3032:237) · 4a Title & fonts (3039:152) · 4b/4c Background (3035:71 / 3035:146) · 5 Share (3032:286).

// MARK: - Shared pieces

/// The Figma cards: white → #F7F9FF (selected: 2 pt #1769FF border, white → #EEF3FF), a #D3DDF5 base 3 pt below and
/// a soft shadow (0 10 24, #0D1433 8 %).
struct ShopCard: ViewModifier {
    var radius: CGFloat
    var selected = false
    var border: CGFloat = 2

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        return content
            .background {
                shape.fill(LinearGradient(colors: [Color(fixed: 0xFFFFFF), Color(fixed: selected ? 0xEEF3FF : 0xF7F9FF)],
                                          startPoint: .top, endPoint: .bottom))
                    .shadow(color: Color(fixed: 0xD3DDF5), radius: 0, x: 0, y: 3)
                    .shadow(color: Color(fixed: 0x0D1433, opacity: 0.08), radius: 12, x: 0, y: 10)
            }
            .overlay { if selected { shape.strokeBorder(Color(fixed: 0x1769FF), lineWidth: border) } }
    }
}

extension View {
    func shopCard(radius: CGFloat, selected: Bool = false, border: CGFloat = 2) -> some View {
        modifier(ShopCard(radius: radius, selected: selected, border: border))
    }
}

/// "‹" in a white 3D circle (40 pt), as on the B-roll summary.
struct ShopBackButton: View {
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Text(verbatim: "‹")
                .font(CS.inter(22, .bold))
                .foregroundStyle(Color(fixed: 0x0B1020))
                .frame(width: 40, height: 40)
                .softPuff(.white, in: Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(L10n.tr("Back"))
        .accessibilityIdentifier("shopBack")
    }
}

/// "App · Botón 3D / Azul" (blue) and "/ Blanco" (white): 52 high, radius 26, Bold 16.
struct ShopBigButton: View {
    let title: String
    var white = false
    var enabled = true
    var id: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(L10n.tr(title))
                .font(CS.inter(16, .bold))
                .foregroundStyle(white ? Color(fixed: 0x0B1020) : .white)
                .frame(maxWidth: .infinity)
                .frame(height: 52)
                .softPuff(white ? .white : .blue, in: Capsule())
                .opacity(enabled ? 1 : 0.45)
        }
        .buttonStyle(BubblePress())
        .disabled(!enabled)
        .accessibilityIdentifier(id)
    }
}

/// A short notice at the bottom (saved, copied…).
struct ShopNotice: View {
    let text: String
    var body: some View {
        Text(verbatim: text)
            .font(CS.inter(13, .semibold))
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 16).padding(.vertical, 10)
            .background(Color(fixed: 0x0B1020, opacity: 0.88), in: Capsule())
            .padding(.horizontal, 24)
            .accessibilityIdentifier("shopNotice")
    }
}

/// The iPhone's photo library, photos only.
struct ShopPhotoPicker: UIViewControllerRepresentable {
    var limit: Int
    let onPicked: ([UIImage]) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration()
        configuration.filter = .images
        configuration.selectionLimit = max(1, limit)
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) { context.coordinator.parent = self }

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        var parent: ShopPhotoPicker
        init(_ parent: ShopPhotoPicker) { self.parent = parent }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            let parent = self.parent
            Task {
                var images: [UIImage] = []
                for result in results where result.itemProvider.canLoadObject(ofClass: UIImage.self) {
                    let image: UIImage? = await withCheckedContinuation { continuation in
                        result.itemProvider.loadObject(ofClass: UIImage.self) { object, _ in
                            continuation.resume(returning: object as? UIImage)
                        }
                    }
                    if let image { images.append(image) }
                }
                // The sheet's binding closes the picker. Calling picker.dismiss here as well closed the sheet
                // AND the screen under it (build 81: Shop board closed right after picking a photo).
                await MainActor.run { parent.onPicked(images) }
            }
        }
    }
}

// MARK: - 1 Home card

/// Figma 1 (3032:65): 350 × 70, radius 22, 2 pt #1769FF border, white → #EEF3FF, 🛍 on a pink → violet circle, "NEW".
struct ShopBoardHomeCard: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Circle()
                    .fill(LinearGradient(colors: [Color(fixed: 0xFF7DBA), Color(fixed: 0xB83DFF)], startPoint: .top, endPoint: .bottom))
                    .frame(width: 42, height: 42)
                    .overlay(Text(verbatim: "🛍").font(CS.inter(18, .bold)))
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.tr("Shop board"))
                        .font(CS.inter(16, .heavy))
                        .foregroundStyle(Color(fixed: 0x0B1020))
                    Text(L10n.tr("Shoppable outfit posts for TikTok & Instagram"))
                        .font(CS.inter(12, .medium))
                        .foregroundStyle(Color(fixed: 0x6B7385))
                        .lineLimit(1).minimumScaleFactor(0.8)
                }
                Spacer(minLength: 0)
            }
            .padding(.leading, 12)
            .padding(.trailing, 12)
            .frame(height: 70)
            .overlay(alignment: .topTrailing) {
                Text(L10n.tr("NEW"))
                    .font(CS.inter(10, .heavy))
                    .foregroundStyle(.white)
                    .frame(width: 40, height: 20)
                    .background(Color(fixed: 0x1769FF), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .padding(.top, 8).padding(.trailing, 14)
            }
            .shopCard(radius: 22, selected: true)
            .contentShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        }
        .buttonStyle(BubblePress())
        .accessibilityIdentifier("hubShopBoard")
    }
}

// MARK: - The flow

struct ShopBoardFlowView: View {
    let onClose: () -> Void

    enum Step: Equatable { case add, shape(UUID), style, board, share }

    @StateObject private var model: ShopBoardModel
    @State private var step: Step

    /// A new board (Home, Edits "+ New Shop board"), or a saved one opening on "Your board".
    init(saved: ShopBoardModel? = nil, onClose: @escaping () -> Void) {
        self.onClose = onClose
        _model = StateObject(wrappedValue: saved ?? ShopBoardModel())
        _step = State(initialValue: saved == nil ? .add : .board)
    }
    @State private var showingTitle = false
    @State private var showingBackground = false

    var body: some View {
        ZStack {
            switch step {
            case .add:
                ShopAddProductsView(model: model, onBack: onClose, onShape: { step = .shape($0) }, onNext: { step = .style })
            case .shape(let id):
                if let item = model.items.first(where: { $0.id == id }) {
                    ShopShapeView(item: item, onBack: { step = .add }, onUse: { model.useShaped($0, for: id); step = .add },
                                  onKeepFlat: { model.keepFlat(id); step = .add })
                }
            case .style:
                ShopStyleView(model: model, onBack: { step = .add }, onNext: { step = .board })
            case .board:
                ShopBoardScreen(model: model, onBack: { step = .style }, onTitle: { showingTitle = true },
                                onBackground: { showingBackground = true }, onNext: { model.prepareShare(); step = .share })
            case .share:
                ShopShareView(model: model, onBack: { step = .board })
            }
            BottomSheetOverlay(isPresented: $showingTitle) { ShopTitleSheet(model: model, onDone: { showingTitle = false }) }
            BottomSheetOverlay(isPresented: $showingBackground) {
                ShopBackgroundSheet(model: model, onDone: { showingBackground = false })
            }
        }
        .environment(\.colorScheme, .light)
        .animation(.easeOut(duration: 0.2), value: step)
        .onAppear { ShopFonts.registerIfNeeded() }
        // Saves itself (Edits › Shop boards) a moment after each change, and when the flow closes.
        .onReceive(model.objectWillChange.debounce(for: .seconds(1), scheduler: RunLoop.main)) { _ in saveBoard() }
        .onDisappear { saveBoard() }
    }

    private func saveBoard() {
        guard model.busy.isEmpty, !model.items.isEmpty else { return }
        let thumb = model.items.contains(where: { $0.cutout != nil }) ? ShopBoardThumbnail.image(model.snapshot()) : nil
        ShopBoardStore.shared.save(model, thumbnail: thumb)
    }
}

/// Title row of each screen: back circle, then the big title under it (Figma: back at 20,56 · title 29 at 24,112).
private struct ShopHeader: View {
    let title: String
    var subtitle: String? = nil
    let onBack: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ShopBackButton(action: onBack)
                .padding(.leading, -4)
                .padding(.top, 8)
            Text(L10n.tr(title))
                .font(CS.inter(29, .heavy))
                .foregroundStyle(Color(fixed: 0x0B1020))
                .padding(.top, 16)
            if let subtitle {
                Text(L10n.tr(subtitle))
                    .font(CS.inter(15, .medium))
                    .foregroundStyle(Color(fixed: 0x6B7385))
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 24)
    }
}

// MARK: - 2 Add products

struct ShopAddProductsView: View {
    @ObservedObject var model: ShopBoardModel
    let onBack: () -> Void
    let onShape: (UUID) -> Void
    let onNext: () -> Void

    @EnvironmentObject private var appState: AppState
    @State private var pickingPhotos = false
    /// Photos for this one item (a product from the list or a pasted link).
    @State private var photoFor: UUID?
    /// The item that asks for its photo once the current sheet has finished closing.
    @State private var photoNext: UUID?
    @State private var showingProducts = false
    @State private var lastSource: ShopItemSource = .photos
    @State private var notice: String?

    private let columns = [GridItem(.fixed(102), spacing: 12), GridItem(.fixed(102), spacing: 12), GridItem(.fixed(102), spacing: 12)]

    var body: some View {
        ZStack(alignment: .bottom) {
            HomeBackdrop()
            VStack(spacing: 0) {
                ShopHeader(title: "Add products", subtitle: "Pick 3–8 items for your board.", onBack: onBack)
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        sourceRow(.myProducts, "From my products", "CeraVe, Dior… (your product list)") { showingProducts = true }
                        sourceRow(.photos, "From photos", "We remove the background") { pickingPhotos = true }
                        if !model.items.isEmpty {
                            Text(verbatim: String(format: L10n.tr("Your items · %d"), model.items.count))
                                .font(CS.inter(13, .bold))
                                .foregroundStyle(Color(fixed: 0x6B7385))
                                .padding(.top, 16)
                            LazyVGrid(columns: columns, alignment: .leading, spacing: 12) {
                                ForEach(model.items) { item in tile(item) }
                            }
                            if model.items.contains(where: { $0.kind.canShape }) {
                                Text(L10n.tr("✦ Shape it = gives clothes & shoes a real 3D shape (AI)"))
                                    .font(CS.inter(11, .semibold))
                                    .foregroundStyle(Color(fixed: 0x1769FF))
                                    .padding(.top, 4)
                            }
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 26)
                    .padding(.bottom, 24)
                }
                ShopBigButton(title: "Choose a style", enabled: ShopBoardRules.canChooseStyle(model.items) && model.busy.isEmpty,
                              id: "shopChooseStyle", action: onNext)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 24)
            }
            if let notice { ShopNotice(text: notice).padding(.bottom, 96).transition(.opacity) }
        }
        .sheet(isPresented: $pickingPhotos) {
            ShopPhotoPicker(limit: ShopBoardRules.remaining(model.items)) { images in
                pickingPhotos = false
                lastSource = .photos
                model.addPhotos(images)
            }
            .ignoresSafeArea()
        }
        .sheet(item: Binding(get: { photoFor.map(PhotoTarget.init) }, set: { photoFor = $0?.id })) { target in
            ShopPhotoPicker(limit: 1) { images in
                photoFor = nil
                if let image = images.first { model.attach(image, to: target.id) }
            }
            .ignoresSafeArea()
        }
        .sheet(isPresented: $showingProducts, onDismiss: {
            // The product list has no product photos: hers comes from Photos, after the list has closed.
            if let id = photoNext { photoNext = nil; photoFor = id }
        }) {
            ShopProductListSheet { name, productID in
                lastSource = .myProducts
                photoNext = model.addProduct(name: name, productID: productID)
                showingProducts = false
            }
            .environmentObject(appState)
        }
        .accessibilityIdentifier("shopAddProducts")
    }

    private struct PhotoTarget: Identifiable { let id: UUID }

    private func show(_ text: String) {
        withAnimation { notice = text }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { withAnimation { if notice == text { notice = nil } } }
    }

    /// Figma 2 rows: 342 × 62, radius 18, a 36 pt blue circle, Bold 15 + Medium 12, "›".
    private func sourceRow(_ source: ShopItemSource, _ title: String, _ detail: String,
                           action: @escaping () -> Void) -> some View {
        let full = ShopBoardRules.remaining(model.items) == 0
        return Button(action: action) {
            HStack(spacing: 12) {
                Circle()
                    .fill(LinearGradient(colors: [Color(fixed: 0x2F7BFF), Color(fixed: 0x0A55F0)], startPoint: .top, endPoint: .bottom))
                    .frame(width: 36, height: 36)
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.tr(title)).font(CS.inter(15, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                    Text(L10n.tr(detail)).font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                        .lineLimit(1).minimumScaleFactor(0.85)
                }
                Spacer(minLength: 0)
                Text(verbatim: "›").font(CS.inter(22, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
            }
            .padding(.leading, 14).padding(.trailing, 18)
            .frame(height: 62)
            .shopCard(radius: 18, selected: lastSource == source)
            .contentShape(Rectangle())
            .opacity(full ? 0.5 : 1)
        }
        .buttonStyle(BubblePress())
        .disabled(full)
        .accessibilityIdentifier("shopSource_\(source.rawValue)")
    }

    /// Figma 2 thumbnails: 102 × 116, white, #E6E9EF border, radius 16; "✓ No background", "×", "✦ Shape it".
    private func tile(_ item: ShopItem) -> some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Color(fixed: 0xFFFFFF))
            if let cutout = item.cutout {
                Image(decorative: cutout, scale: 1).resizable().interpolation(.high).scaledToFit()
                    .frame(width: 84, height: 80)
                    .frame(width: 102, height: 80)
                    .offset(y: 7)
            } else if model.busy.contains(item.id) {
                ProgressView().tint(CS.accent).frame(width: 102, height: 94)
            } else {
                // A product from the list or a link: its photo still has to come from Photos.
                Button { photoFor = item.id } label: {
                    VStack(spacing: 6) {
                        Text(verbatim: item.name.isEmpty ? L10n.tr("Product") : item.name)
                            .font(CS.inter(11, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                            .lineLimit(2).multilineTextAlignment(.center)
                        Text(L10n.tr("+ Add photo"))
                            .font(CS.inter(10, .bold)).foregroundStyle(Color(fixed: 0x1769FF))
                    }
                    .frame(width: 92, height: 86)
                    .offset(x: 5, y: 22)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("shopAddPhoto")
            }
            if item.backgroundRemoved {
                Text(L10n.tr("✓ No background"))
                    .font(CS.inter(9, .bold)).foregroundStyle(Color(fixed: 0x1FA971))
                    .frame(width: 86, height: 18)
                    .background(Color(fixed: 0xEAF8F1), in: Capsule())
                    .offset(x: 7, y: 91)
            }
            if item.kind.canShape && item.cutout != nil {
                Button { onShape(item.id) } label: {
                    Text(L10n.tr("✦ Shape it"))
                        .font(CS.inter(9, .bold)).foregroundStyle(.white)
                        .frame(width: 64, height: 18)
                        .background(Capsule().fill(LinearGradient(colors: [Color(fixed: 0x2F7BFF), Color(fixed: 0x0A55F0)],
                                                                  startPoint: .top, endPoint: .bottom)))
                }
                .buttonStyle(.plain)
                .offset(x: 5, y: 3)
                .accessibilityIdentifier("shopShapeIt")
            }
            Button { model.remove(item.id) } label: {
                Text(verbatim: "×").font(CS.inter(12, .bold)).foregroundStyle(.white)
                    .frame(width: 20, height: 20)
                    .background(Color(fixed: 0x0B1020), in: Circle())
            }
            .buttonStyle(.plain)
            .offset(x: 77, y: 3)
            .accessibilityLabel(L10n.tr("Remove"))
            .accessibilityIdentifier("shopRemoveItem")
        }
        .frame(width: 102, height: 116)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(fixed: 0xE6E9EF), lineWidth: 1))
    }
}

/// "From my products": her product list (Edits › Products), one product at a time.
struct ShopProductListSheet: View {
    let onPick: (String, String) -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var catalog = CatalogStore.shared
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if catalog.allProducts.isEmpty {
                    Text(L10n.tr("No products yet. Add products in Edits › Products."))
                        .font(CS.inter(13, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                }
                ForEach(Array(catalog.allProducts.enumerated()), id: \.offset) { _, entry in
                    Button { onPick(entry.product.name, entry.product.productID) } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(verbatim: entry.product.name).font(CS.inter(15, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                            Text(verbatim: entry.brand.name).font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                        }
                    }
                    .accessibilityIdentifier("shopProduct")
                }
            }
            .navigationTitle(L10n.tr("From my products"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button(L10n.tr("Cancel")) { dismiss() } } }
        }
        .task { await catalog.refresh(session: appState.session, projects: appState.projects) }
    }
}

// MARK: - 2b Shape it

struct ShopShapeView: View {
    let item: ShopItem
    let onBack: () -> Void
    let onUse: (CGImage) -> Void
    let onKeepFlat: () -> Void

    enum Phase { case working, shaped(CGImage), failed(ShopShapeFailure) }
    @State private var phase: Phase = .working

    var body: some View {
        ZStack {
            HomeBackdrop()
            VStack(alignment: .leading, spacing: 0) {
                ShopHeader(title: "Shape it", onBack: onBack)
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(L10n.tr("We give it a real body shape. Same color, fabric and details."))
                            .font(CS.inter(14, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                            .fixedSize(horizontal: false, vertical: true)
                        HStack(spacing: 18) {
                            card(label: "Before · flat", after: false) {
                                if let flat = item.flat ?? item.cutout {
                                    Image(decorative: flat, scale: 1).resizable().interpolation(.high).scaledToFit()
                                }
                            }
                            card(label: "After · shaped", after: true) { afterContent }
                        }
                        .padding(.top, 14)
                        Text(L10n.tr("Works on")).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x6B7385)).padding(.top, 20)
                        Text(L10n.tr("Pants · tops · sweaters · dresses · shoes"))
                            .font(CS.inter(14, .semibold)).foregroundStyle(Color(fixed: 0x0B1020)).padding(.top, 6)
                        Text(L10n.tr("Shoes come out filled and standing, like someone is wearing them. Bags and accessories stay as they are."))
                            .font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                            .fixedSize(horizontal: false, vertical: true).padding(.top, 6)
                        Text(L10n.tr("✦ Uses AI · 1 credit per item"))
                            .font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0xB26A00))
                            .padding(.horizontal, 14)
                            .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
                            .background(Color(fixed: 0xFFF4E0), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                            .padding(.top, 26)
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 4)
                }
                VStack(spacing: 12) {
                    ShopBigButton(title: "Use shaped", enabled: shapedImage != nil, id: "shopUseShaped") {
                        if let shapedImage { onUse(shapedImage) }
                    }
                    ShopBigButton(title: "Keep flat", white: true, id: "shopKeepFlat", action: onKeepFlat)
                    Button(L10n.tr("Retry")) { Task { await run() } }
                        .font(CS.inter(14, .bold))
                        .foregroundStyle(Color(fixed: 0x1769FF))
                        .disabled(isWorking)
                        .accessibilityIdentifier("shopShapeRetry")
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
            }
        }
        .task { await run() }
        .accessibilityIdentifier("shopShape")
    }

    private var shapedImage: CGImage? { if case .shaped(let image) = phase { return image }; return nil }
    private var isWorking: Bool { if case .working = phase { return true }; return false }

    @ViewBuilder
    private var afterContent: some View {
        switch phase {
        case .working: ProgressView().tint(CS.accent)
        case .shaped(let image): Image(decorative: image, scale: 1).resizable().interpolation(.high).scaledToFit()
        case .failed(let failure):
            Text(verbatim: failure.message)
                .font(CS.inter(12, .semibold)).foregroundStyle(CS.wiError)
                .multilineTextAlignment(.center)
                .padding(10)
                .background(CS.wiErrorBg2, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .accessibilityIdentifier("shopShapeError")
        }
    }

    private func run() async {
        guard let kind = item.kind.shapeKind, let image = item.flat ?? item.cutout else { phase = .failed(.failed); return }
        phase = .working
        switch await ShopShapeClient.shape(image, kind: kind) {
        case .success(let shaped): phase = .shaped(shaped)
        case .failure(let failure): phase = .failed(failure)
        }
    }

    /// 162 × 300 cards on #F3EEE6 → #E6DFD4, radius 22; "After" with a 2.5 pt #1769FF border and a blue label.
    private func card<C: View>(label: String, after: Bool, @ViewBuilder content: () -> C) -> some View {
        let shape = RoundedRectangle(cornerRadius: 22, style: .continuous)
        return ZStack(alignment: .topLeading) {
            content()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.horizontal, 22).padding(.top, 46).padding(.bottom, 18)
            Text(L10n.tr(label))
                .font(CS.inter(11, .bold))
                .foregroundStyle(after ? .white : Color(fixed: 0x0B1020))
                .padding(.horizontal, 12)
                .frame(height: 24)
                .frame(minWidth: after ? 122 : 115, alignment: .leading)
                .background(after ? Color(fixed: 0x1769FF) : .white, in: Capsule())
                .padding(after ? 7.5 : 10)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 300)
        .background {
            shape.fill(LinearGradient(colors: [Color(fixed: 0xF3EEE6), Color(fixed: 0xE6DFD4)], startPoint: .top, endPoint: .bottom))
                .shadow(color: Color(fixed: 0xD3DDF5), radius: 0, x: 0, y: 3)
                .shadow(color: Color(fixed: 0x0D1433, opacity: 0.08), radius: 12, x: 0, y: 10)
        }
        .clipShape(shape)
        .overlay { if after { shape.strokeBorder(Color(fixed: 0x1769FF), lineWidth: 2.5) } }
    }
}

// MARK: - 3 Choose a style

struct ShopStyleView: View {
    @ObservedObject var model: ShopBoardModel
    let onBack: () -> Void
    let onNext: () -> Void

    var body: some View {
        ZStack {
            HomeBackdrop()
            VStack(alignment: .leading, spacing: 0) {
                ShopHeader(title: "Choose a style", onBack: onBack)
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(L10n.tr("Templates")).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
                        LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 12) {
                            ForEach(ShopTemplate.allCases) { template in templateCard(template) }
                            Text(L10n.tr("+ More soon"))
                                .font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
                                .frame(maxWidth: .infinity).frame(height: 258)
                                .shopCard(radius: 18)
                                .accessibilityIdentifier("shopMoreSoon")
                        }
                        .padding(.top, 12)
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 12)
                    .padding(.bottom, 20)
                }
                ShopBigButton(title: "Create board", id: "shopCreateBoard", action: onNext)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 24)
            }
        }
        .accessibilityIdentifier("shopChooseStyleScreen")
    }

    /// 166 × 258, radius 18; a 140.4 pt board preview inside (her own products in the template's arrangement).
    private func templateCard(_ template: ShopTemplate) -> some View {
        let selected = model.template == template
        return Button { model.choose(template) } label: {
            ShopBoardCanvas(snapshot: model.snapshot(template: template), width: 140.4)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .frame(maxWidth: .infinity, maxHeight: 258, alignment: .top)
                .padding(.top, selected ? 6 : 8)
                .frame(height: 258, alignment: .top)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .shopCard(radius: 18, selected: selected)
        }
        .buttonStyle(BubblePress())
        .accessibilityLabel(L10n.tr(template.title))
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityIdentifier("shopTemplate_\(template.rawValue)")
    }
}

// MARK: - 4 Your board

struct ShopBoardScreen: View {
    @ObservedObject var model: ShopBoardModel
    let onBack: () -> Void
    let onTitle: () -> Void
    let onBackground: () -> Void
    let onNext: () -> Void
    /// The product being moved by hand (Figma S4e 3053:196): blue outline, pinch and twist apply to it.
    @State private var selected: UUID?

    var body: some View {
        ZStack {
            HomeBackdrop()
            VStack(spacing: 0) {
                HStack(spacing: 12) {
                    ShopBackButton(action: onBack)
                    Text(L10n.tr("Your board")).font(CS.inter(20, .heavy)).foregroundStyle(Color(fixed: 0x0B1020))
                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
                GeometryReader { geo in
                    // As big as the space allows (PO 13 Nov: the 218-wide board was too small to edit by hand):
                    // all the height between the header and the cards, at most the screen width minus 20 a side.
                    let width = min(geo.size.width - 40, (geo.size.height - 8) * 9 / 16)
                    ShopBoardHandEditor(model: model, selected: $selected, width: width)
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .shadow(color: Color(fixed: 0xD3DDF5), radius: 0, x: 0, y: 3)
                        .shadow(color: Color(fixed: 0x0D1433, opacity: 0.08), radius: 12, x: 0, y: 10)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                        .accessibilityIdentifier("shopBoardPreview")
                }
                .padding(.top, 16)
                HStack(spacing: 10) {
                    actionCard("Title", detail: model.bigTitle, id: "shopTitle", action: onTitle)
                    actionCard("Shuffle", detail: L10n.tr("New layout"), id: "shopShuffle") {
                        selected = nil
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) { model.shuffle() }
                    }
                    actionCard("Background", detail: model.style.matchBackground ? L10n.tr("✨ Match") : model.backgroundName,
                               id: "shopBackground", action: onBackground)
                }
                .padding(.horizontal, 24)
                .padding(.top, 14)
                // Figma S4 / S4e: Medium 12, centred, #6B7385 (nothing picked) or #1769FF (a product picked).
                Text(L10n.tr(selected == nil ? "Drag a product or the title · pinch to resize · twist to turn"
                                             : "Pinch to resize · twist to turn · tap outside to finish"))
                    .font(CS.inter(12, .medium))
                    .foregroundStyle(Color(fixed: selected == nil ? 0x6B7385 : 0x1769FF))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 24)
                    .padding(.top, 14)
                    .accessibilityIdentifier("shopMoveHint")
                Spacer(minLength: 20)
                ShopBigButton(title: "Next: share", id: "shopNextShare", action: onNext)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 24)
            }
        }
        .accessibilityIdentifier("shopBoardScreen")
    }

    /// 106 × 52 cards, radius 16: Bold 13 + Medium 10.
    private func actionCard(_ title: String, detail: String, id: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 4) {
                Text(L10n.tr(title)).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                // One line, cut with "…" (build 88: long titles wrapped and broke the card).
                Text(verbatim: detail).font(CS.inter(10, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                    .lineLimit(1).truncationMode(.tail).minimumScaleFactor(0.85)
            }
            .padding(.horizontal, 12)
            .frame(maxWidth: .infinity, minHeight: 52, maxHeight: 52, alignment: .leading)
            .shopCard(radius: 16)
        }
        .buttonStyle(BubblePress())
        .accessibilityIdentifier(id)
    }
}

/// The board on "Your board", with every product movable by hand (Figma S4e 3053:196).
/// One UIKit layer reads all touches together (build 87 feedback: pinch and twist were hard): one finger moves the
/// product under it, two fingers resize and turn it at the same time — the pinch can start right on any product, no
/// tap first. The picked product comes to the front with the blue outline; a tap on the empty board finishes.
struct ShopBoardHandEditor: View {
    @ObservedObject var model: ShopBoardModel
    @Binding var selected: UUID?
    let width: CGFloat

    var body: some View {
        let snapshot = model.snapshot()
        let size = CGSize(width: width, height: width * 16 / 9)
        ZStack(alignment: .topLeading) {
            ShopBoardCanvas(snapshot: snapshot, width: width)
            if selected == ShopBoardModel.titleID {
                let placed = snapshot.titleFrame(board: size)
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(Color(fixed: 0x1769FF), lineWidth: 2)
                    .frame(width: placed.rect.width + 8, height: placed.rect.height + 8)
                    .rotationEffect(.degrees(placed.rotation))
                    .position(x: placed.rect.midX, y: placed.rect.midY)
                    .allowsHitTesting(false)
            }
            ForEach(snapshot.order, id: \.self) { index in
                if let placed = snapshot.frame(at: index, board: size), selected == snapshot.itemIDs[index] {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Color(fixed: 0x1769FF), lineWidth: 2)
                        .frame(width: placed.rect.width + 8, height: placed.rect.height + 8)
                        .rotationEffect(.degrees(placed.rotation))
                        .position(x: placed.rect.midX, y: placed.rect.midY)
                        .allowsHitTesting(false)
                }
            }
            ShopBoardTouchLayer(
                // The title is drawn over the products, so it is grabbed first.
                itemAt: { point in
                    snapshot.isOnTitle(point, board: size) ? ShopBoardModel.titleID
                        : snapshot.item(at: point, board: size).map { snapshot.itemIDs[$0] }
                },
                selected: { selected },
                onTap: { id in
                    selected = id
                    if let id { model.raise(id) }
                },
                onBegin: { id in
                    selected = id
                    model.raise(id)
                    return model.adjusts[id] ?? ShopAdjust()
                },
                onChange: { id, start, move, pinch, twist in
                    var a = ShopAdjust.dragged(start, by: move, board: size)
                    a = ShopAdjust.pinched(a, by: pinch)
                    model.adjusts[id] = ShopAdjust.twisted(a, by: twist)
                })
            .accessibilityIdentifier("shopBoardTouchLayer")
        }
        .frame(width: size.width, height: size.height)
    }
}

/// Pan + pinch + rotation recognised together over the board.
struct ShopBoardTouchLayer: UIViewRepresentable {
    let itemAt: (CGPoint) -> UUID?
    let selected: () -> UUID?
    let onTap: (UUID?) -> Void
    /// A gesture starts on a product: returns its hand edit before the gesture.
    let onBegin: (UUID) -> ShopAdjust
    /// id, start, finger movement (points), pinch factor, twist (degrees).
    let onChange: (UUID, ShopAdjust, CGSize, CGFloat, Double) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        let c = context.coordinator
        let pan = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.pan(_:)))
        pan.maximumNumberOfTouches = 2
        let pinch = UIPinchGestureRecognizer(target: c, action: #selector(Coordinator.pinch(_:)))
        let rotate = UIRotationGestureRecognizer(target: c, action: #selector(Coordinator.rotate(_:)))
        let tap = UITapGestureRecognizer(target: c, action: #selector(Coordinator.tap(_:)))
        for g in [pan, pinch, rotate] as [UIGestureRecognizer] { g.delegate = c; view.addGestureRecognizer(g) }
        view.addGestureRecognizer(tap)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) { context.coordinator.parent = self }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var parent: ShopBoardTouchLayer
        private var active: UUID?
        private var start = ShopAdjust()
        private var move = CGSize.zero, scale: CGFloat = 1, turn: Double = 0
        private var live = 0

        init(_ parent: ShopBoardTouchLayer) { self.parent = parent }

        func gestureRecognizer(_ g: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool { true }

        /// Starts on the product under the fingers (or the picked one for a two-finger gesture beside it).
        func gestureRecognizerShouldBegin(_ g: UIGestureRecognizer) -> Bool {
            if active != nil { return true }
            let point = g.location(in: g.view)
            return parent.itemAt(point) != nil || (g.numberOfTouches >= 2 && parent.selected() != nil)
        }

        private func begin(_ g: UIGestureRecognizer) {
            live += 1
            guard active == nil else { return }
            let point = g.location(in: g.view)
            guard let id = parent.itemAt(point) ?? (g.numberOfTouches >= 2 ? parent.selected() : nil) else { return }
            active = id
            start = parent.onBegin(id)
            move = .zero; scale = 1; turn = 0
        }

        private func end() {
            live = max(0, live - 1)
            if live == 0 { active = nil }
        }

        private func push() { if let id = active { parent.onChange(id, start, move, scale, turn) } }

        @objc func pan(_ g: UIPanGestureRecognizer) {
            switch g.state {
            case .began: begin(g)
            case .changed:
                let t = g.translation(in: g.view)
                move = CGSize(width: t.x, height: t.y)
                push()
            case .ended, .cancelled, .failed: end()
            default: break
            }
        }

        @objc func pinch(_ g: UIPinchGestureRecognizer) {
            switch g.state {
            case .began: begin(g)
            case .changed: scale = g.scale; push()
            case .ended, .cancelled, .failed: end()
            default: break
            }
        }

        @objc func rotate(_ g: UIRotationGestureRecognizer) {
            switch g.state {
            case .began: begin(g)
            case .changed: turn = Double(g.rotation) * 180 / .pi; push()
            case .ended, .cancelled, .failed: end()
            default: break
            }
        }

        @objc func tap(_ g: UITapGestureRecognizer) {
            parent.onTap(parent.itemAt(g.location(in: g.view)))
        }
    }
}

// MARK: - 4a Title & fonts

struct ShopTitleSheet: View {
    @ObservedObject var model: ShopBoardModel
    let onDone: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(L10n.tr("Title")).font(CS.inter(22, .heavy)).foregroundStyle(Color(fixed: 0x0B1020))
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    label("Big title  (e.g. Amazon finds, Walmart, your shop)").padding(.top, 16)
                    field($model.bigTitle, id: "shopBigTitle").padding(.top, 6)
                    label("Script line").padding(.top, 10)
                    field($model.scriptLine, id: "shopScriptLine").padding(.top, 6)
                    label("Letter style · all free").padding(.top, 12)
                    LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10),
                                        GridItem(.flexible(), spacing: 10)], spacing: 12) {
                        ForEach(ShopFontPair.allCases) { pair in pairTile(pair) }
                    }
                    .padding(.top, 10)
                    titleColorRow.padding(.top, 18)
                }
            }
            ShopBigButton(title: "Done", id: "shopTitleDone", action: onDone)
                .padding(.horizontal, -4)
                .padding(.top, 20)
                .padding(.bottom, 16)
        }
        .padding(.horizontal, 24)
        .frame(maxHeight: 600)
        .accessibilityIdentifier("shopTitleSheet")
    }

    private func label(_ text: String) -> some View {
        Text(L10n.tr(text)).font(CS.inter(12, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
    }

    private func field(_ text: Binding<String>, id: String) -> some View {
        TextField("", text: text)
            .font(CS.inter(15, .semibold))
            .foregroundStyle(Color(fixed: 0x0B1020))
            .padding(.horizontal, 13)
            .frame(height: 44)
            .background(Color(fixed: 0xFFFFFF), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(fixed: 0xE6E9EF), lineWidth: 1))
            .accessibilityIdentifier(id)
    }

    /// 106 × 100 tiles, radius 16: "CUTSELL" in the title face, "chic" in the script face, the name.
    private func pairTile(_ pair: ShopFontPair) -> some View {
        let selected = model.fontPair == pair
        let ink = Color(fixed: 0x1E2A4A)
        return Button { model.fontPair = pair } label: {
            VStack(spacing: 0) {
                Text(verbatim: "CUTSELL").font(ShopFonts.font(pair.titleFace, size: 18 * pair.titleScale, opticalSize: 18))
                    .foregroundStyle(ink).frame(height: 24).padding(.top, 12)
                Text(verbatim: "chic").font(ShopFonts.font(pair.scriptFace, size: 24)).foregroundStyle(ink).frame(height: 30)
                Spacer(minLength: 0)
                Text(L10n.tr(pair.title)).font(CS.inter(11, .bold))
                    .foregroundStyle(selected ? Color(fixed: 0x1769FF) : Color(fixed: 0x6B7385))
                    .padding(.bottom, 8)
            }
            .frame(maxWidth: .infinity).frame(height: 100)
            .background(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(LinearGradient(colors: [Color(fixed: 0xFFFFFF), Color(fixed: selected ? 0xEEF3FF : 0xF7F9FF)],
                                     startPoint: .top, endPoint: .bottom)))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(selected ? Color(fixed: 0x1769FF) : Color(fixed: 0xE6E9EF), lineWidth: selected ? 2 : 1))
        }
        .buttonStyle(BubblePress())
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityIdentifier("shopFont_\(pair.rawValue)")
    }

    /// "Title color" · "✨ Matches your products" + four 24 pt swatches (the matched colour first).
    private var titleColorRow: some View {
        let matched = ShopBoardStyle(matchBackground: model.style.matchBackground, background: model.style.background,
                                     matchTitle: true).resolvedTitleColor(palette: model.palette, template: model.template)
        let others = model.palette.filter { $0.distance2(matched) > 0.01 }.prefix(3)
        let swatches = [matched] + others
        return HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.tr("Title color")).font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                Text(L10n.tr(model.style.matchTitle ? "✨ Matches your products" : "Your pick"))
                    .font(CS.inter(11, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
            }
            Spacer(minLength: 8)
            HStack(spacing: 8) {
                ForEach(Array(swatches.enumerated()), id: \.offset) { index, color in
                    let chosen = index == 0 ? model.style.matchTitle : (!model.style.matchTitle && model.style.titleColor == color)
                    Button {
                        if index == 0 { model.style.matchTitle = true } else { model.style.pickTitleColor(color) }
                    } label: {
                        Circle().fill(color.color).frame(width: 24, height: 24)
                            .overlay(Circle().strokeBorder(chosen ? Color(fixed: 0x1769FF) : .white, lineWidth: chosen ? 3 : 2))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(L10n.tr(index == 0 ? "✨ Matches your products" : ShopPalette.name(of: color)))
                    .accessibilityIdentifier("shopTitleColor_\(index)")
                }
            }
        }
        .padding(.horizontal, 14)
        .frame(height: 46)
        .background(Color(fixed: 0xF3F5F9), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

// MARK: - 4b / 4c Background

struct ShopBackgroundSheet: View {
    @ObservedObject var model: ShopBoardModel
    let onDone: () -> Void

    enum Tab: String, CaseIterable { case colors = "Colors", textures = "Textures", photo = "My photo" }
    @State private var tab: Tab = .colors
    @State private var customColor = Color(fixed: 0xF1ECE4)
    @State private var pickingPhoto = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(L10n.tr("Background")).font(CS.inter(22, .heavy)).foregroundStyle(Color(fixed: 0x0B1020))
            tabs.padding(.top, 14)
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    switch tab {
                    case .colors: colors
                    case .textures: textures
                    case .photo: photo
                    }
                }
                .padding(.top, 12)
                .padding(.horizontal, 2)
            }
            ShopBigButton(title: "Done", id: "shopBackgroundDone", action: onDone)
                .padding(.horizontal, -4)
                .padding(.top, 14)
                .padding(.bottom, 16)
        }
        .padding(.horizontal, 24)
        .frame(maxHeight: 660)
        .onAppear {
            if case .texture = model.style.background { tab = .textures }
            if case .photo = model.style.background, !model.style.matchBackground { tab = .photo }
        }
        .sheet(isPresented: $pickingPhoto) {
            ShopPhotoPicker(limit: 1) { images in
                pickingPhoto = false
                if let image = images.first {
                    model.photoBackground = image
                    model.style.pick(.photo)
                }
            }
            .ignoresSafeArea()
        }
        .accessibilityIdentifier("shopBackgroundSheet")
    }

    /// Colors · Textures · My photo: #F3F5F9 pill, the chosen one blue (110 × 32).
    private var tabs: some View {
        HStack(spacing: 2) {
            ForEach(Tab.allCases, id: \.self) { item in
                let selected = item == tab
                Button { withAnimation(.easeOut(duration: 0.15)) { tab = item } } label: {
                    Text(L10n.tr(item.rawValue)).font(CS.inter(14, .bold))
                        .foregroundStyle(selected ? .white : Color(fixed: 0x6B7385))
                        .frame(maxWidth: .infinity).frame(height: 32)
                        .background {
                            if selected {
                                Capsule().fill(LinearGradient(colors: [Color(fixed: 0x2F7BFF), Color(fixed: 0x0A55F0)],
                                                              startPoint: .top, endPoint: .bottom))
                            }
                        }
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(selected ? .isSelected : [])
                .accessibilityIdentifier("shopBgTab_\(item.rawValue)")
            }
        }
        .padding(4)
        .frame(height: 40)
        .background(Color(fixed: 0xF3F5F9), in: Capsule())
    }

    private var matched: (top: ShopRGB, bottom: ShopRGB) { ShopPalette.softBackground(for: model.palette) }

    // 4b
    private var colors: some View {
        let on = model.style.matchBackground
        let matchTitle = ShopPalette.titleColor(for: model.palette, on: matched.top.mixed(matched.bottom, 0.5))
        return VStack(alignment: .leading, spacing: 0) {
            Button { model.style.turnOnMatch() } label: {
                HStack(spacing: 10) {
                    Circle()
                        .fill(on ? AnyShapeStyle(LinearGradient(colors: [Color(fixed: 0x2F7BFF), Color(fixed: 0x0A55F0)],
                                                                startPoint: .top, endPoint: .bottom)) : AnyShapeStyle(Color.white))
                        .overlay(Circle().strokeBorder(on ? Color.clear : Color(fixed: 0xD5DAE3), lineWidth: 1.5))
                        .frame(width: 24, height: 24)
                        .overlay { if on { Text(verbatim: "✓").font(CS.inter(13, .heavy)).foregroundStyle(.white) } }
                    VStack(alignment: .leading, spacing: 4) {
                        Text(L10n.tr("✨ Match my products")).font(CS.inter(15, .heavy)).foregroundStyle(Color(fixed: 0x0B1020))
                        Text(L10n.tr("Auto harmony from your items")).font(CS.inter(11, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                    }
                    Spacer(minLength: 0)
                    Text(verbatim: "Aa").font(CS.inter(16, .heavy)).foregroundStyle(matchTitle.color)
                        .frame(width: 56, height: 40)
                        .background(RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(LinearGradient(colors: [matched.top.color, matched.bottom.color], startPoint: .top, endPoint: .bottom)))
                        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Color(fixed: 0xE6E9EF), lineWidth: 1))
                }
                .padding(.leading, 12).padding(.trailing, 14)
                .frame(height: 66)
                .shopCard(radius: 18, selected: on)
                .overlay(alignment: .topTrailing) {
                    Text(L10n.tr("DEFAULT · ON")).font(CS.inter(8, .heavy)).foregroundStyle(.white)
                        .frame(width: 80, height: 18)
                        .background(Color(fixed: 0x1FA971), in: Capsule())
                        .offset(x: -2, y: -11)
                }
            }
            .buttonStyle(.plain)
            .padding(.top, 11)
            .accessibilityAddTraits(on ? .isSelected : [])
            .accessibilityIdentifier("shopMatch")

            Text(L10n.tr("Or pick a color yourself")).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                .padding(.top, 14)
            HStack(alignment: .top, spacing: 12) {
                ForEach(Array(model.palette.prefix(5).enumerated()), id: \.offset) { _, color in
                    swatch(color.color, name: ShopPalette.name(of: color), chosen: !on && model.style.background == .color(color)) {
                        model.style.pick(.color(color))
                    }
                }
                VStack(spacing: 4) {
                    ZStack {
                        Circle().fill(AngularGradient(colors: [.red, .yellow, .green, .cyan, .blue, .purple, .red], center: .center))
                            .frame(width: 46, height: 46)
                        Text(verbatim: "+").font(CS.inter(22, .heavy)).foregroundStyle(.white)
                            .allowsHitTesting(false)
                        ColorPicker("", selection: $customColor, supportsOpacity: false)
                            .labelsHidden()
                            .opacity(0.02)
                            .frame(width: 46, height: 46)
                            .scaleEffect(1.6)
                            .accessibilityIdentifier("shopCustomColor")
                    }
                    .frame(width: 46, height: 46)
                    .clipShape(Circle())
                    Text(L10n.tr("Custom")).font(CS.inter(11, .semibold)).foregroundStyle(Color(fixed: 0x6B7385))
                }
                .frame(width: 54)
            }
            .padding(.top, 10)
            .onChange(of: customColor) { _, value in model.style.pick(.color(ShopRGB(UIColor(value)))) }
            Text(L10n.tr("Colors from your products, or any color with Custom. Picking one turns off Match."))
                .font(CS.inter(11, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 8)
            Text(L10n.tr("Classic")).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x0B1020)).padding(.top, 22)
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 4), spacing: 10) {
                ForEach(ShopClassicColor.allCases) { classic in
                    tile(name: classic.title, chosen: !on && model.style.background == .classic(classic),
                         id: "shopClassic_\(classic.rawValue)") {
                        LinearGradient(colors: [classic.top.color, classic.bottom.color], startPoint: .top, endPoint: .bottom)
                    } action: { model.style.pick(.classic(classic)) }
                }
            }
            .padding(.top, 10)
        }
    }

    // 4c
    private var textures: some View {
        let on = model.style.matchBackground
        return VStack(alignment: .leading, spacing: 0) {
            Button { model.style.turnOnMatch() } label: {
                HStack(spacing: 0) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(L10n.tr("✨ Match my products")).font(CS.inter(15, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                        Text(L10n.tr("Picks a color that makes your items pop · free")).font(CS.inter(11, .medium))
                            .foregroundStyle(Color(fixed: 0x6B7385)).lineLimit(1).minimumScaleFactor(0.8)
                    }
                    Spacer(minLength: 6)
                    HStack(spacing: 4) {
                        ForEach(Array(model.palette.prefix(3).enumerated()), id: \.offset) { _, color in
                            Circle().fill(color.color).frame(width: 18, height: 18)
                                .overlay(Circle().strokeBorder(.white, lineWidth: 2))
                        }
                    }
                }
                .padding(.horizontal, 12.5)
                .frame(height: 56)
                .shopCard(radius: 16, selected: on, border: 1.5)
            }
            .buttonStyle(.plain)
            .padding(.top, 2)
            .accessibilityAddTraits(on ? .isSelected : [])
            .accessibilityIdentifier("shopMatchTexture")
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 4), spacing: 10) {
                ForEach(ShopTexture.allCases) { texture in
                    let chosen = model.style.background == .texture(texture)
                    tile(name: texture.title, chosen: chosen, id: "shopTexture_\(texture.rawValue)") {
                        Image(uiImage: ShopTextureRenderer.image(texture, size: CGSize(width: 148, height: 148)))
                            .resizable()
                    } action: {
                        // A texture keeps Match on when it was on (Match then tints it).
                        let keepMatch = model.style.matchBackground
                        model.style.pick(.texture(texture))
                        if keepMatch { model.style.turnOnMatch() }
                    }
                }
            }
            .padding(.top, 18)
        }
    }

    // My photo (not in Figma: the minimal picker).
    private var photo: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let image = model.photoBackground {
                Image(uiImage: image).resizable().scaledToFill()
                    .frame(width: 90, height: 160).clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(model.style.background == .photo && !model.style.matchBackground ? Color(fixed: 0x1769FF) : .clear,
                                      lineWidth: 3))
                    .onTapGesture { model.style.pick(.photo) }
            }
            ShopBigButton(title: model.photoBackground == nil ? "Choose a photo" : "Choose another photo", white: true,
                          id: "shopPickPhotoBackground") { pickingPhoto = true }
                .padding(.horizontal, -2)
        }
        .padding(.top, 6)
    }

    private func swatch(_ color: Color, name: String, chosen: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Circle().fill(color).frame(width: 46, height: 46)
                    .overlay(Circle().strokeBorder(chosen ? Color(fixed: 0x1769FF) : Color(fixed: 0xE6E9EF), lineWidth: chosen ? 3 : 1))
                Text(L10n.tr(name)).font(CS.inter(11, .semibold)).foregroundStyle(Color(fixed: 0x6B7385)).lineLimit(1)
            }
            .frame(width: 46)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(chosen ? .isSelected : [])
        .accessibilityIdentifier("shopProductColor")
    }

    /// 74 × 74 tiles, radius 18, #E6E9EF border (chosen: 3 pt #1769FF), name SemiBold 12 under it.
    private func tile<C: View>(name: String, chosen: Bool, id: String, @ViewBuilder fill: () -> C,
                               action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                fill()
                    .aspectRatio(1, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(chosen ? Color(fixed: 0x1769FF) : Color(fixed: 0xE6E9EF), lineWidth: chosen ? 3 : 1))
                Text(L10n.tr(name)).font(CS.inter(12, .semibold)).foregroundStyle(Color(fixed: 0x0B1020))
            }
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(chosen ? .isSelected : [])
        .accessibilityIdentifier(id)
    }
}

// MARK: - 5 Share

struct ShopShareView: View {
    @ObservedObject var model: ShopBoardModel
    let onBack: () -> Void
    @State private var notice: String?
    @State private var working = false
    @State private var showingTikTokTip = false

    var body: some View {
        ZStack(alignment: .bottom) {
            HomeBackdrop()
            VStack(alignment: .leading, spacing: 0) {
                ShopHeader(title: "Share your board", onBack: onBack)
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        HStack(alignment: .top, spacing: 14) {
                            ShopBoardCanvas(snapshot: model.snapshot(), width: 85.8)
                                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                            VStack(alignment: .leading, spacing: 0) {
                                Text(L10n.tr("Amazon / affiliate link")).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
                                TextField(L10n.tr("amazon.com/…  (optional)"), text: $model.affiliateLink)
                                    .font(CS.inter(14, .semibold)).foregroundStyle(Color(fixed: 0x0B1020))
                                    .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                                    .padding(.horizontal, 12).frame(height: 44)
                                    .shopCard(radius: 14)
                                    .padding(.top, 8)
                                    .accessibilityIdentifier("shopAffiliateLink")
                                Text(L10n.tr("Caption")).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
                                    .padding(.top, 14)
                                TextField("", text: $model.caption, axis: .vertical)
                                    .font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x0B1020))
                                    .lineLimit(4...8)
                                    .padding(.horizontal, 12).padding(.vertical, 10)
                                    .frame(minHeight: 90, alignment: .topLeading)
                                    .shopCard(radius: 14)
                                    .padding(.top, 8)
                                    .accessibilityIdentifier("shopCaption")
                            }
                        }
                        VStack(spacing: 12) {
                            row("Post as TikTok photo", "Tag your TikTok Shop product in TikTok", highlighted: true,
                                id: "shopPostTikTok") { showingTikTokTip = true }
                            row("Instagram story", "Your link goes on the link sticker", id: "shopInstagramStory") { await instagram() }
                            row("Save to Photos", "9:16 image · link & caption copied", id: "shopSaveToPhotos") { await save() }
                        }
                        .padding(.top, 22)
                        Text(L10n.tr("TikTok Shop products are tagged inside TikTok. Your Amazon / affiliate link goes in your bio, your story sticker or wherever you paste it. We add \"#ad\" so your post follows the rules for affiliate links."))
                            .font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.top, 24)
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 20)
                    .padding(.bottom, 20)
                }
                ShopBigButton(title: "Copy link & caption", white: true, id: "shopCopyLinkCaption") {
                    model.caption = model.finalCaption
                    UIPasteboard.general.string = ShopCaption.clipboard(caption: model.caption, title: model.bigTitle,
                                                                        link: model.affiliateLink)
                    show(L10n.tr("Link and caption copied"))
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
            if let notice { ShopNotice(text: notice).padding(.bottom, 96).transition(.opacity) }
        }
        .sheet(isPresented: $showingTikTokTip) {
            ShopTikTokTipSheet {
                showingTikTokTip = false
                Task {
                    try? await Task.sleep(nanoseconds: 450_000_000)   // the sheet has closed before TikTok opens
                    await postTikTok()
                }
            }
            .presentationDetents([.height(437)])
            .presentationDragIndicator(.hidden)
            .presentationCornerRadius(28)
        }
        .tiktokReturnPrompt()
        .accessibilityIdentifier("shopShare")
    }

    private func show(_ text: String) {
        withAnimation { notice = text }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { withAnimation { if notice == text { notice = nil } } }
    }

    private func exported() -> UIImage? {
        model.caption = model.finalCaption          // "#ad" (+ the Amazon line) is in what she posts
        return ShopBoardExport.image(model.snapshot())
    }

    private func postTikTok() async {
        guard let image = exported() else { return }
        await TikTokPoster.shared.postPhoto(image, caption: model.caption, onNotice: { show(L10n.tr($0)) })
    }

    private func instagram() async {
        guard let image = exported() else { return }
        if ShopShare.instagramStory(image, link: model.affiliateLink) { return }
        // No Facebook App ID yet: the picture goes to Photos, the link is copied, Instagram opens.
        do {
            try await ShopShare.saveToPhotos(image)
            if !model.affiliateLink.isEmpty { UIPasteboard.general.string = model.affiliateLink }
            show(L10n.tr("Saved to Photos. Link copied for your Shop now sticker."))
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            ShopShare.openInstagram()
        } catch {
            show(L10n.tr("Allow CutSell to add photos to Photos, then try again."))
        }
    }

    private func save() async {
        guard let image = exported() else { return }
        do {
            try await ShopShare.saveToPhotos(image)
            UIPasteboard.general.string = ShopCaption.clipboard(caption: model.caption, title: model.bigTitle,
                                                                link: model.affiliateLink)
            show(L10n.tr("Saved to Photos. Link and caption copied."))
        } catch {
            show(L10n.tr("Allow CutSell to add photos to Photos, then try again."))
        }
    }

    /// 342 × 60, radius 18: Bold 15 + Medium 12, "›".
    private func row(_ title: String, _ detail: String, highlighted: Bool = false, id: String,
                     action: @escaping () async -> Void) -> some View {
        Button {
            guard !working else { return }
            working = true
            Task { await action(); working = false }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.tr(title)).font(CS.inter(15, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                    Text(L10n.tr(detail)).font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                }
                Spacer()
                Text(verbatim: "›").font(CS.inter(22, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
            }
            .padding(.leading, 15).padding(.trailing, 18)
            .frame(height: 60)
            .shopCard(radius: 18, selected: highlighted)
            .contentShape(Rectangle())
        }
        .buttonStyle(BubblePress())
        .accessibilityIdentifier(id)
    }
}

// MARK: - 5b TikTok Shop tip

/// Figma S5b (3051:264): white sheet, radius 28, grabber 40 × 5 #E6E9EF, ExtraBold 22 title, Medium 13 #6B7385,
/// three #F3F5F9 steps (radius 14) with a 24 pt #1769FF number, blue 3D "Open TikTok", Medium 12 centred footer.
struct ShopTikTokTipSheet: View {
    let onOpen: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 3).fill(Color(fixed: 0xE6E9EF)).frame(width: 40, height: 5)
            Text(L10n.tr("Tag your TikTok Shop product"))
                .font(CS.inter(22, .heavy)).foregroundStyle(Color(fixed: 0x0B1020))
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(L10n.tr("TikTok Shop links only work inside TikTok. We open TikTok with your board and copy the caption."))
                .font(CS.inter(13, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            ForEach(Array(ShopTikTokTipSheet.steps.enumerated()), id: \.offset) { index, step in
                HStack(spacing: 12) {
                    Text(verbatim: "\(index + 1)").font(CS.inter(12, .bold)).foregroundStyle(.white)
                        .frame(width: 24, height: 24).background(Color(fixed: 0x1769FF), in: Circle())
                    Text(L10n.tr(step)).font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 14).padding(.vertical, 12)
                .background(Color(fixed: 0xF3F5F9), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            ShopBigButton(title: "Open TikTok", id: "shopOpenTikTok", action: onOpen)
            Text(L10n.tr("Using Amazon instead? Put your link in your bio — the caption already says \"link in bio\"."))
                .font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, 14).padding(.horizontal, 24).padding(.bottom, 34)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(Color.white)
        .accessibilityIdentifier("shopTikTokTip")
    }

    static let steps = ["We open TikTok with your board", "Tap \"Add link\" → \"Products\"", "Pick it from your Showcase and post"]
}

// MARK: - Font credits (Profile › Help & Support)

/// Not in Figma: the twelve Shop board letters, their authors and licence, in the app's list style.
struct ShopFontCreditsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var shown: ShopFontCredit?

    var body: some View {
        YouScreen(title: "Font credits", subtitle: "Shop board letters, bundled free under the SIL Open Font License",
                  back: { dismiss() }) {
            YouSection(label: "FONTS") {
                ForEach(Array(ShopFontCredit.all.enumerated()), id: \.element.id) { index, credit in
                    YouRow(title: credit.family, detail: credit.designer + " · " + ShopFontCredit.license,
                           divider: index < ShopFontCredit.all.count - 1, id: "fontCredit") { shown = credit }
                }
            }
        }
        .sheet(item: $shown) { credit in
            NavigationStack {
                ScrollView {
                    Text(verbatim: Self.licenseText(credit))
                        .font(.system(size: 12, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(16)
                }
                .navigationTitle(credit.family)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button(L10n.tr("Done")) { shown = nil } } }
            }
        }
    }

    static func licenseText(_ credit: ShopFontCredit) -> String {
        guard let url = Bundle.main.url(forResource: credit.licenseFile, withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return ShopFontCredit.license }
        return text
    }
}
