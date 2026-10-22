import Photos
import SwiftUI

// MARK: - Model (/v1/mix …)

struct MixPlanVideo: Codable, Hashable, Identifiable {
    let n: Int
    let code: String
    let names: [String]
    var id: Int { n }
}

struct MixPlan: Codable {
    let total: Int
    let videos: [MixPlanVideo]
}

struct MixVideo: Codable, Hashable, Identifiable {
    let n: Int
    let code: String
    let projectID: String
    var state: String?
    var durationSec: Double?
    var id: String { projectID }
    enum CodingKeys: String, CodingKey { case n, code, projectID = "project_id", state, durationSec = "duration_sec" }
}

struct MixFailedClip: Codable, Hashable {
    let clipID: String
    let section: String?
    let label: String?
    let error: String?
    enum CodingKeys: String, CodingKey { case clipID = "clip_id", section, label, error }
}

struct MixSummary: Codable, Identifiable, Hashable {
    let mixID: String
    var state: String?
    let productID: String?
    let brand: String?
    let product: String?
    var videos: [MixVideo]
    var ready: Int
    let total: Int
    let clipCount: Int
    var failedClips: [MixFailedClip]
    var id: String { mixID }
    var isActive: Bool { ready < total }
    enum CodingKeys: String, CodingKey {
        case mixID = "mix_id", state, productID = "product_id", brand, product, videos, ready, total
        case clipCount = "clip_count", failedClips = "failed_clips"
    }
}

/// One part of the Mix being set up (Hook, Body, Benefit, CTA…) and its recorded versions.
struct MixSectionDraft: Identifiable, Hashable {
    let id = UUID()
    var name: String
    var clips: [MixClipDraft] = []
}

struct MixClipDraft: Identifiable, Hashable {
    let id = UUID()
    let video: LocalVideo
    var thumbnail: UIImage?
}

// MARK: - Store

/// The creator's Mixes: the preview of the combinations, starting one, following it in Edits · Processing and
/// asking again for a clip that failed. `-CutSellMixMock` (simulator probe) answers from memory.
@MainActor
final class MixCenter: ObservableObject {
    static let shared = MixCenter()
    static let maxSections = 6, maxClips = 10, maxVideos = 100

    @Published private(set) var mixes: [MixSummary] = []
    private var loop: Task<Void, Never>?
    let mock = ProcessInfo.processInfo.arguments.contains("-CutSellMixMock")
    private var mockStarted: [String: Date] = [:]

    var active: [MixSummary] { mixes.filter(\.isActive) }
    /// The videos of Mixes still being made (shown as one group in Processing, not one row each).
    var activeProjectIDs: Set<String> { Set(active.flatMap { $0.videos.map(\.projectID) }) }
    var mixIDs: Set<String> { Set(mixes.map(\.mixID)) }
    func mix(containing projectID: String) -> MixSummary? { mixes.first { $0.videos.contains { $0.projectID == projectID } } }

    func plan(sections: [MixSectionDraft], howMany: Int?, session: CutSellSession?) async throws -> MixPlan {
        let names = sections.map(\.name)
        let counts = sections.map { max(1, $0.clips.count) }
        if mock || session == nil { return Self.localPlan(names: names, counts: counts, howMany: howMany) }
        struct Body: Encodable { let user_id: String; let sections: [String]; let counts: [Int]; let how_many: Int? }
        return try await APIClient.shared.request("/v1/mix/plan", method: "POST",
                                                  body: Body(user_id: session!.userID, sections: names, counts: counts,
                                                             how_many: howMany), timeout: 15)
    }

    /// The same order the server makes (every clip used evenly, a new start and a new end each time).
    static func localPlan(names: [String], counts: [Int], howMany: Int?) -> MixPlan {
        var every: [[Int]] = [[]]
        for count in counts { every = every.flatMap { prefix in (0..<count).map { prefix + [$0] } } }
        let n = min(howMany ?? every.count, every.count)
        let last = counts.count - 1
        var used = counts.map { Array(repeating: 0, count: $0) }
        var together: [String: Int] = [:]
        var chosen: [[Int]] = []
        var left = every
        for _ in 0..<n {
            let prev = chosen.last
            func score(_ c: [Int]) -> Double {
                var s = 0.0
                for (i, p) in c.enumerated() { s += (i == 0 || i == last ? 3 : 1) * Double(used[i][p]) }
                for i in 0..<c.count { for j in (i + 1)..<max(i + 1, c.count) { s += 0.5 * Double(together["\(i).\(c[i]).\(j).\(c[j])"] ?? 0) } }
                if let prev {
                    if c[0] == prev[0] && counts[0] > 1 { s += 4 }
                    if c[last] == prev[last] && counts[last] > 1 { s += 4 }
                }
                return s
            }
            guard let best = left.enumerated().min(by: { a, b in
                let sa = score(a.element), sb = score(b.element)
                return sa == sb ? a.offset < b.offset : sa < sb
            }) else { break }
            left.remove(at: best.offset)
            chosen.append(best.element)
            for (i, p) in best.element.enumerated() { used[i][p] += 1 }
            for i in 0..<best.element.count { for j in (i + 1)..<max(i + 1, best.element.count) {
                together["\(i).\(best.element[i]).\(j).\(best.element[j])", default: 0] += 1 } }
        }
        let codes = sectionCodes(names)
        return MixPlan(total: every.count, videos: chosen.enumerated().map { index, picks in
            MixPlanVideo(n: index + 1, code: zip(codes, picks).map { "\($0)\($1 + 1)" }.joined(separator: " · "),
                         names: zip(names, picks).map { "\($0) \($1 + 1)" })
        })
    }

    static func sectionCodes(_ names: [String]) -> [String] {
        var out: [String] = []
        for name in names {
            let clean = name.filter { $0.isLetter || $0.isNumber }
            let letters = Array(clean.isEmpty ? "S" : clean)
            var code = String(letters[0]).uppercased()
            var k = 1
            while out.contains(code) && k < letters.count {
                k += 1
                code = String(letters[0]).uppercased() + String(letters[1..<k]).lowercased()
            }
            while out.contains(code) { code += "x" }
            out.append(code)
        }
        return out
    }

    func refresh(session: CutSellSession?) async {
        if mock { advanceMock(); return }
        guard let session else { return }
        struct Answer: Decodable { let mixes: [MixSummary] }
        do {
            let answer: Answer = try await APIClient.shared.request("/v1/mix", query: [URLQueryItem(name: "user_id", value: session.userID)],
                                                                    timeout: 15)
            mixes = answer.mixes
        } catch APIError.http(404, _) {
            mixes = []                                     // a server without Mix yet
        } catch {}
        follow(session: session)
    }

    /// While a Mix is being made: its group in Processing moves, and each video joins Ready as soon as it is made.
    func follow(session: CutSellSession?) {
        guard loop == nil, !active.isEmpty else { return }
        loop = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 4_000_000_000)
                guard let self else { return }
                let before = self.mixes.map(\.ready)
                await self.refresh(session: session)
                if self.mixes.map(\.ready) != before { NotificationCenter.default.post(name: .mixProgress, object: nil) }
                if self.active.isEmpty { break }
            }
            self?.loop = nil
        }
    }

    func started(_ summary: MixSummary, session: CutSellSession?) {
        mixes.removeAll { $0.mixID == summary.mixID }
        mixes.insert(summary, at: 0)
        if mock { mockStarted[summary.mixID] = Date() }
        follow(session: session)
    }

    func retry(_ mix: MixSummary, session: CutSellSession?) async {
        if mock { mockStarted[mix.mixID] = Date(); if let i = mixes.firstIndex(of: mix) { mixes[i].failedClips = [] }; return }
        guard let session else { return }
        struct Body: Encodable { let user_id: String }
        do {
            let summary: MixSummary = try await APIClient.shared.request("/v1/mix/\(mix.mixID)/retry", method: "POST",
                                                                         body: Body(user_id: session.userID), timeout: 20)
            started(summary, session: session)
            CutSellDiagnostics.log("mix_retry", ["clips": String(mix.failedClips.count)])
        } catch {
            CutSellDiagnostics.log("mix_retry_failed", Retry.fields(error))
        }
    }

    /// Probe: the mock Mix makes one video every 3 seconds.
    private func advanceMock() {
        for (index, mix) in mixes.enumerated() where mix.isActive {
            guard let started = mockStarted[mix.mixID] else { continue }
            let done = min(mix.total, Int(Date().timeIntervalSince(started) / 3))
            mixes[index].ready = done
            for k in 0..<mix.videos.count where k < done {
                mixes[index].videos[k].state = "ready"
                mixes[index].videos[k].durationSec = 33 + Double((k * 7) % 6)
            }
            if done >= mix.total { mixes[index].state = "ready" }
        }
    }
}

extension Notification.Name {
    static let mixProgress = Notification.Name("cutsell.mix.progress")
}

// MARK: - Sending a Mix (voice first, like a normal cut)

@MainActor
final class MixSubmitter: ObservableObject {
    @Published var step = ""
    @Published var fraction = 0.0

    /// Every clip's voice is uploaded first (small): the Mix starts from them; the videos go on uploading in the
    /// background and the server makes the videos as soon as they are there. Returns the Mix as the server sees it.
    func submit(sections: [MixSectionDraft], productID: String, productName: String, howMany: Int?,
                appState: AppState) async throws -> MixSummary {
        if MixCenter.shared.mock { return mockSubmit(sections: sections, productID: productID, howMany: howMany) }
        guard let session = appState.session else { throw UploadFlowError.notSignedIn }
        guard await AIConsentCenter.shared.ensure() else { throw AIConsentError.declined }
        let clips = sections.flatMap(\.clips)
        let started = Date()
        let project = try await Retry.run("project") { try await appState.createProject(title: "Mix · \(productName)") }
        let projectID = project.projectID
        ActiveUploads.shared.insert(projectID)
        defer { ActiveUploads.shared.remove(projectID) }
        let record = PendingCutRecord(projectID: projectID, title: "Mix · \(productName)", mode: "Mix", audioOverlap: false,
                                      autoZoom: true,
                                      videos: clips.map { PendingCutVideo(filePath: $0.video.url.path, name: $0.video.name,
                                                                          duration: $0.video.duration,
                                                                          contentType: $0.video.contentType,
                                                                          cloudAssetID: $0.video.cloudAssetID,
                                                                          expectedBytes: $0.video.expectedBytes) },
                                      createdAt: Date())
        await PendingCutStore.shared.save(record)
        struct Clip: Encodable { let uri: String; let audio_uri: String; let original_name: String; let duration_sec: Double }
        struct Section: Encodable { let name: String; let clips: [Clip] }
        var out: [Section] = []
        var order = 0
        for section in sections {
            var items: [Clip] = []
            for clip in section.clips {
                step = "Clip \(order + 1) of \(clips.count) · Preparing…"
                fraction = Double(order) / Double(max(1, clips.count))
                let voice = try await AudioExtractor.extract(from: clip.video.previewURL ?? clip.video.url)
                let start = try await MultipartUploadManager.shared.begin(fileURL: clip.video.url, projectID: projectID, session: session,
                                                                          contentType: clip.video.contentType,
                                                                          knownSize: clip.video.expectedBytes)
                await SourceVideoCache.shared.store(fileURL: clip.video.previewURL ?? clip.video.url, uri: start.sourceURI)
                step = "Clip \(order + 1) of \(clips.count) · Uploading audio…"
                let voiceURI = try await VoiceUpload.upload(fileURL: voice, projectID: projectID, session: session, order: order)
                try? FileManager.default.removeItem(at: voice)
                items.append(Clip(uri: start.sourceURI, audio_uri: voiceURI, original_name: clip.video.url.lastPathComponent,
                                  duration_sec: clip.video.duration))
                order += 1
            }
            out.append(Section(name: section.name, clips: items))
        }
        step = "Starting your Mix…"
        struct Body: Encodable {
            let user_id: String; let project_id: String; let product_id: String; let sections: [Section]; let how_many: Int?
        }
        let summary: MixSummary = try await Retry.run("start mix") {
            try await APIClient.shared.request("/v1/mix/jobs", method: "POST",
                                               body: Body(user_id: session.userID, project_id: projectID, product_id: productID,
                                                          sections: out, how_many: howMany), timeout: 30)
        }
        await PendingCutStore.shared.markStarted(projectID: projectID, jobID: "mix-" + projectID)
        if let saved = await PendingCutStore.shared.record(projectID: projectID) {
            BackgroundVideoUploads.shared.start(record: saved, session: session)
        }
        CutSellDiagnostics.log("mix_started", ["clips": String(clips.count), "videos": String(summary.total),
                                               "seconds": String(format: "%.1f", Date().timeIntervalSince(started))])
        return summary
    }

    private func mockSubmit(sections: [MixSectionDraft], productID: String, howMany: Int?) -> MixSummary {
        let plan = MixCenter.localPlan(names: sections.map(\.name), counts: sections.map { max(1, $0.clips.count) }, howMany: howMany)
        let mixID = "mix_" + UUID().uuidString.prefix(8).lowercased()
        let videos = plan.videos.map { MixVideo(n: $0.n, code: $0.code, projectID: "\(mixID)_\($0.n)", state: "processing") }
        CatalogStore.shared.mockAddMix(productID: productID, videos: videos.map { ($0.projectID, $0.code) })
        return MixSummary(mixID: mixID, state: "processing", productID: productID, brand: nil, product: nil, videos: videos,
                          ready: 0, total: videos.count, clipCount: sections.reduce(0) { $0 + $1.clips.count }, failedClips: [])
    }
}

// MARK: - Mix · Setup (Figma 1136:2724)

struct MixSetupView: View {
    let onClose: () -> Void
    /// The Mix started: Edits · Processing shows it.
    let onStarted: (MixSummary) -> Void

    @EnvironmentObject private var appState: AppState
    @ObservedObject private var catalog = CatalogStore.shared
    @StateObject private var submitter = MixSubmitter()
    @State private var sections: [MixSectionDraft] = ["Hook", "Body", "Benefit", "CTA"].map { MixSectionDraft(name: $0) }
    @State private var productID: String?
    @State private var choosingProduct = false
    @State private var picking: UUID?
    @State private var recordingFor: UUID?
    @State private var addingTo: UUID?
    @State private var renaming: UUID?
    @State private var newName = ""
    @State private var showingHowMany = false
    @State private var sending = false
    @State private var adding = false
    @State private var errorMessage: String?

    private var product: (brand: CatalogBrand, product: CatalogProduct)? {
        catalog.allProducts.first { $0.product.productID == productID }
    }
    private var filled: Bool { productID != nil && sections.allSatisfy { !$0.clips.isEmpty } }
    private var total: Int { sections.reduce(1) { $0 * max(1, $1.clips.count) } }
    /// "about 35 s each": the length of an average video (the clips are trimmed when cleaned; this is the raw length).
    private var averageSeconds: Int {
        Int(sections.reduce(0.0) { sum, s in
            sum + (s.clips.isEmpty ? 0 : s.clips.reduce(0.0) { $0 + $1.video.duration } / Double(s.clips.count))
        }.rounded())
    }
    private var formula: String {
        sections.map { s in "\(s.clips.count) \(Self.plural(s.name, s.clips.count))" }.joined(separator: " × ")
    }

    /// "3 hooks × 2 bodies × 2 benefits × 2 CTAs" (Figma): lower case, an acronym stays as it is.
    static func plural(_ name: String, _ count: Int) -> String {
        let acronym = name.count > 1 && name.uppercased() == name
        let base = acronym ? name : name.lowercased()
        guard count != 1 else { return base }
        if acronym { return base + "s" }
        if base.hasSuffix("y"), let before = base.dropLast().last, !"aeiou".contains(before) { return String(base.dropLast()) + "ies" }
        return base + "s"
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            LinearGradient(stops: [.init(color: Color(hex: 0x03050D), location: 0), .init(color: Color(hex: 0x050B1C), location: 0.55),
                                   .init(color: Color(hex: 0x030409), location: 1)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ZStack {
                        Text(L10n.tr("Mix")).font(CS.inter(18, .bold)).foregroundStyle(.white)
                        HStack {
                            Button(action: onClose) { Text(L10n.tr("✕")).font(CS.inter(20, .medium)).foregroundStyle(.white) }
                                .accessibilityIdentifier("mixClose")
                            Spacer()
                        }
                        .padding(.leading, 2)
                    }
                    .padding(.top, 10)
                    Text(L10n.tr("Record a few versions of each part. CutSell cleans every clip once and builds every combination for you."))
                        .font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD))
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 22)
                    productCard.padding(.top, 14)
                    VStack(spacing: 8) {
                        ForEach(Array(sections.enumerated()), id: \.element.id) { index, section in
                            sectionCard(index, section)
                        }
                    }
                    .padding(.top, 12)
                    if sections.count < MixCenter.maxSections {
                        Button { addSection() } label: {
                            Text(L10n.tr("+  Add section")).font(CS.inter(14, .semibold)).foregroundStyle(Color(hex: 0x99A3BD))
                                .frame(maxWidth: .infinity).frame(height: 40)
                                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .stroke(Color(hex: 0x596685), style: StrokeStyle(lineWidth: 1, dash: [4, 3])))
                        }
                        .buttonStyle(.plain)
                        .padding(.top, 10)
                        .accessibilityIdentifier("mixAddSection")
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 180)
            }
            bottom
            if showingHowMany {
                Color.black.opacity(0.6).ignoresSafeArea().onTapGesture { showingHowMany = false }
                MixHowManySheet(sections: sections, onMake: { count in
                    showingHowMany = false
                    Task { await make(howMany: count) }
                })
                .environmentObject(appState)
                .transition(.move(edge: .bottom))
            }
            if sending {
                Color.black.opacity(0.7).ignoresSafeArea()
                VStack(spacing: 14) {
                    ProgressView().tint(CS.cyan)
                    Text(L10n.tr(submitter.step.isEmpty ? "Preparing your Mix…" : submitter.step))
                        .font(CS.inter(14, .semibold)).foregroundStyle(.white).multilineTextAlignment(.center)
                    Text(L10n.tr("Keep CutSell open until the audio is sent. The videos keep uploading in the background."))
                        .font(CS.inter(12)).foregroundStyle(Color(hex: 0x99A3BD)).multilineTextAlignment(.center)
                }
                .padding(28)
                .accessibilityIdentifier("mixSending")
            }
            if adding {
                Color.black.opacity(0.5).ignoresSafeArea()
                ProgressView(L10n.tr("Getting your clips…")).tint(CS.cyan).foregroundStyle(.white)
            }
        }
        .animation(.easeOut(duration: 0.25), value: showingHowMany)
        .preferredColorScheme(.dark)
        .task {
            await catalog.refresh(session: appState.session, projects: appState.projects)
            if productID == nil, let first = catalog.allProducts.first { productID = first.product.productID }
        }
        .sheet(isPresented: $choosingProduct) {
            MixProductPicker(selected: $productID).environmentObject(appState)
                .presentationDetents([.height(520), .large])
                .presentationBackground(Color(hex: 0x0D121F))
        }
        .confirmationDialog(L10n.tr("Add clips"), isPresented: Binding(get: { addingTo != nil }, set: { if !$0 { addingTo = nil } })) {
            Button(L10n.tr("Choose from Photos")) { picking = addingTo; addingTo = nil }
            Button(L10n.tr("Record")) { recordingFor = addingTo; addingTo = nil }
            Button(L10n.tr("Cancel"), role: .cancel) { addingTo = nil }
        }
        .fullScreenCover(item: Binding(get: { picking.map { SectionRef(id: $0) } }, set: { picking = $0?.id })) { ref in
            let left = MixCenter.maxClips - (sections.first { $0.id == ref.id }?.clips.count ?? 0)
            GalleryPickerView(
                onCancel: { picking = nil },
                onCreate: { _, assets, _ in
                    picking = nil
                    Task { await add(assets, to: ref.id) }
                },
                initialMode: .combine,
                pickTitle: { count in count == 1 ? "Add 1 clip" : "Add \(count) clips" },
                pickLimit: max(1, left)
            )
        }
        .fullScreenCover(item: Binding(get: { recordingFor.map { SectionRef(id: $0) } }, set: { recordingFor = $0?.id })) { ref in
            CreateCameraView(onBack: { recordingFor = nil }, onUpload: { recordingFor = nil; picking = ref.id }, onRecorded: { video in
                recordingFor = nil
                Task { await append([video], to: ref.id) }
            })
        }
        .alert(L10n.tr("Rename section"), isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField(L10n.tr("Name"), text: $newName).accessibilityIdentifier("mixRenameField")
            Button(L10n.tr("Save")) {
                let name = newName.trimmingCharacters(in: .whitespacesAndNewlines)
                if let id = renaming, !name.isEmpty, let i = sections.firstIndex(where: { $0.id == id }) {
                    sections[i].name = String(name.prefix(40))
                }
                renaming = nil
            }
            Button(L10n.tr("Cancel"), role: .cancel) { renaming = nil }
        }
        .alert(L10n.tr("CutSell"), isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button(L10n.tr("OK"), role: .cancel) {}
        } message: { Text(L10n.tr(errorMessage ?? "")) }
    }

    struct SectionRef: Identifiable { let id: UUID }

    private var productCard: some View {
        Button { choosingProduct = true } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.tr("PRODUCT")).font(CS.inter(10, .bold)).foregroundStyle(CS.cyan)
                    Text(product.map { "\($0.brand.name) · \($0.product.name)" } ?? "Choose the product")
                        .font(CS.inter(15, .semibold)).foregroundStyle(.white).lineLimit(1)
                        .accessibilityIdentifier("mixProductName")
                }
                Spacer()
                Text(L10n.tr(product == nil ? "Choose" : "Change")).font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
            }
            .padding(.horizontal, 15)
            .frame(height: 54)
            .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("mixProduct")
    }

    private func sectionCard(_ index: Int, _ section: MixSectionDraft) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(L10n.tr("\(index + 1)")).font(CS.inter(12, .bold)).foregroundStyle(.white)
                    .frame(width: 24, height: 24).background(CS.brandFull, in: Circle())
                Text(section.name).font(CS.inter(15, .semibold)).foregroundStyle(.white)
                    .accessibilityIdentifier("mixSectionName_\(index)")
                Text(L10n.tr("\(section.clips.count) clip\(section.clips.count == 1 ? "" : "s")"))
                    .font(CS.inter(12)).foregroundStyle(Color(hex: 0x99A3BD))
                Spacer()
                Menu {
                    Button(L10n.tr("Move up")) { move(index, by: -1) }.disabled(index == 0)
                    Button(L10n.tr("Move down")) { move(index, by: 1) }.disabled(index == sections.count - 1)
                } label: {
                    Text(L10n.tr("≡")).font(CS.inter(18, .medium)).foregroundStyle(Color(hex: 0x99A3BD)).frame(width: 22, height: 24)
                }
                .accessibilityIdentifier("mixSectionMove_\(index)")
                Menu {
                    Button(L10n.tr("Rename")) { newName = section.name; renaming = section.id }
                    Button(L10n.tr("Delete section"), role: .destructive) { sections.removeAll { $0.id == section.id } }
                        .disabled(sections.count <= 1)
                } label: {
                    Text(L10n.tr("•••")).font(CS.inter(12, .bold)).foregroundStyle(Color(hex: 0x99A3BD)).frame(width: 30, height: 24)
                }
                .accessibilityIdentifier("mixSectionMore_\(index)")
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(section.clips) { clip in clipThumb(clip, in: section) }
                    if section.clips.count < MixCenter.maxClips {
                        Button { addingTo = section.id } label: {
                            Text(L10n.tr("+")).font(CS.inter(20, .medium)).foregroundStyle(CS.cyan)
                                .frame(width: 42, height: 46)
                                .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .stroke(Color(hex: 0x596685), style: StrokeStyle(lineWidth: 1, dash: [4, 3])))
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("mixAddClip_\(index)")
                    }
                }
            }
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity, minHeight: 98, alignment: .topLeading)
        .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("mixSection_\(index)")
    }

    private func clipThumb(_ clip: MixClipDraft, in section: MixSectionDraft) -> some View {
        Color(hex: 0x1A2236)
            .frame(width: 42, height: 46)
            .overlay { if let image = clip.thumbnail { Image(uiImage: image).resizable().scaledToFill() } }
            .overlay(alignment: .bottom) {
                Text(ProductDetailView.length(clip.video.duration)).font(CS.inter(9, .semibold)).foregroundStyle(.white)
                    .frame(maxWidth: .infinity).frame(height: 16).background(Color.black.opacity(0.55))
            }
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .contextMenu {
                Button(L10n.tr("Remove clip"), role: .destructive) {
                    if let i = sections.firstIndex(where: { $0.id == section.id }) { sections[i].clips.removeAll { $0.id == clip.id } }
                }
            }
            .accessibilityIdentifier("mixClip")
    }

    private var bottom: some View {
        VStack(alignment: .leading, spacing: 0) {
            Rectangle().fill(Color(hex: 0x262E42)).frame(height: 1)
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.tr(filled ? "\(total) video\(total == 1 ? "" : "s") · about \(averageSeconds) s each" : "Add clips to every part"))
                        .font(CS.inter(15, .semibold)).foregroundStyle(.white)
                        .accessibilityIdentifier("mixSummary")
                    Text(formula).font(CS.inter(11)).foregroundStyle(Color(hex: 0x99A3BD)).lineLimit(1).minimumScaleFactor(0.8)
                }
                Spacer()
                Button(L10n.tr("Pick how many ›")) { showingHowMany = true }
                    .font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                    .disabled(!filled || total > MixCenter.maxVideos * 10)
                    .opacity(filled ? 1 : 0.4)
                    .padding(.top, 4)
                    .accessibilityIdentifier("mixPickHowMany")
            }
            .padding(.horizontal, 20)
            .padding(.top, 16)
            Button { Task { await make(howMany: total > MixCenter.maxVideos ? MixCenter.maxVideos : nil) } } label: {
                Text(L10n.tr("Make \(min(total, MixCenter.maxVideos)) video\(total == 1 ? "" : "s")"))
                    .font(CS.sf(15, .semibold)).foregroundStyle(.white)
                    .frame(maxWidth: .infinity).frame(height: 50)
                    .background(Capsule().fill(MixStyle.cta))
                    .shadow(color: Color(hex: 0x1969FF).opacity(0.25), radius: 9, y: 6)
            }
            .disabled(!filled || sending)
            .opacity(filled ? 1 : 0.5)
            .padding(.horizontal, 20)
            .padding(.top, 14)
            .padding(.bottom, 8)
            .accessibilityIdentifier("mixMake")
        }
        .background(Color(hex: 0x090D17).ignoresSafeArea(edges: .bottom))
    }

    private func addSection() {
        let names = ["Hook", "Body", "Benefit", "CTA", "Proof", "Offer", "Demo"]
        let name = names.first { n in !sections.contains { $0.name == n } } ?? "Part \(sections.count + 1)"
        sections.append(MixSectionDraft(name: name))
    }

    private func move(_ index: Int, by delta: Int) {
        let target = index + delta
        guard sections.indices.contains(target) else { return }
        sections.swapAt(index, target)
    }

    private func add(_ assets: [PHAsset], to id: UUID) async {
        adding = true
        defer { adding = false }
        var videos: [LocalVideo] = []
        for asset in assets {
            do {
                var video = try await GalleryExport.export(asset, name: nil)
                video.albumHint = GalleryExport.albumName(of: asset)
                videos.append(video)
            } catch {
                errorMessage = "One clip couldn’t be added. Try another one."
            }
        }
        await append(videos, to: id)
    }

    private func append(_ videos: [LocalVideo], to id: UUID) async {
        for video in videos {
            let image = await CutTracker.firstFrame(of: video.previewURL ?? video.url)
            if let i = sections.firstIndex(where: { $0.id == id }), sections[i].clips.count < MixCenter.maxClips {
                sections[i].clips.append(MixClipDraft(video: video, thumbnail: image))
            }
        }
        CutSellDiagnostics.log("mix_clips_added", ["clips": String(videos.count)])
    }

    private func make(howMany: Int?) async {
        guard filled, let product, !sending else { return }
        sending = true
        defer { sending = false }
        do {
            let summary = try await submitter.submit(sections: sections, productID: product.product.productID,
                                                     productName: product.product.name, howMany: howMany, appState: appState)
            MixCenter.shared.started(summary, session: appState.session)
            try? await appState.refreshProjects()
            onStarted(summary)
        } catch {
            CutSellDiagnostics.log("mix_start_failed", Retry.fields(error))
            errorMessage = "Your Mix couldn’t start. Check your connection and try again."
        }
    }
}

enum MixStyle {
    static let cta = LinearGradient(stops: [.init(color: Color(hex: 0x1969FF), location: 0), .init(color: Color(hex: 0x7C3AED), location: 0.55),
                                            .init(color: Color(hex: 0x19E6FF), location: 1)], startPoint: .leading, endPoint: .trailing)
}

// MARK: - Mix · How many (Figma 1136:2804)

struct MixHowManySheet: View {
    let sections: [MixSectionDraft]
    let onMake: (Int?) -> Void
    @EnvironmentObject private var appState: AppState
    @State private var choice: Int?
    @State private var plan: MixPlan?

    private var total: Int { sections.reduce(1) { $0 * max(1, $1.clips.count) } }
    private var options: [Int] { [6, 12, 18].filter { $0 < total } }
    private var count: Int { choice ?? min(total, MixCenter.maxVideos) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Color(hex: 0x4D5770)).frame(width: 40, height: 5).frame(maxWidth: .infinity).padding(.top, 9)
            Text(L10n.tr("How many videos?")).font(CS.inter(22, .bold)).foregroundStyle(.white).padding(.top, 17)
            Text(L10n.tr("\(total) combinations are possible")).font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 6)
            HStack(spacing: 8) {
                ForEach(options, id: \.self) { n in pill("\(n)", selected: choice == n, width: 70) { choice = n } }
                pill(total > MixCenter.maxVideos ? "\(MixCenter.maxVideos)" : "All \(total)", selected: choice == nil, width: 96) { choice = nil }
            }
            .padding(.top, 16)
            Text(L10n.tr("We mix the clips so each video starts and ends differently."))
                .font(CS.inter(12)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 14)
            Text(L10n.tr("PREVIEW")).font(CS.inter(10, .bold)).foregroundStyle(CS.cyan).padding(.top, 18)
            VStack(spacing: 0) {
                ForEach((plan?.videos ?? []).prefix(4)) { video in
                    HStack(spacing: 0) {
                        Text(L10n.tr("Video \(video.n)")).font(CS.inter(13, .semibold)).foregroundStyle(.white).frame(width: 72, alignment: .leading)
                        Text(L10n.tr(video.names.joined(separator: " · "))).font(CS.inter(12)).foregroundStyle(Color(hex: 0x99A3BD))
                            .lineLimit(1).minimumScaleFactor(0.8)
                        Spacer(minLength: 0)
                    }
                    .frame(height: 34)
                    .overlay(alignment: .bottom) { Rectangle().fill(Color(hex: 0x21293B)).frame(height: 1) }
                    .accessibilityIdentifier("mixPreviewRow")
                }
            }
            .padding(.top, 6)
            if count > 4 {
                Text(L10n.tr("+ \(count - 4) more")).font(CS.inter(12, .semibold)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 6)
            }
            Text(L10n.tr("Tip: post these over several days, between other products. Feeds reward variety."))
                .font(CS.inter(11, .medium)).foregroundStyle(Color(hex: 0xFFB840))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 13).padding(.vertical, 8)
                .frame(maxWidth: .infinity, minHeight: 48, alignment: .leading)
                .background(Color(hex: 0x1F170D), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color(hex: 0x73521F), lineWidth: 1))
                .padding(.top, 14)
            Button { onMake(choice ?? (total > MixCenter.maxVideos ? MixCenter.maxVideos : nil)) } label: {
                Text(L10n.tr("Make \(count) video\(count == 1 ? "" : "s")"))
                    .font(CS.sf(15, .semibold)).foregroundStyle(.white)
                    .frame(maxWidth: .infinity).frame(height: 50)
                    .background(Capsule().fill(MixStyle.cta))
                    .shadow(color: Color(hex: 0x1969FF).opacity(0.25), radius: 9, y: 6)
            }
            .padding(.top, 16)
            .accessibilityIdentifier("mixHowManyMake")
        }
        .padding(.horizontal, 23)
        .padding(.bottom, 30)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(hex: 0x0D121F), in: UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28))
        .overlay(UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28).stroke(Color(hex: 0x33405C), lineWidth: 1))
        .task(id: choice) { await load() }
        .onAppear { if choice == nil, options.contains(12) { choice = 12 } }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("mixHowMany")
    }

    private func pill(_ title: String, selected: Bool, width: CGFloat, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(L10n.tr(title)).font(CS.inter(14, .semibold)).foregroundStyle(.white)
                .frame(width: width, height: 40)
                .background {
                    if selected { Capsule().fill(CS.brandFull) } else { Capsule().fill(Color(hex: 0x0F1726)) }
                }
                .overlay(Capsule().stroke(selected ? .clear : Color(hex: 0x33405C), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("mixCount_\(title)")
    }

    private func load() async {
        plan = try? await MixCenter.shared.plan(sections: sections, howMany: choice ?? min(total, MixCenter.maxVideos),
                                                session: appState.session)
    }
}

// MARK: - Choosing the Mix's product

struct MixProductPicker: View {
    @Binding var selected: String?
    @ObservedObject private var store = CatalogStore.shared
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var addingNew = false
    @State private var newBrand = ""
    @State private var newProduct = ""
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(L10n.tr("Which product is this Mix for?")).font(CS.inter(20, .bold)).foregroundStyle(.white).padding(.top, 28)
            ScrollView {
                VStack(spacing: 0) {
                    ForEach(store.allProducts, id: \.product.productID) { pair in
                        Button { selected = pair.product.productID; dismiss() } label: {
                            HStack(spacing: 12) {
                                ZStack {
                                    Image("ProductsRadio").resizable().frame(width: 20, height: 20)
                                    if selected == pair.product.productID { Circle().fill(CS.cyan).frame(width: 10, height: 10) }
                                }
                                Text(L10n.tr("\(pair.brand.name) · \(pair.product.name)")).font(CS.inter(15)).foregroundStyle(.white)
                                Spacer()
                            }
                            .frame(height: 48)
                            .overlay(alignment: .bottom) { Rectangle().fill(Color(hex: 0x21293B)).frame(height: 1) }
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("mixProductOption_\(pair.brand.name) · \(pair.product.name)")
                    }
                }
            }
            .padding(.top, 10)
            Button { addingNew = true } label: {
                Text(L10n.tr("+  New brand or product")).font(CS.inter(15, .semibold)).foregroundStyle(CS.cyan)
            }
            .padding(.vertical, 14)
            .accessibilityIdentifier("mixProductNew")
        }
        .padding(.horizontal, 23)
        .preferredColorScheme(.dark)
        .alert(L10n.tr("New brand or product"), isPresented: $addingNew) {
            TextField(L10n.tr("Brand (e.g. CeraVe)"), text: $newBrand)
            TextField(L10n.tr("Product (e.g. Hydrating Cleanser)"), text: $newProduct)
            Button(L10n.tr("Use it")) { Task { await create() } }
            Button(L10n.tr("Cancel"), role: .cancel) {}
        }
        .alert(L10n.tr("CutSell"), isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button(L10n.tr("OK"), role: .cancel) {}
        } message: { Text(L10n.tr(errorMessage ?? "")) }
    }

    private func create() async {
        let brand = newBrand.trimmingCharacters(in: .whitespacesAndNewlines)
        let product = newProduct.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !brand.isEmpty else { return }
        do {
            selected = try await store.createProduct(brand: brand, product: product.isEmpty ? brand : product,
                                                     session: appState.session, projects: appState.projects)
            dismiss()
        } catch {
            errorMessage = "Couldn’t add it. Please try again."
        }
    }
}

// MARK: - Edits · Processing: one row per Mix

struct MixGroupRow: View {
    let mix: MixSummary
    let onRetry: () -> Void
    let onOpen: () -> Void

    var body: some View {
        Button(action: onOpen) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Text(L10n.tr("MIX")).font(CS.inter(9, .bold)).foregroundStyle(.white)
                        .frame(width: 34, height: 18).background(CS.brandFull, in: Capsule())
                    VStack(alignment: .leading, spacing: 2) {
                        Text(L10n.tr("Mix · \(mix.total) video\(mix.total == 1 ? "" : "s")")).font(CS.inter(14, .semibold)).foregroundStyle(.white)
                            .accessibilityIdentifier("mixGroupTitle")
                        Text([mix.brand, mix.product].compactMap { $0 }.joined(separator: " · ")
                             + (mix.brand == nil && mix.product == nil ? "\(mix.clipCount) clips" : ""))
                            .font(CS.inter(12)).foregroundStyle(Color(hex: 0x99A3BD)).lineLimit(1)
                    }
                    Spacer()
                    Text(L10n.tr(mix.failedClips.isEmpty ? "\(mix.ready) of \(mix.total) ready" : "Needs a retry"))
                        .font(CS.inter(12, .semibold))
                        .foregroundStyle(mix.failedClips.isEmpty ? CS.cyan : Color(hex: 0xFFB840))
                        .accessibilityIdentifier("mixGroupProgress")
                }
                ProgressView(value: Double(mix.ready), total: Double(max(1, mix.total))).tint(CS.cyan)
                if !mix.failedClips.isEmpty {
                    HStack {
                        Text(L10n.tr("Clip \(mix.failedClips.compactMap(\.label).joined(separator: ", ")) couldn’t be cleaned."))
                            .font(CS.inter(12)).foregroundStyle(Color(hex: 0xFFB840))
                        Spacer()
                        Button(L10n.tr("Retry"), action: onRetry).font(CS.inter(12, .semibold)).foregroundStyle(CS.cyan)
                            .accessibilityIdentifier("mixRetry")
                    }
                }
            }
            .padding(14)
            .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("mixGroup")
    }
}

// MARK: - Export all (Mix · Results)

/// Exports every video of a Mix one after another and saves each to Photos (the same export as the editor's).
@MainActor
final class MixExporter: ObservableObject {
    @Published var running = false
    @Published var done = 0
    @Published var total = 0
    @Published var message: String?

    func exportAll(_ projectIDs: [String], session: CutSellSession?) async {
        guard !running else { return }
        running = true; done = 0; total = projectIDs.count; message = nil
        defer { running = false }
        if MixCenter.shared.mock {
            for _ in projectIDs { try? await Task.sleep(nanoseconds: 150_000_000); done += 1 }
            message = "\(done) videos saved to Photos"
            return
        }
        guard let session else { return }
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { message = "Allow CutSell to add videos to Photos."; return }
        var failed = 0
        for id in projectIDs {
            do {
                let snap: DraftSnapshot = try await APIClient.shared.request("/v1/projects/\(id)/draft",
                                                                             query: [URLQueryItem(name: "user_id", value: session.userID)])
                let sources = snap.sources.compactMap { object -> JSONValue? in
                    guard let sid = object["source_asset_id"], let uri = object["uri"] else { return nil }
                    return .object(["source_asset_id": sid, "original_name": object["original_name"] ?? .string("video.mp4"), "uri": uri])
                }
                let response: ExportSubmitResponse = try await APIClient.shared.request(
                    "/v1/exports/jobs", method: "POST",
                    body: JSONValue.object(["project_id": .string(id), "user_id": .string(session.userID), "draft": snap.draft,
                                            "sources": .array(sources)]))
                var link: URL?
                for _ in 0..<600 {
                    let job: JobStatus = try await APIClient.shared.request("/v1/jobs/\(response.jobID)")
                    if job.state == "finished" { link = job.result?["download_url"]?.stringValue.flatMap(URL.init(string:)); break }
                    if ["failed", "canceled"].contains(job.state) { break }
                    try await Task.sleep(nanoseconds: 1_500_000_000)
                }
                guard let link else { failed += 1; continue }
                let (temporary, _) = try await URLSession.shared.download(from: link)
                let file = FileManager.default.temporaryDirectory.appendingPathComponent("CutSell-Mix-\(UUID().uuidString).mp4")
                try FileManager.default.moveItem(at: temporary, to: file)
                try await PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: file)
                }
                try? FileManager.default.removeItem(at: file)
                done += 1
            } catch {
                failed += 1
                CutSellDiagnostics.log("mix_export_failed", Retry.fields(error))
            }
        }
        message = failed == 0 ? "\(done) videos saved to Photos" : "\(done) saved · \(failed) couldn’t be exported. Try again."
        CutSellDiagnostics.log("mix_export_all", ["saved": String(done), "failed": String(failed)])
    }
}
