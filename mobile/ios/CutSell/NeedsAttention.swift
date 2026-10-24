import SwiftUI

/// White Indigo · ERROR · Bulk/Needs Attention (Figma 1233:3243) and Plan/Needs Attention (1233:3354), on Plan:
/// a Bulk send where a video failed. Summary card (ring with the % ready, "9 of 10 ready", "9 finished cuts are
/// already in Edits"; the Plan variant adds "9 ready · 0 active · 0 queued · 1 failed" and "1 video needs action"),
/// then "Needs attention" with every video of the send (COMPLETED ✓ / FAILED + Retry).
struct NeedsAttentionSummary: Equatable {
    var ready = 0, active = 0, queued = 0, failed = 0
    var total: Int { ready + active + queued + failed }
    /// 1233:3354 (with the counts line) while some videos are still being made; 1233:3243 once they are all done.
    var detailed: Bool { active + queued > 0 }

    init(ready: Int = 0, active: Int = 0, queued: Int = 0, failed: Int = 0) {
        self.ready = ready; self.active = active; self.queued = queued; self.failed = failed
    }

    init(rows: [ProcessingQueue.Row]) {
        for row in rows {
            switch row.state {
            case .ready: ready += 1
            case .queued: queued += 1
            case .uploading, .processing: active += 1
            case .failed: failed += 1
            }
        }
    }
}

struct NeedsAttentionSection: View {
    let summary: NeedsAttentionSummary
    let rows: [ProcessingQueue.Row]
    let onRetry: (ProcessingQueue.Row) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            summaryCard
            WISectionTitle(title: "Needs attention", note: "Retry failed").padding(.top, 22)
            VStack(spacing: 8) {
                ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
                    NeedsAttentionRow(order: index + 1, row: row) { onRetry(row) }
                }
            }
            .padding(.top, 10)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("planNeedsAttention")
    }

    private var fraction: Double { Double(summary.ready) / Double(max(1, summary.total)) }

    private var summaryCard: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle().stroke(Color(fixed: 0xE6EAF2), lineWidth: 6)
                Circle().trim(from: 0, to: fraction)
                    .stroke(CS.accent, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Text(verbatim: "\(Int((fraction * 100).rounded()))%").font(CS.inter(17, .bold)).foregroundStyle(CS.wiInk)
            }
            .frame(width: 70, height: 70)
            VStack(alignment: .leading, spacing: 4) {
                Text(L10n.tr("\(summary.ready) of \(summary.total) ready")).font(CS.inter(19, .bold)).foregroundStyle(CS.wiInk)
                Text(L10n.tr(summary.ready == 1 ? "1 finished cut is already in Edits" : "\(summary.ready) finished cuts are already in Edits"))
                    .font(CS.inter(11, .medium)).foregroundStyle(CS.wiSecondary)
                if summary.detailed {
                    HStack(spacing: 8) {
                        dot(CS.wiReady, L10n.tr("\(summary.ready) ready"), CS.wiReady)
                        dot(CS.accent, L10n.tr("\(summary.active) active"), CS.accent)
                        Text(L10n.tr("\(summary.queued) queued · \(summary.failed) failed"))
                            .foregroundStyle(CS.wiSecondary)
                    }
                    .font(CS.inter(10, .medium))
                    .lineLimit(1).minimumScaleFactor(0.8)
                    .padding(.top, 4)
                    HStack(spacing: 6) {
                        Image("WIClock").renderingMode(.template).resizable().frame(width: 14, height: 14)
                            .foregroundStyle(CS.wiInk)
                        Text(L10n.tr(summary.failed == 1 ? "1 video needs action" : "\(summary.failed) videos need action"))
                            .font(CS.inter(11, .medium)).foregroundStyle(CS.wiInk)
                    }
                    .padding(.top, 6)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 130, alignment: .leading)
        .wiCard()
        .accessibilityIdentifier("planNeedsAttentionSummary")
    }

    private func dot(_ color: Color, _ text: String, _ textColor: Color) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 5, height: 5)
            Text(text).foregroundStyle(textColor)
        }
    }
}

/// One video of the send: number, thumbnail, name, duration, COMPLETED (green check) or FAILED (Retry).
struct NeedsAttentionRow: View {
    let order: Int
    let row: ProcessingQueue.Row
    let onRetry: () -> Void

    private var failed: Bool { if case .failed = row.state { return true }; return false }
    private var status: String {
        switch row.state {
        case .ready: return "COMPLETED"
        case .failed: return "FAILED"
        case .queued: return "QUEUED"
        case .uploading, .processing: return "PROCESSING"
        }
    }

    var body: some View {
        HStack(spacing: 0) {
            Text(verbatim: String(format: "%02d", order))
                .font(CS.inter(11, .semibold)).foregroundStyle(failed ? CS.accent : CS.wiSecondary)
                .frame(width: 30, alignment: .leading)
            Group {
                if let image = row.thumbnail { Image(uiImage: image).resizable().scaledToFill() } else { CS.wiSurface2 }
            }
            .frame(width: 46, height: 60)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            VStack(alignment: .leading, spacing: 4) {
                Text(row.name).font(CS.inter(13, .semibold)).foregroundStyle(CS.wiInk).lineLimit(1).truncationMode(.middle)
                HStack {
                    Text(verbatim: QueueRowView.clock(row.duration)).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
                    Spacer()
                    Text(L10n.tr(status)).font(CS.inter(10, .semibold)).foregroundStyle(CS.accent)
                }
            }
            .padding(.leading, 10)
            Group {
                if failed {
                    Button(action: onRetry) {
                        Text(L10n.tr("Retry")).font(CS.inter(11, .semibold)).foregroundStyle(CS.accent)
                            .padding(.horizontal, 12).frame(height: 28)
                            .background(CS.wiPage, in: Capsule())
                            .overlay(Capsule().stroke(CS.wiError, lineWidth: 1.2))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("needsAttentionRetry")
                } else if row.state == .ready {
                    Image(systemName: "checkmark.circle.fill").font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(CS.wiReady)
                } else if row.state == .queued {
                    Image("WIClock").renderingMode(.template).resizable().frame(width: 22, height: 22)
                        .foregroundStyle(CS.wiSecondary)
                } else {
                    ProgressRing(fraction: nil).frame(width: 22, height: 22)
                }
            }
            .padding(.leading, 10)
        }
        .padding(.horizontal, 14)
        .frame(height: 84)
        .wiCard(shadow: false)
    }
}
