import Foundation
import ObjectiveC
import SwiftUI

// MARK: - Language (You → Language: Automatic · English · Español)
//
// Every English text in the app is the key. Spanish comes from `L10nES.table` (English template → Spanish
// template; `{}` is the next value, `{n}` the n-th value, `{n:es}` writes "es" when the n-th value is not empty).
// Two paths reach it:
// 1. `Text("Literal \(x)")`, `Button("…")`, `Label("…", …)` — SwiftUI looks the key up in Bundle.main; the
//    bundle is swapped for `L10nBundle`, which answers from the table (the key's %@ / %lld become {}).
// 2. Texts built in code (`Text(message)`, toasts, titles) — `L10n.tr(_:)` matches the finished English
//    string against the table (exactly, or by template) and fills the Spanish template with the same values.
// Changing the language re-draws the whole app at once (RootView is keyed by `L10n.shared.code`).

enum AppLanguage: String, CaseIterable, Identifiable {
    case automatic, english = "en", spanish = "es"
    var id: String { rawValue }
    /// Shown in its own language (as iOS does).
    var title: String {
        switch self {
        case .automatic: return L10n.tr("Automatic")
        case .english: return "English"
        case .spanish: return "Español"
        }
    }
}

@MainActor
final class L10n: ObservableObject {
    static let shared = L10n()
    private static let storeKey = "cutsell.language"

    @Published private(set) var choice: AppLanguage
    /// "en" or "es": what the app shows now.
    @Published private(set) var code: String

    private init() {
        let stored = UserDefaults.standard.string(forKey: Self.storeKey).flatMap(AppLanguage.init(rawValue:)) ?? .automatic
        choice = stored
        code = Self.resolve(stored)
        L10nState.code = code
        L10nBundle.install()
    }

    /// Automatic follows the iPhone's language: Spanish when the first preferred language is Spanish.
    nonisolated static func resolve(_ choice: AppLanguage) -> String {
        switch choice {
        case .english: return "en"
        case .spanish: return "es"
        case .automatic:
            #if DEBUG
            if let forced = ProcessInfo.processInfo.environment["CUTSELL_LANGUAGE"], ["en", "es"].contains(forced) { return forced }
            #endif
            return (Locale.preferredLanguages.first ?? "en").lowercased().hasPrefix("es") ? "es" : "en"
        }
    }

    /// After a language change the app re-draws; it comes back on this tab (You).
    nonisolated(unsafe) private static var reopenTab: MainTab?
    static func takeReopenTab() -> MainTab? { defer { reopenTab = nil }; return reopenTab }

    func set(_ choice: AppLanguage, reopen tab: MainTab? = nil) {
        Self.reopenTab = tab
        // The next launch starts in this language everywhere (also texts iOS itself draws).
        if choice == .automatic { UserDefaults.standard.removeObject(forKey: "AppleLanguages") }
        else { UserDefaults.standard.set([choice.rawValue], forKey: "AppleLanguages") }
        UserDefaults.standard.set(choice.rawValue, forKey: Self.storeKey)
        self.choice = choice
        code = Self.resolve(choice)
        L10nState.code = code
        L10nState.cache.removeAll()
        CutSellDiagnostics.log("language_changed", ["choice": choice.rawValue, "code": code])
    }

    /// Dates, numbers and day names follow the app's language.
    nonisolated static var locale: Locale { Locale(identifier: L10nState.code == "es" ? "es_ES" : "en_US") }

    /// The finished English text in the app's language (unchanged in English or when it is not app copy).
    nonisolated static func tr(_ english: String) -> String { L10nState.translate(english) }
    nonisolated static func tr<S: StringProtocol>(_ english: S) -> String { L10nState.translate(String(english)) }
}

/// Lock-free state read from any thread (SwiftUI resolves keys off the main actor too).
enum L10nState {
    nonisolated(unsafe) static var code: String = L10n.resolve(
        UserDefaults.standard.string(forKey: "cutsell.language").flatMap(AppLanguage.init(rawValue:)) ?? .automatic)
    nonisolated(unsafe) static var cache: [String: String] = [:]
    private static let lock = NSLock()

    private struct Template {
        let regex: NSRegularExpression
        let spanish: String
        let literalLength: Int
    }

    /// Templates with values, longest literal text first (the most specific wins).
    private static let templates: [Template] = {
        L10nES.table.compactMap { english, spanish -> Template? in
            guard english.contains("{}") else { return nil }
            let parts = english.components(separatedBy: "{}").map(NSRegularExpression.escapedPattern(for:))
            guard let regex = try? NSRegularExpression(pattern: "^" + parts.joined(separator: "(.*?)") + "$",
                                                       options: [.dotMatchesLineSeparators]) else { return nil }
            return Template(regex: regex, spanish: spanish, literalLength: english.replacingOccurrences(of: "{}", with: "").count)
        }
        .sorted { $0.literalLength > $1.literalLength }
    }()

    static func translate(_ english: String) -> String {
        guard code == "es", !english.isEmpty else { return english }
        lock.lock(); defer { lock.unlock() }
        if let hit = cache[english] { return hit }
        var result = english
        if let exact = L10nES.table[english], !english.contains("{}") {
            result = exact
        } else {
            let range = NSRange(english.startIndex..., in: english)
            for template in templates {
                guard let match = template.regex.firstMatch(in: english, range: range) else { continue }
                let values = (1..<match.numberOfRanges).map { i -> String in
                    guard let r = Range(match.range(at: i), in: english) else { return "" }
                    return String(english[r])
                }
                result = fill(template.spanish, values.map { translateValue($0) })
                break
            }
        }
        if cache.count > 4000 { cache.removeAll() }
        cache[english] = result
        return result
    }

    /// A value inside a template that is itself app copy ("Today · Uploading") is translated too.
    private static func translateValue(_ value: String) -> String {
        if let exact = L10nES.table[value], !value.contains("{}") { return exact }
        return value
    }

    /// Spanish template + values: `{}` next value, `{n}` the n-th, `{n:es}` "es" when the n-th is not empty.
    static func fill(_ spanish: String, _ values: [String]) -> String {
        var out = ""
        var next = 0
        var i = spanish.startIndex
        while i < spanish.endIndex {
            if spanish[i] == "{", let close = spanish[i...].firstIndex(of: "}") {
                let body = spanish[spanish.index(after: i)..<close]
                if body.isEmpty {
                    out += next < values.count ? values[next] : ""
                    next += 1
                    i = spanish.index(after: close); continue
                }
                let pieces = body.split(separator: ":", maxSplits: 1).map(String.init)
                if let n = Int(pieces[0]), n >= 1 {
                    let value = n - 1 < values.count ? values[n - 1] : ""
                    if pieces.count == 2 { out += value.isEmpty ? "" : pieces[1] } else { out += value }
                    i = spanish.index(after: close); continue
                }
            }
            out.append(spanish[i])
            i = spanish.index(after: i)
        }
        return out
    }

    /// For SwiftUI keys: "Keyframe added at %@ · tap ↶ to undo" → the Spanish format with the same specifiers.
    private static let specifier = try! NSRegularExpression(pattern: "%(?:\\d+\\$)?[-+ 0#]*\\d*(?:\\.\\d+)?(?:ll|l|hh|h|q|z|t|j)?[@dDiuUxXoOfeEgGcCsSpaA]")

    static func format(forKey key: String) -> String? {
        guard code == "es" else { return nil }
        if !key.contains("%") { return L10nES.table[key] }
        let range = NSRange(key.startIndex..., in: key)
        let matches = specifier.matches(in: key, range: range)
        guard !matches.isEmpty else { return L10nES.table[key] }
        var specs: [String] = []
        var template = key
        for m in matches.reversed() {
            guard let r = Range(m.range, in: template) else { return nil }
            var spec = String(template[r])
            if let dollar = spec.firstIndex(of: "$") { spec = "%" + spec[spec.index(after: dollar)...] }
            specs.insert(spec, at: 0)
            template.replaceSubrange(r, with: "{}")
        }
        // A Spanish text that needs a plural ending ({n:es}) cannot be a fixed format: SwiftUI shows English there.
        guard let spanish = L10nES.table[template], !spanish.contains(":es}") else { return nil }
        // {} / {n} → %k$spec (positional, so Spanish may use the values in another order)
        var out = ""
        var next = 0
        var i = spanish.startIndex
        while i < spanish.endIndex {
            if spanish[i] == "{", let close = spanish[i...].firstIndex(of: "}") {
                let body = String(spanish[spanish.index(after: i)..<close])
                var n = Int(body)
                if body.isEmpty { next += 1; n = next }
                if let n, n >= 1, n <= specs.count {
                    let spec = specs[n - 1]
                    out += "%\(n)$" + spec.dropFirst()
                    i = spanish.index(after: close); continue
                }
            }
            out.append(spanish[i] == "%" ? "%%" : String(spanish[i]))
            i = spanish.index(after: i)
        }
        return out
    }
}

/// Bundle.main answers SwiftUI's key lookups from the Spanish table while the app is in Spanish.
final class L10nBundle: Bundle, @unchecked Sendable {
    nonisolated(unsafe) private static var installed = false

    static func install() {
        guard !installed else { return }
        installed = true
        object_setClass(Bundle.main, L10nBundle.self)
    }

    override func localizedString(forKey key: String, value: String?, table tableName: String?) -> String {
        if tableName == nil || tableName == "Localizable", let spanish = L10nState.format(forKey: key) { return spanish }
        return super.localizedString(forKey: key, value: value, table: tableName)
    }

    /// Lookups that pick a .lproj folder (es.lproj/Localizable.strings) follow the app's language.
    override var preferredLocalizations: [String] {
        L10nState.code == "es" ? ["es"] : ["en"]
    }
}

extension String {
    /// "jue 22" → "Jue 22" (Spanish day names are lower case).
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
