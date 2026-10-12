import Foundation

/// "Time saved" count, kept separately for each account (keyed by the account's user id).
///
/// Rule (set for CutSell on 2026-10-12): editing a raw video by hand takes about 2 minutes for
/// every minute of footage (watch it all, find the good takes, cut, tidy up), so one cut saves
/// 2 × the footage length. The total adds every cut of the account once.
/// It is stored on this iPhone; a server copy (so it follows the account across phones) is a
/// later step.
enum TimeSavedLedger {
    static let editingMinutesPerFootageMinute = 2.0

    static func saved(forFootage seconds: Double) -> Double {
        max(0, seconds) * editingMinutesPerFootageMinute
    }

    private static func key(_ userID: String) -> String { "cutsell.timeSaved.\(userID)" }

    /// Adds this cut once (re-opening the same cut does not count twice). Returns the new total.
    @discardableResult
    static func record(userID: String, projectID: String, footageSeconds: Double,
                       defaults: UserDefaults = .standard) -> Double {
        var perCut = defaults.dictionary(forKey: key(userID)) as? [String: Double] ?? [:]
        if perCut[projectID] == nil, footageSeconds > 0 {
            perCut[projectID] = saved(forFootage: footageSeconds)
            defaults.set(perCut, forKey: key(userID))
        }
        return perCut.values.reduce(0, +)
    }

    static func total(userID: String, defaults: UserDefaults = .standard) -> Double {
        (defaults.dictionary(forKey: key(userID)) as? [String: Double] ?? [:]).values.reduce(0, +)
    }

    /// "42h 18m" (or "12m" under an hour).
    static func hoursMinutes(_ seconds: Double) -> String {
        let minutes = Int((seconds / 60).rounded())
        return minutes >= 60 ? "\(minutes / 60)h \(minutes % 60)m" : "\(minutes)m"
    }

    /// Whole 8-hour workdays, for the "5+ WORKDAYS" label (nil under one day).
    static func workdays(_ seconds: Double) -> Int? {
        let days = Int(seconds / (8 * 3600))
        return days >= 1 ? days : nil
    }
}
