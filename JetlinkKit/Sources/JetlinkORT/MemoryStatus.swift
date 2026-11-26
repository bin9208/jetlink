import Foundation

/// The process's memory and the phone's, read from /proc on Linux and
/// Android, for the build's log: a compile that runs the phone out of memory
/// is killed without a word, and the last of these lines says how close it was.
enum MemoryStatus {
  /// "app 1.7 GB (heap 1.7 GB), phone 4.2 GB free of 11.2 GB", or nil where
  /// /proc is not there (Apple platforms).
  static func describe() -> String? {
    guard let status = read("/proc/self/status"), let info = read("/proc/meminfo") else { return nil }
    func gb(_ kb: Int64?) -> String { kb.map { String(format: "%.1f GB", Double($0) / 1_048_576) } ?? "?" }
    return "app \(gb(status["VmRSS"])) (heap \(gb(status["RssAnon"]))), phone \(gb(info["MemAvailable"])) free of \(gb(info["MemTotal"]))"
  }

  /// "Key: 123 kB" lines as kB by key.
  static func read(_ path: String) -> [String: Int64]? {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
    return parse(text)
  }

  static func parse(_ text: String) -> [String: Int64] {
    var out: [String: Int64] = [:]
    for line in text.split(separator: "\n") {
      guard let colon = line.firstIndex(of: ":") else { continue }
      let number = line[line.index(after: colon)...].split(separator: " ").first.flatMap { Int64($0) }
      if let number { out[String(line[..<colon])] = number }
    }
    return out
  }
}
