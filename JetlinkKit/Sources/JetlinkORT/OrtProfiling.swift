import Foundation
import JetlinkONNX
import JetlinkServer

/// Where a model's time goes on a Snapdragon's NPU: the model prepared as
/// the NPU runs it, compiled with QNN's detailed profiling, run on changing
/// inputs, and the profile QNN writes read back: how long QNN's execute
/// took against onnxruntime's whole run (the rest is moving inputs and
/// outputs), and the nodes that took longest.
extension OrtBackend {
  public func profileOnAccelerator(model: URL, directory: URL, progress: @escaping (String) -> Void) -> [String] {
    guard profile != .cpu, let unit = profile.sessions.first?.unit, profile.sessions.count == 1, unit == .htp || unit == .qnnGPU else {
      return []
    }
    let fm = FileManager.default
    let dir = directory.appending(path: "profile", directoryHint: .isDirectory)
    defer { try? fm.removeItem(at: dir) }
    do {
      try? fm.removeItem(at: dir)
      try fm.createDirectory(at: dir, withIntermediateDirectories: true)
      progress("profile: preparing")
      let prepared = try CoreMLPreparation.prepare(source: model, into: dir, layout: profile.layout, cacheKey: { "profile\($0)" })
      // A model in chunks is profiled a chunk at a time, each compiled and run
      // alone (on zeros for the hand-offs, which time the same); one CSV of
      // them all, and a run's time their sum.
      var combined = ""
      var wall: [Double] = []
      for (index, part) in prepared.parts.enumerated() {
        let label = prepared.parts.count > 1 ? " \(part.name) (\(index + 1)/\(prepared.parts.count))" : ""
        let csv = dir.appending(path: "qnn_profile_\(part.name).csv")
        var options = providerOptions(unit)
        options["profiling_level"] = "detailed"
        options["profiling_file_path"] = csv.path
        progress("profile: compiling\(label) with QNN's profiling")
        var partWall: [Double] = []
        do {
          let session = try OrtSession(model: part.url, provider: "QNN", options: options, threads: 1)
          final class Buffers {
            var all: [UnsafeMutableRawPointer] = []
            deinit { all.forEach { $0.deallocate() } }
            func alloc(_ s: TensorSpec) -> UnsafeMutableRawPointer {
              let p = UnsafeMutableRawPointer.allocate(byteCount: max(s.byteCount, 1), alignment: 64)
              p.initializeMemory(as: UInt8.self, repeating: 0, count: max(s.byteCount, 1))
              all.append(p)
              return p
            }
          }
          let buffers = Buffers()
          let inputs = session.inputs.map { ($0, buffers.alloc($0)) }
          let binding = try OrtBinding(session: session, inputs: inputs, outputs: session.outputs.map { ($0, buffers.alloc($0)) })
          var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
          let runs = 23
          for run in 0..<runs {
            // changing camera bytes, as a drive gives
            for (spec, p) in inputs where spec.name.contains("img") {
              let bytes = p.assumingMemoryBound(to: UInt8.self)
              for i in stride(from: run % 13, to: spec.byteCount, by: 13) {
                seed ^= seed << 13
                seed ^= seed >> 7
                seed ^= seed << 17
                bytes[i] = UInt8(truncatingIfNeeded: seed >> 24)
              }
            }
            let t = ProcessInfo.processInfo.systemUptime
            try binding.run()
            if run >= 3 { partWall.append((ProcessInfo.processInfo.systemUptime - t) * 1000) }
            progress("profile:\(label) run \(run + 1)/\(runs)")
          }
        }
        wall = wall.isEmpty ? partWall : zip(wall, partWall).map { $0 + $1 }
        // The session is gone: QNN has written its profile.
        guard let text = try? String(contentsOf: csv, encoding: .utf8) else {
          return ["profile: QNN wrote no profile at \(csv.lastPathComponent)"]
        }
        // one header, the first file's
        combined += combined.isEmpty ? text : String(text.drop { $0 != "\n" }.dropFirst())
        if !combined.hasSuffix("\n") { combined += "\n" }
      }
      var lines = QnnProfile(csv: combined).summary(wallMs: wall)
      if prepared.parts.count > 1 {
        lines.insert("profile: \(prepared.parts.count) chunks, each compiled and run alone; the run is their sum", at: 0)
      }
      return lines
    } catch {
      return ["profile failed: \(error)"]
    }
  }
}

/// The CSV onnxruntime's QNN provider writes with `profiling_file_path`:
/// Msg Timestamp, Message, Time, Unit of Measurement, Timing Source, Event
/// Level, Event Identifier. ROOT events are a call's own (an execute), the
/// others its parts (each node's, with detailed profiling).
struct QnnProfile {
  struct Total {
    var sum: Double = 0
    var count = 0
    var unit = ""
  }
  var roots: [String: Total] = [:]
  var nodes: [String: Total] = [:]
  var header = ""
  var rows = 0

  init(csv: String) {
    for (i, line) in csv.split(whereSeparator: \.isNewline).enumerated() {
      if i == 0 {
        header = String(line)
        continue
      }
      let f = line.split(separator: ",", omittingEmptySubsequences: false).map { $0.trimmingCharacters(in: .whitespaces) }
      guard f.count >= 7, let value = Double(f[2]) else { continue }
      rows += 1
      let identifier = f[6...].joined(separator: ",")
      let key = identifier.isEmpty ? f[1] : identifier
      let root = f[5].uppercased().contains("ROOT")
      if root {
        roots[key, default: Total()].sum += value
        roots[key]!.count += 1
        roots[key]!.unit = f[3]
      } else {
        nodes[key, default: Total()].sum += value
        nodes[key]!.count += 1
        nodes[key]!.unit = f[3]
      }
    }
  }

  func summary(wallMs: [Double], top: Int = 20) -> [String] {
    var lines: [String] = []
    let wall = wallMs.isEmpty ? 0 : wallMs.sorted()[wallMs.count / 2]
    lines.append(String(format: "profile: onnxruntime run p50 %.2f ms over %d runs; %d profile rows", wall, wallMs.count, rows))
    if rows == 0 { lines.append("profile: header \(header.prefix(200))") }
    let rootList = roots.sorted { $0.value.sum / Double(max($0.value.count, 1)) > $1.value.sum / Double(max($1.value.count, 1)) }
    if !rootList.isEmpty {
      lines.append(
        "profile: calls (mean): "
          + rootList.prefix(10).map { String(format: "%@ %.1f %@", $0.key, $0.value.sum / Double(max($0.value.count, 1)), $0.value.unit) }
          .joined(separator: "; "))
    }
    // A node's run is counted in cycles; the build's steps (VTCM allocation,
    // graph sequencing) come as sub-events in microseconds. Only the cycles
    // are the run's share: mixing the two put build steps at the top.
    let mean = { (t: Total) in t.sum / Double(max(t.count, 1)) }
    let running = nodes.filter { $0.value.unit.uppercased().contains("CYCLE") }
    let building = nodes.filter { !$0.value.unit.uppercased().contains("CYCLE") }
    let measured = running.isEmpty ? nodes : running
    let total = measured.values.reduce(0) { $0 + mean($1) }
    let nodeList = measured.sorted { mean($0.value) > mean($1.value) }
    if !nodeList.isEmpty {
      lines.append(String(format: "profile: %d nodes, %.0f %@ a run in all", measured.count, total, nodeList[0].value.unit))
      // in chunks, so no log line is too long to read
      let shown = nodeList.prefix(top).map { (key: $0.key, mean: mean($0.value)) }
      for chunk in stride(from: 0, to: shown.count, by: 5) {
        lines.append(
          "profile: top \(chunk + 1)-\(min(chunk + 5, shown.count)): "
            + shown[chunk..<min(chunk + 5, shown.count)].map {
              String(format: "%@ %.0f (%.1f%%)", $0.key, $0.mean, total > 0 ? $0.mean / total * 100 : 0)
            }.joined(separator: "; "))
      }
      // Where the run goes by kind of node (every GELU, every Transpose
      // together), which says which rewrite could pay; a slowest-node list
      // spread over 1000+ nodes does not.
      var kinds: [String: (share: Double, count: Int)] = [:]
      for (key, value) in measured {
        let kind = QnnProfile.kind(key)
        kinds[kind, default: (0, 0)].share += mean(value)
        kinds[kind]!.count += 1
      }
      let kindList = kinds.sorted { $0.value.share > $1.value.share }.prefix(top)
      let kindShown = Array(kindList)
      for chunk in stride(from: 0, to: kindShown.count, by: 5) {
        lines.append(
          "profile: kinds \(chunk + 1)-\(min(chunk + 5, kindShown.count)): "
            + kindShown[chunk..<min(chunk + 5, kindShown.count)].map {
              String(format: "%@ %.1f%% (%d)", $0.key, total > 0 ? $0.value.share / total * 100 : 0, $0.value.count)
            }.joined(separator: "; "))
      }
    }
    if !running.isEmpty && !building.isEmpty {
      let steps = building.sorted { mean($0.value) > mean($1.value) }.prefix(5)
      lines.append(
        "profile: build steps (not the run): "
          + steps.map { String(format: "%@ %.0f %@", $0.key, mean($0.value), $0.value.unit) }.joined(separator: "; "))
    }
    return lines
  }

  /// A node's kind from QNN's name for it: "node_gelu_35_12:OpId_7920
  /// (cycles)" is "gelu", "val_545__reshape_out_12" is "reshape_out".
  static func kind(_ key: String) -> String {
    var name = key.replacingOccurrences(of: " (cycles)", with: "")
    if let opId = name.range(of: ":OpId_") { name = String(name[..<opId.lowerBound]) }
    if name.hasPrefix("node_") { name.removeFirst(5) }
    if name.hasPrefix("val_"), let rest = name.range(of: "__") { name = String(name[rest.upperBound...]) }
    let parts = name.split(separator: "_").filter { !$0.allSatisfy(\.isNumber) }
    return parts.isEmpty ? name : parts.joined(separator: "_")
  }
}
