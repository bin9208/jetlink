import Foundation
import JetlinkONNX
import JetlinkRegistry
import JetlinkServer

/// After a benchmark on a MediaTek APU, the benchmarked model timed a part at
/// a time, to find the part that makes it slow there.
///
/// carrot.17's sweep on the Galaxy Tab S10+ (MT6989) found it: Light Speed 2
/// cut after its first 260 operators (16 MB of weights) ran 2021 ms on the
/// MDLA, and the 2580 after them added 36. Those first operators held the
/// image queue's rank-5 views, which merged whole axes into rows of up to
/// 983040; carrot.18 merges as few as it takes (sliceView's minimal). This
/// sweep cuts the model after the first 8, 16, ... 260 operators, to show
/// whether the start is fast now and, if not, which operators are not.
extension LiteRtBackend {
  static let sweepCuts: [Int?] = [8, 16, 24, 30, 45, 62, 100, 160, 260, nil]
  static let sweepRuns = 5

  public func afterBenchmark(model: URL?, directory: URL, progress: @escaping (String) -> Void) -> [String] {
    #if os(Android)
      guard profile == .npu, LiteRtRuntime.npuVendor == .mediaTek, let model else { return [] }
      let names = Self.sweepCuts.map { $0.map(String.init) ?? "all" }
      var lines = ["NPU sweep: \(model.lastPathComponent) cut after its first \(names.joined(separator: ", ")) operators"]
      var last: (ms: Double, ops: [String: Int]) = (0, [:])
      for (index, cut) in Self.sweepCuts.enumerated() {
        let name = names[index]
        progress("NPU sweep \(index + 1)/\(Self.sweepCuts.count): the model's first \(name) operators")
        log.info("NPU sweep: starting the first \(name) operators")
        let line: String
        switch runCut(model, operators: cut, in: directory) {
        case .success(let result):
          let added = result.ops.compactMap { op, n -> (String, Int)? in
            let d = n - (last.ops[op] ?? 0)
            return d > 0 && op != "DEQUANTIZE" ? (op, d) : nil
          }.sorted { $0.1 > $1.1 }
          line = "first \(name): \(result.line); adds \(String(format: "%+.1f", result.ms - last.ms)) ms with "
            + added.map { "\($0.0) \($0.1)" }.joined(separator: ", ")
          last = (result.ms, result.ops)
        case .failure(let error):
          line = "first \(name): failed: \(error)"
        }
        log.info("NPU sweep: \(line)")
        lines.append("NPU sweep: \(line)")
      }
      try? FileManager.default.removeItem(at: directory)
      return lines
    #else
      return []
    #endif
  }

  struct Cut {
    let ms: Double
    let ops: [String: Int]
    let line: String
  }

  /// The model's first `operators` (all of it for nil) converted for the
  /// NPU, compiled and run on zeros.
  func runCut(_ model: URL, operators: Int?, in directory: URL) -> Result<Cut, Error> {
    let fm = FileManager.default
    let work = directory.appending(path: "cut-\(operators ?? 0)", directoryHint: .isDirectory)
    defer { try? fm.removeItem(at: work) }
    do {
      try? fm.removeItem(at: work)
      let compiled = work.appending(path: LiteRtArtifact.npuCache, directoryHint: .isDirectory)
      try fm.createDirectory(at: compiled, withIntermediateDirectories: true)
      let convertStarted = Date()
      let converted = try LiteRTPreparation.prepare(source: model, into: work, forNPU: true, keepOperators: operators)
      let convertSeconds = Date().timeIntervalSince(convertStarted)
      let compileStarted = Date()
      let engine = try LiteRtEngine(model: converted.url, options: .npu(compiled: compiled, gpu: nil), device: "sweep", label: { $0.name })
      defer { engine.close() }
      let compileSeconds = Date().timeIntervalSince(compileStarted)
      try engine.run()
      var runs: [Double] = []
      let budget = Date().addingTimeInterval(20)
      while runs.count < Self.sweepRuns && (runs.count < 2 || Date() < budget) {
        try engine.run()
        runs.append(Double(engine.lastGpuUs) / 1000)
      }
      let median = runs.sorted()[runs.count / 2]
      let total = converted.operators.values.reduce(0, +)
      let line = "\(total) ops, \(Files.size(of: converted.url) >> 20) MB, \(engine.outputs.count) outputs, on \(engine.hardware.name)"
        + "\(engine.fullyAccelerated ? "" : " (not every op)"), convert \(String(format: "%.0f", convertSeconds)) s, "
        + "compile \(String(format: "%.0f", compileSeconds)) s, model ms median \(String(format: "%.1f", median))"
        + " min \(String(format: "%.1f", runs.min()!)) max \(String(format: "%.1f", runs.max()!))"
      return .success(Cut(ms: median, ops: converted.operators, line: line))
    } catch {
      return .failure(error)
    }
  }
}
