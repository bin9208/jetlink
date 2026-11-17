import Foundation
import JetlinkONNX
import JetlinkRegistry
import JetlinkServer

/// After a benchmark on a MediaTek APU, the benchmarked model timed a part at
/// a time, to find the part that makes it slow there.
///
/// What earlier sweeps found on the Galaxy Tab S10+ (MT6989): Light Speed 2
/// compiled whole for the MDLA runs 2051 ms a frame, all of it the model.
/// Eight of each of its kinds of layers at its shapes ran 0.3 to 2.3 ms on
/// the APU (no slower than on the GPU); fp16 linear layers scaled at about
/// 10 GB/s right up to 768 MB (78 ms), compiling in seconds; int8 weights
/// made them slower, and Light Speed 2 with int8 weights still took 2043 ms.
/// Neither a kind of layer nor the size, then: some part of its graph.
///
/// So the model is converted for the NPU and cut after 10%, 20%, ... of its
/// operators (LiteRTPreparation's keepFraction: what the kept part makes
/// that the rest reads becomes an output), each compiled and timed. Each
/// line says what the cut added to the time and which operators it added.
extension LiteRtBackend {
  static let sweepFractions: [Double] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  static let sweepRuns = 5

  public func afterBenchmark(model: URL?, directory: URL, progress: @escaping (String) -> Void) -> [String] {
    #if os(Android)
      guard profile == .npu, LiteRtRuntime.npuVendor == .mediaTek, let model else { return [] }
      var lines = ["NPU sweep: \(model.lastPathComponent) cut after \(Self.sweepFractions.map { "\(Int($0 * 100))%" }.joined(separator: ", ")) of its operators"]
      var last: (ms: Double, ops: [String: Int]) = (0, [:])
      for (index, fraction) in Self.sweepFractions.enumerated() {
        progress("NPU sweep \(index + 1)/\(Self.sweepFractions.count): the model cut at \(Int(fraction * 100))%")
        log.info("NPU sweep: starting the cut at \(Int(fraction * 100))%")
        let line: String
        switch runCut(model, fraction: fraction, in: directory) {
        case .success(let cut):
          // the operators this cut added to the last one's
          let added = cut.ops.compactMap { op, n -> (String, Int)? in
            let d = n - (last.ops[op] ?? 0)
            return d > 0 && op != "DEQUANTIZE" ? (op, d) : nil
          }.sorted { $0.1 > $1.1 }
          line = "cut \(Int(fraction * 100))%: \(cut.line); adds \(String(format: "%+.1f", cut.ms - last.ms)) ms with "
            + added.map { "\($0.0) \($0.1)" }.joined(separator: ", ")
          last = (cut.ms, cut.ops)
        case .failure(let error):
          line = "cut \(Int(fraction * 100))%: failed: \(error)"
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

  /// The model's first `fraction` converted for the NPU, compiled and run
  /// on zeros.
  func runCut(_ model: URL, fraction: Double, in directory: URL) -> Result<Cut, Error> {
    let fm = FileManager.default
    let work = directory.appending(path: "cut-\(Int(fraction * 100))", directoryHint: .isDirectory)
    defer { try? fm.removeItem(at: work) }
    do {
      try? fm.removeItem(at: work)
      let compiled = work.appending(path: LiteRtArtifact.npuCache, directoryHint: .isDirectory)
      try fm.createDirectory(at: compiled, withIntermediateDirectories: true)
      let convertStarted = Date()
      let converted = try LiteRTPreparation.prepare(source: model, into: work, forNPU: true, keepFraction: fraction < 1 ? fraction : nil)
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
