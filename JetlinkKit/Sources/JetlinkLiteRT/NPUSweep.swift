import Foundation
import JetlinkONNX
import JetlinkServer

/// After a benchmark on a MediaTek APU, the same kind of work at three sizes
/// and two settings, and on the GPU, for telling why a driving model is slow
/// there: Light Speed 2 compiled whole for the Galaxy Tab S10+'s MDLA (every
/// layer on it, NeuroPilot's plan said) and took 2 s a frame where the GPU
/// took 70 to 110 ms.
///
/// Each case is a stack of 2048-wide linear layers over 128 rows (the size of
/// Light Speed 2's widest), converted as for the NPU and compiled, then run
/// ten times on zeros. A time that grows with the weights' size far faster
/// than the GPU's says the APU streams its weights badly at that size; one
/// slow at every size says a setting; the tuned case says whether turbo
/// boost, the L1 cache and the low-latency hint help.
extension LiteRtBackend {
  struct SweepCase {
    let layers: Int
    let npu: Bool
    let tuned: Bool

    /// The weights' size, MB.
    var megabytes: Int { layers * LiteRtBackend.sweepWidth * LiteRtBackend.sweepWidth * 2 >> 20 }
    var name: String { "\(npu ? (tuned ? "NPU tuned" : "NPU") : "GPU") \(megabytes) MB (\(layers) x \(LiteRtBackend.sweepWidth))" }
  }

  static let sweepWidth = 2048
  static let sweepRows = 128
  static let sweepRuns = 10

  static let sweepCases: [SweepCase] = [
    SweepCase(layers: 2, npu: true, tuned: false),
    SweepCase(layers: 8, npu: true, tuned: false),
    SweepCase(layers: 32, npu: true, tuned: false),
    SweepCase(layers: 8, npu: true, tuned: true),
    SweepCase(layers: 8, npu: false, tuned: false),
    SweepCase(layers: 32, npu: false, tuned: false),
  ]

  public func afterBenchmark(directory: URL, progress: @escaping (String) -> Void) -> [String] {
    #if os(Android)
      guard profile == .npu, LiteRtRuntime.npuVendor == .mediaTek else { return [] }
      var lines = ["NPU sweep: \(Self.sweepCases.count) cases of \(Self.sweepWidth)-wide linear layers over \(Self.sweepRows) rows"]
      for (index, sweep) in Self.sweepCases.enumerated() {
        progress("NPU sweep \(index + 1)/\(Self.sweepCases.count): \(sweep.name)")
        let line = runSweepCase(sweep, in: directory)
        log.info("NPU sweep: \(line)")
        lines.append("NPU sweep: \(line)")
      }
      try? FileManager.default.removeItem(at: directory)
      return lines
    #else
      return []
    #endif
  }

  /// One case's line: how long the conversion and the compile took, what
  /// ran it, and its runs' times; or what failed.
  func runSweepCase(_ sweep: SweepCase, in directory: URL) -> String {
    let fm = FileManager.default
    let work = directory.appending(path: "case-\(sweep.layers)-\(sweep.npu)-\(sweep.tuned)", directoryHint: .isDirectory)
    defer { try? fm.removeItem(at: work) }
    do {
      try? fm.removeItem(at: work)
      let compiled = work.appending(path: LiteRtArtifact.npuCache, directoryHint: .isDirectory)
      try fm.createDirectory(at: compiled, withIntermediateDirectories: true)
      let onnx = work.appending(path: "model.onnx")
      let convertStarted = Date()
      try SyntheticModels.writeLinearStack(layers: sweep.layers, width: Self.sweepWidth, rows: Self.sweepRows, to: onnx)
      let converted = try LiteRTPreparation.prepare(source: onnx, into: work, forNPU: sweep.npu)
      try? fm.removeItem(at: onnx)
      let convertSeconds = Date().timeIntervalSince(convertStarted)
      let compileStarted = Date()
      let options: LiteRtCompileOptions = sweep.npu ? .npu(compiled: compiled, gpu: nil, tuned: sweep.tuned) : .gpu(cache: nil)
      let engine = try LiteRtEngine(model: converted.url, options: options, device: "sweep", label: { $0.name })
      defer { engine.close() }
      let compileSeconds = Date().timeIntervalSince(compileStarted)
      try engine.run()
      try engine.run()
      var runs: [Double] = []
      var parts: [String: [Double]] = [:]
      var order: [String] = []
      let budget = Date().addingTimeInterval(30)
      while runs.count < Self.sweepRuns && (runs.count < 2 || Date() < budget) {
        try engine.run()
        runs.append(Double(engine.lastGpuUs) / 1000)
        for part in engine.phases {
          if parts[part.name] == nil { order.append(part.name) }
          parts[part.name, default: []].append(Double(part.us) / 1000)
        }
      }
      func median(_ v: [Double]) -> String { String(format: "%.1f", v.sorted()[v.count / 2]) }
      let phases = order.map { "\($0) \(median(parts[$0]!))" }.joined(separator: ", ")
      return "\(sweep.name) on \(engine.hardware.name)\(engine.fullyAccelerated ? "" : " (not every op)"): "
        + "convert \(String(format: "%.0f", convertSeconds)) s, compile \(String(format: "%.0f", compileSeconds)) s, "
        + "\(runs.count) runs ms median \(median(runs)) min \(String(format: "%.1f", runs.min()!)) max \(String(format: "%.1f", runs.max()!))"
        + (phases.isEmpty ? "" : " (\(phases))")
    } catch {
      return "\(sweep.name): failed: \(error)"
    }
  }
}
