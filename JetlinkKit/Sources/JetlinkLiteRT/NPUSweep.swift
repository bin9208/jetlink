import Foundation
import JetlinkONNX
import JetlinkServer

/// After a benchmark on a MediaTek APU, Light Speed 2's kinds of layers one
/// kind at a time, on the APU and on the GPU, for finding what makes it slow
/// there: compiled whole for the Galaxy Tab S10+'s MDLA (every layer on it,
/// NeuroPilot's plan said), it took 2051 ms a run, where linear layers of
/// its weights' size took about 80 (carrot.12's sweep: 6.0 ms for 64 MB,
/// 26.5 for 256, about 10 GB/s; the GPU 6.6 and 19.4).
///
/// Each case is eight of one kind at Light Speed 2's shapes
/// (SyntheticModels.Block), or 64 MB of 2048-wide linear layers over 128
/// rows for scale, converted for the device that runs it and compiled, then
/// run ten times on zeros. The kind whose APU time is far above its GPU time
/// is what to change.
extension LiteRtBackend {
  struct SweepCase {
    enum Kind {
      case linear(layers: Int)
      case block(SyntheticModels.Block)
    }

    let kind: Kind
    let npu: Bool

    var name: String {
      let what: String
      switch kind {
      case .linear(let layers):
        what = "linear \((layers * LiteRtBackend.sweepWidth * LiteRtBackend.sweepWidth * 2) >> 20) MB (\(layers) x \(LiteRtBackend.sweepWidth))"
      case .block(let block):
        what = "\(LiteRtBackend.sweepRepeats) x \(block.rawValue)"
      }
      return "\(npu ? "NPU" : "GPU") \(what)"
    }

    var slug: String {
      switch kind {
      case .linear(let layers): "linear\(layers)-\(npu)"
      case .block(let block): "\(String(describing: block))-\(npu)"
      }
    }
  }

  static let sweepWidth = 2048
  static let sweepRows = 128
  static let sweepRuns = 10
  static let sweepRepeats = 8

  static let sweepCases: [SweepCase] = [true, false].flatMap { npu in
    [SweepCase(kind: .linear(layers: 8), npu: npu)] + SyntheticModels.Block.allCases.map { SweepCase(kind: .block($0), npu: npu) }
  }

  public func afterBenchmark(directory: URL, progress: @escaping (String) -> Void) -> [String] {
    #if os(Android)
      guard profile == .npu, LiteRtRuntime.npuVendor == .mediaTek else { return [] }
      var lines = ["NPU sweep: \(Self.sweepCases.count) cases, Light Speed 2's kinds of layers on the APU and the GPU"]
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
    let work = directory.appending(path: sweep.slug, directoryHint: .isDirectory)
    defer { try? fm.removeItem(at: work) }
    do {
      try? fm.removeItem(at: work)
      let compiled = work.appending(path: LiteRtArtifact.npuCache, directoryHint: .isDirectory)
      try fm.createDirectory(at: compiled, withIntermediateDirectories: true)
      let onnx = work.appending(path: "model.onnx")
      let convertStarted = Date()
      switch sweep.kind {
      case .linear(let layers):
        try SyntheticModels.writeLinearStack(layers: layers, width: Self.sweepWidth, rows: Self.sweepRows, to: onnx)
      case .block(let block):
        try SyntheticModels.writeBlocks(block, repeats: Self.sweepRepeats, to: onnx)
      }
      let converted = try LiteRTPreparation.prepare(source: onnx, into: work, forNPU: sweep.npu)
      try? fm.removeItem(at: onnx)
      let convertSeconds = Date().timeIntervalSince(convertStarted)
      let compileStarted = Date()
      let options: LiteRtCompileOptions = sweep.npu ? .npu(compiled: compiled, gpu: nil) : .gpu(cache: nil)
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
        + "\(converted.operators.values.reduce(0, +)) ops, convert \(String(format: "%.0f", convertSeconds)) s, compile \(String(format: "%.0f", compileSeconds)) s, "
        + "\(runs.count) runs ms median \(median(runs)) min \(String(format: "%.1f", runs.min()!)) max \(String(format: "%.1f", runs.max()!))"
        + (phases.isEmpty ? "" : " (\(phases))")
    } catch {
      return "\(sweep.name): failed: \(error)"
    }
  }
}
