import Foundation
import JetlinkONNX
import JetlinkRegistry
import JetlinkServer

/// After a benchmark on a MediaTek APU, what tells why the driving model is
/// slow there, and whether int8 weights make it fast.
///
/// What earlier sweeps found on the Galaxy Tab S10+ (MT6989): Light Speed 2
/// compiled whole for the MDLA runs 2051 ms a frame, all of it the model
/// (copies under a millisecond). 2048-wide linear layers ran 1.3, 6.0 and
/// 26.5 ms for 16, 64 and 256 MB of fp16 weights (about 10 GB/s, the GPU
/// about the same), and eight of each of Light Speed 2's other kinds of
/// layers at its shapes ran 0.3 to 2.3 ms on the APU, no slower than on the
/// GPU. None of that adds up to 2 s; Light Speed 2's 770 MB, three times the
/// largest case, and its 11-minute compile (a second for 256 MB) do.
///
/// So this sweep finds the size where fp16 linear layers stop scaling (64 to
/// 768 MB), times int8 weights (half the bytes, one scale a weight: the
/// MDLA takes no per-channel DEQUANTIZE), and compiles and times the model
/// itself with int8 weights (416 MB for Light Speed 2).
extension LiteRtBackend {
  struct SweepCase {
    enum Kind {
      case linear(layers: Int)
      case model(URL)
    }

    let kind: Kind
    let npu: Bool
    let int8: Bool

    var name: String {
      let weights = int8 ? "int8" : "fp16"
      switch kind {
      case .linear(let layers):
        let mb = (layers * LiteRtBackend.sweepWidth * LiteRtBackend.sweepWidth * (int8 ? 1 : 2)) >> 20
        return "\(npu ? "NPU" : "GPU") linear \(weights) \(mb) MB (\(layers) x \(LiteRtBackend.sweepWidth))"
      case .model(let url):
        return "\(npu ? "NPU" : "GPU") \(url.deletingPathExtension().lastPathComponent.prefix(16)) \(weights) weights"
      }
    }

    var slug: String {
      switch kind {
      case .linear(let layers): "linear\(layers)-\(npu)-\(int8)"
      case .model: "model-\(npu)-\(int8)"
      }
    }
  }

  static let sweepWidth = 2048
  static let sweepRows = 128
  static let sweepRuns = 10

  /// Smallest first, the model with int8 weights before the largest fp16
  /// cases: each line is logged as it finishes, so a case that runs the
  /// phone out of memory leaves the ones before it.
  static func sweepCases(model: URL?) -> [SweepCase] {
    var cases = [
      SweepCase(kind: .linear(layers: 8), npu: true, int8: false),
      SweepCase(kind: .linear(layers: 8), npu: true, int8: true),
      SweepCase(kind: .linear(layers: 48), npu: true, int8: false),
      SweepCase(kind: .linear(layers: 64), npu: true, int8: false),
    ]
    if let model { cases.append(SweepCase(kind: .model(model), npu: true, int8: true)) }
    cases += [
      SweepCase(kind: .linear(layers: 80), npu: true, int8: false),
      SweepCase(kind: .linear(layers: 96), npu: true, int8: false),
      SweepCase(kind: .linear(layers: 96), npu: true, int8: true),
      SweepCase(kind: .linear(layers: 96), npu: false, int8: false),
    ]
    return cases
  }

  public func afterBenchmark(model: URL?, directory: URL, progress: @escaping (String) -> Void) -> [String] {
    #if os(Android)
      guard profile == .npu, LiteRtRuntime.npuVendor == .mediaTek else { return [] }
      let cases = Self.sweepCases(model: model)
      var lines = ["NPU sweep: \(cases.count) cases: fp16 linear layers by size, int8 weights, the model with int8 weights"]
      for (index, sweep) in cases.enumerated() {
        progress("NPU sweep \(index + 1)/\(cases.count): \(sweep.name)")
        log.info("NPU sweep: starting \(sweep.name)")
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
      let convertStarted = Date()
      let converted: LiteRTPreparation.Report
      switch sweep.kind {
      case .linear(let layers):
        let onnx = work.appending(path: "model.onnx")
        try SyntheticModels.writeLinearStack(layers: layers, width: Self.sweepWidth, rows: Self.sweepRows, to: onnx)
        converted = try LiteRTPreparation.prepare(source: onnx, into: work, forNPU: sweep.npu, int8Weights: sweep.int8)
        try? fm.removeItem(at: onnx)
      case .model(let onnx):
        converted = try LiteRTPreparation.prepare(source: onnx, into: work, forNPU: sweep.npu, int8Weights: sweep.int8)
      }
      let convertSeconds = Date().timeIntervalSince(convertStarted)
      let compileStarted = Date()
      let options: LiteRtCompileOptions = sweep.npu ? .npu(compiled: compiled, gpu: nil) : .gpu(cache: nil)
      let engine = try LiteRtEngine(model: converted.url, options: options, device: "sweep", label: { $0.name })
      defer { engine.close() }
      let compileSeconds = Date().timeIntervalSince(compileStarted)
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
      // whether the outputs came out as numbers, from the last run
      var finite = true
      for (name, spec) in engine.outputs where spec.type == .float {
        if let out = engine.output(name) {
          let values = UnsafeBufferPointer(start: out.assumingMemoryBound(to: Float.self), count: spec.byteCount / 4)
          finite = finite && values.allSatisfy(\.isFinite)
        }
      }
      func median(_ v: [Double]) -> String { String(format: "%.1f", v.sorted()[v.count / 2]) }
      let phases = order.map { "\($0) \(median(parts[$0]!))" }.joined(separator: ", ")
      let bytes = Files.size(of: converted.url) >> 20
      return "\(sweep.name) on \(engine.hardware.name)\(engine.fullyAccelerated ? "" : " (not every op)"): "
        + "\(converted.operators.values.reduce(0, +)) ops, \(bytes) MB file, convert \(String(format: "%.0f", convertSeconds)) s, "
        + "compile \(String(format: "%.0f", compileSeconds)) s, "
        + "\(runs.count) runs ms median \(median(runs)) min \(String(format: "%.1f", runs.min()!)) max \(String(format: "%.1f", runs.max()!))"
        + (phases.isEmpty ? "" : " (\(phases))") + (finite ? "" : ", outputs NOT finite")
    } catch {
      return "\(sweep.name): failed: \(error)"
    }
  }
}
