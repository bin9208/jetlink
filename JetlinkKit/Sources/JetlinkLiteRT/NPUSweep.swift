import Foundation
import JetlinkONNX
import JetlinkRegistry
import JetlinkServer

/// After a benchmark on a MediaTek phone, what decides how to run the model
/// within the comma's 50 ms there.
///
/// On the Galaxy Tab S10+ (MT6989) carrot.18 brought Light Speed 2 on the
/// APU from 2051 ms to 157 (the image queue's views, kept small), the GPU
/// runs it at 94, and both read fp16 weights at only about 10 to 13 GB/s
/// (768 MB of linear layers: 77 ms on the APU, 57 on the GPU): Light Speed
/// 2's 734 MB cannot get under 50 ms at fp16 on either. int8 weights halve
/// the bytes; the APU ran them slower than fp16, but LiteRT's GPU keeps an
/// int8 weight int8 for its language models. So this sweep times, on the
/// GPU, 768 MB of fp16 linear layers against the same in int8, and the
/// benchmarked model with fp16 and with int8 weights, and says how far the
/// int8 model's outputs are from the fp16 one's.
extension LiteRtBackend {
  struct SweepCase {
    enum Kind {
      case linear(layers: Int)
      case model(URL)
    }

    let kind: Kind
    let int8: Bool

    var name: String {
      let weights = int8 ? "int8" : "fp16"
      switch kind {
      case .linear(let layers):
        return "GPU linear \(weights) \((layers * LiteRtBackend.sweepWidth * LiteRtBackend.sweepWidth * (int8 ? 1 : 2)) >> 20) MB"
      case .model(let url):
        return "GPU \(url.deletingPathExtension().lastPathComponent.prefix(16)) \(weights) weights"
      }
    }

    var slug: String {
      switch kind {
      case .linear(let layers): "linear\(layers)-\(int8)"
      case .model: "model-\(int8)"
      }
    }
  }

  static let sweepWidth = 2048
  static let sweepRows = 128
  static let sweepRuns = 10

  public func afterBenchmark(model: URL?, directory: URL, progress: @escaping (String) -> Void) -> [String] {
    #if os(Android)
      guard LiteRtRuntime.npuVendor == .mediaTek else { return [] }
      var cases = [SweepCase(kind: .linear(layers: 96), int8: false), SweepCase(kind: .linear(layers: 96), int8: true)]
      if let model { cases += [SweepCase(kind: .model(model), int8: false), SweepCase(kind: .model(model), int8: true)] }
      var lines = ["NPU sweep: \(cases.count) cases on the GPU: fp16 against int8 weights, linear layers and the model"]
      var reference: [Float]?
      for (index, sweep) in cases.enumerated() {
        progress("NPU sweep \(index + 1)/\(cases.count): \(sweep.name)")
        log.info("NPU sweep: starting \(sweep.name)")
        var line: String
        switch runSweepCase(sweep, in: directory) {
        case .success(let result):
          line = "\(sweep.name): \(result.line)"
          if case .model = sweep.kind {
            if let reference, let out = result.output, out.count == reference.count {
              // the int8 model's driving output against the fp16 one's, on the same zeros
              let diffs = zip(out, reference).map { abs($0 - $1) }
              let scale = reference.map(abs).max() ?? 0
              line += String(
                format: "; against fp16: max |d| %.4f, mean |d| %.5f (fp16 outputs up to %.3f)", diffs.max() ?? 0,
                diffs.reduce(0, +) / Float(max(diffs.count, 1)), scale)
            } else if !sweep.int8 {
              reference = result.output
            }
          }
        case .failure(let error):
          line = "\(sweep.name): failed: \(error)"
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

  struct SweepResult {
    let line: String
    /// The driving output of the last run, as floats; nil for a model without one.
    let output: [Float]?
  }

  /// One case converted for the GPU, compiled and run on zeros.
  func runSweepCase(_ sweep: SweepCase, in directory: URL) -> Result<SweepResult, Error> {
    let fm = FileManager.default
    let work = directory.appending(path: sweep.slug, directoryHint: .isDirectory)
    defer { try? fm.removeItem(at: work) }
    do {
      try? fm.removeItem(at: work)
      try fm.createDirectory(at: work, withIntermediateDirectories: true)
      let convertStarted = Date()
      let converted: LiteRTPreparation.Report
      switch sweep.kind {
      case .linear(let layers):
        let onnx = work.appending(path: "model.onnx")
        try SyntheticModels.writeLinearStack(layers: layers, width: Self.sweepWidth, rows: Self.sweepRows, to: onnx)
        converted = try LiteRTPreparation.prepare(source: onnx, into: work, int8Weights: sweep.int8)
        try? fm.removeItem(at: onnx)
      case .model(let onnx):
        converted = try LiteRTPreparation.prepare(source: onnx, into: work, int8Weights: sweep.int8)
      }
      let convertSeconds = Date().timeIntervalSince(convertStarted)
      let compileStarted = Date()
      let engine = try LiteRtEngine(model: converted.url, options: .gpu(cache: nil), device: "sweep", label: { $0.name })
      defer { engine.close() }
      let compileSeconds = Date().timeIntervalSince(compileStarted)
      try engine.run()
      var runs: [Double] = []
      let budget = Date().addingTimeInterval(30)
      while runs.count < Self.sweepRuns && (runs.count < 2 || Date() < budget) {
        try engine.run()
        runs.append(Double(engine.lastGpuUs) / 1000)
      }
      var output: [Float]?
      if let spec = engine.outputs[ModelConstants.drivingOutput], let raw = engine.output(ModelConstants.drivingOutput) {
        switch spec.type {
        case .float: output = Array(UnsafeBufferPointer(start: raw.assumingMemoryBound(to: Float.self), count: spec.byteCount / 4))
        case .float16:
          output = UnsafeBufferPointer(start: raw.assumingMemoryBound(to: UInt16.self), count: spec.byteCount / 2).map { Float(Float16(bitPattern: $0)) }
        default: break
        }
      }
      let finite = output?.allSatisfy(\.isFinite) ?? true
      func median(_ v: [Double]) -> String { String(format: "%.1f", v.sorted()[v.count / 2]) }
      let line = "\(converted.operators.values.reduce(0, +)) ops, \(Files.size(of: converted.url) >> 20) MB, on \(engine.hardware.name)"
        + "\(engine.fullyAccelerated ? "" : " (not every op)"), convert \(String(format: "%.0f", convertSeconds)) s, "
        + "compile \(String(format: "%.0f", compileSeconds)) s, \(runs.count) runs ms median \(median(runs))"
        + " min \(String(format: "%.1f", runs.min()!)) max \(String(format: "%.1f", runs.max()!))" + (finite ? "" : ", outputs NOT finite")
      return .success(SweepResult(line: line, output: output))
    } catch {
      return .failure(error)
    }
  }
}
