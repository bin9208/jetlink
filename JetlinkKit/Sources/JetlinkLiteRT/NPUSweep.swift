import Foundation
import JetlinkONNX
import JetlinkRegistry
import JetlinkServer

/// After a benchmark on a MediaTek phone, what decides how to run the model
/// within the comma's 50 ms there.
///
/// On the Galaxy Tab S10+ (MT6989) carrot.18 brought Light Speed 2 on the
/// APU to 157 ms (the image queue's views kept small), 126 of them its
/// body's 718 MB of weights: about 6 GB/s, where a Snapdragon 8 Elite's NPU
/// streams N16's 814 MB in 25 ms. (The linear layers of earlier sweeps
/// looked faster because their repeating weights compressed eightfold on
/// the APU.) LiteRT compiles for the APU with no options of NeuroPilot's,
/// whose compiler knows --num-mdla (the cores to spread over), --opt and
/// --fc-to-conv, and reads them from MTKNN_ADAPTER_COMPILER_OPTIONS: on the
/// PC, --num-mdla=2 changed the compiled model and a made-up option did not.
///
/// So this sweep compiles the benchmarked model for the APU under each of
/// those, and times it; and, on the GPU, times it with fp16 and with int8
/// weights (half the bytes, if LiteRT's GPU keeps them int8) and says how
/// far the int8 one's outputs are. Each NPU line says how big the compiled
/// model came out, which tells whether the options reached the compiler.
extension LiteRtBackend {
  struct SweepCase {
    let name: String
    let npu: Bool
    let int8: Bool
    /// MTKNN_ADAPTER_COMPILER_OPTIONS for the NPU's compile
    var options: String? = nil
    /// MTKNN_MDLA_ENABLE_MULTI_CORE=1 for the compile and the runs
    var multiCore = false
    var tuned = false
  }

  static let sweepRuns = 5

  /// The GPU's first: carrot.20's sweep never reached them. Its options
  /// for the APU (--num-mdla=2, --opt=3, --fc-to-conv, all three) all
  /// reached the compiler (each compiled to another size) and none made
  /// the model faster (115 to 155 ms against the default's 120 to 156);
  /// turbo boost with the L1 cache and the low-latency hint killed the app
  /// in the compile, three times, so it is gone.
  static let sweepCases: [SweepCase] = [
    SweepCase(name: "GPU fp16", npu: false, int8: false),
    SweepCase(name: "GPU int8", npu: false, int8: true),
    SweepCase(name: "NPU default", npu: true, int8: false),
    SweepCase(name: "NPU multi-core", npu: true, int8: false, multiCore: true),
    SweepCase(name: "NPU --num-mdla=2, multi-core", npu: true, int8: false, options: "--num-mdla=2", multiCore: true),
  ]

  public func afterBenchmark(model: URL?, directory: URL, progress: @escaping (String) -> Void) -> [String] {
    #if os(Android)
      guard LiteRtRuntime.npuVendor == .mediaTek, let model else { return [] }
      let cases = Self.sweepCases
      var lines = ["NPU sweep: \(model.lastPathComponent) under \(cases.count) settings: NeuroPilot's options on the APU, int8 weights on the GPU"]
      var gpuReference: [Float]?
      for (index, sweep) in cases.enumerated() {
        progress("NPU sweep \(index + 1)/\(cases.count): \(sweep.name)")
        log.info("NPU sweep: starting \(sweep.name)")
        var line: String
        switch runSweepCase(sweep, model: model, in: directory) {
        case .success(let result):
          line = "\(sweep.name): \(result.line)"
          if !sweep.npu {
            if let reference = gpuReference, let out = result.output, out.count == reference.count {
              // the int8 model's driving output against the fp16 one's, on the same zeros
              let diffs = zip(out, reference).map { abs($0 - $1) }
              line += String(
                format: "; against fp16: max |d| %.4f, mean |d| %.5f (fp16 outputs up to %.3f)", diffs.max() ?? 0,
                diffs.reduce(0, +) / Float(max(diffs.count, 1)), reference.map(abs).max() ?? 0)
            } else if !sweep.int8 {
              gpuReference = result.output
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
    /// The driving output of the last run, as floats.
    let output: [Float]?
  }

  /// The model converted for the case's device, compiled under its settings
  /// and run on zeros.
  func runSweepCase(_ sweep: SweepCase, model: URL, in directory: URL) -> Result<SweepResult, Error> {
    let fm = FileManager.default
    let work = directory.appending(path: "case", directoryHint: .isDirectory)
    defer { try? fm.removeItem(at: work) }
    let environment = [
      ("MTKNN_ADAPTER_COMPILER_OPTIONS", sweep.options), ("MTKNN_MDLA_ENABLE_MULTI_CORE", sweep.multiCore ? "1" : nil),
    ]
    for (name, value) in environment {
      if let value { setenv(name, value, 1) } else { unsetenv(name) }
    }
    defer { for (name, _) in environment { unsetenv(name) } }
    do {
      try? fm.removeItem(at: work)
      let compiled = work.appending(path: LiteRtArtifact.npuCache, directoryHint: .isDirectory)
      try fm.createDirectory(at: compiled, withIntermediateDirectories: true)
      let convertStarted = Date()
      let converted = try LiteRTPreparation.prepare(source: model, into: work, forNPU: sweep.npu, int8Weights: sweep.int8)
      let convertSeconds = Date().timeIntervalSince(convertStarted)
      let compileStarted = Date()
      let options: LiteRtCompileOptions = sweep.npu ? .npu(compiled: compiled, gpu: nil, tuned: sweep.tuned) : .gpu(cache: nil)
      let engine = try LiteRtEngine(model: converted.url, options: options, device: "sweep", label: { $0.name })
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
      // what LiteRT kept of the NPU's compile: its size changes with the options that reached the compiler
      let compiledBytes = sweep.npu ? ", compiled \(Files.size(of: compiled)) bytes" : ""
      let line = "\(Files.size(of: converted.url) >> 20) MB, on \(engine.hardware.name)\(engine.fullyAccelerated ? "" : " (not every op)")"
        + ", convert \(String(format: "%.0f", convertSeconds)) s, compile \(String(format: "%.0f", compileSeconds)) s\(compiledBytes), "
        + "\(runs.count) runs ms median \(median(runs)) min \(String(format: "%.1f", runs.min()!)) max \(String(format: "%.1f", runs.max()!))"
        + (finite ? "" : ", outputs NOT finite")
      return .success(SweepResult(line: line, output: output))
    } catch {
      return .failure(error)
    }
  }
}
