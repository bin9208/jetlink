import Foundation
import JetlinkONNX
import JetlinkRegistry
import JetlinkServer

/// After a benchmark on a MediaTek phone, the last of the NPU sweeps: what
/// is left to make Light Speed 2 fast on a Dimensity's APU.
///
/// Where it stands on the Galaxy Tab S10+ (MT6989): 120 to 165 ms a frame on
/// the APU (carrot.18's small image-queue views), 94 to 120 on the GPU.
/// NeuroPilot's compile options (--num-mdla=2, --opt=3, --fc-to-conv) all
/// reached the phone's compiler and none helped; turbo boost with the L1
/// cache and the low-latency hint, and MTKNN_MDLA_ENABLE_MULTI_CORE=1,
/// killed the app; the GPU took no DEQUANTIZEd int8 weight.
///
/// So this sweep times the model on the APU by default; with a real-time
/// deadline of 50 ms (the scheduler is told it is one), with the CPU cache
/// flush around its inputs and outputs off, and with NeuroPilot's
/// import_forever (the buffers registered once), each through our dispatch
/// library's JETLINK_NEURON_* variables or the compiler's options; with its
/// FULLY_CONNECTEDs fully int8 (the MDLA's own kind of layer, at fixed
/// scales, so for timing alone); and with all of them.
///
/// A case that kills the app is marked beside the sweep's directory before
/// it starts: the next sweep says which one it was and leaves it out.
extension LiteRtBackend {
  struct SweepCase {
    let name: String
    var options: String? = nil
    var deadline: Int? = nil
    var cacheFlush: Int? = nil
    var fullyInt8 = false
  }

  static let sweepRuns = 5
  static let importForever = #"--apusys-config "{ \"import_forever\": true }""#

  static let sweepCases: [SweepCase] = [
    SweepCase(name: "NPU default"),
    SweepCase(name: "NPU deadline 50 ms", deadline: 50),
    // NEURON_CACHE_FLUSH_DISABLE_SYNC_INPUT | ..._DISABLE_INVALIDATE_OUTPUT
    SweepCase(name: "NPU cache flush off", cacheFlush: 3),
    SweepCase(name: "NPU import_forever", options: importForever),
    SweepCase(name: "NPU FULLY_CONNECTEDs fully int8", fullyInt8: true),
    SweepCase(name: "NPU all of them", options: importForever, deadline: 50, cacheFlush: 3, fullyInt8: true),
  ]

  public func afterBenchmark(model: URL?, directory: URL, progress: @escaping (String) -> Void) -> [String] {
    #if os(Android)
      guard LiteRtRuntime.npuVendor == .mediaTek, let model else { return [] }
      let fm = FileManager.default
      let running = directory.deletingLastPathComponent().appending(path: "npu-sweep-running")
      let killers = directory.deletingLastPathComponent().appending(path: "npu-sweep-killed")
      var killed = Set((try? String(contentsOf: killers, encoding: .utf8))?.split(separator: "\n").map(String.init) ?? [])
      let cases = Self.sweepCases
      var lines = ["NPU sweep: \(model.lastPathComponent) under \(cases.count) settings on the APU"]
      if let last = try? String(contentsOf: running, encoding: .utf8), !last.isEmpty {
        let line = "NPU sweep: the last sweep ended with the app killed during \(last); it is left out from now on"
        log.warning(line)
        lines.append(line)
        killed.insert(last)
        try? killed.sorted().joined(separator: "\n").write(to: killers, atomically: true, encoding: .utf8)
        try? fm.removeItem(at: running)
      }
      for (index, sweep) in cases.enumerated() {
        if killed.contains(sweep.name) {
          lines.append("NPU sweep: \(sweep.name): left out, it killed the app before")
          continue
        }
        progress("NPU sweep \(index + 1)/\(cases.count): \(sweep.name)")
        log.info("NPU sweep: starting \(sweep.name) (\(Self.availableMemory()) MB available)")
        try? sweep.name.write(to: running, atomically: true, encoding: .utf8)
        let line: String
        switch runSweepCase(sweep, model: model, in: directory) {
        case .success(let result): line = "\(sweep.name): \(result.line)"
        case .failure(let error): line = "\(sweep.name): failed: \(error)"
        }
        try? fm.removeItem(at: running)
        log.info("NPU sweep: \(line)")
        lines.append("NPU sweep: \(line)")
      }
      try? fm.removeItem(at: directory)
      return lines
    #else
      return []
    #endif
  }

  /// MemAvailable from /proc/meminfo, MB; 0 where it cannot be read.
  static func availableMemory() -> Int {
    guard let text = try? String(contentsOfFile: "/proc/meminfo", encoding: .utf8),
      let line = text.split(separator: "\n").first(where: { $0.hasPrefix("MemAvailable:") }),
      let kb = line.split(separator: " ").dropFirst().first.flatMap({ Int($0) })
    else { return 0 }
    return kb / 1024
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
      ("MTKNN_ADAPTER_COMPILER_OPTIONS", sweep.options), ("JETLINK_NEURON_DEADLINE", sweep.deadline.map(String.init)),
      ("JETLINK_NEURON_CACHE_FLUSH", sweep.cacheFlush.map(String.init)),
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
      let converted = try LiteRTPreparation.prepare(source: model, into: work, forNPU: true, fullyInt8: sweep.fullyInt8)
      let convertSeconds = Date().timeIntervalSince(convertStarted)
      let compileStarted = Date()
      let options = LiteRtCompileOptions.npu(compiled: compiled, gpu: nil)
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
      let compiledBytes = ", compiled \(Files.size(of: compiled)) bytes"
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
