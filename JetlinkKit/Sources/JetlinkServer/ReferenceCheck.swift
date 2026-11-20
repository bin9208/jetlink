import Foundation

/// After a benchmark on an accelerator: the same model on the CPU, fed the
/// same frames, to show whether the accelerator computes the model right.
///
/// The benchmark's own steadiness line cannot: its camera bytes never change,
/// so once the model's remembered frames are all alike, an attention that
/// mixes them wrongly still gives the right row. On a Snapdragon 8 Elite
/// carrot's N16 settled there while its plan jumped metres in the car. Here
/// every frame differs, both engines keep their own state, and the log says
/// how far the accelerator's plan, lane lines and lead are from the CPU's.
struct ReferenceComparison {
  /// Frames compared, each different from the last.
  static let frames = 40
  /// What the frames were, for the log: "changing frames" or "road frames".
  var label = "changing frames"
  private let planEnd: Int
  private let lanes: Range<Int>?
  private let lead: Int?
  private var dy: [Float] = [], dx: [Float] = [], dLanes: [Float] = [], dLead: [Float] = []
  private var worst = (frame: 0, dy: Float(0))

  /// Whether the plan's end is far enough from the CPU's to look inside.
  var wantsProbe: Bool {
    !dy.isEmpty && dy.reduce(0, +) / Float(dy.count) > 0.3
  }

  /// Nil for a model whose plan is not a single 33x15 hypothesis.
  init?(spec: ModelSpec) {
    guard let plan = spec.outputSlices.first(where: { $0.name == "plan" }), let start = plan.start, let stop = plan.stop,
      stop - start == 2 * 33 * 15
    else { return nil }
    planEnd = start + 32 * 15
    if let l = spec.outputSlices.first(where: { $0.name == "lane_lines" }), let a = l.start, let b = l.stop, b > a {
      lanes = a..<b
    } else {
      lanes = nil
    }
    lead = spec.outputSlices.first(where: { $0.name == "lead" })?.start
  }

  mutating func record(frame: Int, accelerator a: [Float], cpu c: [Float]) {
    let y = abs(a[planEnd + 1] - c[planEnd + 1])
    dy.append(y)
    dx.append(abs(a[planEnd] - c[planEnd]))
    if let lanes {
      var m: Float = 0
      for i in lanes { m = max(m, abs(a[i] - c[i])) }
      dLanes.append(m)
    }
    if let lead { dLead.append(abs(a[lead] - c[lead])) }
    if y > worst.dy { worst = (frame, y) }
  }

  var summary: String? {
    guard !dy.isEmpty else { return nil }
    func text(_ v: [Float]) -> String {
      String(format: "mean %.3f max %.3f", Double(v.reduce(0, +)) / Double(v.count), Double(v.max() ?? 0))
    }
    var line = "accelerator against the CPU over \(dy.count) \(label), |difference|: plan end y \(text(dy)) m"
    line += ", plan end x \(text(dx)) m"
    if !dLanes.isEmpty { line += ", lane lines \(text(dLanes))" }
    if !dLead.isEmpty { line += ", lead x \(text(dLead)) m" }
    line += String(format: "; worst plan end y at frame %d", worst.frame)
    return line
  }
}

extension EngineHost {
  /// Builds (once) and loads the model for the backend's CPU reference, runs
  /// both engines on the same changing frames, and returns lines for the log.
  /// Nothing when the backend has no reference or the model no plan to read.
  func compareWithReference(_ l: Loaded, progress: @escaping (String) -> Void) -> [String] {
    guard var comparison = ReferenceComparison(spec: l.spec), let reference = backend.referenceBackend() else { return [] }
    let spec = l.spec
    let started = ProcessInfo.processInfo.systemUptime
    do {
      let model = try cache.modelPath(l.sha256)
      let directory = cache.layout.root.appending(path: "reference", directoryHint: .isDirectory)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      let artifact = directory.appending(path: "\(String(l.sha256.prefix(16))).\(reference.tag())\(reference.suffix)")
      progress("CPU reference: preparing")
      var engine: (any Engine)?
      if FileManager.default.fileExists(atPath: artifact.path) {
        engine = try? reference.load(artifact: artifact) { _, _, _ in }
      }
      if engine == nil {
        try? FileManager.default.removeItem(at: artifact)
        try reference.build(model: model, artifact: artifact, report: { _, _, _ in }, metaExtra: ["spec": spec.dictionary()])
        engine = try reference.load(artifact: artifact) { _, _, _ in }
      }
      guard let cpu = engine else { return [] }
      defer { cpu.close() }
      let cpuStaging = try Staging.forModel(spec, engine: cpu)
      guard let cpuIO = cpu.outputs[ModelConstants.drivingOutput], let io = l.engine.outputs[ModelConstants.drivingOutput] else {
        return ["accelerator against the CPU: no driving output"]
      }

      // Different camera bytes every frame, the same for both engines.
      var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
      func next() -> UInt8 {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        return UInt8(truncatingIfNeeded: seed >> 24)
      }
      let base = (0..<spec.warpedBytes).map { _ in next() }
      let zeros = [UInt8](repeating: 0, count: spec.packedBytes)
      var synthetic: [CameraFrame] = []
      for frame in 0..<ReferenceComparison.frames {
        // A moving picture: the base bytes shifted further each frame, part
        // of them fresh, so remembered frames differ from one another.
        var warped = base
        let shift = (frame * 977) % max(1, warped.count)
        warped = Array(warped[shift...] + warped[..<shift])
        for i in stride(from: frame % 7, to: warped.count, by: 7) { warped[i] = next() }
        synthetic.append(CameraFrame(warped: warped, packed: zeros))
      }
      var cpuMs: [Double] = []
      let sweep = backend.experimentBackends()

      func read(_ staging: any FrameStaging, _ type: ElementType, into out: inout [Float]) -> Bool {
        let p = staging.layout.output!
        out.withUnsafeMutableBytes { o in
          if type == .float16 {
            Convert.f16ToF32(p, o.baseAddress!, count: spec.outputCount)
          } else {
            o.baseAddress!.copyMemory(from: p, byteCount: spec.outputCount * 4)
          }
        }
        return out.withUnsafeBytes { Convert.allFinite($0.baseAddress!, count: spec.outputCount) }
      }
      /// Both engines over `frames` from a fresh state each; the CPU's
      /// outputs are kept for the experiment sweep. Nil once the engine went.
      func compare(_ frames: [CameraFrame], label: String, into comparison: inout ReferenceComparison) throws -> [[Float]?]? {
        var a = [Float](repeating: 0, count: spec.outputCount)
        var c = [Float](repeating: 0, count: spec.outputCount)
        var cpuOutputs: [[Float]?] = []
        comparison.label = label
        lock.lock()
        l.staging.reset()
        lock.unlock()
        cpuStaging.reset()
        defer {
          lock.lock()
          if loaded === l { l.staging.reset() }
          lock.unlock()
        }
        for (frame, f) in frames.enumerated() {
          lock.lock()
          guard loaded === l else {
            lock.unlock()
            return nil
          }
          try f.stage(l.staging)
          try l.engine.run()
          let aFinite = read(l.staging, io.type, into: &a)
          if aFinite { l.staging.keep(outputs: l.staging.layout.output!, type: io.type) }
          lock.unlock()

          let t = ProcessInfo.processInfo.systemUptime
          try f.stage(cpuStaging)
          try cpu.run()
          cpuMs.append((ProcessInfo.processInfo.systemUptime - t) * 1000)
          let cFinite = read(cpuStaging, cpuIO.type, into: &c)
          if cFinite { cpuStaging.keep(outputs: cpuStaging.layout.output!, type: cpuIO.type) }

          if aFinite && cFinite { comparison.record(frame: frame, accelerator: a, cpu: c) }
          if !sweep.isEmpty { cpuOutputs.append(cFinite ? c : nil) }
          progress("CPU reference: \(label) \(frame + 1)/\(frames.count)")
        }
        return cpuOutputs
      }
      guard var sweepOutputs = try compare(synthetic, label: "changing frames", into: &comparison) else {
        return ["accelerator against the CPU: the engine was unloaded"]
      }
      // and on frames of the comma's own drives, which reach what a road
      // does and random bytes do not, when this phone has kept some
      let road = RoadFrames.loadAny(in: roadFramesDirectory, sha256: l.sha256, spec: spec)
      let roadLabel = road?.imported == true ? "imported road frames" : "road frames"
      var onRoad: ReferenceComparison? = nil
      var sweepFrames = synthetic
      if let road, var roadComparison = ReferenceComparison(spec: spec) {
        if let outputs = try compare(road.frames, label: roadLabel, into: &roadComparison) {
          onRoad = roadComparison
          sweepOutputs = outputs
          sweepFrames = road.frames
        }
      }
      let cpuMean = cpuMs.isEmpty ? 0 : cpuMs.reduce(0, +) / Double(cpuMs.count)
      var lines: [String] = []
      for c in [comparison] + (onRoad.map { [$0] } ?? []) {
        if let s = c.summary {
          lines.append(backend.experimentName.map { "\(s) (experiments: \($0))" } ?? s)
        }
      }
      if road == nil {
        lines.append("no road frames on this phone yet for this model: drive with it once and benchmark again to compare on a road too")
      }
      lines.append(
        String(
          format: "CPU reference %@: %.0f ms a frame, %.0f s in all", reference.deviceTag(), cpuMean,
          ProcessInfo.processInfo.systemUptime - started))
      if comparison.wantsProbe {
        for line in lines { log.info("benchmark: \(line)") }
        lines = backend.probeAgainstReference(model: model, directory: directory, progress: progress)
      }
      // where the accelerator's time goes, whatever the comparison said
      for line in lines { log.info("benchmark: \(line)") }
      lines = backend.profileOnAccelerator(model: model, directory: directory, progress: progress)
      // and the same engine under each way of running it, side by side
      if let artifact = try? cache.entry(l.sha256).path {
        for line in lines { log.info("benchmark: \(line)") }
        lines = backend.compareVariants(artifact: artifact, progress: progress)
      }
      // and each other combination of the test build's experiments, built,
      // fed the same frames and timed the same way, one after another
      for (i, x) in sweep.enumerated() {
        for line in lines { log.info("benchmark: \(line)") }
        lines = [
          experiment(
            x.name, x.backend, model: model, spec: spec, sha256: l.sha256, frames: sweepFrames, cpu: sweepOutputs,
            label: onRoad == nil ? "changing frames" : roadLabel,
            progress: { progress("experiments: \(i + 1)/\(sweep.count) \(x.name): \($0)") })
        ]
      }
      return lines
    } catch {
      lock.lock()
      if loaded === l { l.staging.reset() }
      lock.unlock()
      return ["accelerator against the CPU failed: \(error)"]
    }
  }
}

extension EngineHost {
  /// One experiment set: built into a scratch directory, compared with the
  /// CPU's outputs for the same frames, timed, and removed again.
  func experiment(
    _ name: String, _ backend: any EngineBackend, model: URL, spec: ModelSpec, sha256: String, frames: [CameraFrame], cpu: [[Float]?],
    label: String = "changing frames", progress: @escaping (String) -> Void
  ) -> String {
    guard var comparison = ReferenceComparison(spec: spec) else { return "experiment \(name): no plan to compare" }
    comparison.label = label
    let started = ProcessInfo.processInfo.systemUptime
    let directory = cache.layout.root.appending(path: "experiments", directoryHint: .isDirectory)
    let artifact = directory.appending(path: "\(String(sha256.prefix(16))).\(backend.tag())\(backend.suffix)")
    defer { try? FileManager.default.removeItem(at: directory) }
    do {
      try? FileManager.default.removeItem(at: directory)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      progress("building (a few minutes)")
      try backend.build(
        model: model, artifact: artifact, report: { _, _, _ in }, metaExtra: ["spec": spec.dictionary()])
      let buildSeconds = ProcessInfo.processInfo.systemUptime - started
      progress("loading")
      let engine = try backend.load(artifact: artifact) { _, _, _ in }
      defer { engine.close() }
      let staging = try Staging.forModel(spec, engine: engine)
      guard let io = engine.outputs[ModelConstants.drivingOutput] else { return "experiment \(name): no driving output" }
      var a = [Float](repeating: 0, count: spec.outputCount)
      var notFinite = 0
      staging.reset()
      for (frame, f) in frames.enumerated() {
        try f.stage(staging)
        try engine.run()
        let p = staging.layout.output!
        a.withUnsafeMutableBytes { o in
          if io.type == .float16 {
            Convert.f16ToF32(p, o.baseAddress!, count: spec.outputCount)
          } else {
            o.baseAddress!.copyMemory(from: p, byteCount: spec.outputCount * 4)
          }
        }
        let finite = a.withUnsafeBytes { Convert.allFinite($0.baseAddress!, count: spec.outputCount) }
        if finite {
          staging.keep(outputs: p, type: io.type)
          if frame < cpu.count, let c = cpu[frame] { comparison.record(frame: frame, accelerator: a, cpu: c) }
        } else {
          notFinite += 1
        }
        progress("frame \(frame + 1)/\(frames.count)")
      }
      var ms: [Double] = []
      for run in 0..<68 {
        let t = ProcessInfo.processInfo.systemUptime
        try engine.run()
        if run >= 8 { ms.append((ProcessInfo.processInfo.systemUptime - t) * 1000) }
      }
      ms.sort()
      var line = String(format: "experiment %@: run p50 %.2f p90 %.2f ms, built in %.0f s", name, ms[ms.count / 2], ms[ms.count * 9 / 10], buildSeconds)
      if notFinite > 0 { line += ", \(notFinite) frame(s) not finite" }
      if let s = comparison.summary { line += "; \(s)" }
      return line
    } catch {
      return "experiment \(name) failed: \(String(describing: error).prefix(240))"
    }
  }
}

/// A frame as the comma sends it: the warped camera bytes and the packed
/// float32 scalars after them.
typealias CameraFrame = RoadFrames.Frame

extension RoadFrames.Frame {
  func stage(_ staging: any FrameStaging) throws {
    try warped.withUnsafeBytes { w in
      try packed.withUnsafeBytes { p in try staging.stage(warped: w.baseAddress!, packed: p.baseAddress!) }
    }
  }
}
