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
  private let planEnd: Int
  private let lanes: Range<Int>?
  private let lead: Int?
  private var dy: [Float] = [], dx: [Float] = [], dLanes: [Float] = [], dLead: [Float] = []
  private var worst = (frame: 0, dy: Float(0))

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
    var line = "accelerator against the CPU over \(dy.count) changing frames, |difference|: plan end y \(text(dy)) m"
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
  func compareWithReference(_ l: Loaded, progress: (String) -> Void) -> [String] {
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
      let packed = [Float](repeating: 0, count: spec.packedCount)
      var a = [Float](repeating: 0, count: spec.outputCount)
      var c = [Float](repeating: 0, count: spec.outputCount)
      var cpuMs: [Double] = []

      lock.lock()
      l.staging.reset()
      lock.unlock()
      cpuStaging.reset()
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
      for frame in 0..<ReferenceComparison.frames {
        // A moving picture: the base bytes shifted further each frame, part
        // of them fresh, so remembered frames differ from one another.
        var warped = base
        let shift = (frame * 977) % max(1, warped.count)
        warped = Array(warped[shift...] + warped[..<shift])
        for i in stride(from: frame % 7, to: warped.count, by: 7) { warped[i] = next() }

        lock.lock()
        guard loaded === l else {
          lock.unlock()
          return ["accelerator against the CPU: the engine was unloaded"]
        }
        try warped.withUnsafeBytes { w in
          try packed.withUnsafeBytes { p in try l.staging.stage(warped: w.baseAddress!, packed: p.baseAddress!) }
        }
        try l.engine.run()
        let aFinite = read(l.staging, io.type, into: &a)
        if aFinite { l.staging.keep(outputs: l.staging.layout.output!, type: io.type) }
        lock.unlock()

        let t = ProcessInfo.processInfo.systemUptime
        try warped.withUnsafeBytes { w in
          try packed.withUnsafeBytes { p in try cpuStaging.stage(warped: w.baseAddress!, packed: p.baseAddress!) }
        }
        try cpu.run()
        cpuMs.append((ProcessInfo.processInfo.systemUptime - t) * 1000)
        let cFinite = read(cpuStaging, cpuIO.type, into: &c)
        if cFinite { cpuStaging.keep(outputs: cpuStaging.layout.output!, type: cpuIO.type) }

        if aFinite && cFinite { comparison.record(frame: frame, accelerator: a, cpu: c) }
        progress("CPU reference: frame \(frame + 1)/\(ReferenceComparison.frames)")
      }
      lock.lock()
      l.staging.reset()
      lock.unlock()
      let cpuMean = cpuMs.isEmpty ? 0 : cpuMs.reduce(0, +) / Double(cpuMs.count)
      var lines: [String] = []
      if let s = comparison.summary { lines.append(s) }
      lines.append(
        String(
          format: "CPU reference %@: %.0f ms a frame, %.0f s in all", reference.deviceTag(), cpuMean,
          ProcessInfo.processInfo.systemUptime - started))
      return lines
    } catch {
      lock.lock()
      if loaded === l { l.staging.reset() }
      lock.unlock()
      return ["accelerator against the CPU failed: \(error)"]
    }
  }
}
