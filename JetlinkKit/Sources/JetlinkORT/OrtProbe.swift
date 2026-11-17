import Foundation
import JetlinkONNX
import JetlinkServer

/// Where an accelerator starts computing a model wrong: the model prepared
/// as the accelerator runs it with up to 160 inner tensors made outputs,
/// that graph on the accelerator, the CPU's graph with the same tensors, the
/// same inputs into both, and each tensor's difference. Each frame starts
/// both from the CPU's state, so a difference is that frame's, not one the
/// state carried in.
extension OrtBackend {
  public func probeAgainstReference(model: URL, directory: URL, progress: @escaping (String) -> Void) -> [String] {
    guard profile != .cpu, let unit = profile.sessions.first?.unit, profile.sessions.count == 1, unit == .htp || unit == .qnnGPU else {
      return []
    }
    return probe(
      model: model, directory: directory, layout: profile.layout, provider: "QNN", options: providerOptions(unit), label: profile.rawValue,
      progress: progress)
  }

  /// The probe with the accelerator's session spelled out: `provider` nil
  /// runs that side on the CPU too, which a test can.
  func probe(
    model: URL, directory: URL, layout: CoreMLPreparation.Layout, provider: String?, options: [String: String], label: String,
    progress: @escaping (String) -> Void
  ) -> [String] {
    let started = ProcessInfo.processInfo.systemUptime
    let fm = FileManager.default
    let accelDir = directory.appending(path: "probe-accelerator", directoryHint: .isDirectory)
    let cpuDir = directory.appending(path: "probe-cpu", directoryHint: .isDirectory)
    defer {
      try? fm.removeItem(at: accelDir)
      try? fm.removeItem(at: cpuDir)
    }
    do {
      try? fm.removeItem(at: accelDir)
      try? fm.removeItem(at: cpuDir)
      try fm.createDirectory(at: accelDir, withIntermediateDirectories: true)
      try fm.createDirectory(at: cpuDir, withIntermediateDirectories: true)
      progress("probe: preparing")
      let accelReport = try CoreMLPreparation.prepare(
        source: model, into: accelDir, layout: layout, cacheKey: { "probe\($0)" }, probes: .init())
      let names = accelReport.probes.map(\.name)
      let cpuLayout = OrtProfile.cpu.layout
      let cpuReport = try CoreMLPreparation.prepare(
        source: model, into: cpuDir, layout: cpuLayout, cacheKey: { "probecpu\($0)" }, probes: .init(names: names))
      let ops = Dictionary(accelReport.probes.map { ($0.name, $0.op) }, uniquingKeysWith: { a, _ in a })
      progress("probe: compiling \(names.count) tensors for the accelerator")
      let accel = try OrtSession(
        model: accelDir.appending(path: "model.onnx"), provider: provider, options: options, threads: provider == nil ? 4 : 1)
      let cpu = try OrtSession(model: cpuDir.appending(path: "model.onnx"), provider: nil, threads: 4)
      _ = cpuReport

      // Caller-owned buffers for every input and output of each session.
      final class Buffers {
        var data: [String: UnsafeMutableRawPointer] = [:]
        var specs: [String: TensorSpec] = [:]
        deinit { data.values.forEach { $0.deallocate() } }
        func alloc(_ s: TensorSpec) -> UnsafeMutableRawPointer {
          let p = UnsafeMutableRawPointer.allocate(byteCount: max(s.byteCount, 1), alignment: 64)
          p.initializeMemory(as: UInt8.self, repeating: 0, count: max(s.byteCount, 1))
          data[s.name] = p
          specs[s.name] = s
          return p
        }
      }
      let a = Buffers()
      let c = Buffers()
      let aBinding = try OrtBinding(
        session: accel, inputs: accel.inputs.map { ($0, a.alloc($0)) }, outputs: accel.outputs.map { ($0, a.alloc($0)) })
      let cBinding = try OrtBinding(
        session: cpu, inputs: cpu.inputs.map { ($0, c.alloc($0)) }, outputs: cpu.outputs.map { ($0, c.alloc($0)) })

      func floats(_ p: UnsafeMutableRawPointer, _ s: TensorSpec) -> [Float] {
        switch s.type {
        case .float: return Array(UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self), count: s.count))
        case .float16:
          let h = p.assumingMemoryBound(to: UInt16.self)
          return (0..<s.count).map { Float(Float16(bitPattern: h[$0])) }
        default: return []
        }
      }
      func set(_ b: Buffers, _ name: String, _ values: (Int) -> Float) {
        guard let s = b.specs[name], let p = b.data[name] else { return }
        switch s.type {
        case .float:
          let f = p.assumingMemoryBound(to: Float.self)
          for i in 0..<s.count { f[i] = values(i) }
        case .float16:
          let h = p.assumingMemoryBound(to: UInt16.self)
          for i in 0..<s.count { h[i] = Float16(values(i)).bitPattern }
        case .uint8:
          let u = p.assumingMemoryBound(to: UInt8.self)
          for i in 0..<s.count { u[i] = UInt8(clamping: Int(values(i))) }
        default: break
        }
      }

      var seed: UInt64 = 0x2545_F491_4F6C_DD1D
      func next() -> UInt8 {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        return UInt8(truncatingIfNeeded: seed >> 24)
      }
      // worst relative error of each compared tensor over the frames
      var worst: [String: (rel: Float, diff: Float, ref: Float)] = [:]
      let compared = accel.outputs.map(\.name).filter { c.specs[$0] != nil }
      let frames = 3
      for frame in 0..<frames {
        progress("probe: frame \(frame + 1)/\(frames)")
        for spec in accel.inputs {
          let name = spec.name
          if name.hasPrefix("state_") { continue }
          if name.contains("img") {
            let bytes = (0..<spec.count).map { _ in Float(next()) }
            set(a, name, { bytes[$0] })
            set(c, name, { bytes[$0] })
          } else if name == "traffic_convention" {
            set(a, name, { $0 == 0 ? 1 : 0 })
            set(c, name, { $0 == 0 ? 1 : 0 })
          } else {
            set(a, name, { _ in 0 })
            set(c, name, { _ in 0 })
          }
        }
        // the same state into both: the CPU's
        for spec in accel.inputs where spec.name.hasPrefix("state_") {
          guard let cs = c.specs[spec.name], let cp = c.data[spec.name] else { continue }
          let values = floats(cp, cs)
          if values.count == spec.count { set(a, spec.name, { values[$0] }) }
        }
        try aBinding.run()
        try cBinding.run()
        for name in compared {
          let x = floats(a.data[name]!, a.specs[name]!)
          let y = floats(c.data[name]!, c.specs[name]!)
          guard x.count == y.count, !x.isEmpty else { continue }
          var diff: Float = 0
          var ref: Float = 0
          for i in 0..<x.count {
            let d = abs(x[i] - y[i])
            if d.isNaN || d > diff { diff = d.isNaN ? .infinity : d }
            ref = max(ref, abs(y[i]))
          }
          let rel = diff / (ref + 1e-3)
          if rel > (worst[name]?.rel ?? -1) { worst[name] = (rel, diff, ref) }
        }
        // the CPU's next state becomes both engines' next state
        for spec in cpu.inputs where spec.name.hasPrefix("state_") {
          let out = "next_" + spec.name
          guard let os = c.specs[out], let op = c.data[out], os.byteCount == spec.byteCount, let ip = c.data[spec.name] else { continue }
          ip.copyMemory(from: op, byteCount: spec.byteCount)
        }
      }
      let order = names + compared.filter { ops[$0] == nil }
      let rows = order.compactMap { n in worst[n].map { (n, $0) } }
      func row(_ r: (String, (rel: Float, diff: Float, ref: Float))) -> String {
        String(format: "%@ (%@) %.1f%% |diff| %.3g of |ref| %.3g", r.0, ops[r.0] ?? "output", r.1.rel * 100, r.1.diff, r.1.ref)
      }
      var lines = ["probe: \(rows.count) tensors on \(label) against the CPU, \(frames) frames, \(Int(ProcessInfo.processInfo.systemUptime - started)) s"]
      let firstBad = rows.filter { $0.1.rel > 0.05 }.prefix(5)
      lines.append(
        firstBad.isEmpty
          ? "probe: no tensor over 5% from the CPU"
          : "probe: first over 5% in graph order: " + firstBad.map(row).joined(separator: "; "))
      let top = rows.sorted { $0.1.rel > $1.1.rel }.prefix(8)
      lines.append("probe: worst: " + top.map(row).joined(separator: "; "))
      return lines
    } catch {
      return ["probe failed: \(error)"]
    }
  }
}
