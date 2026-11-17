import Foundation
import JetlinkONNX
import JetlinkServer

extension OrtBackend {
  /// Calibration is intentionally local to the benchmark. These changing
  /// synthetic images exercise the timing candidate; they are not a road
  /// calibration set and do not establish driving accuracy.
  func prepareW8A16(in directory: URL, visionOnly: Bool, report: @escaping ProgressFn) throws {
    let source = directory.appending(path: "model.onnx")
    let probeFile = directory.appending(path: "calibration.onnx")
    defer { try? FileManager.default.removeItem(at: probeFile) }
    report("patch", 0, "W8A16: preparing activation calibration (synthetic, benchmark only)")
    let probe = try W8A16Preparation.writeProbe(
      source: source, to: probeFile, visionOnly: visionOnly)
    let ranges = try calibrateW8A16(model: probeFile, names: probe.names, report: report)
    report("patch", 0.8, "W8A16: quantizing \(probe.layers) constant layers")
    let output = directory.appending(path: "quantized.onnx")
    let count = try W8A16Preparation.quantize(
      source: source, to: output, ranges: ranges, visionOnly: visionOnly)
    try FileManager.default.removeItem(at: source)
    try FileManager.default.moveItem(at: output, to: source)
    ServerLog(category: "ort").info(
      "W8A16 timing candidate: \(count) layers, \(ranges.count) measured ranges, 20 synthetic calibration frames; not road accuracy"
    )
  }

  func calibrateW8A16(model: URL, names: [String], report: @escaping ProgressFn) throws -> [String:
    W8A16Preparation.Range]
  {
    let session = try OrtSession(model: model, provider: "QNN", options: providerOptions(.htp), config: ["session.disable_cpu_ep_fallback": "1"], threads: 1)
    final class Buffers {
      var pointers: [String: UnsafeMutableRawPointer] = [:]
      var specs: [String: TensorSpec] = [:]
      deinit { pointers.values.forEach { $0.deallocate() } }
      func alloc(_ s: TensorSpec) -> UnsafeMutableRawPointer {
        let p = UnsafeMutableRawPointer.allocate(byteCount: max(1, s.byteCount), alignment: 64)
        p.initializeMemory(as: UInt8.self, repeating: 0, count: max(1, s.byteCount))
        pointers[s.name] = p
        specs[s.name] = s
        return p
      }
    }
    let buffers = Buffers()
    let binding = try OrtBinding(
      session: session, inputs: session.inputs.map { ($0, buffers.alloc($0)) },
      outputs: session.outputs.map { ($0, buffers.alloc($0)) })
    var ranges: [String: W8A16Preparation.Range] = [:]
    var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    for frame in 0..<20 {
      report(
        "patch", Double(frame) / 25,
        "W8A16: activation calibration \(frame + 1)/20 (NPU, synthetic)")
      for s in session.inputs where !s.name.hasPrefix("state_") {
        guard let p = buffers.pointers[s.name] else { continue }
        for i in 0..<s.count {
          let value: Float
          if s.name.contains("img") {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            value = Float(UInt8(truncatingIfNeeded: seed >> 24))
          } else {
            value = s.name == "traffic_convention" && i == 0 ? 1 : 0
          }
          switch s.type {
          case .float: p.assumingMemoryBound(to: Float.self)[i] = value
          case .float16: p.assumingMemoryBound(to: UInt16.self)[i] = Float16(value).bitPattern
          case .uint8: p.assumingMemoryBound(to: UInt8.self)[i] = UInt8(clamping: Int(value))
          default: break
          }
        }
      }
      try binding.run()
      for name in names {
        guard let s = buffers.specs[name], let p = buffers.pointers[name] else {
          throw HostError.failed("calibration tensor missing: \(name)")
        }
        var r = ranges[name] ?? .init(min: 0, max: 0)
        for i in 0..<s.count {
          let value: Float
          switch s.type {
          case .float: value = p.assumingMemoryBound(to: Float.self)[i]
          case .float16:
            value = Float(Float16(bitPattern: p.assumingMemoryBound(to: UInt16.self)[i]))
          default: throw HostError.failed("unsupported calibration type for \(name)")
          }
          guard value.isFinite else {
            throw HostError.failed("non-finite calibration tensor: \(name)")
          }
          r.min = min(r.min, value)
          r.max = max(r.max, value)
        }
        ranges[name] = r
      }
      for s in session.inputs where s.name.hasPrefix("state_") {
        if let output = buffers.specs["next_" + s.name], output.type == s.type,
          output.byteCount == s.byteCount,
          let input = buffers.pointers[s.name], let next = buffers.pointers[output.name]
        {
          input.copyMemory(from: next, byteCount: s.byteCount)
        }
      }
    }
    return ranges
  }
}
