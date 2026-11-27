import Foundation
import JetlinkONNX
import JetlinkServer

/// How fast the Snapdragon's NPU reads weights from memory: linear stacks
/// with almost no arithmetic (8 rows through 4096-wide fp16 layers) and
/// 128, 256 and 512 MB of weights, each compiled for the NPU and timed; the
/// line through the three gives the bytes a millisecond more weight costs
/// (the read rate) and what a run costs with none (the fixed part).
///
/// carrot.60's sweep on a Galaxy Tab S9 left BMRLNAP (1.76 GB) at 55-58 ms
/// whatever the graph did, and N16 (0.82 GB) runs 32 ms there: if the NPU
/// reads ~41 GB/s of the memory's 68, the weights alone take ~42 ms.
extension OrtBackend {
  static let bandwidthWidth = 4096
  static let bandwidthRows = 8
  /// 32 MB a layer: 128, 256 and 512 MB.
  static let bandwidthLayers = [4, 8, 16]

  public func measureBandwidth(directory: URL, progress: @escaping (String) -> Void) -> [String] {
    guard profile != .cpu, profile.sessions.contains(where: { $0.unit == .htp }) else { return [] }
    let fm = FileManager.default
    let dir = directory.appending(path: "bandwidth", directoryHint: .isDirectory)
    defer { try? fm.removeItem(at: dir) }
    var points: [(bytes: Double, ms: Double)] = []
    var lines: [String] = []
    do {
      try? fm.removeItem(at: dir)
      try fm.createDirectory(at: dir, withIntermediateDirectories: true)
      for layers in OrtBackend.bandwidthLayers {
        let bytes = Double(layers * OrtBackend.bandwidthWidth * OrtBackend.bandwidthWidth * 2)
        let label = String(format: "%.0f MB", bytes / 1_048_576)
        let url = dir.appending(path: "stack\(layers).onnx")
        progress("bandwidth: writing \(label)")
        try SyntheticModels.writeLinearStack(
          layers: layers, width: OrtBackend.bandwidthWidth, rows: OrtBackend.bandwidthRows, fp16: true, to: url)
        progress("bandwidth: compiling \(label) for the NPU")
        var options = providerOptions(.htp)
        options["htp_performance_mode"] = "burst"
        var ms: [Double] = []
        do {
          let session = try OrtSession(model: url, provider: "QNN", options: options, threads: 1)
          final class Buffers {
            var all: [UnsafeMutableRawPointer] = []
            deinit { all.forEach { $0.deallocate() } }
            func alloc(_ s: TensorSpec) -> UnsafeMutableRawPointer {
              let p = UnsafeMutableRawPointer.allocate(byteCount: max(s.byteCount, 1), alignment: 64)
              p.initializeMemory(as: UInt8.self, repeating: 0, count: max(s.byteCount, 1))
              all.append(p)
              return p
            }
          }
          let buffers = Buffers()
          let binding = try OrtBinding(
            session: session, inputs: session.inputs.map { ($0, buffers.alloc($0)) },
            outputs: session.outputs.map { ($0, buffers.alloc($0)) })
          for run in 0..<35 {
            let t = ProcessInfo.processInfo.systemUptime
            try binding.run()
            if run >= 5 { ms.append((ProcessInfo.processInfo.systemUptime - t) * 1000) }
          }
        }
        try? fm.removeItem(at: url)
        ms.sort()
        let p50 = ms[ms.count / 2]
        points.append((bytes, p50))
        lines.append(
          String(
            format: "bandwidth: %@ of weights p50 %.2f p90 %.2f ms, %.1f GB/s whole run", label, p50, ms[ms.count * 9 / 10],
            bytes / 1e9 / (p50 / 1000)))
      }
    } catch {
      lines.append("bandwidth failed: \(String(describing: error).prefix(240))")
      return lines
    }
    // least squares through the points: ms = fixed + bytes / rate
    let n = Double(points.count)
    let mx = points.reduce(0) { $0 + $1.bytes } / n
    let my = points.reduce(0) { $0 + $1.ms } / n
    let sxx = points.reduce(0) { $0 + ($1.bytes - mx) * ($1.bytes - mx) }
    let sxy = points.reduce(0) { $0 + ($1.bytes - mx) * ($1.ms - my) }
    if sxx > 0, sxy > 0 {
      let slope = sxy / sxx  // ms a byte
      let fixed = my - slope * mx
      lines.append(
        String(
          format: "bandwidth: the NPU reads %.1f GB/s (each GB %.1f ms), %.2f ms a run fixed; a 1.76 GB model's weights alone %.1f ms",
          1 / slope / 1e6, slope * 1e9, fixed, slope * 1.76e9))
    }
    return lines
  }
}
