import Foundation
import JetlinkServer

/// Carrot's auxiliary camera models in onnxruntime (`AuxHub`): on a
/// Snapdragon on QNN's GPU backend, which leaves the Hexagon NPU to the
/// driving model, and on the CPU provider when QNN cannot take the model or
/// there is no QNN. Nodes QNN's GPU does not support run on onnxruntime's CPU
/// provider inside the same session.
public final class OrtAuxBackend: AuxBackend {
  /// Try QNN's GPU first.
  let qnn: Bool
  /// The CPU provider's intra-op threads.
  let threads: Int
  private let log = ServerLog(category: "ort-aux")

  public init(qnn: Bool, threads: Int = 2) {
    self.qnn = qnn
    self.threads = threads
  }

  public func load(model: URL) throws -> any AuxEngine {
    if qnn {
      do {
        let session = try OrtSession(model: model, provider: "QNN", options: ["backend_type": "gpu"])
        return try OrtAuxEngine(session: session, device: "QNN(gpu)")
      } catch {
        log.warning("QNN's GPU cannot take \(model.lastPathComponent), using the CPU: \(String(describing: error))")
      }
    }
    let session = try OrtSession(model: model, provider: nil, threads: threads)
    return try OrtAuxEngine(session: session, device: "CPU")
  }
}

/// One auxiliary model's session with its buffers bound once.
final class OrtAuxEngine: AuxEngine, @unchecked Sendable {
  let input: TensorSpec
  let outputs: [TensorSpec]
  let device: String
  private let session: OrtSession
  private var binding: OrtBinding?
  private let inputBuffer: UnsafeMutableRawPointer
  private let outputBuffers: [UnsafeMutableRawPointer]

  init(session: OrtSession, device: String) throws {
    guard session.inputs.count == 1, let input = session.inputs.first else {
      throw OrtError("an auxiliary model takes one input, this one \(session.inputs.count)")
    }
    guard !session.outputs.isEmpty else { throw OrtError("the auxiliary model has no outputs") }
    for output in session.outputs where output.type != .float && output.type != .float16 {
      throw OrtError("output \(output.name) is \(output.type), not float32 or float16")
    }
    self.session = session
    self.input = input
    outputs = session.outputs
    self.device = device
    inputBuffer = .allocate(byteCount: max(input.byteCount, 1), alignment: 64)
    outputBuffers = session.outputs.map { .allocate(byteCount: max($0.byteCount, 1), alignment: 64) }
    do {
      binding = try OrtBinding(
        session: session, inputs: [(input, inputBuffer)], outputs: Array(zip(session.outputs, outputBuffers)))
    } catch {
      inputBuffer.deallocate()
      outputBuffers.forEach { $0.deallocate() }
      throw error
    }
  }

  deinit {
    binding = nil
    inputBuffer.deallocate()
    outputBuffers.forEach { $0.deallocate() }
  }

  func run(input bytes: UnsafeRawBufferPointer, outputs names: [String]) throws -> [(TensorSpec, [Float])] {
    guard let binding else { throw OrtError("auxiliary session closed") }
    guard bytes.count == input.byteCount, let base = bytes.baseAddress else {
      throw OrtError("input is \(bytes.count) bytes, the model takes \(input.byteCount)")
    }
    inputBuffer.copyMemory(from: base, byteCount: bytes.count)
    try binding.run()
    let wanted = names.isEmpty ? outputs.map(\.name) : names
    return try wanted.map { name in
      guard let index = outputs.firstIndex(where: { $0.name == name }) else {
        throw OrtError("the model has no output \(name)")
      }
      let spec = outputs[index]
      var values = [Float](repeating: 0, count: spec.count)
      values.withUnsafeMutableBytes { out in
        switch spec.type {
        case .float16: auxF16ToF32(outputBuffers[index], out.baseAddress!, count: spec.count)
        default: out.copyMemory(from: UnsafeRawBufferPointer(start: outputBuffers[index], count: spec.byteCount))
        }
      }
      return (TensorSpec(name: spec.name, type: .float, shape: spec.shape), values)
    }
  }
}
