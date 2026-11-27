import Foundation
import JetlinkServer
import Testing

@testable import JetlinkONNX
@testable import JetlinkORT

/// The bandwidth step's synthetic stack in fp16 end to end: onnxruntime
/// takes it (the FLOAT form's x would not type-check against fp16 weights
/// under the QNN provider), runs it, and the CPU profile skips the step.
@Suite struct BandwidthTests {
  @Test func theFp16StackRunsOnOnnxruntime() throws {
    let dir = FileManager.default.temporaryDirectory.appending(path: "jetlink-bw-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let url = dir.appending(path: "stack.onnx")
    try SyntheticModels.writeLinearStack(layers: 3, width: 64, rows: 8, fp16: true, to: url)
    let session = try OrtSession(model: url, provider: nil)
    #expect(session.inputs.map(\.name) == ["x"])
    #expect(session.inputs.first?.type == .float16)
    #expect(session.outputs.first?.type == .float16)
    let input = UnsafeMutableRawPointer.allocate(byteCount: session.inputs[0].byteCount, alignment: 64)
    let output = UnsafeMutableRawPointer.allocate(byteCount: session.outputs[0].byteCount, alignment: 64)
    defer {
      input.deallocate()
      output.deallocate()
    }
    input.initializeMemory(as: UInt8.self, repeating: 0, count: session.inputs[0].byteCount)
    let binding = try OrtBinding(session: session, inputs: [(session.inputs[0], input)], outputs: [(session.outputs[0], output)])
    try binding.run()
  }

  @Test func theCPUHasNoBandwidthStep() {
    let backend = OrtBackend(profile: .cpu, preparer: ONNXPreparer(), keepAlive: false, keepCPUWarm: false, chip: "test")
    #expect(backend.measureBandwidth(directory: FileManager.default.temporaryDirectory) { _ in }.isEmpty)
  }
}
