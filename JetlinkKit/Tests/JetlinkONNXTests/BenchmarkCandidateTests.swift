import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkONNX

@Suite struct BenchmarkCandidateTests {
  private let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))

  func model(_ op: String = "MatMul") -> OnnxGraphBuilder {
    var b = OnnxGraphBuilder(opset: 20)
    let conv = op == "Conv"
    b.input("new_img", DataType.float16, conv ? [1, 64, 4, 4] : [1, 256])
    b.fp16(
      "weight", conv ? [64, 64, 4, 4] : [256, 256], (0..<65536).map { Float($0 % 31 - 15) * 0.002 })
    if op == "MatMul" {
      b.node(op, ["new_img", "weight"], ["y"])
    } else {
      b.fp16("bias", [conv ? 64 : 256], Array(repeating: 0.01, count: conv ? 64 : 256))
      b.node(op, ["new_img", "weight", "bias"], ["y"], op == "Gemm" ? [.int("transB", 1)] : [])
    }
    b.output("y", DataType.float16, conv ? [1, 64, 1, 1] : [1, 256])
    return b
  }

  @Test(arguments: ["MatMul", "Gemm", "Conv"])
  func qdqHasIntegerActivationsAndWeightsAndRunsFromONNX(_ op: String) throws {
    let dir = try TemporaryDirectory()
    let source = dir.url.appending(path: "source.onnx")
    let output = dir.url.appending(path: "quantized.onnx")
    try Data(model(op).bytes).write(to: source)
    let probe = try W8A16Preparation.writeProbe(
      source: source, to: dir.url.appending(path: "probe.onnx"), visionOnly: true)
    #expect(probe.layers == 1 && Set(probe.names) == ["new_img", "y"])
    #expect(
      try W8A16Preparation.quantize(
        source: source, to: output,
        ranges: ["new_img": .init(min: -1, max: 1), "y": .init(min: -100, max: 100)],
        visionOnly: false) == 1)
    let data = try Data(contentsOf: output)
    try data.withUnsafeBytes { bytes in
      let (m, g) = try Decode.preparable(Source(bytes: bytes))
      #expect(m.opsets.first { $0.domain.isEmpty }?.version == 20)
      #expect(m.opsets.first { $0.domain == "com.microsoft" }?.version == 1)
      #expect(g.outputs.map(\.key) == ["y"] && g.inputs.map(\.key) == ["new_img"])
      #expect(g.nodes.filter { $0.op == "QuantizeLinear" }.count == 2)
      #expect(g.initializers.contains { $0.elementType == DataType.uint8 })
      #expect(g.initializers.filter { $0.elementType == DataType.uint16 }.count == 2)
      #expect(!g.initializers.contains { $0.key == "weight" })
      if op != "MatMul" { #expect(g.initializers.contains { $0.elementType == DataType.int32 }) }
    }
    // Optional export for numerical verification with the official Python ORT.
    if let export = ProcessInfo.processInfo.environment["JETLINK_QDQ_OUT"] {
      let folder = URL(fileURLWithPath: export, isDirectory: true)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      try Data(contentsOf: source).write(to: folder.appending(path: "\(op)-source.onnx"))
      try data.write(to: folder.appending(path: "\(op)-w8a16.onnx"))
    }
  }

  @Test func missingOrInvalidCalibrationIsAnError() throws {
    let dir = try TemporaryDirectory()
    let source = dir.url.appending(path: "source.onnx")
    try Data(model().bytes).write(to: source)
    #expect(throws: (any Error).self) {
      try W8A16Preparation.quantize(
        source: source, to: dir.url.appending(path: "bad.onnx"), ranges: [:], visionOnly: false)
    }
    #expect(throws: (any Error).self) {
      try W8A16Preparation.parameters(.init(min: .nan, max: 1), levels: 65535)
    }
    let p = try W8A16Preparation.parameters(.init(min: -2, max: 3), levels: 65535)
    #expect(p.scale > 0 && p.zero > 0 && p.zero < 65535)
  }

  @Test func preparedFlatGemmIsIncludedInCalibration() throws {
    var b = OnnxGraphBuilder(opset: 20)
    b.input("new_img", DataType.float16, [1, 2, 256])
    b.valueInfo("mm", DataType.float16, [1, 2, 256])
    b.fp16("w", [256, 256], Array(repeating: 0.001, count: 65536))
    b.fp16("b", [256], Array(repeating: 0.01, count: 256))
    b.node("MatMul", ["new_img", "w"], ["mm"])
    b.node("Add", ["mm", "b"], ["y"])
    b.output("y", DataType.float16, [1, 2, 256])
    let dir = try TemporaryDirectory()
    let file = dir.url.appending(path: "source.onnx")
    try Data(b.bytes).write(to: file)
    let prepared = dir.url.appending(path: "prepared")
    _ = try CoreMLPreparation.prepare(source: file, into: prepared, layout: .whole, cacheKey: { $0 })
    let probe = try W8A16Preparation.writeProbe(source: prepared.appending(path: "model.onnx"), to: dir.url.appending(path: "probe.onnx"), visionOnly: true)
    #expect(probe.layers == 1)
    #expect(probe.names.contains("mm__flat") && probe.names.contains("mm__gemm"))
  }

  @Test func visionCandidateLeavesPolicyWeightsAlone() throws {
    var b = model()
    b.input("state_feat_q", DataType.float16, [1, 256])
    b.fp16("policy_weight", [256, 256], Array(repeating: 0.01, count: 65536))
    b.node("MatMul", ["state_feat_q", "policy_weight"], ["policy"])
    b.output("policy", DataType.float16, [1, 256])
    #expect(try W8A16Preparation.targets(b.graph, visionOnly: true).count == 1)
    #expect(try W8A16Preparation.targets(b.graph, visionOnly: false).count == 2)
  }

  @Test func realModelStructureWhenRequested() throws {
    guard let path = ProcessInfo.processInfo.environment["JETLINK_QDQ_MODEL"],
      let output = ProcessInfo.processInfo.environment["JETLINK_QDQ_OUT"] else { return }
    let folder = URL(fileURLWithPath: output, isDirectory: true)
    let prepared = folder.appending(path: "real-prepared")
    _ = try CoreMLPreparation.prepare(source: URL(fileURLWithPath: path), into: prepared, layout: .hexagonWhole, cacheKey: { $0 })
    let model = prepared.appending(path: "model.onnx")
    let probe = try W8A16Preparation.writeProbe(source: model, to: folder.appending(path: "real-probe.onnx"), visionOnly: false)
    // Structural smoke check only. These ranges are deliberately not a
    // calibration result, and this file is never shipped or used to drive.
    let ranges = Dictionary(uniqueKeysWithValues: probe.names.map { ($0, W8A16Preparation.Range(min: -100, max: 100)) })
    let count = try W8A16Preparation.quantize(source: model, to: folder.appending(path: "real-structure-only.onnx"), ranges: ranges, visionOnly: false)
    #expect(count == probe.layers && count > 10)
    print("real model structural check: \(count) layers, \(probe.names.count) calibration tensors")
  }

  @Test func inverseTransposesBecomeAnIdentityWithoutChangingTheOutputName() {
    var b = OnnxGraphBuilder()
    b.input("x", DataType.float, [2, 3])
    b.valueInfo("mid", DataType.float, [3, 2])
    b.node("Transpose", ["x"], ["mid"], [.ints("perm", [1, 0])])
    b.node("Transpose", ["mid"], ["y"], [.ints("perm", [1, 0])])
    b.output("y", DataType.float, [2, 3])
    var g = b.graph
    #expect(LayoutPreparation.simplify(&g) == 2)
    #expect(
      g.nodes.count == 1 && g.nodes[0].op == "Identity" && g.nodes[0].inputs == ["x"]
        && g.nodes[0].outputs == ["y"])
  }

  @Test func aSharedOrExposedIntermediateTransposeIsPreserved() {
    var b = OnnxGraphBuilder()
    b.input("x", DataType.float, [2, 3])
    b.valueInfo("mid", DataType.float, [3, 2])
    b.node("Transpose", ["x"], ["mid"], [.ints("perm", [1, 0])])
    b.node("Transpose", ["mid"], ["y"], [.ints("perm", [1, 0])])
    b.output("mid", DataType.float, [3, 2])
    b.output("y", DataType.float, [2, 3])
    var g = b.graph
    #expect(LayoutPreparation.simplify(&g) == 0)
    #expect(g.nodes.count == 2)
  }
}
