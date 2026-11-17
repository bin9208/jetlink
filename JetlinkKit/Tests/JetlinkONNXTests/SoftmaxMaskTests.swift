import Foundation
import Testing

@testable import JetlinkONNX

@Suite struct SoftmaxMaskTests {
  private let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))

  /// N16's temporal attention: Softmax(Where(Not(mask), -inf, scores)), twice
  /// over the same causal mask.
  private func n16Like() -> OnnxGraphBuilder {
    var b = OnnxGraphBuilder()
    b.input("scores", DataType.float16, [1, 1, 2, 2])
    b.input("scores2", DataType.float16, [1, 1, 2, 2])
    b.bool("mask", [1, 1, 2, 2], [true, false, true, true])
    b.initializer("ninf", DataType.float16, [], [0x00, 0xFC])
    b.node("Not", ["mask"], ["keep_not"])
    b.node("Where", ["keep_not", "ninf", "scores"], ["masked"])
    b.node("Softmax", ["masked"], ["attn"])
    b.node("Not", ["mask"], ["keep_not2"])
    b.node("Where", ["keep_not2", "ninf", "scores2"], ["masked2"])
    b.node("Softmax", ["masked2"], ["attn2"])
    b.output("attn", DataType.float16, [1, 1, 2, 2])
    b.output("attn2", DataType.float16, [1, 1, 2, 2])
    return b
  }

  @Test func constantMasksBecomeOneSharedAddedBias() throws {
    var g = n16Like().graph
    #expect(try Patches.additiveSoftmaxMasks(&g, none) == 2)
    #expect(!g.nodes.contains { $0.op == "Where" || $0.op == "Not" })
    let adds = g.nodes.filter { $0.op == "Add" }
    #expect(adds.map(\.inputs[0]) == ["scores", "scores2"])
    #expect(adds.map(\.outputs) == [["masked"], ["masked2"]])
    #expect(Set(adds.map(\.inputs[1])).count == 1)
    let bias = try #require(g.initializers.first { $0.key == adds[0].inputs[1] })
    #expect(bias.dims == [1, 1, 2, 2] && bias.elementType == DataType.float16)
    let values = try Elements.littleEndian(bias, none)
    let fill = Patches.softmaxMaskFill.bitPattern.littleEndian
    let filled: [UInt8] = [UInt8(truncatingIfNeeded: fill), UInt8(truncatingIfNeeded: fill >> 8)]
    // Masked (mask false, so Not true, so the fill) only at index 1.
    #expect(Array(values[0..<2]) == [0, 0])
    #expect(Array(values[2..<4]) == filled)
    #expect(Array(values[4..<8]) == [0, 0, 0, 0])
    // Nothing left for the finite fill.
    #expect(try Patches.finiteSoftmaxMasks(&g, none) == 0)
  }

  @Test func aConditionWithoutNotAndAFillOnTheOtherSide() throws {
    var b = OnnxGraphBuilder()
    b.input("scores", DataType.float16, [1, 2])
    b.bool("keep", [1, 2], [true, false])
    b.initializer("ninf", DataType.float16, [], [0x00, 0xFC])
    b.node("Where", ["keep", "scores", "ninf"], ["masked"])
    b.node("Softmax", ["masked"], ["attn"])
    b.output("attn", DataType.float16, [1, 2])
    var g = b.graph
    #expect(try Patches.additiveSoftmaxMasks(&g, none) == 1)
    let add = try #require(g.nodes.first { $0.op == "Add" })
    let bias = try #require(g.initializers.first { $0.key == add.inputs[1] })
    let values = try Elements.littleEndian(bias, none)
    #expect(Array(values[0..<2]) == [0, 0])
    #expect(Array(values[2..<4]) != [0, 0])
  }

  @Test func aMaskComputedAtRunTimeIsLeftAlone() throws {
    var b = OnnxGraphBuilder()
    b.input("scores", DataType.float16, [1, 2])
    b.input("keep", DataType.bool, [1, 2])
    b.initializer("ninf", DataType.float16, [], [0x00, 0xFC])
    b.node("Where", ["keep", "scores", "ninf"], ["masked"])
    b.node("Softmax", ["masked"], ["attn"])
    b.output("attn", DataType.float16, [1, 2])
    var g = b.graph
    #expect(try Patches.additiveSoftmaxMasks(&g, none) == 0)
    #expect(g.nodes.contains { $0.op == "Where" })
  }
}

@Suite struct VisionLayerNormTests {
  private let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))

  @Test func visionNormsGetTheirInputDividedAndTheirEpsilonWithIt() throws {
    var b = OnnxGraphBuilder()
    b.input("x", DataType.float16, [1, 4, 8])
    b.valueInfo("x", DataType.float16, [1, 4, 8])
    b.fp16("w", [8], Array(repeating: 1, count: 8))
    b.fp16("bias", [8], Array(repeating: 0, count: 8))
    b.node("LayerNormalization", ["x", "w", "bias"], ["y"], [.floatValue("epsilon", 1e-5), .int("axis", -1)])
    // one the policy prescale already fed
    b.fp16("__layernorm_prescale_8", [], [0.125])
    b.node("Mul", ["x", "__layernorm_prescale_8"], ["x__scaled"])
    b.node("LayerNormalization", ["x__scaled", "w", "bias"], ["z"])
    b.output("y", DataType.float16, [1, 4, 8])
    b.output("z", DataType.float16, [1, 4, 8])
    var g = b.graph
    #expect(try Patches.prescaleVisionLayerNorm(&g) == 1)
    let norms = g.nodes.filter { $0.op == "LayerNormalization" }
    #expect(norms[0].inputs[0] == "x__vision_scaled")
    #expect(norms[1].inputs[0] == "x__scaled")
    let eps = try #require(norms[0].attribute("epsilon")?.f)
    #expect(abs(eps - 1e-5 / 1024) < 1e-12)
    #expect(norms[0].attribute("axis")?.i == -1)
    let mul = try #require(g.nodes.first { $0.outputs == ["x__vision_scaled"] })
    #expect(mul.op == "Mul" && mul.inputs == ["x", "__layernorm_vision_prescale_32"])
    // and once is enough
    #expect(try Patches.prescaleVisionLayerNorm(&g) == 0)
  }
}

@Suite struct SafeLayerNormTests {
  @Test func everyNormBecomesItsFp16SafeFormAndKeepsItsOutput() throws {
    var b = OnnxGraphBuilder(opset: 20)
    b.input("x", DataType.float16, [1, 4, 8])
    b.valueInfo("y", DataType.float16, [1, 4, 8])
    b.fp16("w", [8], Array(repeating: 1, count: 8))
    b.fp16("bias", [8], Array(repeating: 0, count: 8))
    b.node("LayerNormalization", ["x", "w", "bias"], ["y"], [.floatValue("epsilon", 1e-5), .int("axis", -1)])
    b.output("y", DataType.float16, [1, 4, 8])
    var g = b.graph
    #expect(Patches.safeLayerNorm(&g) == 1)
    #expect(!g.nodes.contains { $0.op == "LayerNormalization" })
    #expect(g.nodes.filter { $0.op == "ReduceMean" }.count == 2 && g.nodes.contains { $0.op == "ReduceMax" })
    let last = try #require(g.nodes.last)
    #expect(last.outputs == ["y"] && last.op == "Add" && last.displayName == Patches.safeLayerNormOutput("y"))
    let axes = try #require(g.initializers.first { $0.key == "y__safe_axes" })
    #expect(axes.dims == [1])
    // a probe still sees it as a norm
    let probed = Patches.addProbes(&g, .init(ops: ["LayerNormalization"]))
    #expect(probed.map(\.op) == [] || probed.map(\.op) == ["LayerNormalization"])
  }
}

@Suite struct TanhGeluTests {
  @Test func anExactFp16GeluBecomesItsTanhFormAndKeepsItsOutput() throws {
    var b = OnnxGraphBuilder(opset: 20)
    b.input("x", DataType.float16, [1, 4])
    b.valueInfo("x", DataType.float16, [1, 4])
    b.node("Gelu", ["x"], ["y"])
    b.input("f", DataType.float, [1, 4])
    b.valueInfo("f", DataType.float, [1, 4])
    b.node("Gelu", ["f"], ["z"])
    b.output("y", DataType.float16, [1, 4])
    b.output("z", DataType.float, [1, 4])
    var g = b.graph
    #expect(Patches.tanhGelu(&g) == 1)
    #expect(g.nodes.filter { $0.op == "Gelu" }.map(\.outputs) == [["z"]])
    #expect(g.nodes.contains { $0.op == "Tanh" } && g.nodes.contains { $0.op == "Clip" })
    let last = try #require(g.nodes.first { $0.outputs == ["y"] })
    #expect(last.op == "Mul" && last.displayName == Patches.tanhGeluOutput("y"))
  }
}

@Suite struct FP16StateIOTests {
  @Test func aStatePairIsFp16AtTheEdgeAndFp32Inside() throws {
    var b = OnnxGraphBuilder(opset: 20)
    b.input("state_feat_q", DataType.float, [4, 2])
    b.input("x", DataType.float, [1, 2])
    b.input("state_img_q", DataType.uint8, [4])
    b.node("Concat", ["state_feat_q", "x"], ["next_state_feat_q"], [.int("axis", 0)])
    b.node("Identity", ["state_img_q"], ["next_state_img_q"])
    b.node("Identity", ["state_feat_q"], ["outputs"])
    b.output("next_state_feat_q", DataType.float, [5, 2])
    b.output("next_state_img_q", DataType.uint8, [4])
    b.output("outputs", DataType.float, [4, 2])
    var g = b.graph
    #expect(Patches.fp16StateIO(&g) == 1)
    #expect(g.inputs.first { $0.key == "state_feat_q" }?.elemType == DataType.float16)
    #expect(g.outputs.first { $0.key == "next_state_feat_q" }?.elemType == DataType.float16)
    #expect(g.inputs.first { $0.key == "state_img_q" }?.elemType == DataType.uint8)
    #expect(g.nodes.first?.op == "Cast" && g.nodes.first?.inputs == ["state_feat_q"])
    #expect(g.nodes.last?.op == "Cast" && g.nodes.last?.outputs == ["next_state_feat_q"])
    let concat = try #require(g.nodes.first { $0.op == "Concat" })
    #expect(concat.inputs == ["state_feat_q__fp32", "x"] && concat.outputs == ["next_state_feat_q__fp32"])
    #expect(g.nodes.first { $0.outputs == ["outputs"] }?.inputs == ["state_feat_q__fp32"])
  }
}

@Suite struct FoldNormalizationTests {
  private let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))

  @Test func aNormalizationBeforeAnUnpaddedConvGoesIntoItsWeights() throws {
    var b = OnnxGraphBuilder(opset: 20)
    b.input("x", DataType.float16, [1, 2, 2, 2])
    b.fp16("mean", [1, 2, 1, 1], [10, 20])
    b.fp16("std", [1, 2, 1, 1], [2, 4])
    b.fp16("w", [1, 2, 2, 2], [1, 1, 1, 1, 2, 2, 2, 2])
    b.fp16("b", [1], [0.5])
    b.node("Sub", ["x", "mean"], ["s"])
    b.node("Div", ["s", "std"], ["d"])
    b.node("Conv", ["d", "w", "b"], ["y"], [.ints("pads", [0, 0, 0, 0]), .ints("strides", [2, 2])])
    b.output("y", DataType.float16, [1, 1, 1, 1])
    var g = b.graph
    #expect(try Patches.foldInputNormalization(&g, none) == 1)
    #expect(g.nodes.map(\.op) == ["Conv"])
    let conv = g.nodes[0]
    #expect(conv.inputs[0] == "x")
    let w = try Elements.floats(Elements.littleEndian(try #require(g.initializers.first { $0.key == conv.inputs[1] }), none), as: DataType.float16, for: "w")
    let bias = try Elements.floats(Elements.littleEndian(try #require(g.initializers.first { $0.key == conv.inputs[2] }), none), as: DataType.float16, for: "b")
    #expect(w == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    // 0.5 - (4 * 1 * 10 / 2 + 4 * 2 * 20 / 4) = 0.5 - (20 + 40)
    #expect(bias == [-59.5])
  }

  @Test func aPaddedConvKeepsItsNormalization() throws {
    var b = OnnxGraphBuilder(opset: 20)
    b.input("x", DataType.float16, [1, 1, 2, 2])
    b.fp16("mean", [1, 1, 1, 1], [10])
    b.fp16("std", [1, 1, 1, 1], [2])
    b.fp16("w", [1, 1, 3, 3], Array(repeating: 1, count: 9))
    b.node("Sub", ["x", "mean"], ["s"])
    b.node("Div", ["s", "std"], ["d"])
    b.node("Conv", ["d", "w"], ["y"], [.ints("pads", [1, 1, 1, 1])])
    b.output("y", DataType.float16, [1, 1, 2, 2])
    var g = b.graph
    #expect(try Patches.foldInputNormalization(&g, none) == 0)
  }
}

@Suite struct Int8WeightsTests {
  private let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))

  @Test func aMatMulsWeightsGoInt8WithAScalePerColumn() throws {
    var b = OnnxGraphBuilder(opset: 20)
    b.input("x", DataType.float16, [1, 2])
    b.fp16("w", [2, 2], [1, -0.5, -2, 0.25])
    b.node("MatMul", ["x", "w"], ["y"])
    b.output("y", DataType.float16, [1, 2])
    var g = b.graph
    #expect(try Patches.int8Weights(&g, none, minimum: 1) == 1)
    #expect(g.nodes.map(\.op) == ["DequantizeLinear", "MatMul"])
    #expect(g.nodes[1].inputs[1] == g.nodes[0].outputs[0])
    #expect(g.nodes[0].attribute("axis")?.i == 1)
    #expect(!g.initializers.contains { $0.key == "w" })
    let q = try #require(g.initializers.first { $0.key == "w__int8" })
    let bytes = try Elements.littleEndian(q, none).map { Int8(bitPattern: $0) }
    // columns' largest: 2 and 0.5, so scales 2/127 and 0.5/127
    #expect(bytes == [64, -127, -127, 64])
    let scale = try Elements.floats(Elements.littleEndian(try #require(g.initializers.first { $0.key == "w__int8_scale" }), none), as: DataType.float16, for: "s")
    #expect(abs(scale[0] - 2 / 127) < 1e-4 && abs(scale[1] - 0.5 / 127) < 1e-4)
  }

  @Test func smallWeightsStay() throws {
    var b = OnnxGraphBuilder(opset: 20)
    b.input("x", DataType.float16, [1, 2])
    b.fp16("w", [2, 2], [1, 2, 3, 4])
    b.node("MatMul", ["x", "w"], ["y"])
    b.output("y", DataType.float16, [1, 2])
    var g = b.graph
    #expect(try Patches.int8Weights(&g, none) == 0)
    #expect(g.nodes.map(\.op) == ["MatMul"])
  }
}
