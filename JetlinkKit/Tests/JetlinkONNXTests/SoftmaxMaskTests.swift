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

  @Test("Folded, a norm after a convolution takes the 1/32 in the convolution's weight and bias; any other keeps its Mul")
  func foldsTheDivisionIntoTheConvolution() throws {
    var b = OnnxGraphBuilder()
    b.input("x", DataType.float16, [1, 2, 3, 3])
    b.fp16("w", [2, 1, 1, 1], [64, -0.5])
    b.fp16("cb", [2], [32, 1])
    b.node("Conv", ["x", "w", "cb"], ["c"], [.int("group", 2)])
    b.node("Transpose", ["c"], ["t"], [.ints("perm", [0, 2, 3, 1])])
    b.valueInfo("t", DataType.float16, [1, 3, 3, 2])
    b.fp16("s", [2], [1, 1])
    b.fp16("z", [2], [0, 0])
    b.node("LayerNormalization", ["t", "s", "z"], ["y"], [.floatValue("epsilon", 1e-5), .int("axis", -1)])
    // a norm whose input is no convolution's keeps its Mul
    b.input("u", DataType.float16, [1, 4, 2])
    b.valueInfo("u", DataType.float16, [1, 4, 2])
    b.node("LayerNormalization", ["u", "s", "z"], ["v"], [.floatValue("epsilon", 1e-5), .int("axis", -1)])
    b.output("y", DataType.float16, [1, 3, 3, 2])
    b.output("v", DataType.float16, [1, 4, 2])
    var g = b.graph
    #expect(try Patches.prescaleVisionLayerNorm(&g, none, fold: true) == 2)

    let conv = try #require(g.nodes.first { $0.op == "Conv" })
    #expect(conv.inputs[1].hasSuffix("__vision_div_32") && conv.inputs[2].hasSuffix("__vision_div_32"))
    func values(_ name: String) throws -> [Float] {
      let t = try #require(g.initializers.first { $0.key == name })
      return try Elements.floats(Elements.littleEndian(t, none), as: t.elementType, for: name)
    }
    #expect(try values(conv.inputs[1]) == [2, -0.015625])
    #expect(try values(conv.inputs[2]) == [1, 0.03125])
    let norms = g.nodes.filter { $0.op == "LayerNormalization" }
    #expect(norms[0].inputs[0] == "t")
    #expect(abs(try #require(norms[0].attribute("epsilon")?.f) - 1e-5 / 1024) < 1e-12)
    #expect(norms[1].inputs[0] == "u__vision_scaled")
    #expect(g.nodes.filter { $0.op == "Mul" }.count == 1)
    // the weights the copies replaced are gone
    #expect(!g.initializers.contains { $0.key == "w" || $0.key == "cb" })

    // and once is enough: neither the fold nor a Mul again
    #expect(try Patches.prescaleVisionLayerNorm(&g, none, fold: true) == 0)
    #expect(g.nodes.filter { $0.op == "Mul" }.count == 1)
    #expect(try values(conv.inputs[1]) == [2, -0.015625])
  }

  @Test("The sweep tries none and each precision experiment alone, never int8 or the vision-prescale ones")
  func sweepsThePrecisionExperiments() {
    let sets = CoreMLPreparation.Experiment.combinations
    #expect(sets == [[], [.normGuard224], [.normGuardRes224], [.normGuardHot224], [.normGuardHot192], [.normGuardHot240]])
    #expect(CoreMLPreparation.Experiment.name([.foldVisionPrescale]) == "fold-vision-prescale")
    #expect(CoreMLPreparation.Experiment.name([.policyNormGuard]) == "policy-norm-guard")
  }

  @Test("The hot-norm guards are N16's alone among the phone's models; a hand-named file keeps what was asked")
  func hotNormGuardsAreN16s() {
    let hot: Set<CoreMLPreparation.Experiment> = [.normGuardHot192]
    #expect(CoreMLPreparation.Experiment.applicable(hot, modelStem: "01effc679eb46d60") == hot)
    #expect(CoreMLPreparation.Experiment.applicable(hot, modelStem: "ae90227c1ba51d32") == [])
    #expect(CoreMLPreparation.Experiment.applicable([.normGuard224, .normGuardHot240], modelStem: "ae90227c1ba51d32") == [.normGuard224])
    #expect(CoreMLPreparation.Experiment.applicable(hot, modelStem: "n16") == hot)
  }

  @Test("A guarded policy norm reads its prescaled input times peak / max(row peak, peak); a vision norm is left alone")
  func policyNormsGetARowGuard() throws {
    var g = Graph()
    g.inputs = [ValueInfo.tensor("x", DataType.float16, [1, 288, 512]), ValueInfo.tensor("v", DataType.float16, [1, 64])]
    g.nodes = [
      Node(inputs: ["x", Patches.policyPrescaleConst], outputs: ["x__scaled"], name: "x__prescale", opType: "Mul"),
      Node(inputs: ["x__scaled", "w", "b"], outputs: ["y"], name: "ln", opType: "LayerNormalization"),
      Node(inputs: ["x__scaled", "w", "b"], outputs: ["y2"], name: "ln2", opType: "LayerNormalization"),
      Node(inputs: ["v", "w", "b"], outputs: ["z"], name: "vision", opType: "LayerNormalization"),
    ]
    #expect(Patches.guardPolicyLayerNorm(&g) == 2)
    let ln = try #require(g.nodes.first { $0.name == "ln" })
    #expect(ln.inputs[0] == "x__scaled__guard_x")
    // one guard for the input both norms share
    #expect(g.nodes.filter { $0.op == "ReduceMax" }.count == 1)
    #expect(g.nodes.first { $0.name == "ln2" }?.inputs[0] == "x__scaled__guard_x")
    #expect(g.nodes.first { $0.name == "vision" }?.inputs[0] == "v")
    // 512 along the axis: a peak of 8, so a row's sum of squares stays under 32768
    #expect(Patches.layerNormGuardPeak(512) == 8)
    #expect(Patches.layerNormGuardPeak(64) == (512 as Float).squareRoot())
    #expect(Patches.layerNormGuardPeak(nil) == 8)
    let peak = try #require(g.initializers.first { $0.key == "x__scaled__guard_peak" })
    #expect(peak.dataType == DataType.float16)
  }

  @Test("The residual selection guards a norm reading an Add 512 wide, not a query norm from a Split; a named one only its names")
  func guardSelections() throws {
    func graph() -> Graph {
      var g = Graph()
      g.inputs = [ValueInfo.tensor("a", DataType.float16, [1, 288, 512]), ValueInfo.tensor("b", DataType.float16, [1, 288, 512])]
      g.valueInfo = [ValueInfo.tensor("r", DataType.float16, [1, 288, 512]), ValueInfo.tensor("q", DataType.float16, [1, 8, 288, 64])]
      g.nodes = [
        Node(inputs: ["a", "b"], outputs: ["r"], name: "add", opType: "Add"),
        Node(inputs: ["a"], outputs: ["q"], name: "split", opType: "Split"),
        Node(inputs: ["r", Patches.policyPrescaleConst], outputs: ["r__scaled"], name: "r__prescale", opType: "Mul"),
        Node(inputs: ["q", Patches.policyPrescaleConst], outputs: ["q__scaled"], name: "q__prescale", opType: "Mul"),
        Node(inputs: ["r__scaled", "w", "b"], outputs: ["y"], name: "node_layer_norm_67", opType: "LayerNormalization"),
        Node(inputs: ["q__scaled", "w", "b"], outputs: ["z"], name: "node_layer_norm_77", opType: "LayerNormalization"),
      ]
      return g
    }
    var g = graph()
    #expect(Patches.guardPolicyLayerNorm(&g, fixedPeak: 224, only: .residual) == 1)
    #expect(g.nodes.first { $0.name == "node_layer_norm_67" }?.inputs[0] == "r__scaled__guard_x")
    #expect(g.nodes.first { $0.name == "node_layer_norm_77" }?.inputs[0] == "q__scaled")
    var h = graph()
    #expect(Patches.guardPolicyLayerNorm(&h, fixedPeak: 224, only: .named(["node_layer_norm_77"])) == 1)
    #expect(h.nodes.first { $0.name == "node_layer_norm_77" }?.inputs[0] == "q__scaled__guard_x")
    #expect(Patches.n16HotNorms.count == 18 && Patches.n16HotNorms.contains("node_layer_norm_101"))
  }

  @Test("Folded, the guard replaces the 1/8 of the policy's prescale Mul with peak / max(row peak, 8 peak)")
  func guardFoldsIntoThePrescale() throws {
    var g = Graph()
    g.inputs = [ValueInfo.tensor("x", DataType.float16, [1, 288, 512])]
    g.nodes = [
      Node(inputs: ["x", Patches.policyPrescaleConst], outputs: ["x__scaled"], name: "x__prescale", opType: "Mul"),
      Node(inputs: ["x__scaled", "w", "b"], outputs: ["y"], name: "ln", opType: "LayerNormalization"),
    ]
    #expect(Patches.guardPolicyPrescale(&g, peak: 224) == 1)
    let mul = try #require(g.nodes.first { $0.name == "x__prescale" })
    #expect(mul.inputs == ["x", "x__scaled__guard_c"])
    // the norm reads the Mul's output as before
    #expect(g.nodes.first { $0.name == "ln" }?.inputs[0] == "x__scaled")
    // and the guard comes before the Mul
    let order = g.nodes.map(\.name)
    #expect(try #require(order.firstIndex(of: "x__scaled__guard_c")) < #require(order.firstIndex(of: "x__prescale")))
    #expect(g.initializers.contains { $0.key == "__layernorm_prescale_guard_floor" })
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
