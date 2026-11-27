import Foundation
import Testing

@testable import JetlinkONNX

/// What carrot.62's nav students needed of the preparation: types for the
/// tensors a merged-in encoder did not record (LS2 Nav), a uint8 Cast on the
/// image chain dropped (BM Nav's stateful wrapper), and its chunks.
@Suite struct NavModelTests {
  @Test func typesFollowTheInputsTheFileDoesNotRecord() {
    var b = OnnxGraphBuilder()
    b.input("x", DataType.float16, [1, 4, 8])
    b.fp16("w", [8], [Float](repeating: 1, count: 8))
    b.node("Transpose", ["x"], ["t"], [.ints("perm", [0, 2, 1])])
    b.node("LayerNormalization", ["t", "w"], ["n", "mean"])
    b.node("Cast", ["n"], ["c"], [.int("to", Int64(DataType.float))])
    b.node("Less", ["c", "c"], ["l"])
    b.node("Shape", ["c"], ["s"])
    b.output("c", DataType.float, [1, 8, 4])
    let types = Patches.inferredTypes(b.graph)
    #expect(types["t"] == DataType.float16)
    #expect(types["n"] == DataType.float16)
    #expect(types["mean"] == nil)
    #expect(types["c"] == DataType.float)
    #expect(types["l"] == DataType.bool)
    #expect(types["s"] == DataType.int64)
  }

  @Test func aRecordedTypeStands() {
    var b = OnnxGraphBuilder()
    b.input("x", DataType.float16, [4])
    b.node("Relu", ["x"], ["r"])
    b.valueInfo("r", DataType.float, [4])
    #expect(Patches.inferredTypes(b.graph)["r"] == DataType.float)
  }

  /// new_img -> Cast(to: UINT8) -> Concat -> Cast(to: FLOAT16) -> Relu -> y:
  /// both Casts go and the input is fp16.
  @Test func aUint8CastOnTheImageChainGoes() throws {
    var b = OnnxGraphBuilder()
    b.input("new_img", DataType.uint8, [1, 6, 4, 4])
    b.node("Cast", ["new_img"], ["img"], [.int("to", Int64(DataType.uint8))])
    b.node("Concat", ["img", "img"], ["cat"], [.int("axis", 1)])
    b.node("Cast", ["cat"], ["f"], [.int("to", Int64(DataType.float16))])
    b.node("Relu", ["f"], ["y"])
    b.output("y", DataType.float16, [1, 12, 4, 4])
    var g = b.graph
    try Patches.patchUint8Inputs(&g)
    #expect(g.nodes.map(\.op) == ["Concat", "Relu"])
    #expect(g.nodes[0].inputs == ["new_img", "new_img"])
    #expect(g.nodes[1].inputs == ["cat"])
    #expect(g.inputs.first?.elemType == DataType.float16)
  }
}
