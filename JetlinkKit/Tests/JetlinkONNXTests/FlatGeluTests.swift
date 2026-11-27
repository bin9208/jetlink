import Foundation
import Testing

@testable import JetlinkONNX

/// `flat-gelu`: a ConvNeXt MLP (Linear -> Gelu -> Linear on [1, H, W, C])
/// after the Gemm rewrite, with the Gelu on the first Gemm's [N, C] and the
/// second Gemm reading it, both reshapes between them gone.
@Suite struct FlatGeluTests {
  /// x [1, 2, 4, 32] -> MatMul+Add (32 -> 64) -> Gelu -> MatMul+Add (64 -> 32) -> y
  static func mlp() -> Graph {
    var b = OnnxGraphBuilder()
    var values = SeededValues(seed: 7)
    b.input("x", DataType.float16, [1, 2, 4, 32])
    b.fp16("w1", [32, 64], values(32 * 64))
    b.fp16("b1", [64], values(64))
    b.fp16("w2", [64, 32], values(64 * 32))
    b.fp16("b2", [32], values(32))
    b.node("MatMul", ["x", "w1"], ["m1"])
    b.node("Add", ["m1", "b1"], ["h"])
    b.node("Gelu", ["h"], ["g"])
    b.node("MatMul", ["g", "w2"], ["m2"])
    b.node("Add", ["m2", "b2"], ["y"])
    b.valueInfo("m1", DataType.float16, [1, 2, 4, 64])
    b.valueInfo("h", DataType.float16, [1, 2, 4, 64])
    b.valueInfo("g", DataType.float16, [1, 2, 4, 64])
    b.valueInfo("m2", DataType.float16, [1, 2, 4, 32])
    b.output("y", DataType.float16, [1, 2, 4, 32])
    return b.graph
  }

  @Test func reshapesBetweenTheGemmsGo() throws {
    var g = Self.mlp()
    let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))
    #expect(try Patches.gemmWithTransposedWeight(&g, none) == 2)
    #expect(g.nodes.map(\.op) == ["Reshape", "Gemm", "Reshape", "Gelu", "Reshape", "Gemm", "Reshape"])
    #expect(Patches.flatElementwiseBetweenGemms(&g) == 1)
    #expect(g.nodes.map(\.op) == ["Reshape", "Gemm", "Gelu", "Gemm", "Reshape"])
    let gelu = g.nodes[2]
    // the Gelu reads the first Gemm's output and writes what the second reads
    #expect(gelu.inputs == g.nodes[1].outputs)
    #expect(g.nodes[3].inputs.first == gelu.outputs.first)
    // the first reshape flattens x, the last gives y back its shape
    #expect(g.nodes[0].inputs.first == "x")
    #expect(g.nodes[4].outputs == ["y"])
    // no 4-D shape left for the tensors that went, and every input made by something
    #expect(!g.valueInfo.contains { $0.key == "h" || $0.key == "g" })
    let made = Set(g.nodes.flatMap(\.outputs) + g.initializers.map(\.key) + g.inputs.map(\.key))
    #expect(g.nodes.flatMap(\.inputs).allSatisfy(made.contains))
    // nothing more to do the second time
    #expect(Patches.flatElementwiseBetweenGemms(&g) == 0)
  }

  /// A Gelu whose output something else reads too keeps its reshapes.
  @Test func aSharedGeluStays() throws {
    var g = Self.mlp()
    g.outputs.append(.tensor("g", DataType.float16, [1, 2, 4, 64]))
    let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))
    _ = try Patches.gemmWithTransposedWeight(&g, none)
    #expect(Patches.flatElementwiseBetweenGemms(&g) == 0)
  }
}
