import Foundation
import Testing

@testable import JetlinkONNX

/// `fold-layer-scale`: ConvNeXt's gamma after fc2 (Gemm -> Reshape ->
/// Transpose to NCHW -> Mul by [1, C, 1, 1]) taken into fc2's weight rows and
/// bias, the Mul gone.
@Suite struct FoldLayerScaleTests {
  static let gamma: [Float] = (0..<32).map { 1e-3 * Float($0 + 1) }

  /// x [1, 2, 4, 64] -> MatMul+Add (64 -> 32) -> Transpose(0, 3, 1, 2) -> Mul(gamma) -> y
  static func block(gammaDims: [Int64] = [1, 32, 1, 1]) -> Graph {
    var b = OnnxGraphBuilder()
    var values = SeededValues(seed: 11)
    b.input("x", DataType.float16, [1, 2, 4, 64])
    b.fp16("w2", [64, 32], values(64 * 32))
    b.fp16("b2", [32], values(32))
    b.fp16("gamma", gammaDims, gamma)
    b.node("MatMul", ["x", "w2"], ["m2"])
    b.node("Add", ["m2", "b2"], ["h"])
    b.node("Transpose", ["h"], ["t"], [.ints("perm", [0, 3, 1, 2])])
    b.node("Mul", ["t", "gamma"], ["y"])
    b.valueInfo("m2", DataType.float16, [1, 2, 4, 32])
    b.valueInfo("h", DataType.float16, [1, 2, 4, 32])
    b.valueInfo("t", DataType.float16, [1, 32, 2, 4])
    b.output("y", DataType.float16, [1, 32, 2, 4])
    return b.graph
  }

  @Test func theScaleGoesIntoFc2() throws {
    var g = Self.block()
    let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))
    let bias = try Elements.floats(Elements.littleEndian(g.initializers[1], none), as: DataType.float16, for: "b2")
    #expect(try Patches.gemmWithTransposedWeight(&g, none) == 1)
    #expect(try Patches.foldLayerScale(&g, none) == 1)
    #expect(g.nodes.map(\.op) == ["Reshape", "Gemm", "Reshape", "Transpose"])
    #expect(g.nodes.last?.outputs == ["y"])
    #expect(!g.initializers.contains { $0.key == "gamma" })
    #expect(!g.valueInfo.contains { $0.key == "t" })
    let weight = try #require(g.initializers.first { $0.key == g.nodes[1].inputs[1] })
    guard case .transposed(let tr) = weight.raw else {
      Issue.record("fc2's weight is no longer a transpose")
      return
    }
    // the scale as the model holds it, fp16
    let gamma16 = Self.gamma.map { Float(Float16($0)) }
    #expect(tr.rowScale == gamma16)
    let b2 = try #require(g.initializers.first { $0.key == "b2" })
    let scaled = try Elements.floats(Elements.littleEndian(b2, none), as: DataType.float16, for: "b2")
    let expected = zip(bias, gamma16).map { Float(Float16($0 * $1)) }
    #expect(scaled == expected)
    // nothing more the second time
    #expect(try Patches.foldLayerScale(&g, none) == 0)
  }

  @Test func aScaleOfTheWrongShapeStays() throws {
    var g = Self.block(gammaDims: [1, 1, 1, 32])
    let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))
    _ = try Patches.gemmWithTransposedWeight(&g, none)
    #expect(try Patches.foldLayerScale(&g, none) == 0)
    #expect(g.nodes.last?.op == "Mul")
  }

  /// The rows as they are written: (w.astype(float32) * s).astype(float16).
  @Test func rowsAreScaledAsWritten() throws {
    let rows = 3
    let values: [Float] = [1, -2, 0.5, 4, 1000, -0.001, 7, 8, 9]  // 3 output rows of 3
    var bytes = Elements.encode(values, as: DataType.float16)
    try bytes.withUnsafeMutableBytes { p in
      try PartWriter.scaleRows(p.baseAddress!, rows: rows, from: 0, to: 3, [0.5, 100, -1], elementSize: 2)
    }
    let out = try Elements.floats(bytes, as: DataType.float16, for: "w")
    let expected = values.enumerated().map { i, v in Float(Float16(Float(Float16(v)) * [0.5, 100, -1][i / rows])) }
    #expect(out == expected)
  }
}
