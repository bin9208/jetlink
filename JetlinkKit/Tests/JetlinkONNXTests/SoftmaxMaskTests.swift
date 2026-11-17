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
