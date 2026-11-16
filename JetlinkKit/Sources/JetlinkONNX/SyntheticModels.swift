import Foundation

/// Small ONNX models made on the phone to measure an accelerator with, the
/// NPU sweep's (LiteRtBackend): a stack of linear layers shaped like a
/// driving model's, whose size alone varies.
public enum SyntheticModels {
  /// `layers` linear layers of `width` in and out, each a MatMul by an fp16
  /// [width, width] weight and an Add of an fp16 bias, with GELUs between,
  /// over an input "x" of [1, rows, width] to an output "y" of the same
  /// shape, written to `url`. Its weights are `layers * width * width * 2`
  /// bytes. The values are small and fixed, so every run computes the same.
  public static func writeLinearStack(layers: Int, width: Int, rows: Int, to url: URL) throws {
    let w = Int64(width)
    var graph = Graph(name: "linear_stack")
    graph.inputs = [.tensor("x", DataType.float, [1, Int64(rows), w])]
    graph.outputs = [.tensor("y", DataType.float, [1, Int64(rows), w])]
    // One weight's and one bias's bytes, shared by every layer's initializer
    // (copy on write): the file holds a copy each, memory one.
    let weight = Elements.encode((0..<(width * width)).map { Float(($0 * 7919) % 2001 - 1000) * 2e-5 }, as: DataType.float16)
    let bias = Elements.encode((0..<width).map { Float($0 % 17) * 1e-3 }, as: DataType.float16)
    var x = "x"
    for i in 0..<layers {
      let last = i == layers - 1
      graph.initializers.append(Tensor(name: "w\(i)", dims: [w, w], dataType: DataType.float16, raw: .owned(weight)))
      graph.initializers.append(Tensor(name: "b\(i)", dims: [w], dataType: DataType.float16, raw: .owned(bias)))
      graph.nodes.append(Node(inputs: [x, "w\(i)"], outputs: ["mm\(i)"], name: "MatMul_\(i)", opType: "MatMul"))
      let sum = last ? "y" : "sum\(i)"
      graph.nodes.append(Node(inputs: ["mm\(i)", "b\(i)"], outputs: [sum], name: "Add_\(i)", opType: "Add"))
      if !last {
        graph.nodes.append(Node(inputs: [sum], outputs: ["act\(i)"], name: "Gelu_\(i)", opType: "Gelu"))
        x = "act\(i)"
      }
    }
    // opset 20 has Gelu, as Light Speed 2 imports it
    try write(graph, opset: 20, to: url)
  }

  /// `graph` as onnx.save writes it, piece by piece into `url`, never whole
  /// in memory.
  static func write(_ graph: Graph, opset: Int64, to url: URL) throws {
    let none = Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0))
    var model = Encode.model(Model(irVersion: 8, graph: graph), none)
    var opsetImport = Encoded()
    opsetImport.stringField(1, "")
    opsetImport.intField(2, opset)
    model.message(8, opsetImport)
    FileManager.default.createFile(atPath: url.path, contents: nil)
    let file = try FileHandle(forWritingTo: url)
    defer { try? file.close() }
    for piece in model.allPieces {
      guard case .bytes(let bytes) = piece else { throw OnnxError("a synthetic model has only bytes of its own") }
      try file.write(contentsOf: bytes)
    }
  }
}
