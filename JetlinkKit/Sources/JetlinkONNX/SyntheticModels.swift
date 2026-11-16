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

  /// One kind of Light Speed 2's layers, at its shapes, for timing a kind
  /// alone on an accelerator.
  public enum Block: String, CaseIterable, Sendable {
    /// LayerNormalization over [1, 288, 512], which LiteRT's conversion
    /// writes as its fp16-safe form (two MEANs, ABS, REDUCE_MAX, DIV, RSQRT).
    case layerNorm = "layer norm [1,288,512]"
    /// ReduceMean over the last axis of [1, 288, 512], kept, from x.
    case mean = "mean [1,288,512]"
    /// ReduceMax of |x| over the last axis of [1, 288, 512], from x.
    case reduceMax = "max |x| [1,288,512]"
    /// GELU over [1, 288, 2048].
    case gelu = "gelu [1,288,2048]"
    /// A 7x7 depthwise convolution of 1024 channels over 8x16, padded to
    /// keep the size.
    case depthwise = "depthwise 7x7 1024ch 8x16"
    /// Attention of 8 heads over 288 positions of 64: q kT, Softmax, times v.
    case attention = "attention 8x288x64"
  }

  /// `repeats` of `block` one after another, written to `url`: inputs "x"
  /// (and "k", "v" for attention) of fixed small values, output "y".
  public static func writeBlocks(_ block: Block, repeats: Int, to url: URL) throws {
    var graph = Graph(name: block.rawValue)
    let f16 = DataType.float16
    func input(_ name: String, _ dims: [Int64]) { graph.inputs.append(.tensor(name, DataType.float, dims)) }
    func constant(_ name: String, _ dims: [Int64], _ value: (Int) -> Float) {
      let count = dims.reduce(1, *)
      graph.initializers.append(
        Tensor(name: name, dims: dims, dataType: f16, raw: .owned(Elements.encode((0..<Int(count)).map(value), as: f16))))
    }
    func node(_ op: String, _ inputs: [String], _ output: String, _ attributes: [Attribute] = []) {
      graph.nodes.append(Node(inputs: inputs, outputs: [output], name: "\(op)_\(graph.nodes.count)", opType: op, attributes: attributes))
    }
    let row: [Int64] = [1, 288, 512]
    var shape = row
    var x = "x"
    switch block {
    case .layerNorm:
      input("x", row)
      constant("scale", [512]) { 1 + Float($0 % 7) * 1e-2 }
      constant("bias", [512]) { Float($0 % 5) * 1e-2 }
      for i in 0..<repeats {
        node("LayerNormalization", [x, "scale", "bias"], "ln\(i)", [.int("axis", -1)])
        x = "ln\(i)"
      }
    case .mean, .reduceMax:
      input("x", row)
      graph.initializers.append(Patches.int64Tensor([-1], "axes"))
      for i in 0..<repeats {
        if block == .mean {
          node("ReduceMean", [x, "axes"], "m\(i)", [.int("keepdims", 1)])
        } else {
          node("Abs", [x], "a\(i)")
          node("ReduceMax", ["a\(i)", "axes"], "m\(i)", [.int("keepdims", 1)])
        }
        node("Sub", [x, "m\(i)"], "d\(i)")
        x = "d\(i)"
      }
    case .gelu:
      shape = [1, 288, 2048]
      input("x", shape)
      for i in 0..<repeats {
        node("Gelu", [x], "g\(i)")
        x = "g\(i)"
      }
    case .depthwise:
      shape = [1, 1024, 8, 16]
      input("x", shape)
      constant("w", [1024, 1, 7, 7]) { Float($0 % 11 - 5) * 1e-3 }
      constant("b", [1024]) { Float($0 % 3) * 1e-3 }
      for i in 0..<repeats {
        node("Conv", [x, "w", "b"], "c\(i)", [.int("group", 1024), .ints("kernel_shape", [7, 7]), .ints("pads", [3, 3, 3, 3])])
        x = "c\(i)"
      }
    case .attention:
      shape = [1, 8, 288, 64]
      input("x", shape)
      input("k", shape)
      input("v", shape)
      node("Transpose", ["k"], "kT", [.ints("perm", [0, 1, 3, 2])])
      for i in 0..<repeats {
        node("MatMul", [x, "kT"], "s\(i)")
        node("Softmax", ["s\(i)"], "p\(i)", [.int("axis", -1)])
        node("MatMul", ["p\(i)", "v"], "o\(i)")
        x = "o\(i)"
      }
    }
    // the last node's output is y
    graph.nodes[graph.nodes.count - 1].outputs = ["y"]
    graph.outputs = [.tensor("y", DataType.float, shape)]
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
