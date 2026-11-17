import Foundation

/// Benchmark-only static QDQ: uint16 activations and uint8 constant weights.
/// Ranges come from a separate calibration run, never from a fixed scale.
/// Floating-point regions (attention, normalization and heads) stay as exported.
public enum W8A16Preparation {
  public struct Range: Sendable {
    public var min: Float
    public var max: Float
    public init(min: Float, max: Float) {
      self.min = min
      self.max = max
    }
  }

  public struct Probe: Sendable {
    public let names: [String]
    public let layers: Int
  }

  struct Target {
    let index: Int
    let weight: Tensor
    let bias: Tensor?
  }

  static func targets(_ g: Graph, visionOnly: Bool, minimum: Int = 65_536) throws -> [Target] {
    let constants = Dictionary(g.initializers.map { ($0.key, $0) }, uniquingKeysWith: { _, b in b })
    let types = Patches.staticTypes(g)
    let vision = visionOnly ? try Split.visionMask(g) : Array(repeating: true, count: g.nodes.count)
    return g.nodes.enumerated().compactMap { i, n in
      guard vision[i], ["MatMul", "Gemm", "Conv"].contains(n.op), n.inputs.count >= 2,
        n.outputs.count == 1,
        let w = constants[n.inputs[1]], w.elementCount >= minimum,
        [DataType.float, DataType.float16].contains(w.elementType),
        [DataType.float, DataType.float16].contains(types[n.inputs[0]] ?? 0),
        [DataType.float, DataType.float16].contains(types[n.outputs[0]] ?? 0)
      else { return nil }
      if w.key.hasSuffix("__fp32") { return nil }  // protected vision heads
      if n.op == "MatMul", w.dims.count != 2 { return nil }
      if n.op == "Gemm", (n.attribute("alpha")?.f ?? 1) != 1 || (n.attribute("beta")?.f ?? 1) != 1 {
        return nil
      }
      let bias = n.inputs.count > 2 && !n.inputs[2].isEmpty ? constants[n.inputs[2]] : nil
      if n.inputs.count > 2 && !n.inputs[2].isEmpty && bias == nil { return nil }
      return Target(index: i, weight: w, bias: bias)
    }
  }

  public static func writeProbe(source: URL, to output: URL, visionOnly: Bool) throws -> Probe {
    let data = try Data(contentsOf: source, options: .alwaysMapped)
    return try data.withUnsafeBytes { bytes in
      let src = Source(bytes: bytes)
      var (model, g) = try Decode.preparable(src)
      try preparedShapes(&g, src)
      let selected = try targets(g, visionOnly: visionOnly)
      guard !selected.isEmpty else {
        throw OnnxError("no constant Conv/MatMul/Gemm layers suitable for W8A16")
      }
      let names = Set(
        selected.flatMap { [g.nodes[$0.index].inputs[0], g.nodes[$0.index].outputs[0]] })
      _ = Patches.addProbes(&g, .init(names: names.sorted()))
      let visible = Set((g.inputs + g.outputs).map(\.key))
      guard names.isSubset(of: visible) else {
        throw OnnxError("missing W8A16 calibration tensor shapes")
      }
      model.graph = g
      try PartWriter.write(Encode.model(model, src), src, to: output, onWrite: { _ in })
      return Probe(names: names.sorted(), layers: selected.count)
    }
  }

  /// Explicit Q/DQ on both activations and weights is essential: weight-only
  /// DQ made carrot.41's MatMuls fall to the CPU. Boundaries are float32 for
  /// QNN's QDQ recognizer; Casts preserve the original graph's float16 I/O.
  public static func quantize(
    source: URL, to output: URL, ranges: [String: Range], visionOnly: Bool
  ) throws -> Int {
    let data = try Data(contentsOf: source, options: .alwaysMapped)
    return try data.withUnsafeBytes { bytes in
      let src = Source(bytes: bytes)
      var (model, g) = try Decode.preparable(src)
      try preparedShapes(&g, src)
      let selected = try targets(g, visionOnly: visionOnly)
      guard !selected.isEmpty else { throw OnnxError("no W8A16 targets") }
      let count = try rewrite(&g, src, selected: selected, ranges: ranges)
      // uint16 Q/DQ entered the standard ONNX domain at opset 21.
      model.opsets = model.opsets.map {
        $0.domain.isEmpty && $0.version < 21 ? OpsetImport(raw: nil, domain: "", version: 21) : $0
      }
      model.graph = g
      try PartWriter.write(Encode.model(model, src), src, to: output, onWrite: { _ in })
      return count
    }
  }

  static func parameters(_ r: Range, levels: Float) throws -> (scale: Float, zero: Int) {
    guard r.min.isFinite, r.max.isFinite, r.min <= r.max else {
      throw OnnxError("non-finite or reversed calibration range")
    }
    let lo = Swift.min(0, r.min)
    let hi = Swift.max(0, r.max)
    let scale = Swift.max((hi - lo) / levels, 1e-8)
    return (scale, Int(Swift.max(0, Swift.min(levels, (-lo / scale).rounded(.toNearestOrEven)))))
  }

  /// The existing MatMul+Add preparation introduces flat Reshape/Gemm
  /// tensors without value_info. Recover their static shapes so the large
  /// FCs are included, rather than silently testing only the remaining Conv.
  static func preparedShapes(_ g: inout Graph, _ src: Source) throws {
    var types = Patches.staticTypes(g)
    var dims = Patches.staticDims(g)
    let constants = Dictionary(g.initializers.map { ($0.key, $0) }, uniquingKeysWith: { _, b in b })
    for n in g.nodes where n.outputs.count == 1 && !n.inputs.isEmpty {
      let y = n.outputs[0]
      guard let type = types[n.inputs[0]], let x = dims[n.inputs[0]], x.allSatisfy({ $0 > 0 }) else { continue }
      var shape: [Int64]?
      if n.op == "Reshape", n.inputs.count == 2, let t = constants[n.inputs[1]] {
        var s = try Elements.integers(t, src)
        if (n.attribute("allowzero")?.i ?? 0) == 0 {
          for i in s.indices where s[i] == 0 && i < x.count { s[i] = x[i] }
        }
        if s.filter({ $0 == -1 }).count == 1, let i = s.firstIndex(of: -1) {
          let product = s.filter { $0 > 0 }.reduce(Int64(1), *)
          let total = x.reduce(Int64(1), *)
          if product > 0 && total % product == 0 { s[i] = total / product }
        }
        if s.allSatisfy({ $0 > 0 }), s.reduce(Int64(1), *) == x.reduce(Int64(1), *) { shape = s }
      } else if n.op == "Gemm", x.count == 2, n.inputs.count >= 2, let w = constants[n.inputs[1]], w.dims.count == 2 {
        shape = [x[(n.attribute("transA")?.i ?? 0) == 1 ? 1 : 0], w.dims[(n.attribute("transB")?.i ?? 0) == 1 ? 0 : 1]]
      }
      if let shape, types[y] == nil || dims[y] == nil {
        g.valueInfo.append(.tensor(y, type, shape))
        types[y] = type
        dims[y] = shape
      }
    }
  }

  static func rewrite(_ g: inout Graph, _ src: Source, selected: [Target], ranges: [String: Range])
    throws -> Int
  {
    let types = Patches.staticTypes(g)
    let dims = Patches.staticDims(g)
    let byIndex = Dictionary(uniqueKeysWithValues: selected.map { ($0.index, $0) })
    var nodes: [Node] = []
    var replaced = Set<String>()
    var extra: [Tensor] = []
    var info: [ValueInfo] = []
    func constant(_ name: String, _ type: Int32, _ values: [Float]) {
      extra.append(
        Tensor(name: name, dims: [], dataType: type, raw: .owned(Elements.encode(values, as: type)))
      )
    }
    func zero(_ name: String, _ type: Int32, _ value: Int) {
      let bytes: [UInt8] =
        type == DataType.uint8
        ? [UInt8(value)]
        : [UInt8(truncatingIfNeeded: value), UInt8(truncatingIfNeeded: value >> 8)]
      extra.append(Tensor(name: name, dims: [], dataType: type, raw: .owned(bytes)))
    }
    func qdq(_ input: String, _ prefix: String, _ p: (scale: Float, zero: Int)) -> String {
      constant(prefix + "_scale", DataType.float, [p.scale])
      zero(prefix + "_zero", DataType.uint16, p.zero)
      nodes.append(
        Node(
          inputs: [input, prefix + "_scale", prefix + "_zero"], outputs: [prefix + "_q"],
          name: prefix + "_Q", opType: "QuantizeLinear"))
      nodes.append(
        Node(
          inputs: [prefix + "_q", prefix + "_scale", prefix + "_zero"], outputs: [prefix + "_dq"],
          name: prefix + "_DQ", opType: "DequantizeLinear"))
      return prefix + "_dq"
    }
    for (i, original) in g.nodes.enumerated() {
      guard let target = byIndex[i] else {
        nodes.append(original)
        continue
      }
      let x = original.inputs[0]
      let y = original.outputs[0]
      guard let xr = ranges[x], let yr = ranges[y] else {
        throw OnnxError("missing calibration range for \(x) or \(y)")
      }
      let xp = try parameters(xr, levels: 65535)
      let yp = try parameters(yr, levels: 65535)
      let values = try Elements.floats(
        Elements.littleEndian(target.weight, src), as: target.weight.elementType,
        for: target.weight.key)
      guard let lo = values.min(), let hi = values.max(), values.allSatisfy(\.isFinite) else {
        throw OnnxError("invalid W8 weight")
      }
      let wp = try parameters(Range(min: lo, max: hi), levels: 255)
      let prefix = "__jetlink_w8a16_\(i)"
      let quantized = values.map {
        UInt8(
          Swift.max(0, Swift.min(255, Int(($0 / wp.scale).rounded(.toNearestOrEven)) + wp.zero)))
      }
      extra.append(
        Tensor(
          name: prefix + "_w", dims: target.weight.dims, dataType: DataType.uint8,
          raw: .owned(quantized)))
      constant(prefix + "_ws", DataType.float, [wp.scale])
      zero(prefix + "_wz", DataType.uint8, wp.zero)
      var input = x
      if types[x] == DataType.float16 {
        input = prefix + "_xf"
        nodes.append(
          Node(
            inputs: [x], outputs: [input], name: input, opType: "Cast",
            attributes: [.int("to", Int64(DataType.float))]))
      }
      let dqInput = qdq(input, prefix + "_a", xp)
      nodes.append(
        Node(
          inputs: [prefix + "_w", prefix + "_ws", prefix + "_wz"], outputs: [prefix + "_wdq"],
          name: prefix + "_WDQ", opType: "DequantizeLinear"))
      var node = original
      node.inputs[0] = dqInput
      node.inputs[1] = prefix + "_wdq"
      if let b = target.bias {
        let bv = try Elements.floats(Elements.littleEndian(b, src), as: b.elementType, for: b.key)
        let bs = xp.scale * wp.scale
        let q = try bv.flatMap { v -> [UInt8] in
          let scaled = Double(v) / Double(bs)
          guard scaled.isFinite, scaled >= Double(Int32.min), scaled <= Double(Int32.max) else {
            throw OnnxError("W8A16 bias exceeds int32")
          }
          let bits = UInt32(bitPattern: Int32(scaled.rounded(.toNearestOrEven)))
          return (0..<4).map { UInt8(truncatingIfNeeded: bits >> ($0 * 8)) }
        }
        extra.append(
          Tensor(name: prefix + "_b", dims: b.dims, dataType: DataType.int32, raw: .owned(q)))
        constant(prefix + "_bs", DataType.float, [bs])
        extra.append(
          Tensor(
            name: prefix + "_bz", dims: [], dataType: DataType.int32, raw: .owned([0, 0, 0, 0])))
        nodes.append(
          Node(
            inputs: [prefix + "_b", prefix + "_bs", prefix + "_bz"], outputs: [prefix + "_bdq"],
            name: prefix + "_BDQ", opType: "DequantizeLinear"))
        node.inputs[2] = prefix + "_bdq"
        replaced.insert(b.key)
      }
      node.outputs = [prefix + "_yf"]
      nodes.append(node)
      let dqOutput = qdq(prefix + "_yf", prefix + "_o", yp)
      nodes.append(
        Node(
          inputs: [dqOutput], outputs: [y], name: prefix + "_out", opType: "Cast",
          attributes: [.int("to", Int64(types[y]!))]))
      if let shape = dims[x] { info.append(.tensor(dqInput, DataType.float, shape)) }
      if types[x] == DataType.float16, let shape = dims[x] {
        info.append(.tensor(prefix + "_xf", DataType.float, shape))
      }
      if let shape = dims[y] { info.append(.tensor(prefix + "_yf", DataType.float, shape)) }
      replaced.insert(target.weight.key)
    }
    let live = Set(nodes.flatMap(\.inputs))
    g.initializers.removeAll { replaced.contains($0.key) && !live.contains($0.key) }
    g.initializers.append(contentsOf: extra)
    g.valueInfo.append(contentsOf: info)
    g.nodes = nodes
    return selected.count
  }
}
