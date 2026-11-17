import Foundation

#if canImport(os)
  import os
#else
  import JetlinkLog
#endif

/// The graph rewrites of jetlink/onnx_patch.py on the `simplify` branch, on
/// the decoded model. Each one follows its Python function step by step,
/// including the order it appends initializers in and the errors it raises,
/// so the file it leads to is the one Python writes. The reasons for each
/// rewrite are in the Python docstrings; the comments here are about the port.
///
/// There is no shape inferrer in Swift. Where the Python asks onnx's shape
/// inferrer for a shape the file does not record, the rewrite that needed it
/// is skipped here, as the Python skips it when the inferrer finds none;
/// where it asks for a type (the ane-whole passes), the preparation refuses,
/// since the pass would otherwise go a different way. The driving models'
/// exports record every tensor's shape and type, so on them nothing is
/// skipped or refused and the output is the Python's byte for byte.
enum Patches {
  /// Where the vision trunk starts (onnx_patch.IMG_INPUTS).
  static let imageInputs = ["img", "big_img", "new_img", "state_img_q"]
  static let imageInputsRepr = "('img', 'big_img', 'new_img', 'state_img_q')"
  /// Ops that only move image bytes around (onnx_patch.LAYOUT_OPS).
  static let layoutOps: Set<String> = [
    "Concat", "Slice", "Gather", "Reshape", "Unsqueeze", "Squeeze", "Transpose", "Flatten", "Identity", "Expand",
  ]
  static let tinygradDomain = "org.tinygrad"
  static let passthroughOps: Set<String> = ["Contiguous"]
  /// A weight smaller than this stays as it is (onnx_patch.BLOB_MIN_ELEMENTS).
  static let blobMinElements: Int64 = 1024
  /// LayerNorm(x / k) is LayerNorm(x) with epsilon scaled by k^2
  /// (onnx_patch.LAYERNORM_PRESCALE).
  static let layerNormPrescale = 8
  /// The ops heads_in_fp32 moves to fp32 (onnx_patch.HEAD_OPS).
  static let headOps: Set<String> = [
    "Gemm", "MatMul", "LayerNormalization", "Gelu", "Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh",
  ]
  /// Larger than that is the trunk itself (onnx_patch.HEAD_MAX_NODES).
  static let headMaxNodes = 64

  /// Where the Python's `log.warning` lines go: the cases a pass declines
  /// and leaves the graph as it was.
  private static let log = Logger(subsystem: "io.zoompilot.jetlink", category: "onnx")

  // MARK: strip_tinygrad_ops

  /// Bypasses tinygrad's layout-hint nodes and returns how many went.
  static func stripTinygradOps(_ g: inout Graph, _ opsets: inout [OpsetImport]) throws -> Int {
    let graphOutputs = Set(g.outputs.map(\.key))
    var removed = 0
    var k = 0
    while k < g.nodes.count {
      let node = g.nodes[k]
      guard node.domainName == tinygradDomain else {
        k += 1
        continue
      }
      guard passthroughOps.contains(node.op) else {
        throw OnnxError("unknown \(tinygradDomain) op \(pyRepr(node.op)); it may not be a no-op, so dropping it is not safe")
      }
      guard node.inputs.count == 1, node.outputs.count == 1, node.attributes.isEmpty else {
        throw OnnxError("\(node.op) is not a plain one-in one-out passthrough")
      }
      let source = node.inputs[0]
      let produced = node.outputs[0]
      if graphOutputs.contains(produced) {
        // The output keeps its name, so its producer takes the name over.
        rename(&g, outputs: source, to: produced)
        rename(&g, inputs: source, to: produced)
      } else {
        rename(&g, inputs: produced, to: source)
      }
      // Nothing before it moved, so it is still at k; the next node is now there.
      g.nodes.remove(at: k)
      removed += 1
    }

    if removed > 0 {
      if let i = opsets.firstIndex(where: { $0.domain == tinygradDomain }) {
        opsets.remove(at: i)
      }
      var live = Set<String>()
      for node in g.nodes {
        live.formUnion(node.inputs)
        live.formUnion(node.outputs)
      }
      g.valueInfo.removeAll { !live.contains($0.key) }
    }
    return removed
  }

  // MARK: patch_uint8_inputs

  static func needsPatch(_ g: Graph) -> Bool {
    g.inputs.contains { $0.elemType == DataType.uint8 }
  }

  /// Retypes the uint8 inputs, the images, to fp16 and drops the head Cast.
  static func patchUint8Inputs(_ g: inout Graph) throws {
    let sources = g.inputs.filter { $0.elemType == DataType.uint8 }.map(\.key)
    guard !sources.isEmpty else {
      throw OnnxError("model has no uint8 inputs; already patched?")
    }

    // _image_dataflow: nodes are in topological order, so one pass sees every
    // producer before its consumers.
    var retyped = Set(sources)
    var casts: [Int] = []
    for (k, node) in g.nodes.enumerated() where node.inputs.contains(where: retyped.contains) {
      if node.op == "Cast" {
        casts.append(k)
      } else if layoutOps.contains(node.op) {
        retyped.formUnion(node.outputs)
      } else {
        throw OnnxError(
          "\(node.op) \(pyRepr(node.displayName)) reads the uint8 images; only layout ops "
            + "and the head Cast are expected there")
      }
    }
    guard !casts.isEmpty else {
      throw OnnxError("could not find the head Cast: no Cast ends the uint8 image chain")
    }
    let graphOutputs = Set(g.outputs.map(\.key))
    for k in casts {
      let cast = g.nodes[k]
      guard let to = cast.attribute("to") else {
        throw OnnxError("head Cast \(cast.displayName) has no 'to' attribute")
      }
      guard to.i == Int64(DataType.float16) else {
        throw OnnxError("head Cast targets \(to.i), expected FLOAT16 (\(DataType.float16))")
      }
      if let out = cast.outputs.first, graphOutputs.contains(out) {
        throw OnnxError("head Cast \(cast.displayName) feeds a graph output; dropping it would rename one")
      }
    }

    // Everything downstream reads each cast's source directly now. Python
    // removes each cast right after its renames; removing them all after all
    // the renames ends the same, since a rename that reaches a cast only
    // touches a node that is dropped anyway.
    for k in casts {
      let cast = g.nodes[k]
      if let source = cast.inputs.first, let produced = cast.outputs.first {
        rename(&g, inputs: produced, to: source)
      }
    }
    for k in casts.reversed() {
      g.nodes.remove(at: k)
    }

    retype(&g.inputs, retyped)
    retype(&g.valueInfo, retyped)
    retype(&g.outputs, retyped)
  }

  private static func retype(_ values: inout [ValueInfo], _ names: Set<String>) {
    for i in values.indices where names.contains(values[i].key) {
      var type = values[i].type ?? TypeInfo()
      if type.tensor == nil {
        // Setting tensor_type in Python selects it in TypeProto's oneof,
        // which clears whichever other member was set.
        type.extras.removeAll { [4, 5, 8, 9].contains($0.number) }
        type.tensor = TensorType()
      }
      type.tensor!.elemType = DataType.float16
      values[i].type = type
    }
  }

  // MARK: normalize_gather_indices

  /// Rewrites Gathers whose constant index is negative to the positive
  /// equivalent, each with an index initializer of its own.
  static func normalizeGatherIndices(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    let gathers = g.nodes.indices.filter { k in
      let n = g.nodes[k]
      return n.op == "Gather" && n.inputs.count > 1 && initializers[n.inputs[1]] != nil
    }
    let dims = staticDims(g)
    var rewritten = 0
    for k in gathers {
      let node = g.nodes[k]
      let index = g.initializers[initializers[node.inputs[1]]!]
      guard DataType.isInteger(index.elementType) else { continue }
      let values = try Elements.integers(index, src)
      guard let lowest = values.min(), lowest < 0 else { continue }
      let axis = node.attribute("axis")?.i ?? 0
      guard let shape = dims[node.inputs[0]], axis < Int64(shape.count) else { continue }
      // Python indexes the shape tuple with the axis, so a negative one counts from the end.
      let position = axis < 0 ? Int(axis) + shape.count : Int(axis)
      guard position >= 0 else {
        throw OnnxError("\(node.displayName): Gather axis \(axis) is out of range for a rank-\(shape.count) input")
      }
      let size = shape[position]
      guard size > 0 else { continue }
      let fixed = values.map { $0 < 0 ? $0 + size : $0 }
      if fixed.contains(where: { $0 < 0 || $0 >= size }) {
        throw OnnxError(
          "\(node.displayName): Gather index \(pyList(values, index.dims)) out of range "
            + "for axis \(axis) of size \(size)")
      }
      let name = "\(node.outputs.first ?? "")__index"
      g.initializers.append(
        Tensor(
          name: name, dims: index.dims, dataType: index.dataType,
          raw: .owned(try Elements.encode(fixed, as: index.elementType, for: name))))
      g.nodes[k].inputs[1] = name
      rewritten += 1
    }
    return rewritten
  }

  // MARK: expand_to_tile

  /// Rewrites an Expand with a constant shape of the input's rank, which only
  /// repeats size-1 axes, as the equivalent Tile.
  static func expandToTile(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    let dims = staticDims(g)
    var done = 0
    for k in g.nodes.indices {
      let n = g.nodes[k]
      guard n.op == "Expand", n.inputs.count == 2, let shapeIndex = initializers[n.inputs[1]] else { continue }
      let shape = dims[n.inputs[0]]
      let target = try Elements.integers(g.initializers[shapeIndex], src)
      guard let shape, shape.count == target.count, !shape.contains(where: { $0 <= 0 }) else { continue }
      var repeats: [Int64] = []
      var repeatsOnly = true
      for (have, want) in zip(shape, target) {
        if want == 1 || want == have {
          repeats.append(1)
        } else if have == 1 {
          repeats.append(want)
        } else {
          repeatsOnly = false
          break
        }
      }
      guard repeatsOnly else { continue }
      let name = "\(n.outputs.first ?? "")__repeats"
      g.initializers.append(int64Tensor(repeats, name))
      g.nodes[k].opType = "Tile"
      g.nodes[k].inputs[1] = name
      done += 1
    }
    return done
  }

  // MARK: gemm_with_transposed_weight

  /// Rewrites `MatMul(x, W)` followed by `Add(b)` as `Gemm(x, W.T, b,
  /// transB=1)`, with a Reshape either side when x has more than two axes.
  ///
  /// The transposed weights are not made here. Each new initializer is a
  /// recipe over the original's bytes in the mapped file, carried out block
  /// by block when the part is written, so no two are ever in memory at once.
  static func gemmWithTransposedWeight(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    var consumers: [String: [Int]] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in node.inputs {
        consumers[name, default: []].append(k)
      }
    }
    let outputs = Set(g.outputs.map(\.key))
    let candidates = g.nodes.indices.filter { k in
      let n = g.nodes[k]
      return n.op == "MatMul" && n.inputs.count == 2 && initializers[n.inputs[1]] != nil
        && !(n.outputs.first.map(outputs.contains) ?? true)
    }
    let dims = staticDims(g)

    var replacements: [Int: [Node]] = [:]
    var drop = Set<Int>()
    var rewritten = 0
    for k in candidates {
      let node = g.nodes[k]
      let weight = g.initializers[initializers[node.inputs[1]]!]
      guard weight.dims.count == 2, weight.dims[0] * weight.dims[1] >= blobMinElements else { continue }
      let after = consumers[node.outputs[0]] ?? []
      guard after.count == 1, g.nodes[after[0]].op == "Add" else { continue }
      let add = g.nodes[after[0]]
      // The bias has to broadcast over the output's last axis; anything else
      // is a real elementwise Add and not a Gemm's C.
      guard let bias = add.inputs.first(where: { initializers[$0] != nil }),
        g.initializers[initializers[bias]!].dims == [weight.dims[1]]
      else { continue }
      guard let shape = dims[node.inputs[0]], shape.count >= 2, shape.last == weight.dims[0],
        !shape.contains(where: { $0 <= 0 })
      else { continue }

      let stem = node.outputs[0]
      let transposed = "\(stem)__wt"
      g.initializers.append(try transposedTensor(weight, transposed, src))

      var new: [Node] = []
      var a = node.inputs[0]
      if shape.count > 2 {
        let flat = "\(stem)__flat_shape"
        g.initializers.append(int64Tensor([-1, weight.dims[0]], flat))
        a = "\(stem)__flat"
        new.append(Node(inputs: [node.inputs[0], flat], outputs: [a], name: "\(stem)__reshape_in", opType: "Reshape"))
      }
      let gemmOut = shape.count == 2 ? add.outputs[0] : "\(stem)__gemm"
      new.append(
        Node(
          inputs: [a, transposed, bias], outputs: [gemmOut], name: "\(stem)__gemm", opType: "Gemm",
          attributes: [.int("transB", 1)]))
      if shape.count > 2 {
        let back = "\(stem)__out_shape"
        g.initializers.append(int64Tensor(Array(shape.dropLast()) + [weight.dims[1]], back))
        new.append(
          Node(
            inputs: [gemmOut, back], outputs: [add.outputs[0]], name: "\(stem)__reshape_out",
            opType: "Reshape"))
      }
      replacements[k] = new
      drop.insert(after[0])
      rewritten += 1
    }

    guard rewritten > 0 else { return 0 }

    var rebuilt: [Node] = []
    rebuilt.reserveCapacity(g.nodes.count + 2 * rewritten)
    for (k, node) in g.nodes.enumerated() where !drop.contains(k) {
      if let new = replacements[k] {
        rebuilt.append(contentsOf: new)
      } else {
        rebuilt.append(node)
      }
    }
    g.nodes = rebuilt
    dropUnusedInitializers(&g)
    return rewritten
  }

  /// The weights the rewrite left behind (_drop_unused_initializers).
  static func dropUnusedInitializers(_ g: inout Graph) {
    var used = Set<String>()
    for node in g.nodes {
      used.formUnion(node.inputs)
    }
    g.initializers.removeAll { !used.contains($0.key) }
  }

  /// `numpy_helper.from_array(np.ascontiguousarray(W.T), name)`: the same
  /// element type, the dims swapped, the data as raw_data.
  private static func transposedTensor(_ weight: Tensor, _ name: String, _ src: Source) throws -> Tensor {
    guard let size = DataType.size(weight.elementType) else {
      throw OnnxError("\(weight.key): cannot transpose a weight of element type \(weight.elementType)")
    }
    let transpose = Transpose(
      elements: try transposeElements(of: weight, src), rows: Int(weight.dims[0]), cols: Int(weight.dims[1]), elementSize: size)
    return Tensor(
      name: name, dims: [weight.dims[1], weight.dims[0]], dataType: weight.dataType,
      raw: .transposed(transpose))
  }

  /// A weight's elements as a transpose reads them while it is written:
  /// raw_data or a packed float or double field where it lies, a weight in
  /// the other typed fields decoded then.
  static func transposeElements(of weight: Tensor, _ src: Source) throws -> Transpose.Elements {
    let type = weight.elementType
    guard let size = DataType.size(type) else {
      throw OnnxError("\(weight.key): cannot transpose a weight of element type \(type)")
    }
    let expected = weight.elementCount * size
    switch weight.raw {
    case .source(let r):
      guard r.count == expected else { throw sizeMismatch(weight, r.count, expected) }
      return .source(r)
    case .owned(let bytes):
      guard bytes.count == expected else { throw sizeMismatch(weight, bytes.count, expected) }
      return .owned(bytes)
    case .transposed:
      throw OnnxError("\(weight.key): a transposed weight cannot be transposed again")
    case .widened:
      throw OnnxError("\(weight.key): a widened weight cannot be transposed")
    case nil:
      // float_data and double_data, packed in one run, are already the raw
      // layout. Anything else is decoded when it is written.
      let field = Elements.typedField(type)
      if let field, field == 4 || field == 10, type == DataType.float || type == DataType.double,
        let ranges = weight.typed[field], ranges.count == 1, ranges[0].count == expected
      {
        return .source(ranges[0])
      }
      let count = try Elements.typedCount(weight, src)
      guard count == weight.elementCount else { throw sizeMismatch(weight, count * size, expected) }
      return .typed(weight)
    }
  }

  private static func sizeMismatch(_ t: Tensor, _ have: Int, _ want: Int) -> OnnxError {
    OnnxError("initializer \(t.key) has \(have) bytes of data where its dims \(t.dims) need \(want)")
  }

  // MARK: prescale_layernorm

  /// Feeds every fp16 LayerNormalization on the policy side its input times
  /// 1/k, one Mul per distinct input, and returns how many norms.
  ///
  /// Python looks a norm's input type up in the file and asks the shape
  /// inferrer for one the file does not record; Swift has no inferrer, so a
  /// policy norm whose input type is not recorded is an error rather than a
  /// silent difference. The driving models' exports record every tensor's.
  static func prescaleLayerNorm(_ g: inout Graph, k: Int = layerNormPrescale) throws -> Int {
    let const = "__layernorm_prescale_\(k)"
    let mask = try Split.visionMask(g)
    var norms = Set<Int>()
    for (i, node) in g.nodes.enumerated() where node.op == "LayerNormalization" && !mask[i] {
      norms.insert(i)
    }
    var already = Set<String>()
    for node in g.nodes where node.op == "Mul" && node.inputs.count == 2 && node.inputs[1] == const {
      if let out = node.outputs.first { already.insert(out) }
    }
    let types = staticTypes(g)
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count + norms.count)
    var scaled: [String: String] = [:]
    var done = 0
    for (i, node) in g.nodes.enumerated() {
      var node = node
      if norms.contains(i), let x = node.inputs.first, !already.contains(x) {
        guard let type = types[x] else {
          throw OnnxError("the export records no type for \(x); this model cannot be prepared on iPhone or iPad")
        }
        if type == DataType.float16 {
          if scaled[x] == nil {
            scaled[x] = "\(x)__scaled"
            new.append(Node(inputs: [x, const], outputs: ["\(x)__scaled"], name: "\(x)__prescale", opType: "Mul"))
          }
          node.inputs[0] = scaled[x]!
          done += 1
        }
      }
      new.append(node)
    }
    if done > 0 {
      // np.float16(1 / k), a scalar: no dims, the two bytes as raw_data.
      let bits = Float16(1 / Float(k)).bitPattern.littleEndian
      g.initializers.append(
        Tensor(
          name: const, dims: [], dataType: DataType.float16,
          raw: .owned([UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)])))
      g.nodes = new
    }
    return done
  }

  // MARK: safe_layernorm

  /// Rewrites every LayerNormalization (opset 17+, one output) as
  ///
  ///     d = x - mean(x);  s = max(max|d|, 1e-2);  y = (d/s) / sqrt(mean((d/s)^2) + eps/s^2)
  ///
  /// then scale and bias: the fp16-safe form LiteRT's lowering gives a
  /// MediaTek APU (LiteRTOps.layerNorm), here as ONNX nodes for the Hexagon
  /// NPU. It is LayerNorm exactly in real arithmetic, and nothing in it
  /// leaves fp16's range: the deviations are at most 1 before squaring and
  /// their mean square at least 1/D.
  ///
  /// On a Snapdragon 8 Elite (HTP v79) carrot's N16 left the CPU by metres:
  /// with its vision norms prescaled by 1/32 the probe still found the
  /// policy's query and key norms (288 x 64, already prescaled by 1/8) 40%
  /// off and the attention scores after them 30%. One fixed scale cannot
  /// serve norms whose inputs reach 2000 and norms whose inputs are small;
  /// this form needs none. Returns how many norms.
  static func safeLayerNormOutput(_ y: String) -> String { "\(y)__safe_layernorm" }

  static func safeLayerNorm(_ g: inout Graph) -> Int {
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count * 2)
    var added: [Tensor] = []
    var done = 0
    func half(_ name: String, _ value: Float) -> String {
      let bits = Float16(value).bitPattern.littleEndian
      added.append(
        Tensor(name: name, dims: [], dataType: DataType.float16, raw: .owned([UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)])))
      return name
    }
    let floor = "__safe_layernorm_floor"
    let one = "__safe_layernorm_one"
    var constants = false
    for node in g.nodes {
      guard node.op == "LayerNormalization", node.inputs.count >= 2, node.outputs.count == 1 || node.outputs.dropFirst().allSatisfy(\.isEmpty)
      else {
        new.append(node)
        continue
      }
      if !constants {
        _ = half(floor, 1e-2)
        _ = half(one, 1)
        constants = true
      }
      let y = node.outputs[0]
      let p = "\(y)__safe"
      let axis = node.attribute("axis")?.i ?? -1
      let epsilon = node.attribute("epsilon")?.f ?? 1e-5
      // axes from `axis` to the last; a negative axis names them from the end
      let axes = "\(p)_axes"
      if axis < 0 {
        var t = int64Tensor(Array(axis...(-1)), axes)
        t.dims = [Int64(-axis)]
        added.append(t)
      } else {
        // a non-negative axis needs the rank, which the norms of the driving
        // models never use; leave such a norm as it is
        new.append(node)
        continue
      }
      let keep = [Attribute.int("keepdims", 1)]
      func n(_ op: String, _ inputs: [String], _ out: String, _ attributes: [Attribute] = []) {
        new.append(Node(inputs: inputs, outputs: [out], name: out, opType: op, attributes: attributes))
      }
      let x = node.inputs[0]
      n("ReduceMean", [x, axes], "\(p)_mean", keep)
      n("Sub", [x, "\(p)_mean"], "\(p)_d")
      n("Abs", ["\(p)_d"], "\(p)_abs")
      n("ReduceMax", ["\(p)_abs", axes], "\(p)_peak", keep)
      n("Max", ["\(p)_peak", floor], "\(p)_s")
      n("Div", [one, "\(p)_s"], "\(p)_r")
      n("Mul", ["\(p)_d", "\(p)_r"], "\(p)_dn")
      n("Mul", ["\(p)_dn", "\(p)_dn"], "\(p)_sq")
      n("ReduceMean", ["\(p)_sq", axes], "\(p)_var", keep)
      n("Mul", ["\(p)_r", "\(p)_r"], "\(p)_r2")
      n("Mul", ["\(p)_r2", half("\(p)_eps", epsilon)], "\(p)_epsr")
      n("Add", ["\(p)_var", "\(p)_epsr"], "\(p)_ve")
      n("Sqrt", ["\(p)_ve"], "\(p)_std")
      n("Div", ["\(p)_dn", "\(p)_std"], "\(p)_norm")
      let hasBias = node.inputs.count > 2 && !node.inputs[2].isEmpty
      n("Mul", ["\(p)_norm", node.inputs[1]], hasBias ? "\(p)_scaled" : y)
      if hasBias {
        n("Add", ["\(p)_scaled", node.inputs[2]], y)
      }
      // the norm's own output, which a probe looks for by this name
      new[new.count - 1].name = safeLayerNormOutput(y)
      done += 1
    }
    if done > 0 {
      g.nodes = new
      g.initializers.append(contentsOf: added)
    }
    return done
  }

  // MARK: int8_weights

  /// The weights of MatMul, Gemm and Conv with at least `minimum` elements as
  /// int8, one scale per output channel (symmetric, zero point 0), read
  /// through a DequantizeLinear to the type they had: a quarter of fp32's
  /// bytes and half of fp16's for the NPU to read each run, if QNN runs them
  /// so. Returns how many. Before the Gemm rewrite, which leaves its
  /// weights transposed and unread until written.
  static func int8Weights(_ g: inout Graph, _ src: Source, minimum: Int = 65_536) throws -> Int {
    var inits: [String: Int] = [:]
    for (i, t) in g.initializers.enumerated() { inits[t.key] = i }
    var readers: [String: Int] = [:]
    for n in g.nodes { for x in n.inputs { readers[x, default: 0] += 1 } }
    var done: [String: String] = [:]
    var added: [Tensor] = []
    var dequant: [Node] = []
    var replaced = Set<String>()
    for (k, node) in g.nodes.enumerated() where ["MatMul", "Gemm", "Conv"].contains(node.op) && node.inputs.count >= 2 {
      let name = node.inputs[1]
      if let made = done[name] {
        g.nodes[k].inputs[1] = made
        continue
      }
      guard let i = inits[name] else { continue }
      let t = g.initializers[i]
      guard t.elementType == DataType.float16 || t.elementType == DataType.float, t.elementCount >= minimum, t.dims.count >= 2 else { continue }
      if case .transposed = t.raw { continue }
      if case .widened = t.raw { continue }
      // the output channel's axis: B's columns, a Gemm's rows when transB, a Conv's first
      let axis: Int
      switch node.op {
      case "MatMul":
        guard t.dims.count == 2 else { continue }
        axis = 1
      case "Gemm": axis = (node.attribute("transB")?.i ?? 0) == 1 ? 0 : 1
      default: axis = 0
      }
      let values = try Elements.floats(Elements.littleEndian(t, src), as: t.elementType, for: name)
      let dims = t.dims.map(Int.init)
      let channels = dims[axis]
      let inner = dims[(axis + 1)...].reduce(1, *)
      let outer = dims[..<axis].reduce(1, *)
      var amax = [Float](repeating: 0, count: channels)
      for o in 0..<outer {
        for c in 0..<channels {
          let base = (o * channels + c) * inner
          for j in 0..<inner { amax[c] = max(amax[c], abs(values[base + j])) }
        }
      }
      let scale = amax.map { $0 > 0 ? $0 / 127 : 1 }
      var q = [UInt8](repeating: 0, count: values.count)
      for o in 0..<outer {
        for c in 0..<channels {
          let base = (o * channels + c) * inner
          let r = 1 / scale[c]
          for j in 0..<inner {
            let v = Int(max(-127, min(127, (values[base + j] * r).rounded())))
            q[base + j] = UInt8(bitPattern: Int8(v))
          }
        }
      }
      let qName = "\(name)__int8"
      let sName = "\(name)__int8_scale"
      let zName = "\(name)__int8_zero"
      let out = "\(name)__dequantized"
      added.append(Tensor(name: qName, dims: t.dims, dataType: DataType.int8, raw: .owned(q)))
      added.append(Tensor(name: sName, dims: [Int64(channels)], dataType: t.elementType, raw: .owned(Elements.encode(scale, as: t.elementType))))
      added.append(Tensor(name: zName, dims: [Int64(channels)], dataType: DataType.int8, raw: .owned([UInt8](repeating: 0, count: channels))))
      dequant.append(Node(inputs: [qName, sName, zName], outputs: [out], name: out, opType: "DequantizeLinear", attributes: [.int("axis", Int64(axis))]))
      done[name] = out
      g.nodes[k].inputs[1] = out
      replaced.insert(name)
    }
    guard !done.isEmpty else { return 0 }
    // the fp weights a rewritten node alone read go
    var still = Set<String>()
    for n in g.nodes { still.formUnion(n.inputs) }
    g.initializers.removeAll { replaced.contains($0.key) && !still.contains($0.key) }
    g.initializers.append(contentsOf: added)
    g.nodes = dequant + g.nodes
    return done.count
  }

  // MARK: fold_input_normalization

  /// Folds `Div(Sub(x, mean), std)` into the convolution that alone reads it,
  /// when that convolution pads nothing (a patchify stem): conv(W, (x-m)/s)
  /// is conv(W/s, x) + b - sum(W m/s), exactly, with no padding to tell the
  /// two apart. `mean` and `std` are per input channel ([1, C, 1, 1]). The
  /// new weights are made in fp32 and stored as the old ones were. Returns
  /// how many.
  ///
  /// carrot's N16 normalizes its 24-channel image stack ([1, 24, 128, 256])
  /// this way before a 4x4 stride-4 stem; on a Snapdragon 8 Gen 2 the Sub
  /// and the Div took 1.35 M of the NPU's 85 M cycles a run.
  static func foldInputNormalization(_ g: inout Graph, _ src: Source) throws -> Int {
    var producer: [String: Int] = [:]
    var readers: [String: [Int]] = [:]
    for (i, n) in g.nodes.enumerated() {
      for o in n.outputs { producer[o] = i }
      for x in n.inputs { readers[x, default: []].append(i) }
    }
    let outputs = Set(g.outputs.map(\.key))
    var inits: [String: Int] = [:]
    for (i, t) in g.initializers.enumerated() { inits[t.key] = i }
    func floats(_ name: String) throws -> (values: [Float], type: Int32, dims: [Int64])? {
      guard let i = inits[name] else { return nil }
      let t = g.initializers[i]
      guard t.elementType == DataType.float16 || t.elementType == DataType.float else { return nil }
      return (try Elements.floats(Elements.littleEndian(t, src), as: t.elementType, for: name), t.elementType, t.dims)
    }
    var drop = Set<Int>()
    var done = 0
    for (c, conv) in g.nodes.enumerated() where conv.op == "Conv" && conv.inputs.count >= 2 {
      let pads = conv.attribute("pads")?.ints ?? []
      let autoPad = conv.attribute("auto_pad")?.s ?? "NOTSET"
      guard pads.allSatisfy({ $0 == 0 }), autoPad == "NOTSET" || autoPad == "VALID", (conv.attribute("group")?.i ?? 1) == 1,
        let d = producer[conv.inputs[0]], g.nodes[d].op == "Div", readers[conv.inputs[0]] == [c], !outputs.contains(conv.inputs[0]),
        let sb = producer[g.nodes[d].inputs[0]], g.nodes[sb].op == "Sub", readers[g.nodes[d].inputs[0]] == [d],
        let std = try floats(g.nodes[d].inputs[1]), let mean = try floats(g.nodes[sb].inputs[1]),
        let w = try floats(conv.inputs[1]), w.dims.count == 4
      else { continue }
      let (o, ch, kh, kw) = (Int(w.dims[0]), Int(w.dims[1]), Int(w.dims[2]), Int(w.dims[3]))
      guard std.values.count == ch, mean.values.count == ch, std.values.allSatisfy({ $0 != 0 }) else { continue }
      var bias = [Float](repeating: 0, count: o)
      var biasType = w.type
      if conv.inputs.count > 2, !conv.inputs[2].isEmpty {
        guard let b = try floats(conv.inputs[2]), b.values.count == o else { continue }
        bias = b.values
        biasType = b.type
      }
      var weights = w.values
      let k = kh * kw
      for oc in 0..<o {
        var shift: Float = 0
        for ic in 0..<ch {
          let scale = 1 / std.values[ic]
          for j in 0..<k {
            let at = (oc * ch + ic) * k + j
            shift += weights[at] * mean.values[ic] * scale
            weights[at] *= scale
          }
        }
        bias[oc] -= shift
      }
      let wName = "\(conv.inputs[1])__folded"
      let bName = "\(conv.displayName)__folded_bias"
      g.initializers.append(Tensor(name: wName, dims: w.dims, dataType: w.type, raw: .owned(Elements.encode(weights, as: w.type))))
      g.initializers.append(Tensor(name: bName, dims: [Int64(o)], dataType: biasType, raw: .owned(Elements.encode(bias, as: biasType))))
      var node = g.nodes[c]
      node.inputs = [g.nodes[sb].inputs[0], wName, bName]
      g.nodes[c] = node
      drop.formUnion([d, sb])
      done += 1
    }
    if !drop.isEmpty {
      g.nodes = g.nodes.enumerated().filter { !drop.contains($0.offset) }.map(\.element)
    }
    return done
  }

  // MARK: fp16_state_io

  /// Makes a stateful graph's fp32 queues (`state_X` in, `next_state_X` out)
  /// fp16 at the graph's edge, with a Cast inside each, and returns how many
  /// pairs. The graph computes as before; only what crosses into and out of
  /// it each run halves.
  ///
  /// A Hexagon NPU computes fp16 (enable_htp_fp16_precision), so every value
  /// such a queue holds was fp16 already: the fp32 queue only doubled what
  /// onnxruntime moved between the CPU's memory and the NPU's every frame,
  /// which N16's 128 x 16384 feature queue makes 8.4 MB each way. On a
  /// Galaxy Tab S9, whose FastRPC has no shared memory for the NPU, about
  /// 3.5 ms of Light Speed 2's 31 ms a run was outside the NPU's execute.
  static func fp16StateIO(_ g: inout Graph) -> Int {
    let outputs = Dictionary(uniqueKeysWithValues: g.outputs.enumerated().map { ($1.key, $0) })
    var pairs: [(input: Int, output: Int)] = []
    for (i, vi) in g.inputs.enumerated() where vi.key.hasPrefix("state_") && vi.elemType == DataType.float {
      if let o = outputs["next_" + vi.key], g.outputs[o].elemType == DataType.float {
        pairs.append((i, o))
      }
    }
    guard !pairs.isEmpty else { return 0 }
    var rename: [String: String] = [:]
    var added: [Node] = []
    var tail: [Node] = []
    for (i, o) in pairs {
      let input = g.inputs[i].key
      let output = g.outputs[o].key
      // the graph reads the fp32 queue from a Cast of the fp16 input
      rename[input] = "\(input)__fp32"
      added.append(
        Node(inputs: [input], outputs: ["\(input)__fp32"], name: "\(input)__to_fp32", opType: "Cast", attributes: [.int("to", Int64(DataType.float))]))
      // and writes the next one through a Cast to the fp16 output
      rename[output] = "\(output)__fp32"
      tail.append(
        Node(inputs: ["\(output)__fp32"], outputs: [output], name: "\(output)__to_fp16", opType: "Cast", attributes: [.int("to", Int64(DataType.float16))]))
      g.inputs[i].type?.tensor?.elemType = DataType.float16
      g.outputs[o].type?.tensor?.elemType = DataType.float16
    }
    for n in g.nodes.indices {
      g.nodes[n].inputs = g.nodes[n].inputs.map { rename[$0] ?? $0 }
      g.nodes[n].outputs = g.nodes[n].outputs.map { rename[$0] ?? $0 }
    }
    let renamed = Set(rename.keys)
    g.valueInfo.removeAll { renamed.contains($0.key) }
    g.nodes = added + g.nodes + tail
    return pairs.count
  }

  // MARK: tanh_gelu

  static func tanhGeluOutput(_ y: String) -> String { "\(y)__tanh_gelu" }

  /// Rewrites every exact (erf) Gelu as its tanh form, in ops a Hexagon NPU
  /// runs fast:
  ///
  ///     c = clip(x, -10, 10);  y = 0.5 x (1 + tanh(c (0.7978846 + 0.0356774 c^2)))
  ///
  /// The tanh form is within about 3e-4 of the exact one everywhere, and past
  /// |x| = 10 both are x or 0 to fp16's precision, so the clip loses nothing
  /// and keeps c^3 inside fp16. On a Snapdragon 8 Gen 2 QNN's profile of
  /// carrot's N16 put its 11 policy GELUs ([1, 288, 2048]) at about 29% of
  /// the NPU's cycles. Returns how many.
  static func tanhGelu(_ g: inout Graph) -> Int {
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count * 2)
    var added: [Tensor] = []
    var done = 0
    var constants = false
    func half(_ name: String, _ value: Float) {
      let bits = Float16(value).bitPattern.littleEndian
      added.append(
        Tensor(name: name, dims: [], dataType: DataType.float16, raw: .owned([UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)])))
    }
    let lo = "__tanh_gelu_lo", hi = "__tanh_gelu_hi", c1 = "__tanh_gelu_c1", c3 = "__tanh_gelu_c3", one = "__tanh_gelu_one",
      halfName = "__tanh_gelu_half"
    let types = staticTypes(g)
    for node in g.nodes {
      // fp16 exact Gelu only: an fp32 one keeps its precision as it is
      guard node.op == "Gelu", node.inputs.count == 1, node.outputs.count == 1, (node.attribute("approximate")?.s ?? "none") == "none",
        types[node.inputs[0]] == DataType.float16
      else {
        new.append(node)
        continue
      }
      if !constants {
        half(lo, -10)
        half(hi, 10)
        half(c1, 0.7978846)
        half(c3, 0.0356774)
        half(one, 1)
        half(halfName, 0.5)
        constants = true
      }
      let x = node.inputs[0]
      let y = node.outputs[0]
      let p = "\(y)__tg"
      func n(_ op: String, _ inputs: [String], _ out: String) {
        new.append(Node(inputs: inputs, outputs: [out], name: out, opType: op))
      }
      n("Clip", [x, lo, hi], "\(p)_c")
      n("Mul", ["\(p)_c", "\(p)_c"], "\(p)_c2")
      n("Mul", ["\(p)_c2", c3], "\(p)_k")
      n("Add", ["\(p)_k", c1], "\(p)_s")
      n("Mul", ["\(p)_c", "\(p)_s"], "\(p)_inner")
      n("Tanh", ["\(p)_inner"], "\(p)_t")
      n("Add", ["\(p)_t", one], "\(p)_t1")
      n("Mul", [x, halfName], "\(p)_hx")
      n("Mul", ["\(p)_hx", "\(p)_t1"], y)
      new[new.count - 1].name = tanhGeluOutput(y)
      done += 1
    }
    if done > 0 {
      g.nodes = new
      g.initializers.append(contentsOf: added)
    }
    return done
  }

  // MARK: prescale_vision_layernorm

  /// How much `prescaleVisionLayerNorm` divides a vision norm's input by.
  static let visionLayerNormPrescale: Float = 32

  /// Feeds every fp16 LayerNormalization the policy prescale left alone (the
  /// vision trunk's) its input times 1/k and its epsilon divided by k², so it
  /// computes the same norm, and returns how many norms.
  ///
  /// carrot's N16 reaches its vision norms with values up to about 2000 on a
  /// real road: 24 of its 44 have a sum of squares over the normalized axis
  /// past fp16's 65504, 22 a single square past it. A Snapdragon 8 Gen 2
  /// (HTP v73) ran N16 right that way; on an 8 Elite (v79) its plan and lead
  /// were metres from the CPU's on the same frames. Divided by 32 the worst
  /// sum is about 4000. LayerNorm(x / k) with epsilon / k² is LayerNorm(x)
  /// with epsilon exactly.
  static func prescaleVisionLayerNorm(_ g: inout Graph, k: Float = visionLayerNormPrescale) throws -> Int {
    let const = "__layernorm_vision_prescale_\(Int(k))"
    let policyConst = "__layernorm_prescale_\(layerNormPrescale)"
    var scaledByPolicy = Set<String>()
    for node in g.nodes where node.op == "Mul" && node.inputs.count == 2 && (node.inputs[1] == policyConst || node.inputs[1] == const) {
      if let out = node.outputs.first { scaledByPolicy.insert(out) }
    }
    let types = staticTypes(g)
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count + 64)
    var scaled: [String: String] = [:]
    var done = 0
    for node in g.nodes {
      var node = node
      if node.op == "LayerNormalization", let x = node.inputs.first, !scaledByPolicy.contains(x), types[x] == DataType.float16 {
        if scaled[x] == nil {
          scaled[x] = "\(x)__vision_scaled"
          new.append(Node(inputs: [x, const], outputs: ["\(x)__vision_scaled"], name: "\(x)__vision_prescale", opType: "Mul"))
        }
        node.inputs[0] = scaled[x]!
        let epsilon = node.attribute("epsilon")?.f ?? 1e-5
        node.attributes.removeAll { $0.name == "epsilon" }
        node.attributes.append(.floatValue("epsilon", epsilon / (k * k)))
        done += 1
      }
      new.append(node)
    }
    if done > 0 {
      let bits = Float16(1 / k).bitPattern.littleEndian
      g.initializers.append(
        Tensor(
          name: const, dims: [], dataType: DataType.float16,
          raw: .owned([UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)])))
      g.nodes = new
    }
    return done
  }

  // MARK: probes

  /// Makes graph outputs of tensors to compare between two runtimes: the
  /// outputs of the request's ops whose value info is static and float, at
  /// most `limit` of them evenly over the graph, or the request's names that
  /// the graph has so. Returns them in graph order with their op.
  static func addProbes(_ g: inout Graph, _ request: CoreMLPreparation.ProbeRequest) -> [(name: String, op: String)] {
    var info: [String: ValueInfo] = [:]
    for vi in g.valueInfo + g.inputs where vi.elemType == DataType.float16 || vi.elemType == DataType.float {
      guard let dims = vi.shape?.dims, !dims.isEmpty, dims.allSatisfy({ ($0.value ?? 0) > 0 }) else { continue }
      info[vi.key] = vi
    }
    let outputs = Set(g.outputs.map(\.key))
    var candidates: [(name: String, op: String)] = []
    let wanted = request.names.map(Set.init)
    for n in g.nodes {
      // a norm in its fp16-safe form is still a norm here
      let op =
        n.outputs.count == 1 && n.displayName == safeLayerNormOutput(n.outputs[0])
        ? "LayerNormalization" : n.outputs.count == 1 && n.displayName == tanhGeluOutput(n.outputs[0]) ? "Gelu" : n.op
      for out in n.outputs where info[out] != nil && !outputs.contains(out) {
        if let wanted {
          if wanted.contains(out) { candidates.append((out, op)) }
        } else if request.ops.contains(op) {
          candidates.append((out, op))
        }
      }
    }
    var chosen = candidates
    if request.names == nil, candidates.count > request.limit, request.limit > 0 {
      let step = Double(candidates.count) / Double(request.limit)
      chosen = (0..<request.limit).map { candidates[Int(Double($0) * step)] }
    }
    for c in chosen { g.outputs.append(info[c.name]!) }
    return chosen
  }

  // MARK: finite_softmax_masks

  /// What a masked attention score becomes instead of fp16 -inf. Any score
  /// left unmasked is within a few hundred of it at most, so exp(fill - max)
  /// underflows to 0 in fp16 exactly as exp(-inf) does.
  static let softmaxMaskFill = Float16(-10000)

  /// Gives every Softmax that reads `Where(cond, a, b)`, where a or b is an
  /// fp16 initializer holding -inf alone, a finite copy of that fill
  /// (`<name>__finite`), and returns how many Where inputs changed.
  ///
  /// carrot's N16 masks its 12 temporal attentions so. A row that keeps an
  /// unmasked score (a causal mask always keeps the diagonal) gives the same
  /// Softmax either way; onnxruntime's CPU gives N16 bit for bit the same
  /// outputs and state. QNN's HTP on a Snapdragon 8 Elite (v79) did not
  /// treat the -inf as 0: N16's plan, lead and speed came out as fresh noise
  /// every frame while its lanes stayed put. Only a fully masked row differs
  /// (NaN before, uniform weights after), and the outputs' finite check
  /// fails such a frame before either reaches the comma anyway.
  static func finiteSoftmaxMasks(_ g: inout Graph, _ src: Source) throws -> Int {
    var producer: [String: Int] = [:]
    for (i, n) in g.nodes.enumerated() {
      for out in n.outputs { producer[out] = i }
    }
    var initializers: [String: Tensor] = [:]
    for t in g.initializers { initializers[t.key] = t }
    let negInf: (UInt8, UInt8) = (0x00, 0xFC)  // fp16 -inf, little-endian
    var finite: [String: String] = [:]
    var added: [Tensor] = []
    var changed = 0
    for softmax in g.nodes where softmax.op == "Softmax" {
      guard let x = softmax.inputs.first, let w = producer[x], g.nodes[w].op == "Where", g.nodes[w].inputs.count == 3 else { continue }
      for j in 1...2 {
        let name = g.nodes[w].inputs[j]
        if let made = finite[name] {
          g.nodes[w].inputs[j] = made
          changed += 1
          continue
        }
        guard let t = initializers[name], t.elementType == DataType.float16, t.elementCount > 0 else { continue }
        let bytes = try Elements.littleEndian(t, src)
        let allNegInf = stride(from: 0, to: bytes.count, by: 2).allSatisfy { bytes[$0] == negInf.0 && bytes[$0 + 1] == negInf.1 }
        guard allNegInf else { continue }
        let made = "\(name)__finite"
        let bits = softmaxMaskFill.bitPattern.littleEndian
        let one = [UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)]
        added.append(
          Tensor(
            name: made, dims: t.dims, dataType: DataType.float16,
            raw: .owned(Array([[UInt8]](repeating: one, count: t.elementCount).joined()))))
        finite[name] = made
        g.nodes[w].inputs[j] = made
        changed += 1
      }
    }
    g.initializers.append(contentsOf: added)
    return changed
  }

  // MARK: additive_softmax_masks

  /// Rewrites every Softmax input `Where(c, a, b)` whose condition is a
  /// constant bool mask, `M` or `Not(M)`, and whose one other branch is an
  /// fp16 initializer holding -inf alone, as `Add(scores, bias)`: `bias` an
  /// fp16 constant of the mask's shape, 0 where the scores pass and
  /// `softmaxMaskFill` where the fill did. Returns how many Wheres changed.
  ///
  /// carrot's N16 masks its 12 temporal attentions with a fixed causal
  /// `_supercombo_mask` (bool, 1x1x288x288) through Not and Where. On a
  /// Snapdragon 8 Elite (HTP v79) its plan still jumped metres in a frame
  /// while driving after the -inf fill went finite, and the benchmark,
  /// whose fixed input makes every remembered frame alike, cannot see a
  /// wrong mask: any mix of equal rows is the same row. Adding a constant
  /// leaves the HTP no bool tensor, Not or Where to get wrong. A passing
  /// score gains 0; a masked one, at most a few hundred, ends near -10000,
  /// whose exp is 0 in fp16 as the fill's was. onnxruntime's CPU gives N16
  /// the same outputs both ways.
  static func additiveSoftmaxMasks(_ g: inout Graph, _ src: Source) throws -> Int {
    var producer: [String: Int] = [:]
    for (i, n) in g.nodes.enumerated() {
      for out in n.outputs { producer[out] = i }
    }
    var initializers: [String: Tensor] = [:]
    for t in g.initializers { initializers[t.key] = t }
    func negInfFill(_ name: String) throws -> Bool {
      guard let t = initializers[name], t.elementType == DataType.float16, t.elementCount == 1 else { return false }
      let bytes = try Elements.littleEndian(t, src)
      return bytes.count == 2 && bytes[0] == 0x00 && bytes[1] == 0xFC
    }
    let pass = [UInt8](repeating: 0, count: 2)
    let fillBits = softmaxMaskFill.bitPattern.littleEndian
    let fill = [UInt8(truncatingIfNeeded: fillBits), UInt8(truncatingIfNeeded: fillBits >> 8)]
    var biases: [[UInt8]: String] = [:]
    var added: [Tensor] = []
    var changed = 0
    for softmax in g.nodes where softmax.op == "Softmax" {
      guard let x = softmax.inputs.first, let w = producer[x], g.nodes[w].op == "Where", g.nodes[w].inputs.count == 3 else { continue }
      let wh = g.nodes[w]
      // The bool mask and whether the condition is its negation.
      var maskName = wh.inputs[0]
      var negated = false
      if initializers[maskName] == nil, let c = producer[maskName], g.nodes[c].op == "Not", g.nodes[c].inputs.count == 1 {
        maskName = g.nodes[c].inputs[0]
        negated = true
      }
      guard let mask = initializers[maskName], mask.elementType == DataType.bool, mask.elementCount > 0 else { continue }
      // Which branch is the -inf fill; the other carries the scores.
      let fillWhenTrue: Bool
      let scores: String
      if try negInfFill(wh.inputs[1]), initializers[wh.inputs[2]] == nil {
        fillWhenTrue = true
        scores = wh.inputs[2]
      } else if try negInfFill(wh.inputs[2]), initializers[wh.inputs[1]] == nil {
        fillWhenTrue = false
        scores = wh.inputs[1]
      } else {
        continue
      }
      let bits = try Elements.littleEndian(mask, src)
      guard bits.count == mask.elementCount else { continue }
      var raw = [UInt8]()
      raw.reserveCapacity(bits.count * 2)
      for b in bits {
        let condition = (b != 0) != negated
        raw.append(contentsOf: condition == fillWhenTrue ? fill : pass)
      }
      let name: String
      if let made = biases[raw] {
        name = made
      } else {
        name = "jetlink_softmax_mask_bias_\(biases.count)"
        added.append(Tensor(name: name, dims: mask.dims, dataType: DataType.float16, raw: .owned(raw)))
        biases[raw] = name
      }
      g.nodes[w] = Node(inputs: [scores, name], outputs: wh.outputs, name: wh.name, opType: "Add", domain: wh.domain)
      changed += 1
    }
    guard changed > 0 else { return 0 }
    g.initializers.append(contentsOf: added)
    // The Nots that only fed the rewritten Wheres.
    var read = Set(g.outputs.map(\.key))
    for n in g.nodes { read.formUnion(n.inputs) }
    g.nodes.removeAll { $0.op == "Not" && !$0.outputs.contains(where: read.contains) }
    return changed
  }

  // MARK: vision_heads

  /// The indices, in graph order, of the heads that end the vision trunk:
  /// grown backwards to a fixed point from the Concats that make a graph
  /// output. Empty when there is no such Concat or the set would exceed
  /// headMaxNodes.
  static func visionHeads(_ g: Graph) throws -> [Int] {
    let mask = try Split.visionMask(g)
    let outputs = Set(g.outputs.map(\.key))
    var ends = Set<Int>()
    for (i, n) in g.nodes.enumerated() where n.op == "Concat" && n.outputs.contains(where: outputs.contains) {
      ends.insert(i)
    }
    guard !ends.isEmpty else { return [] }
    var readers: [String: Set<Int>] = [:]
    for (i, n) in g.nodes.enumerated() {
      for x in n.inputs {
        readers[x, default: []].insert(i)
      }
    }
    var region = Set<Int>()
    var grew = true
    while grew {
      grew = false
      for i in g.nodes.indices.reversed() {
        let n = g.nodes[i]
        if region.contains(i) || !mask[i] || !headOps.contains(n.op) || n.outputs.contains(where: outputs.contains) {
          continue
        }
        let allowed = region.union(ends)
        let joins = n.outputs.allSatisfy { o in
          guard let r = readers[o], !r.isEmpty else { return false }
          return r.isSubset(of: allowed)
        }
        if joins {
          region.insert(i)
          grew = true
        }
      }
    }
    return region.count <= headMaxNodes ? region.sorted() : []
  }

  // MARK: heads_in_fp32

  /// Runs `visionHeads` in fp32: a Cast up before the first head reads each
  /// entry, fp32 copies of the fp16 head weights, a Cast down after each
  /// exit. Returns how many nodes moved, 0 with a warning when it found no
  /// heads it could move.
  ///
  /// The fp32 copies are not made here: each is a recipe over the fp16
  /// bytes, carried out band by band when the part is written.
  static func headsInFP32(_ g: inout Graph, _ src: Source) throws -> Int {
    let index = try visionHeads(g)
    guard !index.isEmpty else {
      log.warning(
        "heads_in_fp32: no heads found after the vision trunk (no output Concat they feed, or more than \(headMaxNodes) nodes); the whole graph stays fp16"
      )
      return 0
    }
    let heads = Set(index)
    let initializers = lastIndexByName(g.initializers)
    var produced = Set<String>()
    for i in index {
      produced.formUnion(g.nodes[i].outputs)
    }
    var entries = Set<String>()
    var weights = Set<String>()
    for i in index {
      for x in g.nodes[i].inputs where !x.isEmpty {
        if initializers[x] != nil {
          weights.insert(x)
        } else if !produced.contains(x) {
          entries.insert(x)
        }
      }
    }
    let types = staticTypes(g)
    // Python asks the shape inferrer for an entry the file does not type;
    // Swift has none (see prescaleLayerNorm).
    if let untyped = entries.filter({ types[$0] == nil }).sorted(by: pyLess).first {
      throw OnnxError("the export records no type for \(untyped); this model cannot be prepared on iPhone or iPad")
    }
    let notFP16 = entries.filter { types[$0] != DataType.float16 }.sorted(by: pyLess)
    guard notFP16.isEmpty else {
      log.warning("heads_in_fp32: a head reads \(pyStrList(notFP16)), which is not fp16; the whole graph stays fp16")
      return 0
    }
    let odd = weights.filter {
      let type = g.initializers[initializers[$0]!].elementType
      return type != DataType.float16 && type != DataType.float
    }.sorted(by: pyLess)
    guard odd.isEmpty else {
      log.warning("heads_in_fp32: head weights \(pyStrList(odd)) are neither fp16 nor fp32; the whole graph stays fp16")
      return 0
    }
    var readOutside = Set<String>()
    for (j, n) in g.nodes.enumerated() where !heads.contains(j) {
      readOutside.formUnion(n.inputs)
    }
    let exits = produced.intersection(readOutside)
    var wide: [String: String] = [:]
    for w in weights.sorted(by: pyLess) {
      let weight = g.initializers[initializers[w]!]
      guard weight.elementType == DataType.float16 else { continue }
      wide[w] = "\(w)__fp32"
      if initializers[wide[w]!] == nil {
        g.initializers.append(try widenedTensor(weight, wide[w]!, src))
      }
    }
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count + entries.count + exits.count)
    var cast = Set<String>()
    for (j, node) in g.nodes.enumerated() {
      guard heads.contains(j) else {
        new.append(node)
        continue
      }
      var n = node
      for i in n.inputs.indices {
        let x = n.inputs[i]
        if entries.contains(x) {
          if !cast.contains(x) {
            cast.insert(x)
            new.append(
              Node(
                inputs: [x], outputs: ["\(x)__fp32"], name: "\(x)__cast_fp32", opType: "Cast",
                attributes: [.int("to", Int64(DataType.float))]))
          }
          n.inputs[i] = "\(x)__fp32"
        } else if exits.contains(x) {
          n.inputs[i] = "\(x)__fp32"
        } else if let w = wide[x] {
          n.inputs[i] = w
        }
      }
      let back = n.outputs.filter(exits.contains)
      for i in n.outputs.indices where exits.contains(n.outputs[i]) {
        n.outputs[i] = "\(n.outputs[i])__fp32"
      }
      new.append(n)
      for o in back {
        new.append(
          Node(
            inputs: ["\(o)__fp32"], outputs: [o], name: "\(o)__cast_fp16", opType: "Cast",
            attributes: [.int("to", Int64(DataType.float16))]))
      }
    }
    g.nodes = new
    // The fp16 originals, unless something outside the heads reads them too.
    g.initializers.removeAll { wide[$0.key] != nil && !readOutside.contains($0.key) }
    // What the heads compute inside is fp32 now; the value infos said fp16.
    let inside = produced.subtracting(exits)
    g.valueInfo.removeAll { inside.contains($0.key) }
    return index.count
  }

  /// `numpy_helper.from_array(numpy_helper.to_array(w).astype(np.float32),
  /// name)`: the same dims, FLOAT, the data as raw_data.
  private static func widenedTensor(_ weight: Tensor, _ name: String, _ src: Source) throws -> Tensor {
    let count = weight.elementCount
    let expected = count * 2
    let elements: Widen.Elements
    switch weight.raw {
    case .source(let r):
      guard r.count == expected else { throw sizeMismatch(weight, r.count, expected) }
      elements = .source(r)
    case .owned(let bytes):
      guard bytes.count == expected else { throw sizeMismatch(weight, bytes.count, expected) }
      elements = .owned(bytes)
    case .transposed(let t):
      guard t.byteCount == expected, t.elementSize == 2 else { throw sizeMismatch(weight, t.byteCount, expected) }
      elements = .transposed(t)
    case .widened:
      throw OnnxError("\(weight.key): a widened weight cannot be widened again")
    case nil:
      let have = try Elements.typedCount(weight, src)
      guard have == count else { throw sizeMismatch(weight, have * 2, expected) }
      elements = .typed(weight)
    }
    return Tensor(name: name, dims: weight.dims, dataType: DataType.float, raw: .widened(Widen(elements: elements, count: count)))
  }

  // MARK: helpers

  /// Static shapes from what the file carries (_static_dims), inputs first,
  /// then value_info, then outputs, a later entry replacing an earlier one.
  /// A dimension without a value is -1.
  static func staticDims(_ g: Graph) -> [String: [Int64]] {
    var dims: [String: [Int64]] = [:]
    for vi in g.inputs + g.valueInfo + g.outputs {
      if let shape = vi.shape {
        dims[vi.key] = shape.dims.map { $0.value ?? -1 }
      }
    }
    return dims
  }

  /// Element types from what the file carries (_static_types), in the same
  /// order as `staticDims`. A value info without an element type is absent.
  static func staticTypes(_ g: Graph) -> [String: Int32] {
    var types: [String: Int32] = [:]
    for vi in g.inputs + g.valueInfo + g.outputs where vi.elemType != 0 {
      types[vi.key] = vi.elemType
    }
    return types
  }

  /// `{t.name: t for t in g.initializer}` as positions: a repeated name maps
  /// to its last tensor, as a dict built that way does.
  static func lastIndexByName(_ tensors: [Tensor]) -> [String: Int] {
    var map: [String: Int] = [:]
    for (i, t) in tensors.enumerated() {
      map[t.key] = i
    }
    return map
  }

  static func int64Tensor(_ values: [Int64], _ name: String) -> Tensor {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(values.count * 8)
    for v in values {
      withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) }
    }
    return Tensor(name: name, dims: [Int64(values.count)], dataType: DataType.int64, raw: .owned(bytes))
  }

  private static func rename(_ g: inout Graph, inputs from: String, to: String) {
    for k in g.nodes.indices {
      for i in g.nodes[k].inputs.indices where g.nodes[k].inputs[i] == from {
        g.nodes[k].inputs[i] = to
      }
    }
  }

  private static func rename(_ g: inout Graph, outputs from: String, to: String) {
    for k in g.nodes.indices {
      for i in g.nodes[k].outputs.indices where g.nodes[k].outputs[i] == from {
        g.nodes[k].outputs[i] = to
      }
    }
  }
}

// MARK: Python's formatting, for error messages that match

/// Python's repr of a str.
func pyRepr(_ s: String) -> String {
  let quote: Character = (s.contains("'") && !s.contains("\"")) ? "\"" : "'"
  var out = String(quote)
  for scalar in s.unicodeScalars {
    switch scalar {
    case "\\": out += "\\\\"
    case "\n": out += "\\n"
    case "\r": out += "\\r"
    case "\t": out += "\\t"
    default:
      if Character(scalar) == quote {
        out += "\\\(quote)"
      } else if scalar.value < 0x20 || scalar.value == 0x7f {
        out += String(format: "\\x%02x", scalar.value)
      } else {
        out.unicodeScalars.append(scalar)
      }
    }
  }
  out.append(quote)
  return out
}

/// numpy's `tolist()` of integers, as Python prints it.
func pyList(_ values: [Int64], _ dims: [Int64]) -> String {
  guard let first = dims.first else { return values.first.map(String.init) ?? "[]" }
  let rest = Array(dims.dropFirst())
  let stride = max(1, rest.reduce(1) { $0 * Int($1) })
  let items = (0..<Int(first)).map { i -> String in
    if rest.isEmpty { return String(values[i]) }
    return pyList(Array(values[(i * stride)..<((i + 1) * stride)]), rest)
  }
  return "[" + items.joined(separator: ", ") + "]"
}

/// Python's print of a list of str.
func pyStrList(_ values: [String]) -> String {
  "[" + values.map(pyRepr).joined(separator: ", ") + "]"
}

/// Python's sort order for str: by code point, with no normalisation.
func pyLess(_ a: String, _ b: String) -> Bool {
  a.unicodeScalars.lexicographicallyPrecedes(b.unicodeScalars) { $0.value < $1.value }
}
