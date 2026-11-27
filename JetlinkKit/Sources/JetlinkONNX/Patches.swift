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
    // A Cast to uint8 on the uint8 chain changes nothing and goes with the
    // head Casts: BM Nav (carrot.62) wraps the queued BMRLNAP in N16's
    // stateful interface and hands the queue's frames to `img` and `big_img`
    // through Cast(to: UINT8); left in, it would turn the fp16 images back.
    var identities: [Int] = []
    for (k, node) in g.nodes.enumerated() where node.inputs.contains(where: retyped.contains) {
      if node.op == "Cast", node.attribute("to")?.i == Int64(DataType.uint8) {
        identities.append(k)
        retyped.formUnion(node.outputs)
      } else if node.op == "Cast" {
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
    for k in identities {
      if let out = g.nodes[k].outputs.first, graphOutputs.contains(out) {
        throw OnnxError("uint8 Cast \(g.nodes[k].displayName) feeds a graph output; dropping it would rename one")
      }
    }
    let dropped = (casts + identities).sorted()
    for k in dropped {
      let cast = g.nodes[k]
      if let source = cast.inputs.first, let produced = cast.outputs.first {
        rename(&g, inputs: produced, to: source)
      }
    }
    for k in dropped.reversed() {
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
  /// Ops that act on each element alone, so a reshape around them changes nothing.
  static let elementwiseUnary: Set<String> = ["Gelu", "Relu", "Sigmoid", "Tanh", "Erf"]

  /// Where `gemmWithTransposedWeight` left a Gemm's output reshaped back to
  /// [.., C] only for one elementwise op and a reshape to [-1, C] into the
  /// next Gemm (a ConvNeXt MLP: fc1 -> Gelu -> fc2), the op reads the first
  /// Gemm's [N, C] and the second Gemm reads the op's: both reshapes go.
  /// Row-major [.., C] flattened to [-1, C] is the Gemm's own [N, C], so the
  /// values are the same. Returns how many pairs went.
  static func flatElementwiseBetweenGemms(_ g: inout Graph) -> Int {
    var consumers: [String: [Int]] = [:]
    var producer: [String: Int] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in node.inputs { consumers[name, default: []].append(k) }
      for name in node.outputs { producer[name] = k }
    }
    let outputs = Set(g.outputs.map(\.key))
    var drop = Set<Int>()
    var gone = Set<String>()
    var fused = 0
    for (k, back) in g.nodes.enumerated() where back.op == "Reshape" && (back.name ?? "").hasSuffix("__reshape_out") {
      guard let gemmOut = back.inputs.first, let p = producer[gemmOut], g.nodes[p].op == "Gemm",
        let wide = back.outputs.first, !outputs.contains(wide), let c1 = consumers[wide], c1.count == 1
      else { continue }
      let u = c1[0]
      guard Self.elementwiseUnary.contains(g.nodes[u].op), g.nodes[u].inputs.count == 1, g.nodes[u].outputs.count == 1 else { continue }
      let unaryOut = g.nodes[u].outputs[0]
      guard !outputs.contains(unaryOut), let c2 = consumers[unaryOut], c2.count == 1 else { continue }
      let flat = g.nodes[c2[0]]
      guard flat.op == "Reshape", (flat.name ?? "").hasSuffix("__reshape_in"), let flatOut = flat.outputs.first,
        let next = consumers[flatOut], next.allSatisfy({ g.nodes[$0].op == "Gemm" })
      else { continue }
      // [-1, C] with the C the Gemm gave: the same stem's shapes
      g.nodes[u].inputs = [gemmOut]
      g.nodes[u].outputs = [flatOut]
      drop.formUnion([k, c2[0]])
      gone.formUnion([wide, unaryOut])
      fused += 1
    }
    guard fused > 0 else { return 0 }
    g.nodes = g.nodes.enumerated().filter { !drop.contains($0.offset) }.map(\.element)
    // the 4-D shapes recorded for the tensors that went, and none for the op's new output
    g.valueInfo.removeAll { gone.contains($0.key) }
    dropUnusedInitializers(&g)
    return fused
  }

  /// ConvNeXt's layer scale (`gamma * x` after the MLP's fc2) taken into fc2:
  /// where a Gemm the rewrite gave a transposed weight is reshaped back,
  /// transposed to NCHW and multiplied by a constant [1, C, 1, 1] (or [C])
  /// with no other reader on the way, each of the weight's output rows and
  /// the bias go times their channel's scale (in fp32, rounded back to fp16
  /// as the weight is written) and the Mul goes. BMRLNAP on a Galaxy Tab S9:
  /// the Muls were 11.7% of the NPU's cycles (carrot.59's profile). Not the
  /// same values: each scaled weight rounds once more. Returns how many.
  static func foldLayerScale(_ g: inout Graph, _ src: Source) throws -> Int {
    var consumers: [String: [Int]] = [:]
    var producer: [String: Int] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in node.inputs { consumers[name, default: []].append(k) }
      for name in node.outputs { producer[name] = k }
    }
    let initializers = lastIndexByName(g.initializers)
    let outputs = Set(g.outputs.map(\.key))
    func only(_ name: String) -> Int? {
      guard !outputs.contains(name), let c = consumers[name], c.count == 1 else { return nil }
      return c[0]
    }
    var drop = Set<Int>()
    var gone = Set<String>()
    var folded = 0
    for (m, mul) in g.nodes.enumerated() where mul.op == "Mul" && mul.inputs.count == 2 {
      // which input is the scale, which the transposed activations
      guard let si = mul.inputs.firstIndex(where: { initializers[$0] != nil }) else { continue }
      let scaleName = mul.inputs[si]
      let x = mul.inputs[1 - si]
      guard let t = producer[x], g.nodes[t].op == "Transpose", only(x) == m,
        g.nodes[t].attribute("perm")?.ints == [0, 3, 1, 2],
        let r = producer[g.nodes[t].inputs[0]], g.nodes[r].op == "Reshape", only(g.nodes[r].outputs[0]) == t,
        let k = producer[g.nodes[r].inputs[0]], g.nodes[k].op == "Gemm", only(g.nodes[k].outputs[0]) == r,
        g.nodes[k].inputs.count == 3, g.nodes[k].attribute("transB")?.i == 1,
        let wi = initializers[g.nodes[k].inputs[1]], let bi = initializers[g.nodes[k].inputs[2]],
        consumers[g.nodes[k].inputs[1]]?.count == 1, consumers[g.nodes[k].inputs[2]]?.count == 1,
        consumers[scaleName]?.count == 1,
        case .transposed(var tr) = g.initializers[wi].raw, tr.rowScale == nil
      else { continue }
      let scaleTensor = g.initializers[initializers[scaleName]!]
      let channels = tr.cols
      guard scaleTensor.elementCount == channels,
        scaleTensor.dims == [1, Int64(channels), 1, 1] || scaleTensor.dims == [Int64(channels)]
      else { continue }
      let scale = try Elements.floats(Elements.littleEndian(scaleTensor, src), as: scaleTensor.elementType, for: scaleName)
      let bias = g.initializers[bi]
      guard bias.elementCount == channels else { continue }
      let b = try Elements.floats(Elements.littleEndian(bias, src), as: bias.elementType, for: bias.key)
      tr.rowScale = scale
      g.initializers[wi].raw = .transposed(tr)
      g.initializers[bi].raw = .owned(Elements.encode(zip(b, scale).map { $0 * $1 }, as: bias.elementType))
      g.initializers[bi].typed = [:]
      // the Transpose writes what the Mul wrote
      gone.insert(x)
      g.nodes[t].outputs = mul.outputs
      drop.insert(m)
      folded += 1
    }
    guard folded > 0 else { return 0 }
    g.nodes = g.nodes.enumerated().filter { !drop.contains($0.offset) }.map(\.element)
    g.valueInfo.removeAll { gone.contains($0.key) }
    dropUnusedInitializers(&g)
    return folded
  }

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
    let types = inferredTypes(g)
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

  // MARK: guard_policy_layernorm

  /// The row peak a guarded norm's input is brought down to for a normalized
  /// axis of `d` values: d·T² = 32768, so the sum of squares fp16 adds up
  /// stays under half of 65504 whatever the row, and 8 for d = 512.
  static func layerNormGuardPeak(_ d: Int64?) -> Float {
    guard let d, d > 0 else { return 8 }
    return max(1, min(64, (32768 / Float(d)).squareRoot()))
  }

  /// Feeds every policy LayerNormalization that `prescaleLayerNorm` scaled
  /// by 1/8 its input times T / max(max|x|, T) along the normalized axes,
  /// one row at a time (`layerNormGuardPeak`), and returns how many norms.
  ///
  /// carrot's N16 on a real road reaches its temporal transformer's
  /// residual norms (288 x 512) with values up to about 2160: over 1/8 a
  /// single square is 73000 and a row's sum of squares 121000, past fp16's
  /// 65504, so a Hexagon NPU's fp16 norm overflows there; the benchmark's
  /// random camera bytes keep the same norms under 1470 (sum 57000) and never
  /// show it. A row already under T is left exactly as it was; a larger row
  /// is scaled so its peak is T. LayerNorm does not see the scale but for its
  /// epsilon, about 1e-4 of those rows' variance at most.
  static var policyPrescaleConst: String { "__layernorm_prescale_\(layerNormPrescale)" }

  /// Which of the policy's prescaled norms `guardPolicyLayerNorm` guards.
  enum GuardSelection: Sendable {
    /// Every one (carrot.49/50).
    case all
    /// Those reading the residual stream: an Add's output (or a reshape of
    /// one) at least 256 wide. A query or key norm (288 x 64, from a Split)
    /// never gets past fp16 on a road and keeps the shipped norm as it is.
    case residual
    /// Those named: on a PC over 40 real road frames N16's norms whose input
    /// over 1/8 peaked past 128 (`n16HotNorms`).
    case named(Set<String>)

    func takes(norm: String, feeder: String?, width: Int64?) -> Bool {
      switch self {
      case .all: true
      case .residual: (feeder == "Add" || feeder?.hasSuffix("<Add") == true) && (width ?? 0) >= 256
      case .named(let names): names.contains(norm)
      }
    }
  }

  /// carrot's N16 (01effc679eb46d60): the policy norms whose input over 1/8
  /// peaked past 128 on a PC over real road frames (past 224, fp16's square,
  /// for 67, 70, 71, 74, 75, 78 and their second copies, and 54); the rest
  /// stay under 8.
  static let n16HotNorms: Set<String> = Set(
    [54, 57, 63, 66, 67, 70, 71, 74, 75, 78, 86, 89, 90, 93, 94, 97, 98, 101].map { "node_layer_norm_\($0)" })

  /// `fixedPeak` instead of `layerNormGuardPeak`: a high one (224) touches
  /// only the rows past fp16's square, so on the benchmark's random bytes
  /// the values are the shipped ones and only the ops are new (carrot.50-test:
  /// the guard at 8 put N16 on a Y700 4x further from the CPU than without).
  static func guardPolicyLayerNorm(_ g: inout Graph, fixedPeak: Float? = nil, only select: GuardSelection = .all) -> Int {
    let policyConst = policyPrescaleConst
    // what writes each norm's own input, a reshaping op looked through once
    var producer: [String: Node] = [:]
    for node in g.nodes { for o in node.outputs { producer[o] = node } }
    func feeder(_ x: String) -> String? {
        guard let p = producer[x] else { return nil }
        if ["Transpose", "Reshape", "Squeeze", "Unsqueeze"].contains(p.op), let i = p.inputs.first, let q = producer[i] {
            return "\(p.op)<\(q.op)"
        }
        return p.op
    }
    var policyScaled = Set<String>()
    for node in g.nodes where node.op == "Mul" && node.inputs.count == 2 && node.inputs[1] == policyConst {
      if let out = node.outputs.first { policyScaled.insert(out) }
    }
    guard !policyScaled.isEmpty else { return 0 }
    let dims = staticDims(g)
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count + 6 * 80)
    var added: [Tensor] = []
    var guarded: [String: String] = [:]
    var done = 0
    func half(_ name: String, _ value: Float) {
      let bits = Float16(value).bitPattern.littleEndian
      added.append(
        Tensor(name: name, dims: [], dataType: DataType.float16, raw: .owned([UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)])))
    }
    for node in g.nodes {
      var node = node
      let axis = node.attribute("axis")?.i ?? -1
      // a non-negative axis needs the rank, which the driving models' norms never use
      guard node.op == "LayerNormalization", let x = node.inputs.first, policyScaled.contains(x), axis < 0 else {
        new.append(node)
        continue
      }
      let original = x.hasSuffix("__scaled") ? String(x.dropLast("__scaled".count)) : x
      // the Mul's output has no value info; its input, the norm's own, does
      let shape = dims[x] ?? dims[original]
      var d: Int64? = nil
      if let shape, shape.count >= Int(-axis), shape.suffix(Int(-axis)).allSatisfy({ $0 > 0 }) {
        d = shape.suffix(Int(-axis)).reduce(1, *)
      }
      guard select.takes(norm: node.name ?? "", feeder: feeder(original), width: d) else {
        new.append(node)
        continue
      }
      if let y = guarded[x] {
        node.inputs[0] = y
        new.append(node)
        done += 1
        continue
      }
      let p = "\(x)__guard"
      let peak = fixedPeak ?? layerNormGuardPeak(d)
      let axes = "\(p)_axes"
      var t = int64Tensor(Array(axis...(-1)), axes)
      t.dims = [Int64(-axis)]
      added.append(t)
      half("\(p)_peak", peak)
      func n(_ op: String, _ inputs: [String], _ out: String, _ attributes: [Attribute] = []) {
        new.append(Node(inputs: inputs, outputs: [out], name: out, opType: op, attributes: attributes))
      }
      n("Abs", [x], "\(p)_abs")
      n("ReduceMax", ["\(p)_abs", axes], "\(p)_rowmax", [Attribute.int("keepdims", 1)])
      n("Max", ["\(p)_rowmax", "\(p)_peak"], "\(p)_s")
      n("Div", ["\(p)_peak", "\(p)_s"], "\(p)_c")
      n("Mul", [x, "\(p)_c"], "\(p)_x")
      guarded[x] = "\(p)_x"
      node.inputs[0] = "\(p)_x"
      new.append(node)
      done += 1
    }
    if done > 0 {
      g.nodes = new
      g.initializers.append(contentsOf: added)
    }
    return done
  }

  /// The guard folded into the policy prescale itself: each `x * 1/8` that
  /// feeds a norm becomes `x * peak / max(max|x|, 8 peak)` per row, one Mul as
  /// before, so the norm still reads a Mul's output (QNN may fuse the pair).
  /// A row whose peak stays under 8 peak gets exactly 1/8 again. Returns how
  /// many prescale Muls.
  static func guardPolicyPrescale(_ g: inout Graph, peak: Float) -> Int {
    let policyConst = policyPrescaleConst
    // the axes each prescaled tensor is normalized over, from the norms reading it
    var axisOf: [String: Int64] = [:]
    for node in g.nodes where node.op == "LayerNormalization" {
      if let x = node.inputs.first { axisOf[x] = node.attribute("axis")?.i ?? -1 }
    }
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count + 5 * 80)
    var added: [Tensor] = []
    var done = 0
    func half(_ name: String, _ value: Float) {
      let bits = Float16(value).bitPattern.littleEndian
      added.append(
        Tensor(name: name, dims: [], dataType: DataType.float16, raw: .owned([UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)])))
    }
    let top = "__layernorm_prescale_guard_peak"
    let floor = "__layernorm_prescale_guard_floor"
    for node in g.nodes {
      guard node.op == "Mul", node.inputs.count == 2, node.inputs[1] == policyConst, let out = node.outputs.first,
        let axis = axisOf[out], axis < 0
      else {
        new.append(node)
        continue
      }
      if done == 0 {
        half(top, peak)
        half(floor, peak * Float(layerNormPrescale))
      }
      let x = node.inputs[0]
      let p = "\(out)__guard"
      let axes = "\(p)_axes"
      var t = int64Tensor(Array(axis...(-1)), axes)
      t.dims = [Int64(-axis)]
      added.append(t)
      func n(_ op: String, _ inputs: [String], _ o: String, _ attributes: [Attribute] = []) {
        new.append(Node(inputs: inputs, outputs: [o], name: o, opType: op, attributes: attributes))
      }
      n("Abs", [x], "\(p)_abs")
      n("ReduceMax", ["\(p)_abs", axes], "\(p)_rowmax", [Attribute.int("keepdims", 1)])
      n("Max", ["\(p)_rowmax", floor], "\(p)_s")
      n("Div", [top, "\(p)_s"], "\(p)_c")
      var mul = node
      mul.inputs[1] = "\(p)_c"
      new.append(mul)
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
  /// so. Returns how many. Last of the patches, as the norms' vision mask
  /// reads the graph as exported; a weight the Gemm rewrite transposes is
  /// transposed here first.
  /// Each DequantizeLinear sits just before what reads it: carrot.40-test
  /// put them all first and a Galaxy Tab S9's NPU, holding every widened
  /// weight at once, could not create the context (QNN error 1002).
  static func int8Weights(_ g: inout Graph, _ src: Source, minimum: Int = 65_536) throws -> Int {
    var inits: [String: Int] = [:]
    for (i, t) in g.initializers.enumerated() { inits[t.key] = i }
    var readers: [String: Int] = [:]
    for n in g.nodes { for x in n.inputs { readers[x, default: 0] += 1 } }
    var done: [String: String] = [:]
    var added: [Tensor] = []
    var dequant: [Int: [Node]] = [:]
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
      let values: [Float]
      if case .transposed(let tr) = t.raw {
        // the Gemm rewrite's [rows, cols] weight, as the [cols, rows] it writes
        let bytes: [UInt8]
        switch tr.elements {
        case .source(let r): bytes = Array(src.slice(r))
        case .owned(let b): bytes = b
        case .typed: continue
        }
        let v = try Elements.floats(bytes, as: t.elementType, for: name)
        var out = [Float](repeating: 0, count: v.count)
        for r in 0..<tr.rows {
          for c in 0..<tr.cols { out[c * tr.rows + r] = v[r * tr.cols + c] }
        }
        values = out
      } else {
        values = try Elements.floats(Elements.littleEndian(t, src), as: t.elementType, for: name)
      }
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
      dequant[k, default: []].append(
        Node(inputs: [qName, sName, zName], outputs: [out], name: out, opType: "DequantizeLinear", attributes: [.int("axis", Int64(axis))]))
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
    var nodes: [Node] = []
    nodes.reserveCapacity(g.nodes.count + done.count)
    for (k, n) in g.nodes.enumerated() {
      nodes.append(contentsOf: dequant[k] ?? [])
      nodes.append(n)
    }
    g.nodes = nodes
    return done.count
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
    let types = inferredTypes(g)
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
  ///
  /// With `fold` (and `src` to read weights from), a norm whose input is a
  /// convolution's output alone, through a Transpose (N16's ConvNeXt blocks:
  /// Conv -> Transpose -> LayerNorm), takes the 1/k in that convolution's
  /// weight and bias instead of a Mul: the same values divided by a power of
  /// two, exact in fp16 but where one falls below its normal range.
  static func prescaleVisionLayerNorm(
    _ g: inout Graph, _ src: Source? = nil, k: Float = visionLayerNormPrescale, fold: Bool = false
  ) throws -> Int {
    let const = "__layernorm_vision_prescale_\(Int(k))"
    let policyConst = "__layernorm_prescale_\(layerNormPrescale)"
    let divided = "__vision_div_\(Int(k))"
    var scaledByPolicy = Set<String>()
    for node in g.nodes where node.op == "Mul" && node.inputs.count == 2 && (node.inputs[1] == policyConst || node.inputs[1] == const) {
      if let out = node.outputs.first { scaledByPolicy.insert(out) }
    }
    let types = inferredTypes(g)
    var producer: [String: Int] = [:]
    var uses: [String: Int] = [:]
    for (i, n) in g.nodes.enumerated() {
      for o in n.outputs { producer[o] = i }
      for input in n.inputs where !input.isEmpty { uses[input, default: 0] += 1 }
    }
    let outputs = Set(g.outputs.map(\.key))
    /// The convolution whose output alone, through one Transpose, is `x`.
    func convolution(feeding x: String) -> Int? {
      guard uses[x] == 1, !outputs.contains(x), let t = producer[x], g.nodes[t].op == "Transpose",
        let c = g.nodes[t].inputs.first, uses[c] == 1, !outputs.contains(c),
        let conv = producer[c], g.nodes[conv].op == "Conv", g.nodes[conv].inputs.count == 3
      else { return nil }
      return conv
    }
    // norms a fold before this one already took the 1/k for
    var folded = Set<String>()
    for node in g.nodes where node.op == "LayerNormalization" {
      if let x = node.inputs.first, let conv = convolution(feeding: x), g.nodes[conv].inputs[1].hasSuffix(divided) {
        folded.insert(x)
      }
    }
    var foldNow = Set<String>()
    var convInputs: [Int: [String]] = [:]
    var added: [Tensor] = []
    var originals = Set<String>()
    if fold, let src {
      let initializers = Dictionary(g.initializers.map { ($0.key, $0) }, uniquingKeysWith: { a, _ in a })
      for node in g.nodes where node.op == "LayerNormalization" {
        guard let x = node.inputs.first, !scaledByPolicy.contains(x), !folded.contains(x), types[x] == DataType.float16,
          let conv = convolution(feeding: x), convInputs[conv] == nil,
          let w = initializers[g.nodes[conv].inputs[1]], let b = initializers[g.nodes[conv].inputs[2]],
          w.elementType == DataType.float16, b.elementType == DataType.float16
        else { continue }
        func scaled(_ t: Tensor) throws -> Tensor {
          let values = try Elements.floats(Elements.littleEndian(t, src), as: t.elementType, for: t.key)
          return Tensor(
            name: "\(t.key)__\(conv)\(divided)", dims: t.dims, dataType: DataType.float16,
            raw: .owned(Elements.encode(values.map { $0 / k }, as: DataType.float16)))
        }
        // a weight kept where the preparation cannot read it keeps its Mul
        guard let w2 = try? scaled(w), let b2 = try? scaled(b) else { continue }
        added += [w2, b2]
        originals.formUnion([w.key, b.key])
        var inputs = g.nodes[conv].inputs
        inputs[1] = w2.key
        inputs[2] = b2.key
        convInputs[conv] = inputs
        foldNow.insert(x)
      }
    }
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count + 64)
    var scaled: [String: String] = [:]
    var done = 0
    var multiplied = false
    for (i, node) in g.nodes.enumerated() {
      var node = node
      if let inputs = convInputs[i] { node.inputs = inputs }
      if node.op == "LayerNormalization", let x = node.inputs.first, !scaledByPolicy.contains(x), !folded.contains(x),
        types[x] == DataType.float16
      {
        if foldNow.contains(x) {
          let epsilon = node.attribute("epsilon")?.f ?? 1e-5
          node.attributes.removeAll { $0.name == "epsilon" }
          node.attributes.append(.floatValue("epsilon", epsilon / (k * k)))
          done += 1
          new.append(node)
          continue
        }
        multiplied = true
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
      if multiplied && !g.initializers.contains(where: { $0.key == const }) {
        let bits = Float16(1 / k).bitPattern.littleEndian
        g.initializers.append(
          Tensor(
            name: const, dims: [], dataType: DataType.float16,
            raw: .owned([UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)])))
      }
      g.initializers.append(contentsOf: added)
      g.nodes = new
      // the weights the divided copies replaced, where nothing else reads them
      if !originals.isEmpty {
        let read = Set(g.nodes.flatMap(\.inputs))
        g.initializers.removeAll { originals.contains($0.key) && !read.contains($0.key) }
      }
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
    let types = inferredTypes(g)
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

  /// Ops whose outputs take the type of their first typed data input, as
  /// onnx's type inference gives them.
  static let typeFollowsInput: Set<String> = [
    "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Neg", "Abs", "Sqrt", "Exp", "Log", "Reciprocal", "Erf",
    "Relu", "Gelu", "Sigmoid", "Tanh", "Softmax", "LayerNormalization", "MatMul", "Gemm", "Conv", "ConvTranspose",
    "ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "Clip", "Transpose", "Reshape", "Squeeze", "Unsqueeze",
    "Flatten", "Expand", "Tile", "Identity", "Concat", "Split", "Slice", "Gather", "GatherND", "Pad", "Resize",
    "AveragePool", "MaxPool", "GlobalAveragePool", "BatchNormalization", "InstanceNormalization", "Dropout",
  ]
  /// Ops whose outputs are bool, and the int64 ones.
  static let boolOutputs: Set<String> = ["Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Not", "And", "Or", "Xor", "IsNaN", "IsInf"]
  static let int64Outputs: Set<String> = ["Shape", "Size", "NonZero", "ArgMax", "ArgMin"]

  /// `staticTypes`, and for a node output the file does not type, the type
  /// onnx's inference would give it from the node's inputs (Cast its `to`,
  /// Where its values', a Constant its tensor's, most ops their first typed
  /// data input's), in graph order. Python's preparation asks onnx's shape
  /// inferrer for these; LS2 Nav (carrot.62) merged N16's map and navigation
  /// encoders in without their 177 tensors' types, and its policy norms could
  /// not be prescaled ("the export records no type for nav/permute_83"). A
  /// recorded type always stands.
  static func inferredTypes(_ g: Graph) -> [String: Int32] {
    var types = staticTypes(g)
    for t in g.initializers where t.elementType != 0 && types[t.key] == nil {
      types[t.key] = t.elementType
    }
    for node in g.nodes {
      let missing = node.outputs.filter { !$0.isEmpty && types[$0] == nil }
      guard !missing.isEmpty else { continue }
      var type: Int32?
      switch node.op {
      case "Cast":
        type = node.attribute("to").map { Int32(truncatingIfNeeded: $0.i) }
      case "Where":
        type = node.inputs.dropFirst().lazy.compactMap { types[$0] }.first
      case "Constant":
        type = node.attribute("value")?.t?.elementType
      case _ where boolOutputs.contains(node.op):
        type = DataType.bool
      case _ where int64Outputs.contains(node.op):
        type = DataType.int64
      case _ where typeFollowsInput.contains(node.op):
        type = node.inputs.lazy.filter { !$0.isEmpty }.compactMap { types[$0] }.first
      default:
        type = nil
      }
      guard let type, type != 0 else { continue }
      // LayerNormalization's and Dropout's other outputs (mean, mask) are not
      // what is read here; only the first is typed for them.
      let typedOutputs = ["LayerNormalization", "Dropout", "BatchNormalization"].contains(node.op) ? [node.outputs[0]] : node.outputs
      for name in typedOutputs where !name.isEmpty && types[name] == nil {
        types[name] = type
      }
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
