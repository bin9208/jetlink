import Foundation

/// onnx_patch.split_vision_policy: the graph cut where the image-only trunk
/// ends, as (vision, policy), each part made the way
/// `onnx.utils.Extractor.extract_model` makes it.
enum Split {
  static func visionPolicy(_ model: Model) throws -> (vision: Model, policy: Model) {
    guard let g = model.graph else { throw OnnxError("the model has no graph") }
    let mask = try visionMask(g)
    var made = Set<String>()
    for (node, vision) in zip(g.nodes, mask) where vision {
      made.formUnion(node.outputs)
    }
    let policyNodes = zip(g.nodes, mask).filter { !$0.1 }.map(\.0)
    let images = g.inputs.map(\.key).filter(Patches.imageInputs.contains)
    let others = g.inputs.map(\.key).filter { !Patches.imageInputs.contains($0) }
    if policyNodes.contains(where: { $0.inputs.contains(where: images.contains) }) {
      throw OnnxError("a policy node reads the image inputs directly; the graph has no clean vision trunk")
    }
    var handedSet = Set<String>()
    for node in policyNodes {
      handedSet.formUnion(node.inputs.filter(made.contains))
    }
    let handed = handedSet.sorted(by: pyLess)
    guard !handed.isEmpty else {
      throw OnnxError("nothing crosses from the vision trunk to the policy")
    }
    let ends = trunkEnd(g, mask, images: images, handed: handed).map { [$0] } ?? handed
    let outputs = g.outputs.map(\.key)

    // The hand-off tensors become graph inputs and outputs, which need a type
    // and a fixed shape. Python runs onnx's shape inferrer for one the export
    // does not record; there is none here.
    let typed = Set((g.inputs + g.valueInfo + g.outputs).filter { $0.shape != nil }.map(\.key))
    if let missing = ends.first(where: { !typed.contains($0) }) {
      throw OnnxError("the export records no shape for \(missing); this model cannot be prepared on iPhone or iPad")
    }

    let extractor = try Extractor(model)
    let vision = try extractor.extract(inputs: images, outputs: ends + outputs.filter(made.contains))
    let policy = try extractor.extract(inputs: ends + others, outputs: outputs.filter { !made.contains($0) })
    return (vision, policy)
  }

  /// The model in `count` chunks run one after the other, cut inside the
  /// vision trunk where a single tensor alone crosses (a block's output),
  /// each cut where the weights read so far come nearest an equal share.
  /// Chunk i reads the cut before it and the graph inputs its outputs reach
  /// back to, and gives the cut after it and any graph output it makes; the last
  /// holds the rest of the trunk and the policy. Fewer chunks when the trunk
  /// has fewer such places; one when it has none.
  ///
  /// For the Hexagon NPU: QNN compiles a 1.76 GB model in one piece with
  /// more memory than a 12 GB phone has free, and a chunk at a time with a
  /// share of it. The nodes, their order and the weights are the model's.
  static func chunks(_ model: Model, count: Int) throws -> [Model] {
    guard count > 1, let g = model.graph else { return [model] }
    let cuts = chunkCuts(g, count: count)
    guard !cuts.isEmpty else { return [model] }
    let extractor = try Extractor(model)
    let graphInputs = g.inputs.map(\.key)
    let graphOutputs = g.outputs.map(\.key)
    var producedAt: [String: Int] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in node.outputs where !name.isEmpty { producedAt[name] = k }
    }
    let graphInputSet = Set(graphInputs)
    var models: [Model] = []
    var start = 0
    var previous: String?
    for (end, tensor) in cuts.map({ ($0.index, $0.tensor) }) + [(g.nodes.count - 1, nil as String?)] {
      let made = graphOutputs.filter { producedAt[$0].map { (start...end).contains($0) } ?? false }
      let outputs = (tensor.map { [$0] } ?? []) + made
      // The graph inputs the chunk's outputs reach back to, past the cut
      // before: a node that works on graph inputs alone (a queue's Slice
      // and Cast) is made again in each chunk that reads it, wherever it
      // sits in the graph's order.
      var reached = Set<String>()
      var seen = Set<String>()
      var stack = outputs
      while let name = stack.popLast() {
        guard seen.insert(name).inserted, name != previous else { continue }
        if graphInputSet.contains(name) {
          reached.insert(name)
        } else if let k = producedAt[name] {
          stack.append(contentsOf: g.nodes[k].inputs.filter { !$0.isEmpty })
        }
      }
      let inputs = (previous.map { [$0] } ?? []) + graphInputs.filter(reached.contains)
      models.append(try extractor.extract(inputs: inputs, outputs: outputs))
      previous = tensor
      start = end + 1
    }
    return models
  }

  /// Tensors made from the graph's inputs other than the images, and
  /// initializers, alone (and those inputs): what any chunk can make again
  /// for itself, so they do not count as crossing a cut. BM Nav's wrapper
  /// slices and casts its desire and feature queues ahead of the trunk.
  static func sideTensors(_ g: Graph) -> Set<String> {
    let initializers = Set(g.initializers.map(\.key))
    var side = Set(g.inputs.map(\.key).filter { !Patches.imageInputs.contains($0) })
    for node in g.nodes {
      let data = node.inputs.filter { !$0.isEmpty && !initializers.contains($0) }
      if data.allSatisfy(side.contains) {
        side.formUnion(node.outputs.filter { !$0.isEmpty })
      }
    }
    return side
  }

  /// Where `chunks` cuts: (the node after which, the one tensor crossing),
  /// in order.
  static func chunkCuts(_ g: Graph, count: Int) -> [(index: Int, tensor: String)] {
    guard let mask = try? visionMask(g) else { return [] }
    var weightOf: [String: Int] = [:]
    for t in g.initializers { weightOf[t.key] = t.raw?.count ?? 0 }
    var lastUse: [String: Int] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in node.inputs where !name.isEmpty { lastUse[name] = k }
    }
    // A graph output crosses a cut only if a later node reads it: the chunk
    // that makes it gives it out (`chunks`). BM Nav's wrapper makes
    // next_state_img_q first; counted live to the end, it left no cut.
    let outputs = Set(g.outputs.map(\.key))
    let shaped = Set(
      (g.inputs + g.valueInfo + g.outputs).filter { vi in
        guard let dims = vi.shape?.dims else { return false }
        return dims.allSatisfy { $0.value != nil }
      }.map(\.key))

    // Weights read up to and including each node, each counted at its first reader.
    var counted = Set<String>()
    var weight = 0
    var weightAt: [Int] = []
    for node in g.nodes {
      for name in node.inputs where !counted.contains(name) {
        if let w = weightOf[name] {
          counted.insert(name)
          weight += w
        }
      }
      weightAt.append(weight)
    }
    let total = weight

    var candidates: [(index: Int, tensor: String, weight: Int)] = []
    var live = Set<String>()
    let side = sideTensors(g)
    for (k, node) in g.nodes.enumerated() {
      live.formUnion(node.outputs.filter { !$0.isEmpty && !side.contains($0) })
      live = live.filter { (lastUse[$0] ?? -1) > k }
      if mask[k], live.count == 1, let t = live.first, shaped.contains(t), !outputs.contains(t) {
        candidates.append((k, t, weightAt[k]))
      }
    }
    var chosen: [(index: Int, tensor: String)] = []
    for j in 1..<count {
      let share = total * j / count
      guard let best = candidates.filter({ $0.index > (chosen.last?.index ?? -1) }).min(by: { abs($0.weight - share) < abs($1.weight - share) })
      else { break }
      chosen.append((best.index, best.tensor))
    }
    return chosen
  }

  /// Per node, in graph order: whether it depends on the image inputs alone
  /// (_vision_mask). Nodes reading only initializers count as neither.
  static func visionMask(_ g: Graph) throws -> [Bool] {
    let initializers = Set(g.initializers.map(\.key))
    let imageInputs = Set(g.inputs.map(\.key).filter(Patches.imageInputs.contains))
    guard !imageInputs.isEmpty else {
      throw OnnxError("model has none of \(Patches.imageInputsRepr) as graph inputs")
    }
    var visionTensors = imageInputs
    return g.nodes.map { node in
      let data = node.inputs.filter { !$0.isEmpty && !initializers.contains($0) }
      let vision = !data.isEmpty && data.allSatisfy(visionTensors.contains)
      if vision {
        visionTensors.formUnion(node.outputs)
      }
      return vision
    }
  }

  /// The latest vision tensor every one of `handed` is computed from and
  /// nothing else of the images (_trunk_end), or nil. The check comes before
  /// each step, as in the Python, so a set that narrows to one tensor only
  /// at the very first node is not found.
  static func trunkEnd(_ g: Graph, _ mask: [Bool], images: [String], handed: [String]) -> String? {
    let initializers = Set(g.initializers.map(\.key))
    let never = Set(g.outputs.map(\.key)).union(images)
    var live = Set(handed)
    for (node, vision) in zip(g.nodes, mask).reversed() {
      if live.count == 1, live.isDisjoint(with: never) {
        return live.first
      }
      if vision, !live.isDisjoint(with: node.outputs) {
        live.subtract(node.outputs)
        live.formUnion(node.inputs.filter { !$0.isEmpty && !initializers.contains($0) })
      }
    }
    return nil
  }
}

/// onnx.utils.Extractor (onnx 1.22), for a model already in memory.
///
/// What each extracted model gets, as extract_model builds it:
/// - nodes: those reachable backwards from the outputs, stopping at the
///   inputs, in the original order;
/// - initializers and value_info: those named by any of those nodes' inputs or
///   outputs, in the order of the name maps below. The value_info map is the
///   graph's value_info updated with its inputs and then its outputs, so it
///   holds the graph inputs and outputs too, and they land in value_info;
/// - inputs and outputs: the map's entries for the names asked for;
/// - the graph named `Extracted from {<name>}`, with no doc_string or
///   metadata_props, and no sparse initializers or quantization annotations
///   (either one in the source is an error);
/// - the model: ir_version and opset_import copied, producer_name
///   "onnx.utils.extract_model", the local functions the nodes call, and
///   nothing else: no producer_version, domain, model_version, doc_string or
///   metadata_props. The COREML_CACHE_KEY is added afterwards.
struct Extractor {
  private let model: Model
  private let graph: Graph
  private let initializerOrder: [String]
  private let initializers: [String: Tensor]
  private let valueInfoOrder: [String]
  private let valueInfos: [String: ValueInfo]
  private let producers: [String: Int]

  init(_ model: Model) throws {
    guard let graph = model.graph else { throw OnnxError("the model has no graph") }
    self.model = model
    self.graph = graph

    // Python dicts: a repeated name keeps its first position and its last value.
    var order: [String] = []
    var tensors: [String: Tensor] = [:]
    for t in graph.initializers {
      if tensors.updateValue(t, forKey: t.key) == nil { order.append(t.key) }
    }
    initializerOrder = order
    initializers = tensors

    var viOrder: [String] = []
    var vis: [String: ValueInfo] = [:]
    for vi in graph.valueInfo + graph.inputs + graph.outputs {
      if vis.updateValue(vi, forKey: vi.key) == nil { viOrder.append(vi.key) }
    }
    valueInfoOrder = viOrder
    valueInfos = vis

    var producers: [String: Int] = [:]
    for (k, node) in graph.nodes.enumerated() {
      for name in node.outputs where !name.isEmpty {
        guard producers[name] == nil else { throw OnnxError("two nodes produce \(name)") }
        producers[name] = k
      }
    }
    self.producers = producers
  }

  func extract(inputs: [String], outputs: [String]) throws -> Model {
    let newInputs = try collectIO(inputs)
    let newOutputs = try collectIO(outputs)
    let nodes = reachableNodes(inputs: inputs, outputs: outputs).map { graph.nodes[$0] }

    var names = Set<String>()
    for node in nodes {
      names.formUnion(node.inputs)
      names.formUnion(node.outputs)
    }
    let tensors = initializerOrder.filter(names.contains).map { initializers[$0]! }
    let valueInfo = valueInfoOrder.filter(names.contains).map { valueInfos[$0]! }
    if graph.sparseInitializers != 0 {
      throw OnnxError("len_sparse_initializer is \(graph.sparseInitializers), it must be 0.")
    }
    if graph.quantizationAnnotations != 0 {
      throw OnnxError("len_quantization_annotation is \(graph.quantizationAnnotations), it must be 0.")
    }

    var g = Graph()
    g.nodes = nodes
    g.name = "Extracted from {\(graph.name ?? "")}"
    g.initializers = tensors
    g.inputs = newInputs
    g.outputs = newOutputs
    g.valueInfo = valueInfo

    var m = Model()
    // make_model sets ir_version, so it is always written, 0 included.
    m.irVersion = model.irVersion ?? 0
    m.producerName = "onnx.utils.extract_model"
    m.graph = g
    m.opsets = model.opsets
    m.functions = referredFunctions(nodes)
    return m
  }

  private func collectIO(_ names: [String]) throws -> [ValueInfo] {
    let missing = names.filter { valueInfos[$0] == nil }
    if !missing.isEmpty {
      throw OnnxError("The following names were not found in value_infos: \(missing.joined(separator: ", "))")
    }
    return names.map { valueInfos[$0]! }
  }

  private func reachableNodes(inputs: [String], outputs: [String]) -> [Int] {
    let stops = Set(inputs)
    var reachable = Set<Int>()
    for name in outputs {
      var stack = [name]
      while let current = stack.popLast() {
        if stops.contains(current) { continue }
        if let k = producers[current], !reachable.contains(k) {
          reachable.insert(k)
          stack.append(contentsOf: graph.nodes[k].inputs.filter { !$0.isEmpty })
        }
      }
    }
    return reachable.sorted()
  }

  /// The model's local functions the nodes call, directly or through another
  /// function, in the order a breadth-first walk finds them.
  private func referredFunctions(_ nodes: [Node]) -> [LocalFunction] {
    struct Key: Hashable {
      let name: String
      let domain: String
    }
    var available: [Key: LocalFunction] = [:]
    for f in model.functions {
      available[Key(name: f.name, domain: f.domain)] = f
    }
    var found: [LocalFunction] = []
    var queue = nodes.map { Key(name: $0.op, domain: $0.domainName) }
    var head = 0
    while head < queue.count {
      let key = queue[head]
      head += 1
      if let f = available.removeValue(forKey: key) {
        found.append(f)
        queue.append(contentsOf: f.calls.map { Key(name: $0.opType, domain: $0.domain) })
      }
    }
    return found
  }
}
