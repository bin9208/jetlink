import Foundation

/// Exact layout rewrites, separately benchmarked instead of changing the
/// model all users run. Does not approximate arithmetic or shorten history.
public enum LayoutPreparation {
  public static func prepare(source: URL, to output: URL) throws -> Int {
    let data = try Data(contentsOf: source, options: .alwaysMapped)
    return try data.withUnsafeBytes { bytes in
      let src = Source(bytes: bytes)
      var (model, g) = try Decode.preparable(src)
      let count = simplify(&g)
      model.graph = g
      try PartWriter.write(Encode.model(model, src), src, to: output, onWrite: { _ in })
      return count
    }
  }

  static func simplify(_ g: inout Graph) -> Int {
    let dims = Patches.staticDims(g)
    let exposed = Set(g.outputs.map(\.key))
    var readers: [String: Int] = [:]
    var producer: [String: Int] = [:]
    for (i, n) in g.nodes.enumerated() {
      for x in n.inputs { readers[x, default: 0] += 1 }
      for y in n.outputs { producer[y] = i }
    }
    var removed = Set<Int>()
    var count = 0
    for i in g.nodes.indices {
      var n = g.nodes[i]
      guard n.op == "Transpose", n.inputs.count == 1, n.outputs.count == 1,
        let shape = dims[n.inputs[0]], shape.allSatisfy({ $0 > 0 })
      else { continue }
      var p = n.attribute("perm")?.ints.map(Int.init) ?? Array(shape.indices.reversed())
      guard p.sorted() == Array(shape.indices) else { continue }
      if let j = producer[n.inputs[0]], !removed.contains(j), g.nodes[j].op == "Transpose",
        readers[n.inputs[0]] == 1, !exposed.contains(n.inputs[0]),
        let before = dims[g.nodes[j].inputs[0]]
      {
        let q = g.nodes[j].attribute("perm")?.ints.map(Int.init) ?? Array(before.indices.reversed())
        if q.sorted() == Array(shape.indices) {
          p = p.map { q[$0] }
          n.inputs = g.nodes[j].inputs
          n.attributes = [.ints("perm", p.map(Int64.init))]
          removed.insert(j)
          count += 1
        }
      }
      let original = dims[n.inputs[0]] ?? shape
      if p == Array(original.indices) {
        n.opType = "Identity"
        n.attributes = []
        count += 1
      } else if p.filter({ original[$0] != 1 }) == original.indices.filter({ original[$0] != 1 }) {
        let name = "__jetlink_layout_shape_\(i)"
        g.initializers.append(Patches.int64Tensor(p.map { original[$0] }, name))
        n.opType = "Reshape"
        n.inputs.append(name)
        n.attributes = []
        count += 1
      }
      g.nodes[i] = n
    }
    g.nodes = g.nodes.enumerated().filter { !removed.contains($0.offset) }.map(\.element)
    return count
  }
}
