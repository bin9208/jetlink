import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkONNX

/// The Swift preparation held to what Python's makes of the same graphs.
/// Scripts/make_onnx_fixtures.py wrote the graphs, Python's prepared files and
/// python.json (the counts, the weight bytes, or the error Python raised),
/// with cache keys from the prefix "fixture". The files must match byte for
/// byte; Scripts/check_onnx_prep.py does the field-by-field comparison and
/// runs both with onnxruntime.
@Suite struct PreparationTests {
  struct PythonResult: Decodable, Sendable {
    let error: String?
    let stripped: Int?
    let retypedImages: Bool?
    let gathers: Int?
    let gemms: Int?
    let tiles: Int?
    let norms: Int?
    let heads: Int?
    let parts: [String: Int64]?
  }

  static let python = results("python.json")

  /// A results file Python wrote, by case.
  static func results(_ name: String) -> [String: PythonResult] {
    (try? JSONDecoder().decode([String: PythonResult].self, from: Data(contentsOf: Fixtures.url(name)))) ?? [:]
  }

  /// "<fixture>.<layout>" for every case Python ran.
  static let cases: [String] = python.keys.sorted()

  static let layouts: [String: CoreMLPreparation.Layout] = ["split": .split, "whole": .whole, "ane-whole": .aneWhole]

  @Test func fixturesArePresent() {
    #expect(Self.cases.count == 54)
    #expect(Self.cases.contains("stateful.split"))
    #expect(Self.cases.contains("variants.ane-whole"))
  }

  @Test(arguments: cases)
  func matchesPython(_ name: String) throws {
    let expected = try #require(Self.python[name])
    let fixture = String(name.split(separator: ".")[0])
    let layoutName = String(name.split(separator: ".")[1])
    let layout = try #require(Self.layouts[layoutName])
    let out = try TemporaryDirectory()

    let prepare = {
      try CoreMLPreparation.prepare(
        source: Fixtures.url("\(fixture).onnx"), into: out.url, layout: layout,
        cacheKey: { CoreMLPreparation.cacheKey(stem: "fixture", part: $0) })
    }

    if fixture == "unrecorded", layout != .split {
      // The one place the two differ by design: Python asks onnx's shape
      // inferrer for the Gather's input and rewrites it; Swift has no
      // inferrer and leaves a Gather whose input shape is not recorded alone.
      #expect(expected.gathers == 1)
      let report = try prepare()
      #expect(report.gathers == 0)
      #expect(report.retypedImages)
      return
    }
    if fixture == "noshape", layout == .split {
      // Python asks onnx's shape inferrer for the trunk's shape; Swift has none.
      #expect(throws: OnnxError("the export records no shape for trunk; this model cannot be prepared on iPhone or iPad")) {
        try prepare()
      }
      #expect(try FileManager.default.contentsOfDirectory(atPath: out.url.path).isEmpty)
      return
    }
    if fixture == "notype" || fixture == "noentry", layout == .aneWhole {
      // Python asks onnx's shape inferrer for the type of a policy norm's
      // input (notype) or a head's entry (noentry) and carries on; Swift
      // refuses rather than prepare something else.
      #expect(expected.norms == 3 && expected.heads == 6)
      let tensor = fixture == "notype" ? "p" : "vm"
      #expect(throws: OnnxError("the export records no type for \(tensor); this model cannot be prepared on iPhone or iPad")) {
        try prepare()
      }
      #expect(try FileManager.default.contentsOfDirectory(atPath: out.url.path).isEmpty)
      return
    }
    if let message = expected.error {
      #expect(throws: OnnxError(message)) { try prepare() }
      #expect(try FileManager.default.contentsOfDirectory(atPath: out.url.path).isEmpty)
      return
    }

    let report = try prepare()
    #expect(report.stripped == expected.stripped)
    #expect(report.retypedImages == expected.retypedImages)
    #expect(report.gathers == expected.gathers)
    #expect(report.gemms == expected.gemms)
    #expect(report.tiles == expected.tiles)
    #expect(report.norms == (expected.norms ?? 0))
    #expect(report.heads == (expected.heads ?? 0))
    #expect(report.parts.map(\.name) == (layout == .split ? ["vision", "policy"] : ["model"]))
    for part in report.parts {
      #expect(part.weightBytes == expected.parts?[part.name], "\(part.name)")
      #expect(part.url == out.url.appendingPathComponent("\(part.name).onnx"))
      let mine = try Data(contentsOf: part.url)
      let theirs = try Data(contentsOf: Fixtures.url("\(fixture).\(layoutName).\(part.name).expected.onnx"))
      #expect(mine == theirs, "\(name) \(part.name): \(mine.count) bytes against Python's \(theirs.count)")
    }
  }

  /// The names and order tests/test_ane_whole.py pins, read back from the
  /// file the Swift preparation wrote (the bytes are checked above).
  @Test func aneWholeVariantsLayout() throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(
      source: Fixtures.url("variants.onnx"), into: out.url, layout: .aneWhole, cacheKey: { $0 })
    #expect(report.norms == 3 && report.heads == 6)
    let data = try Data(contentsOf: report.parts[0].url)
    let model = try data.withUnsafeBytes { try Decode.model(Source(bytes: $0)) }
    let g = try #require(model.graph)
    let names = g.nodes.map(\.displayName)
    let i = try #require(names.firstIndex(of: "vm"))
    #expect(
      Array(names[(i + 1)..<(i + 10)]) == [
        "vm__cast_fp32", "hln", "h1mm__gemm", "hg", "h2", "hres", "hres__cast_fp16", "hsc", "hsc__cast_fp16",
      ])
    #expect(names.firstIndex(of: "p__prescale") == names.firstIndex(of: "ln1")! - 1)
    let node = { (name: String) in g.nodes.first { $0.displayName == name }! }
    #expect(node("ln1").inputs[0] == "p__scaled" && node("ln3").inputs[0] == "p__scaled" && node("ln4").inputs[0] == "p32")
    #expect(node("hln").inputs == ["vm__fp32", "hs__fp32", "hb__fp32"])
    #expect(node("h1mm__gemm").inputs == ["hln", "h1mm__wt__fp32", "hb1__fp32"])
    #expect(node("hsc").inputs == ["hres__fp32", "s__fp32"] && node("hsc").outputs == ["hsc__fp32"])
    #expect(node("vm__cast_fp32").attribute("to")?.i == 1 && node("hres__cast_fp16").attribute("to")?.i == 10)
    #expect(node("pre").inputs.suffix(2) == ["hres", "hsc"])
    let inits = g.initializers.map(\.key)
    #expect(inits.filter { $0.hasSuffix("__fp32") } == ["h1mm__wt__fp32", "hW2__fp32", "hb__fp32", "hb1__fp32", "hs__fp32", "s__fp32"])
    #expect(inits.contains("s") && !inits.contains("hW2") && !inits.contains("h1mm__wt"))
    #expect(inits.last == "s__fp32")
    let const = try #require(g.initializers.first { $0.key == "__layernorm_prescale_8" })
    #expect(const.dims.isEmpty && const.elementType == DataType.float16)
    let vi = Set(g.valueInfo.map(\.key))
    #expect(vi.isDisjoint(with: ["hln", "hg", "h2"]) && vi.isSuperset(of: ["vm", "hres", "hsc"]))
  }

  @Test func progressRisesToOne() throws {
    let out = try TemporaryDirectory()
    var seen: [Double] = []
    _ = try CoreMLPreparation.prepare(
      source: Fixtures.url("stateful.onnx"), into: out.url, layout: .split,
      cacheKey: { "k\($0)" }, progress: { seen.append($0) })
    #expect(!seen.isEmpty)
    #expect(seen == seen.sorted())
    #expect(seen.last == 1.0)
    #expect(seen.allSatisfy { (0...1).contains($0) })
  }

  /// The caller's key is written as it is given, replacing CACHE_KEY.
  @Test func cacheKeyReplacesOldKeys() throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(
      source: Fixtures.url("queued.onnx"), into: out.url, layout: .whole, cacheKey: { "abc\($0)" })
    let meta = try OnnxMeta.read(contentsOf: report.parts[0].url)
    #expect(meta.props.map(\.key) == ["output_slices", "model_checkpoint", "COREML_CACHE_KEY"])
    #expect(meta.prop("COREML_CACHE_KEY") == "abcmodel")
  }

  /// The split parts carry only the key: Extractor copies no metadata_props.
  @Test func splitPartsCarryOnlyTheKey() throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(
      source: Fixtures.url("queued.onnx"), into: out.url, layout: .split, cacheKey: { $0 })
    let vision = try OnnxMeta.read(contentsOf: report.parts[0].url)
    #expect(vision.props == [.init(key: "COREML_CACHE_KEY", value: "vision")])
    #expect(vision.inputs.map(\.name) == ["img", "big_img"])
    #expect(vision.inputs.allSatisfy { $0.elemType == 10 })
    #expect(vision.outputs.map(\.name) == ["trunk"])
    let policy = try OnnxMeta.read(contentsOf: report.parts[1].url)
    #expect(policy.inputs.map(\.name) == ["trunk", "desire_pulse", "traffic_convention", "features_buffer"])
    #expect(policy.outputs.map(\.name) == ["outputs"])
  }

  /// Two branches that never meet: the cut is every tensor the policy reads.
  @Test func noSingleCut() throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(
      source: Fixtures.url("nocut.onnx"), into: out.url, layout: .split, cacheKey: { $0 })
    let vision = try OnnxMeta.read(contentsOf: report.parts[0].url)
    #expect(vision.outputs.map(\.name) == ["a", "b", "vision_out"])
    let policy = try OnnxMeta.read(contentsOf: report.parts[1].url)
    #expect(policy.inputs.map(\.name) == ["a", "b", "desire"])
  }

  @Test func cacheKeyMatchesPython() {
    // re.sub('[^A-Za-z0-9]', '', stem + part)[:63]
    #expect(
      CoreMLPreparation.cacheKey(stem: "404a18cfd86d2963.ort1.29.0.ane-Apple M1 Pro", part: "vision")
        == "404a18cfd86d2963ort1290aneAppleM1Provision")
    #expect(
      CoreMLPreparation.cacheKey(stem: String(repeating: "é-x", count: 70), part: "policy")
        == String(repeating: "x", count: 63))
  }

  @Test func missingSourceIsAnError() throws {
    let out = try TemporaryDirectory()
    #expect(throws: (any Error).self) {
      try CoreMLPreparation.prepare(
        source: out.url.appendingPathComponent("none.onnx"), into: out.url, layout: .whole, cacheKey: { $0 })
    }
  }

  /// A real model prepared on request, to check against the original with
  /// onnxruntime or to measure: JETLINK_PREPARE=<layout>:<model.onnx>:<directory>,
  /// the layout whole, ane-whole, split or plain.
  @Test(.enabled(if: ProcessInfo.processInfo.environment["JETLINK_PREPARE"] != nil))
  func preparesARealModel() throws {
    let spec = ProcessInfo.processInfo.environment["JETLINK_PREPARE"]!.split(separator: ":", maxSplits: 2).map(String.init)
    let layouts: [String: CoreMLPreparation.Layout] = [
      "whole": .whole, "ane-whole": .aneWhole, "htp-whole": .hexagonWhole, "split": .split, "plain": .plain,
    ]
    let layout = try #require(layouts[spec[0]])
    // JETLINK_PREPARE_EXPERIMENTS=policy-norm-guard,safe-norms for the Hexagon NPU's
    let experiments = Set(
      (ProcessInfo.processInfo.environment["JETLINK_PREPARE_EXPERIMENTS"] ?? "").split(separator: ",").compactMap {
        CoreMLPreparation.Experiment(rawValue: String($0))
      })
    // JETLINK_PREPARE_EXTERNAL_FROM=0 moves any model's weights out; a huge
    // value keeps a large one's inside, as before carrot.57
    let externalFrom = ProcessInfo.processInfo.environment["JETLINK_PREPARE_EXTERNAL_FROM"].flatMap { Int64($0) }
      ?? CoreMLPreparation.externalWeightsFrom
    // JETLINK_PREPARE_CHUNK_FROM: a huge value keeps a large model in one piece
    let chunkFrom = ProcessInfo.processInfo.environment["JETLINK_PREPARE_CHUNK_FROM"].flatMap { Int64($0) }
      ?? CoreMLPreparation.chunkFrom
    let start = Date()
    let report = try CoreMLPreparation.prepare(
      source: URL(fileURLWithPath: spec[1]), into: URL(fileURLWithPath: spec[2]), layout: layout, cacheKey: { "jetlink\($0)" },
      experiments: experiments, externalWeightsFrom: externalFrom, chunkFrom: chunkFrom)
    print("prepared in \(Date().timeIntervalSince(start)) s: \(report)")
  }

  /// The Hexagon NPU's layout with the weights in a file of their own: the
  /// same model with each initializer of 1 KB or more moved to
  /// model.onnx.data at an aligned offset, byte for byte. Below the
  /// threshold, and in any other layout, one file as before.
  @Test func largeWeightsGoBesideTheModel() throws {
    let inline = try TemporaryDirectory()
    let external = try TemporaryDirectory()
    let source = Fixtures.url("stateful.onnx")
    let kept = try CoreMLPreparation.prepare(
      source: source, into: inline.url, layout: .hexagonWhole, cacheKey: { $0 })
    let moved = try CoreMLPreparation.prepare(
      source: source, into: external.url, layout: .hexagonWhole, cacheKey: { $0 }, externalWeightsFrom: 0)
    #expect(kept.parts[0].dataURL == nil)
    #expect(!FileManager.default.fileExists(atPath: inline.url.appending(path: "model.onnx.data").path))
    let dataURL = try #require(moved.parts[0].dataURL)
    #expect(dataURL.lastPathComponent == "model.onnx.data")
    #expect(moved.parts[0].weightBytes == kept.parts[0].weightBytes)

    // Every tensor the inline model holds 1 KB or more of is in the data file
    // where the moved model's external_data says, and nothing else changed.
    let inlineModel = try Data(contentsOf: kept.parts[0].url)
    let movedModel = try Data(contentsOf: moved.parts[0].url)
    let data = try Data(contentsOf: dataURL)
    let inlineTensors = try Self.initializers(inlineModel)
    let movedTensors = try Self.initializers(movedModel)
    #expect(inlineTensors.map(\.name) == movedTensors.map(\.name))
    var movedCount = 0
    for (a, b) in zip(inlineTensors, movedTensors) {
      if let entries = b.external {
        movedCount += 1
        #expect(b.raw == nil)
        #expect(entries["location"] == "model.onnx.data")
        let offset = try #require(entries["offset"].flatMap { Int($0) })
        let length = try #require(entries["length"].flatMap { Int($0) })
        #expect(offset % CoreMLPreparation.externalAlignment == 0)
        #expect(length == a.raw?.count)
        #expect(data[offset..<(offset + length)] == a.raw)
        #expect(b.dataLocation == 1)
      } else {
        #expect(b.raw == a.raw)
        #expect((a.raw?.count ?? 0) < CoreMLPreparation.externalTensorFrom)
      }
      #expect(a.header == b.header)
    }
    #expect(movedCount > 0)
    #expect(movedModel.count < inlineModel.count)

    // Other layouts never move them.
    let whole = try TemporaryDirectory()
    let wholeReport = try CoreMLPreparation.prepare(
      source: source, into: whole.url, layout: .aneWhole, cacheKey: { $0 }, externalWeightsFrom: 0)
    #expect(wholeReport.parts[0].dataURL == nil)
  }

  /// A model in chunks: the same nodes in the same order, each chunk reading
  /// the one tensor the chunk before gives, the graph's inputs where its
  /// nodes read them, and the graph's outputs from the last; every weight in
  /// the chunk that reads it.
  @Test func chunksCutTheTrunkWhereOneTensorCrosses() throws {
    let bytes = try Data(contentsOf: Fixtures.url("stateful.onnx"))
    try bytes.withUnsafeBytes { buf in
      let src = Source(bytes: buf)
      let (model, g) = try Decode.preparable(src)
      let chunks = try Split.chunks(model, count: 2)
      try #require(chunks.count == 2)
      let graphs = chunks.map { $0.graph! }
      #expect(graphs.flatMap(\.nodes).map(\.outputs) == g.nodes.map(\.outputs))
      // a small constant both read (an axes tensor) goes with each
      #expect(Set(graphs.flatMap(\.initializers).map(\.key)) == Set(g.initializers.map(\.key).filter { name in
        g.nodes.contains { $0.inputs.contains(name) }
      }))
      for (chunk, graph) in graphs.enumerated() {
        let reads = Set(graph.nodes.flatMap(\.inputs))
        #expect(graph.initializers.allSatisfy { reads.contains($0.key) }, "chunk \(chunk + 1) holds a weight it never reads")
      }
      let cut = graphs[0].outputs.map(\.key)
      #expect(cut.count == 1)
      #expect(graphs[1].inputs.map(\.key).first == cut[0])
      #expect(graphs[1].outputs.map(\.key) == g.outputs.map(\.key))
      let images = Set(Patches.imageInputs)
      #expect(graphs[0].inputs.map(\.key).allSatisfy(images.contains))
      // the rest are graph inputs (this fixture's trunk reads its image queue
      // past the cut too), and every one the graph has is read somewhere
      let graphInputs = Set(g.inputs.map(\.key))
      #expect(Set(graphs[1].inputs.map(\.key).dropFirst()).isSubset(of: graphInputs))
      #expect(Set(graphs.flatMap { $0.inputs.map(\.key) }).isSuperset(of: graphInputs))
      // one chunk asked for, or none possible: the model as it was
      #expect(try Split.chunks(model, count: 1).count == 1)
    }
  }

  /// A source tensor that writes data_location DEFAULT out (BMRLNAP v3's do)
  /// is still EXTERNAL once moved: its own field goes, it does not follow and win.
  @Test func movedTensorDropsItsOwnDataLocation() throws {
    // data_location (14) = DEFAULT, as the exporter wrote it
    let sourceBytes: [UInt8] = [0x70, 0x00]
    try sourceBytes.withUnsafeBytes { buf in
      let src = Source(bytes: buf)
      var t = Tensor(name: "w", dims: [4], dataType: 1, raw: .owned([UInt8](repeating: 7, count: 16)))
      t.extras = [RawField(number: 14, range: 0..<2)]
      let kept = try Self.fields(Data(flatten(Encode.tensor(t, src), src)))
      #expect(kept.filter { $0.0 == 14 }.map { [UInt8]($0.1) } == [[0]])

      t.raw = nil
      t.movedTo = Tensor.ExternalData(location: "model.onnx.data", offset: 16384, length: 16)
      let moved = try Self.fields(Data(flatten(Encode.tensor(t, src), src)))
      #expect(moved.filter { $0.0 == 14 }.map { [UInt8]($0.1) } == [[1]])
      #expect(moved.filter { $0.0 == 13 }.count == 3)
      #expect(!moved.contains { $0.0 == 9 })
    }
  }

  struct TensorFields {
    var name = ""
    var raw: Data?
    var external: [String: String]?
    var dataLocation: UInt64?
    /// dims and data type, as written
    var header = Data()
  }

  /// A prepared model's initializers, read field by field.
  static func initializers(_ model: Data) throws -> [TensorFields] {
    var out: [TensorFields] = []
    for (number, payload) in try fields(model) where number == 7 {
      for (gn, tensor) in try fields(payload) where gn == 5 {
        var t = TensorFields()
        for (tn, value) in try fields(tensor) {
          switch tn {
          case 1, 2: t.header.append(value)
          case 8: t.name = String(decoding: value, as: UTF8.self)
          case 9: t.raw = value
          case 13:
            let kv = Dictionary(uniqueKeysWithValues: try fields(value).map { ($0.0, String(decoding: $0.1, as: UTF8.self)) })
            t.external = (t.external ?? [:]).merging([kv[1] ?? "": kv[2] ?? ""]) { $1 }
          case 14: t.dataLocation = value.first.map { UInt64($0) }
          default: break
          }
        }
        out.append(t)
      }
    }
    return out
  }

  /// A message's fields: (number, payload) for length-delimited ones, (number,
  /// the varint's bytes) for varints.
  static func fields(_ d: Data) throws -> [(Int, Data)] {
    let b = [UInt8](d)
    var i = 0
    var out: [(Int, Data)] = []
    func varint() throws -> UInt64 {
      var v: UInt64 = 0
      var shift: UInt64 = 0
      while true {
        guard i < b.count else { throw OnnxError("truncated varint") }
        let c = b[i]
        i += 1
        v |= UInt64(c & 0x7f) << shift
        if c < 0x80 { return v }
        shift += 7
      }
    }
    while i < b.count {
      let key = try varint()
      let number = Int(key >> 3)
      switch key & 7 {
      case 0:
        let start = i
        _ = try varint()
        out.append((number, Data(b[start..<i])))
      case 2:
        let n = Int(try varint())
        out.append((number, Data(b[i..<(i + n)])))
        i += n
      case 1: i += 8
      case 5: i += 4
      default: throw OnnxError("wire type \(key & 7)")
      }
    }
    return out
  }
}
