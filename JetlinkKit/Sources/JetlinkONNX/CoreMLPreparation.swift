import Foundation

/// A driving model prepared for onnxruntime's CoreML provider, as the
/// `simplify` branch's ORT backend prepares it (`_prepared_model(for_coreml=
/// True)`, `_stage` and `_with_cache_key` in jetlink/server/backends/ort):
///
/// 1. strip tinygrad's `org.tinygrad` layout ops;
/// 2. retype uint8 image inputs to fp16 and drop the head Cast, if any input is uint8;
/// 3. write negative constant Gather indices from the front;
/// 4. rewrite MatMul+Add as Gemm with a transposed weight (transB=1);
/// 5. rewrite repeat-only Expands as Tiles;
/// 6. for `.aneWhole` only (`layout='ane-whole'`), feed the policy's fp16
///    LayerNormalizations their input times 1/8, run the heads after the
///    vision trunk in fp32, and give attention masks a finite fill for -inf
///    (`Patches.finiteSoftmaxMasks`, which Python's preparation does not do);
/// 7. cut the graph where the vision trunk ends (`.split`, writing
///    vision.onnx and policy.onnx), or keep it whole (`.whole` and
///    `.aneWhole`, model.onnx);
/// 8. give each file a COREML_CACHE_KEY in its metadata_props.
///
/// `.plain` is TensorRT's preparation (`onnx_patch.patch_file`, what
/// trt/build.py parsed), and onnxruntime's CPU provider's: steps 1 and 2
/// only, the whole graph, no cache key. The other steps work around CoreML's
/// provider, which neither needs; the CPU provider on Android runs step 4's
/// transposed fp16 Gemm on one thread, a hundred times slower than MatMul.
///
/// The files are what Python's onnx.save writes for the same model: field
/// for field, and byte for byte on a model Python wrote.
///
/// Memory is the point, because an iPhone runs this. The source is
/// memory-mapped and only the graph's structure is decoded; each weight stays
/// a range of the mapped file until it is copied to the output. The
/// transposed Gemm weights, about 671 MB on the big models, and the fp32
/// head weights are made a few MB at a time while they are written. Every message's length is known
/// before it is written, so the output streams to disk in one pass.
public enum CoreMLPreparation {
  public enum Layout: Sendable {
    /// The vision trunk and the rest as two models: vision.onnx and policy.onnx.
    case split
    /// One model: model.onnx.
    case whole
    /// One model prepared for the whole graph on the Neural Engine
    /// (`--device ane-whole`): model.onnx, with the policy's norms prescaled
    /// and the vision heads in fp32.
    case aneWhole
    /// `aneWhole` for a Snapdragon's Hexagon NPU (`htp-whole`): the vision's
    /// norms prescaled too (`prescaleVisionLayerNorm`) and GELU in its tanh
    /// form (`tanhGelu`).
    case hexagonWhole
    /// One model for TensorRT or the CPU: model.onnx, tinygrad's ops
    /// stripped and the images fp16, nothing else changed.
    case plain
  }

  /// Which tensors a probe preparation makes graph outputs: up to `limit`
  /// of those the listed ops write, spread over the graph, or exactly
  /// `names` where the graph has them.
  public struct ProbeRequest: Sendable {
    public var ops: Set<String>
    public var limit: Int
    public var names: [String]?

    public init(ops: Set<String> = ["LayerNormalization", "Softmax", "Gelu", "MatMul", "Gemm", "Conv"], limit: Int = 160, names: [String]? = nil) {
      self.ops = ops
      self.limit = limit
      self.names = names
    }
  }

  /// What a test build may try on the Hexagon NPU beyond what it ships,
  /// each measured against the rest (the test app's experiment sweep).
  public enum Experiment: String, CaseIterable, Sendable, Comparable {
    /// Big weights as int8 with a DequantizeLinear (`int8Weights`).
    case int8
    case w8a16Vision = "w8a16-vision"
    case w8a16 = "w8a16"
    case layouts

    public static func < (a: Experiment, b: Experiment) -> Bool { a.rawValue < b.rawValue }

    /// "fold+int8", or "none".
    public static func name(_ set: Set<Experiment>) -> String {
      set.isEmpty ? "none" : set.sorted().map(\.rawValue).joined(separator: "+")
    }

    /// Independent benchmark candidates, with the unchanged baseline first.
    public static var combinations: [Set<Experiment>] {
      [[], [.layouts], [.w8a16Vision], [.w8a16]]
    }
  }

  public struct Part: Sendable, Equatable {
    /// "vision", "policy" or "model"; the file is `<name>.onnx`.
    public let name: String
    public let url: URL
    /// The initializers' raw_data bytes in this part, which is what
    /// onnxruntime writes out as the CoreML weight file. Python sums
    /// `len(t.raw_data)` the same way, so a tensor kept in the typed fields
    /// counts as 0.
    public let weightBytes: Int64
  }

  public struct Report: Sendable, Equatable {
    public static func == (a: Report, b: Report) -> Bool {
      a.stripped == b.stripped && a.retypedImages == b.retypedImages && a.gathers == b.gathers && a.gemms == b.gemms
        && a.tiles == b.tiles && a.norms == b.norms && a.heads == b.heads && a.masks == b.masks && a.parts == b.parts
        && a.biasMasks == b.biasMasks && a.visionNorms == b.visionNorms && a.safeNorms == b.safeNorms && a.gelus == b.gelus && a.stateIO == b.stateIO && a.int8Weights == b.int8Weights && a.probes.map(\.name) == b.probes.map(\.name)
    }
    /// tinygrad layout ops bypassed.
    public let stripped: Int
    /// Whether uint8 image inputs were retyped to fp16.
    public let retypedImages: Bool
    /// Gathers whose negative index was rewritten.
    public let gathers: Int
    /// MatMul+Add pairs rewritten as Gemm.
    public let gemms: Int
    /// Expands rewritten as Tile.
    public let tiles: Int
    /// Policy LayerNormalizations prescaled; 0 unless the layout is `.aneWhole`.
    public let norms: Int
    /// Vision head nodes moved to fp32; 0 unless the layout is `.aneWhole`.
    public let heads: Int
    /// Softmax masks given a finite fill for -inf (`finiteSoftmaxMasks`); 0
    /// unless the layout is `.aneWhole`.
    public let masks: Int
    /// Constant Softmax masks made an fp16 bias the scores add
    /// (`additiveSoftmaxMasks`); 0 unless the layout is `.aneWhole`.
    public var biasMasks: Int = 0
    /// Vision LayerNormalizations prescaled (`prescaleVisionLayerNorm`); 0
    /// unless the layout is `.aneWhole`.
    public var visionNorms: Int = 0
    /// LayerNormalizations in their fp16-safe form (`safeLayerNorm`); 0
    /// unless the layout is `.hexagonWhole`.
    public var safeNorms: Int = 0
    /// Exact GELUs made their tanh form (`tanhGelu`); 0 unless the layout is
    /// `.hexagonWhole`.
    public var gelus: Int = 0
    /// State queue pairs made fp16 at the graph's edge (`fp16StateIO`); 0
    /// unless the layout is `.hexagonWhole`.
    public var stateIO: Int = 0
    /// Weights made int8 (`int8Weights`); 0 unless asked for.
    public var int8Weights: Int = 0
    /// The tensors made graph outputs for a probe, in graph order, with the
    /// op that writes each (`Patches.addProbes`); empty unless asked for.
    public var probes: [(name: String, op: String)] = []
    public let parts: [Part]
  }

  /// The metadata_props key onnxruntime's CoreML provider keys its
  /// compiled-model cache by.
  public static let cacheKeyProp = "COREML_CACHE_KEY"

  /// The Python server's `_cache_key(out_path, part)` with `stem` for the
  /// path's stem: ASCII letters and digits only, at most 63 characters. The
  /// stem is the artifact directory's, which carries the device tag, so the
  /// `ane`, `coreml` and `ane-whole` layouts key compile caches of their own.
  public static func cacheKey(stem: String, part: String) -> String {
    let kept = (stem + part).unicodeScalars.filter {
      ("A"..."Z").contains($0) || ("a"..."z").contains($0) || ("0"..."9").contains($0)
    }
    return String(String.UnicodeScalarView(kept.prefix(63)))
  }

  public static func prepare(
    source: URL, into directory: URL, layout: Layout,
    cacheKey: (String) -> String,
    progress: ((Double) -> Void)? = nil,
    probes: ProbeRequest? = nil,
    experiments: Set<Experiment> = []
  ) throws -> Report {
    let data = try Data(contentsOf: source, options: .alwaysMapped)
    return try data.withUnsafeBytes { buf in
      try prepare(
        Source(bytes: buf), into: directory, layout: layout, cacheKey: cacheKey, progress: progress, probes: probes, experiments: experiments)
    }
  }

  private static func prepare(
    _ src: Source, into directory: URL, layout: Layout,
    cacheKey: (String) -> String,
    progress: ((Double) -> Void)?,
    probes probeRequest: ProbeRequest? = nil,
    experiments: Set<Experiment> = []
  ) throws -> Report {
    var (model, g) = try Decode.preparable(src)

    let stripped = try Patches.stripTinygradOps(&g, &model.opsets)
    // Not for the Hexagon NPU either: kept uint8 there (carrot.38-test), the
    // image queue's Slice, Concat and Gather left the NPU for the CPU, and
    // N16 ran in 39.7 ms a frame instead of 33.5, further from the CPU's.
    let retyped = Patches.needsPatch(g)
    if retyped {
      try Patches.patchUint8Inputs(&g)
    }
    var gathers = 0
    var gemms = 0
    var tiles = 0
    var int8 = 0
    if layout != .plain {
      gathers = try Patches.normalizeGatherIndices(&g, src)
      gemms = try Patches.gemmWithTransposedWeight(&g, src)
      tiles = try Patches.expandToTile(&g, src)
    }
    var norms = 0
    var heads = 0
    var masks = 0
    var biasMasks = 0
    var visionNorms = 0
    var safeNorms = 0
    var gelus = 0
    var stateIO = 0
    if layout == .aneWhole {
      norms = try Patches.prescaleLayerNorm(&g)
      heads = try Patches.headsInFP32(&g, src)
    } else if layout == .hexagonWhole {
      // the native norm, fed a scaled input: the policy's by 1/8, the vision's
      // by 1/32 with its epsilon (carrot.15). The
      // fp16-safe norms of carrot.23 were no more exact on the NPU and made a
      // Snapdragon 8 Gen 2 run N16 in 55 ms instead of about 40.
      norms = try Patches.prescaleLayerNorm(&g)
      heads = try Patches.headsInFP32(&g, src)
      visionNorms = try Patches.prescaleVisionLayerNorm(&g)
      stateIO = Patches.fp16StateIO(&g)
      // Not Patches.tanhGelu: QNN runs the native Gelu well. On a Snapdragon
      // 8 Gen 2 the tanh form took Light Speed 2 from 84 M NPU cycles a run
      // to 168 M (33.4 ms to 39.3 ms a frame) and twice as far from the CPU.
    }
    if layout == .aneWhole || layout == .hexagonWhole {
      biasMasks = try Patches.additiveSoftmaxMasks(&g, src)
      masks = try Patches.finiteSoftmaxMasks(&g, src)
    }
    if layout == .hexagonWhole && experiments.contains(.int8) {
      int8 = try Patches.int8Weights(&g, src)
    }
    var probed: [(name: String, op: String)] = []
    if let probeRequest {
      probed = Patches.addProbes(&g, probeRequest)
    }
    model.graph = g

    var parts: [(name: String, model: Model)]
    switch layout {
    case .split:
      let (vision, policy) = try Split.visionPolicy(model)
      parts = [("vision", vision), ("policy", policy)]
    case .whole, .aneWhole, .hexagonWhole, .plain:
      parts = [("model", model)]
    }
    for i in parts.indices where layout != .plain {
      // _with_cache_key: any key already there goes, the part's own is added last.
      parts[i].model.props.removeAll { $0.key == cacheKeyProp || $0.key == "CACHE_KEY" }
      parts[i].model.props.append(Prop(raw: nil, key: cacheKeyProp, value: cacheKey(parts[i].name)))
    }
    let reported = try write(parts, src, into: directory, progress: progress)
    var report = Report(
      stripped: stripped, retypedImages: retyped, gathers: gathers, gemms: gemms, tiles: tiles,
      norms: norms, heads: heads, masks: masks, parts: reported)
    report.biasMasks = biasMasks
    report.visionNorms = visionNorms
    report.safeNorms = safeNorms
    report.gelus = gelus
    report.stateIO = stateIO
    report.int8Weights = int8
    report.probes = probed
    return report
  }

  /// Encodes each part and streams it to `<name>.onnx` in `directory`; on a
  /// failure, the files already written go.
  private static func write(
    _ parts: [(name: String, model: Model)], _ src: Source, into directory: URL, progress: ((Double) -> Void)?
  ) throws -> [Part] {
    let encoded = parts.map { part in
      (
        name: part.name, bytes: Encode.model(part.model, src),
        weights: part.model.graph!.initializers.reduce(Int64(0)) { $0 + Int64($1.raw?.count ?? 0) }
      )
    }
    let total = max(1, encoded.reduce(0) { $0 + $1.bytes.count })

    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    var done = 0
    var written: [URL] = []
    var reported: [Part] = []
    do {
      for part in encoded {
        let url = directory.appendingPathComponent("\(part.name).onnx")
        written.append(url)
        let base = done
        try PartWriter.write(part.bytes, src, to: url) { bytes in
          progress?(Double(base + bytes) / Double(total))
        }
        done += part.bytes.count
        reported.append(Part(name: part.name, url: url, weightBytes: part.weights))
      }
    } catch {
      for url in written {
        try? FileManager.default.removeItem(at: url)
      }
      throw error
    }
    progress?(1.0)
    return reported
  }
}

// MARK: writing a part

/// Streams an encoded model to a file: owned bytes through a 1 MB buffer,
/// weights straight from the mapped source, transposed and widened weights a
/// block at a time.
final class PartWriter {
  private static let bufferSize = 1 << 20
  /// Weights go to the file in pieces of this size, so progress moves.
  private static let chunkSize = 8 << 20
  /// A transposed or widened weight is produced this many bytes at a time, at most.
  private static let blockSize = 4 << 20

  private let handle: FileHandle
  private var buffer: [UInt8] = []
  private var written = 0
  private let onWrite: (Int) -> Void

  private init(url: URL, onWrite: @escaping (Int) -> Void) throws {
    guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
      throw OnnxError("could not create \(url.path)")
    }
    handle = try FileHandle(forWritingTo: url)
    buffer.reserveCapacity(PartWriter.bufferSize)
    self.onWrite = onWrite
  }

  static func write(_ encoded: Encoded, _ src: Source, to url: URL, onWrite: @escaping (Int) -> Void) throws {
    let writer = try PartWriter(url: url, onWrite: onWrite)
    do {
      for piece in encoded.allPieces {
        switch piece {
        case .bytes(let b): try b.withUnsafeBytes { try writer.write($0) }
        case .source(let r): try writer.write(src.slice(r))
        case .transposed(let t): try writer.transpose(t, src)
        case .widened(let w): try writer.widen(w, src)
        }
      }
      try writer.flush()
      try writer.handle.close()
    } catch {
      try? writer.handle.close()
      throw error
    }
    guard writer.written == encoded.count else {
      throw OnnxError("wrote \(writer.written) bytes of \(url.lastPathComponent) where \(encoded.count) were planned")
    }
  }

  private func write(_ bytes: UnsafeRawBufferPointer) throws {
    if buffer.count + bytes.count <= PartWriter.bufferSize {
      buffer.append(contentsOf: bytes)
      return
    }
    try flush()
    if bytes.count < PartWriter.bufferSize {
      buffer.append(contentsOf: bytes)
      return
    }
    var offset = 0
    while offset < bytes.count {
      let n = min(PartWriter.chunkSize, bytes.count - offset)
      try handle.write(contentsOf: UnsafeRawBufferPointer(rebasing: bytes[offset..<(offset + n)]))
      offset += n
      written += n
      onWrite(written)
    }
  }

  private func flush() throws {
    guard !buffer.isEmpty else { return }
    try handle.write(contentsOf: buffer)
    written += buffer.count
    buffer.removeAll(keepingCapacity: true)
    onWrite(written)
  }

  /// Each produced band goes to `sink`: the file, or the widening.
  private func transpose(_ t: Transpose, _ src: Source, sink: ((UnsafeRawBufferPointer) throws -> Void)? = nil) throws {
    switch t.elements {
    case .source(let r):
      try transpose(src.slice(r), t, sink: sink)
    case .owned(let bytes):
      try bytes.withUnsafeBytes { try transpose($0, t, sink: sink) }
    case .typed(let tensor):
      // Only this one weight is decoded, and it goes when this returns.
      let bytes = try Elements.littleEndian(tensor, src)
      try bytes.withUnsafeBytes { try transpose($0, t, sink: sink) }
    }
  }

  /// Writes the [cols, rows] transpose of a row-major [rows, cols] matrix, a
  /// block of output rows at a time. Each block reads a narrow column band
  /// of every source row, so the reads stay sequential within a row.
  private func transpose(_ elements: UnsafeRawBufferPointer, _ t: Transpose, sink: ((UnsafeRawBufferPointer) throws -> Void)?) throws {
    let rows = t.rows
    let cols = t.cols
    let size = t.elementSize
    guard elements.count == t.byteCount else {
      throw OnnxError("a weight has \(elements.count) bytes where [\(rows), \(cols)] needs \(t.byteCount)")
    }
    guard rows > 0, cols > 0, let from = elements.baseAddress else { return }
    let rowBytes = rows * size
    let band = max(1, min(cols, PartWriter.blockSize / rowBytes))
    var scratch = [UInt8](repeating: 0, count: band * rowBytes)
    try flush()
    try scratch.withUnsafeMutableBytes { scratchBytes in
      let to = scratchBytes.baseAddress!
      var j0 = 0
      while j0 < cols {
        let j1 = min(cols, j0 + band)
        switch size {
        case 1: PartWriter.transposeBand(UInt8.self, from, to, rows, cols, j0, j1)
        case 2: PartWriter.transposeBand(UInt16.self, from, to, rows, cols, j0, j1)
        case 4: PartWriter.transposeBand(UInt32.self, from, to, rows, cols, j0, j1)
        case 8: PartWriter.transposeBand(UInt64.self, from, to, rows, cols, j0, j1)
        default:
          for i in 0..<rows {
            for j in j0..<j1 {
              (to + ((j - j0) * rows + i) * size).copyMemory(from: from + (i * cols + j) * size, byteCount: size)
            }
          }
        }
        let n = (j1 - j0) * rowBytes
        let band = UnsafeRawBufferPointer(start: to, count: n)
        if let sink {
          try sink(band)
        } else {
          try handle.write(contentsOf: band)
          written += n
          onWrite(written)
        }
        j0 = j1
      }
    }
  }

  private func widen(_ w: Widen, _ src: Source) throws {
    switch w.elements {
    case .source(let r):
      try widen(src.slice(r), w.sourceByteCount)
    case .owned(let bytes):
      try bytes.withUnsafeBytes { try widen($0, w.sourceByteCount) }
    case .typed(let tensor):
      let bytes = try Elements.littleEndian(tensor, src)
      try bytes.withUnsafeBytes { try widen($0, w.sourceByteCount) }
    case .transposed(let t):
      guard t.byteCount == w.sourceByteCount, t.elementSize == 2 else {
        throw OnnxError("a weight has \(t.byteCount) bytes where its fp32 copy expects \(w.sourceByteCount)")
      }
      try flush()
      try transpose(t, src) { band in try self.widen(band, band.count) }
    }
  }

  /// Writes the fp32 of `expected` bytes of little-endian fp16, a band at a time.
  private func widen(_ elements: UnsafeRawBufferPointer, _ expected: Int) throws {
    guard elements.count == expected else {
      throw OnnxError("a weight has \(elements.count) bytes where its fp32 copy expects \(expected)")
    }
    guard elements.count > 0, let from = elements.baseAddress else { return }
    let count = elements.count / 2
    let band = max(1, min(count, PartWriter.blockSize / 4))
    var scratch = [UInt8](repeating: 0, count: band * 4)
    try flush()
    try scratch.withUnsafeMutableBytes { scratchBytes in
      let to = scratchBytes.baseAddress!
      var i0 = 0
      while i0 < count {
        let i1 = min(count, i0 + band)
        PartWriter.widenBand(from + i0 * 2, to, i1 - i0)
        let n = (i1 - i0) * 4
        try handle.write(contentsOf: UnsafeRawBufferPointer(start: to, count: n))
        written += n
        onWrite(written)
        i0 = i1
      }
    }
  }

  /// `count` little-endian fp16 values at `from` as little-endian fp32 at
  /// `to`: numpy's `astype(np.float32)`, which is exact. Subnormals widen to
  /// normals, an infinity stays one, and a NaN keeps its payload shifted up
  /// 13 bits with the quiet bit set, as numpy's cast does on Apple silicon.
  @inline(__always)
  static func widenBand(_ from: UnsafeRawPointer, _ to: UnsafeMutableRawPointer, _ count: Int) {
    for i in 0..<count {
      let bits = UInt16(littleEndian: from.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
      let value = Float(Float16(bitPattern: bits))
      to.storeBytes(of: value.bitPattern.littleEndian, toByteOffset: i * 4, as: UInt32.self)
    }
  }

  @inline(__always)
  private static func transposeBand<T: FixedWidthInteger>(
    _: T.Type, _ from: UnsafeRawPointer,
    _ to: UnsafeMutableRawPointer,
    _ rows: Int, _ cols: Int, _ j0: Int, _ j1: Int
  ) {
    let size = MemoryLayout<T>.size
    // The source may sit at any offset in the file, so its loads are unaligned.
    for i in 0..<rows {
      let row = from + i * cols * size
      for j in j0..<j1 {
        let value = row.loadUnaligned(fromByteOffset: j * size, as: T.self)
        to.storeBytes(of: value, toByteOffset: ((j - j0) * rows + i) * size, as: T.self)
      }
    }
  }
}
