import Foundation

/// Camera frames from the comma's own drives, kept on the phone for the
/// benchmark's CPU comparison (`compareWithReference`).
///
/// The benchmark's own frames are random bytes, and a model does not reach
/// on them what it reaches on a road: carrot's N16's temporal norms stay
/// under 1470 on random bytes and reach 2160 on a road, past what a Hexagon
/// NPU's fp16 norm holds at its prescale. Here one frame in `every` that a
/// comma sends (about one each 3 s at 20 Hz) is remembered with its scalars,
/// the last `count` of them, and written to the cache now and then and when
/// the comma goes. They never leave the phone; a benchmark of the same model
/// compares on them as well as on its random frames.
final class RoadFrames {
  static let count = ReferenceComparison.frames
  static let every = 60
  /// At most one write a while, so a long drive is a few writes, not one a minute.
  static let saveInterval: TimeInterval = 600
  private static let magic: UInt32 = 0x4A4C_5246  // "JLRF"

  struct Frame {
    var warped: [UInt8]
    var packed: [UInt8]
  }

  private let directory: URL
  private var sha256 = ""
  private var frames: [Frame] = []
  private var seen = 0
  private var dirty = false
  private var savedAt: TimeInterval = 0

  init(directory: URL) {
    self.directory = directory
  }

  static func file(in directory: URL, sha256: String) -> URL {
    directory.appending(path: "\(String(sha256.prefix(16))).roadframes")
  }

  /// A frame the comma sent and the model ran, as it arrived. Caller holds
  /// the host's lock.
  func offer(sha256: String, warped: UnsafeRawPointer, warpedBytes: Int, packed: UnsafeRawPointer, packedBytes: Int) {
    if sha256 != self.sha256 {
      save()
      self.sha256 = sha256
      frames = []
      seen = 0
      savedAt = ProcessInfo.processInfo.systemUptime
    }
    seen += 1
    guard seen % RoadFrames.every == 0 else { return }
    let frame = Frame(
      warped: Array(UnsafeRawBufferPointer(start: warped, count: warpedBytes)),
      packed: Array(UnsafeRawBufferPointer(start: packed, count: packedBytes)))
    if frames.count == RoadFrames.count { frames.removeFirst() }
    frames.append(frame)
    dirty = true
    let now = ProcessInfo.processInfo.systemUptime
    if frames.count == RoadFrames.count && now - savedAt >= RoadFrames.saveInterval {
      save()
    }
  }

  /// Writes the frames held, if a full set changed since the last write.
  func save() {
    guard dirty, frames.count == RoadFrames.count, let first = frames.first else { return }
    dirty = false
    savedAt = ProcessInfo.processInfo.systemUptime
    var data = Data()
    func put(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    put(RoadFrames.magic)
    put(UInt32(frames.count))
    put(UInt32(first.warped.count))
    put(UInt32(first.packed.count))
    for f in frames {
      data.append(contentsOf: f.warped)
      data.append(contentsOf: f.packed)
    }
    do {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      try data.write(to: RoadFrames.file(in: directory, sha256: sha256), options: .atomic)
    } catch {
      ServerLog(category: "server").error("road frames: not written: \(error)")
    }
  }

  /// The frames written for this model, if their sizes are the spec's.
  static func load(in directory: URL, sha256: String, spec: ModelSpec) -> [Frame]? {
    guard let data = try? Data(contentsOf: file(in: directory, sha256: sha256)), data.count >= 16 else { return nil }
    func get(_ at: Int) -> Int {
      Int(UInt32(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: at, as: UInt32.self) }))
    }
    let n = get(4)
    let w = get(8)
    let p = get(12)
    guard get(0) == Int(magic), n > 0, w == spec.warpedBytes, p == spec.packedBytes, data.count == 16 + n * (w + p) else { return nil }
    var out: [Frame] = []
    out.reserveCapacity(n)
    var at = 16
    for _ in 0..<n {
      let start = data.startIndex + at
      out.append(Frame(warped: [UInt8](data[start..<start + w]), packed: [UInt8](data[start + w..<start + w + p])))
      at += w + p
    }
    return out
  }
}
