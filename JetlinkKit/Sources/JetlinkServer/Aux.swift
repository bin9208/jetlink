import Foundation
import JetlinkKit
import JetlinkRegistry

// Carrot's auxiliary vision: the comma's small camera models (lane-marking
// type, the visual blind spot) run on this device beside the driving model,
// so the comma's little cores stop running them on the CPU.
//
// The comma uploads each model once (it is cached here by sha256), then sends
// one float32 input tensor at a time and gets the model's float32 outputs
// back; it decodes them itself, exactly as when the model runs on the comma.
// It sends one auxiliary message per frame, right after the driving reply,
// so a driving frame never queues behind one. Here the model runs on its own
// thread, and the reply is written from it, never from the request loop.
//
// The driving model comes first: a job is refused (`shed`) while the device
// is hot or the last driving frames were slow, and the comma then runs that
// one on its own CPU.

/// A model the auxiliary vision runs: one float32 input, fixed shapes.
public protocol AuxEngine: AnyObject, Sendable {
  var input: TensorSpec { get }
  var outputs: [TensorSpec] { get }
  /// What runs it, for the comma's status: "QNN(gpu)", "CPU".
  var device: String { get }
  /// Runs on `input` (`input.byteCount` bytes of float32) and returns the
  /// named outputs (all of them, in model order, for none) as float32.
  func run(input: UnsafeRawBufferPointer, outputs: [String]) throws -> [(TensorSpec, [Float])]
}

/// float16 to float32, for an auxiliary engine whose graph outputs float16.
package func auxF16ToF32(_ source: UnsafeRawPointer, _ destination: UnsafeMutableRawPointer, count: Int) {
  Convert.f16ToF32(source, destination, count: count)
}

/// Loads auxiliary models: JetlinkORT's `OrtAuxBackend`, or the tests' fake.
public protocol AuxBackend: AnyObject, Sendable {
  func load(model: URL) throws -> any AuxEngine
}

/// An auxiliary message's payload: a little-endian u32 JSON length, the JSON
/// object, then raw bytes.
enum AuxPayload {
  static func decode(_ payload: UnsafeRawBufferPointer) -> (json: [String: Any], blob: UnsafeRawBufferPointer)? {
    guard payload.count >= 4 else { return nil }
    let length = Int(UInt32(littleEndian: payload.loadUnaligned(as: UInt32.self)))
    guard length <= payload.count - 4 else { return nil }
    guard let json = JSONLine.decode(UnsafeRawBufferPointer(rebasing: payload[4..<(4 + length)])) else { return nil }
    return (json, UnsafeRawBufferPointer(rebasing: payload[(4 + length)...]))
  }

  /// The JSON part with its length in front; the raw bytes follow as more parts.
  static func head(_ json: [String: Any]) -> Data {
    let body = JSONLine.encode(json)
    var head = Data(capacity: 4 + body.count)
    withUnsafeBytes(of: UInt32(body.count).littleEndian) { head.append(contentsOf: $0) }
    head.append(body)
    return head
  }
}

/// The auxiliary models this device holds, and the thread they run on. One
/// per host: models and engines outlive a connection, as the driving
/// model's do.
final class AuxHub: @unchecked Sendable {
  /// Jobs waiting at most; past it a new one is refused as busy.
  static let maxQueued = 4
  /// A driving frame slower than this, on average over the last few, sheds
  /// auxiliary jobs until it is fast again. The driving model has the NPU and
  /// the auxiliary ones the GPU, so they share no accelerator; what matters is
  /// that a driving frame still answers well inside the comma's 50 ms. At
  /// 35 ms N16 (37 ms on a Snapdragon 8 Gen 2's NPU) shed every job, so the
  /// lane and BSD models never left the comma's CPU. Should they slow the
  /// driving frames past this, shedding takes them off again by itself.
  static let slowFrameUs: UInt32 = 44_000
  /// Upload pieces past this are refused (a model is ~10 MB).
  static let maxModelBytes: Int64 = 64 << 20

  private let backend: any AuxBackend
  private let directory: URL
  private let thermal: @Sendable () -> String
  private let log = ServerLog(category: "aux")
  private let lock = NSLock()
  private var engines: [String: any AuxEngine] = [:]
  private var loading: Set<String> = []
  private var failed: [String: String] = [:]
  private var uploads: [String: Int64] = [:]
  private var queued = 0
  private var recentFrames: [UInt32] = []
  private let worker = DispatchQueue(label: "jetlink.aux", qos: .utility)

  init(backend: any AuxBackend, directory: URL, thermal: @escaping @Sendable () -> String) {
    self.backend = backend
    self.directory = directory
    self.thermal = thermal
  }

  private func modelPath(_ sha256: String) -> URL {
    directory.appending(path: "\(sha256).onnx")
  }

  static func validSHA(_ sha: String) -> Bool {
    sha.count == 64 && sha.allSatisfy { $0.isHexDigit && !$0.isUppercase }
  }

  /// A driving frame's total, for shedding.
  func noteFrame(totalUs: UInt32) {
    lock.lock()
    recentFrames.append(totalUs)
    if recentFrames.count > 5 { recentFrames.removeFirst(recentFrames.count - 5) }
    lock.unlock()
  }

  /// Why a job must wait for the driving model, or nil when it may run.
  func shedReason() -> String? {
    let state = thermal()
    if state == "serious" || state == "critical" {
      return "thermal \(state)"
    }
    lock.lock()
    let frames = recentFrames
    lock.unlock()
    if frames.count >= 3 {
      let mean = frames.reduce(0) { $0 + UInt64($1) } / UInt64(frames.count)
      if mean > UInt64(AuxHub.slowFrameUs) {
        return "driving frames average \(mean / 1000) ms"
      }
    }
    return nil
  }

  // MARK: models

  /// The state of a model, as AUX_STATE says it.
  private func state(_ sha: String, _ state: String, detail: String = "") -> [String: Any] {
    var out: [String: Any] = ["sha256": sha, "state": state]
    if !detail.isEmpty { out["detail"] = detail }
    if state == "ready", let engine = engines[sha] {
      out["device"] = engine.device
      out["input"] = ["name": engine.input.name, "shape": engine.input.shape]
      out["outputs"] = engine.outputs.map { ["name": $0.name, "shape": $0.shape] }
    }
    return out
  }

  /// AUX_LOAD: what the comma must do next for a model. `push` sends a later
  /// state (the load finishing) from the worker.
  func load(sha256 sha: String, nbytes: Int64, push: @escaping @Sendable ([String: Any]) -> Void) -> [String: Any] {
    guard AuxHub.validSHA(sha), nbytes > 0, nbytes <= AuxHub.maxModelBytes else {
      return ["sha256": sha, "state": "failed", "detail": "bad model identity"]
    }
    lock.lock()
    defer { lock.unlock() }
    if engines[sha] != nil { return state(sha, "ready") }
    if loading.contains(sha) { return state(sha, "loading") }
    if let reason = failed[sha] { return state(sha, "failed", detail: reason) }
    let path = modelPath(sha)
    let size = (try? FileManager.default.attributesOfItem(atPath: path.path)[.size] as? NSNumber)?.int64Value
    if size == nbytes {
      startLoad(sha, push: push)
      return state(sha, "loading")
    }
    do {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      FileManager.default.createFile(atPath: path.path, contents: nil)
    } catch {
      return state(sha, "failed", detail: "cannot store the model: \(error.localizedDescription)")
    }
    uploads[sha] = nbytes
    return state(sha, "need_upload")
  }

  /// AUX_CHUNK: a piece of an upload at `offset`. The last piece verifies the
  /// whole file and starts the load; its state is returned, and nil for the
  /// pieces before it.
  func chunk(sha256 sha: String, offset: Int64, data: UnsafeRawBufferPointer, push: @escaping @Sendable ([String: Any]) -> Void) -> [String: Any]? {
    lock.lock()
    defer { lock.unlock() }
    guard let nbytes = uploads[sha] else {
      return state(sha, "failed", detail: "no upload of this model was asked for")
    }
    guard offset >= 0, offset + Int64(data.count) <= nbytes else {
      uploads[sha] = nil
      return state(sha, "failed", detail: "piece past the model's size")
    }
    let path = modelPath(sha)
    do {
      let handle = try FileHandle(forWritingTo: path)
      defer { try? handle.close() }
      if offset == 0 { try handle.truncate(atOffset: 0) }
      try handle.seek(toOffset: UInt64(offset))
      try handle.write(contentsOf: Data(data))
    } catch {
      uploads[sha] = nil
      return state(sha, "failed", detail: "cannot store the model: \(error.localizedDescription)")
    }
    guard offset + Int64(data.count) == nbytes else { return nil }
    uploads[sha] = nil
    let digest = (try? Registry.hashFile(path))?.0
    guard digest == sha else {
      try? FileManager.default.removeItem(at: path)
      return state(sha, "failed", detail: "sha256 mismatch after upload")
    }
    startLoad(sha, push: push)
    return state(sha, "loading")
  }

  /// Caller holds `lock`.
  private func startLoad(_ sha: String, push: @escaping @Sendable ([String: Any]) -> Void) {
    loading.insert(sha)
    let path = modelPath(sha)
    worker.async { [self] in
      let started = DispatchTime.now().uptimeNanoseconds
      let result: Result<any AuxEngine, any Error> = Result { try backend.load(model: path) }
      let took = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
      lock.lock()
      loading.remove(sha)
      let reply: [String: Any]
      switch result {
      case .success(let engine):
        if engine.input.type != .float {
          failed[sha] = "input \(engine.input.name) is not float32"
          reply = state(sha, "failed", detail: failed[sha]!)
        } else {
          engines[sha] = engine
          reply = state(sha, "ready")
          log.info("aux model \(sha.prefix(12)) ready on \(engine.device) in \(String(format: "%.1f", took)) s")
        }
      case .failure(let error):
        failed[sha] = String(describing: error)
        reply = state(sha, "failed", detail: failed[sha]!)
        log.error("aux model \(sha.prefix(12)) failed to load: \(String(describing: error))")
      }
      lock.unlock()
      push(reply)
    }
  }

  // MARK: jobs

  /// AUX_REQ: queues the job and returns nil, or returns the refusal to send
  /// at once. `reply` sends the outputs from the worker.
  func submit(
    id: NSNumber, sha256 sha: String, outputs: [String], input: UnsafeRawBufferPointer,
    reply: @escaping @Sendable (_ json: [String: Any], _ blob: [[Float]]) -> Void
  ) -> [String: Any]? {
    func refuse(_ code: String, _ detail: String) -> [String: Any] {
      ["id": id, "ok": false, "error": code, "detail": detail]
    }
    if let reason = shedReason() {
      return refuse("shed", reason)
    }
    lock.lock()
    guard let engine = engines[sha] else {
      let known = loading.contains(sha) || uploads[sha] != nil
      lock.unlock()
      return refuse(known ? "loading" : "not_loaded", "model \(sha.prefix(12)) is not ready")
    }
    guard input.count == engine.input.byteCount else {
      lock.unlock()
      return refuse("bad_shape", "input is \(input.count) bytes, the model takes \(engine.input.byteCount)")
    }
    guard queued < AuxHub.maxQueued else {
      lock.unlock()
      return refuse("busy", "\(queued) jobs queued")
    }
    queued += 1
    lock.unlock()
    let copy = Data(input)
    let received = DispatchTime.now().uptimeNanoseconds
    worker.async { [self] in
      lock.lock()
      queued -= 1
      lock.unlock()
      let started = DispatchTime.now().uptimeNanoseconds
      do {
        let results = try copy.withUnsafeBytes { try engine.run(input: $0, outputs: outputs) }
        let done = DispatchTime.now().uptimeNanoseconds
        reply(
          [
            "id": id, "ok": true, "device": engine.device,
            "queue_us": Int((started - received) / 1000), "run_us": Int((done - started) / 1000),
            "outputs": results.map { ["name": $0.0.name, "shape": $0.0.shape] },
          ], results.map(\.1))
      } catch {
        reply(["id": id, "ok": false, "error": "infer_failed", "detail": String(describing: error)], [])
      }
    }
    return nil
  }
}
