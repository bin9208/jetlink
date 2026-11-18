import Foundation
import JetlinkKit

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif
#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Android)
  import Android
#endif

// Can the auxiliary camera models run here beside the driving model without
// slowing it? The benchmark runs the driving model at 20 Hz again, and sends
// the auxiliary jobs through `AuxHub` as a comma would: one at most after
// each driving reply, one at a time a model (the comma waits for each
// answer), at the comma's default rates. It reports the driving frames
// beside the plain run's, each model's times and the jobs it lost, and the
// process's CPU.

/// An auxiliary model as a comma runs it, for the benchmark.
public struct AuxBenchmarkModel: Sendable {
  public let name: String
  public let sha256: String
  public let bytes: Int64
  /// Where to download it when no comma has uploaded it here yet.
  public let url: String
  /// Seconds between the comma's rounds, and the jobs in each.
  public let interval: Double
  public let perRound: Int

  public init(name: String, sha256: String, bytes: Int64, url: String, interval: Double, perRound: Int) {
    self.name = name
    self.sha256 = sha256
    self.bytes = bytes
    self.url = url
    self.interval = interval
    self.perRound = perRound
  }

  var perSecond: Double { Double(perRound) / interval }

  /// Carrot's (xiaoge) at the comma's defaults: the visual blind spot every
  /// 250 ms for each side (OnnxBsdIntervalMs), the lane-marking type every
  /// 400 ms (OnnxLaneIntervalMs). The files are the comma's, by sha256, from
  /// the commit of the test branch they were checked at.
  public static let carrot: [AuxBenchmarkModel] = {
    let base =
      "https://media.githubusercontent.com/media/ajouatom/openpilot/8aacad7178fb9ab49687e3d8f0c0021b8f2f2dc1/openpilot/selfdrive/carrot/xiaoge/assets/"
    return [
      AuxBenchmarkModel(
        name: "bsd", sha256: "5d20cdbb457ba18db51a537ee2e305bbe442264b1613956068d473e35d15900d", bytes: 6_166_229,
        url: base + "v_asm_model.onnx", interval: 0.25, perRound: 2),
      AuxBenchmarkModel(
        name: "lane", sha256: "d3761c185daf33897ff3ff5edf28115c5dc829c06c538033accb616d6c67528c", bytes: 13_194_895,
        url: base + "lane.onnx", interval: 0.4, perRound: 1),
    ]
  }()
}

/// The process's CPU time so far, every thread's, in seconds.
func processCPUSeconds() -> Double {
  var ts = timespec()
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts)
  return Double(ts.tv_sec) + Double(ts.tv_nsec) / 1e9
}

/// Downloads `url` to `path`, blocking.
func downloadFile(_ url: String, to path: URL, timeout: TimeInterval = 120) throws {
  guard let target = URL(string: url) else { throw HostError.failed("bad url \(url)") }
  var request = URLRequest(url: target)
  request.timeoutInterval = timeout
  let done = DispatchSemaphore(value: 0)
  final class Box: @unchecked Sendable { var error: (any Error)? }
  let box = Box()
  let task = URLSession.shared.downloadTask(with: request) { location, response, error in
    defer { done.signal() }
    if let error {
      box.error = error
      return
    }
    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
      box.error = HostError.failed("HTTP \(http.statusCode) for \(url)")
      return
    }
    guard let location else {
      box.error = HostError.failed("no file for \(url)")
      return
    }
    do {
      try FileManager.default.moveItem(at: location, to: path)
    } catch {
      box.error = error
    }
  }
  task.resume()
  done.wait()
  if let error = box.error { throw error }
}

/// The jobs of one model through the run.
final class AuxBenchLane: @unchecked Sendable {
  let model: AuxBenchmarkModel
  let engine: any AuxEngine
  let input: [UInt8]
  private let lock = NSLock()
  var nextDue: Double
  var pending = 0
  var busy = false
  var wanted = 0
  var missed = 0
  var sent = 0
  var runMs: [Double] = []
  var waitMs: [Double] = []
  var replyMs: [Double] = []
  var refused: [String: Int] = [:]
  var failed = 0

  init(model: AuxBenchmarkModel, engine: any AuxEngine, start: Double) {
    self.model = model
    self.engine = engine
    var generator = SystemRandomNumberGenerator()
    input = (0..<(engine.input.byteCount / 4)).map { _ in UInt8.random(in: 0...255, using: &generator) }
    nextDue = start
  }

  func withLock<T>(_ body: () -> T) -> T { lock.withLock(body) }
}

extension EngineHost {
  /// Runs `hooks.auxBenchmark` beside the driving model for `seconds` and
  /// returns the log lines; none when there are no auxiliary models here.
  func auxBenchmark(_ l: Loaded, seconds: Double, alone: BenchmarkReport, cpuAlone: Double, run: BenchmarkRun) throws -> [String] {
    guard let hub = aux, !hooks.auxBenchmark.isEmpty else { return [] }
    var lines: [String] = []
    func say(_ detail: String) {
      emit(
        .benchmark(
          BenchmarkEvent(
            state: "running", elapsed: alone.seconds, total: alone.seconds, frames: alone.frames, frame: alone.frame, report: nil,
            detail: detail)))
    }
    let start = ProcessInfo.processInfo.systemUptime
    var lanes: [AuxBenchLane] = []
    for model in hooks.auxBenchmark {
      if run.cancelled { return lines }
      say("aux: preparing \(model.name)")
      do {
        let got = try hub.benchmarkEngine(sha256: model.sha256, bytes: model.bytes) { path in
          say("aux: downloading \(model.name)")
          try downloadFile(model.url, to: path)
        }
        lines.append(
          "aux: \(model.name) on \(got.engine.device), \(got.source)"
            + (got.seconds > 0 ? String(format: ", loaded in %.1f s", got.seconds) : ""))
        lanes.append(AuxBenchLane(model: model, engine: got.engine, start: start))
      } catch {
        lines.append("aux: \(model.name) not run: \(error)")
      }
    }
    guard !lanes.isEmpty else { return lines }

    let replies = DispatchGroup()
    var turn = 0
    let cpuAtStart = processCPUSeconds()
    let report = try measure(l, seconds: seconds, run: run, phase: "with aux") { totalUs in
      hub.noteFrame(totalUs: totalUs)
      let now = ProcessInfo.processInfo.systemUptime
      // rounds due: a model whose last rounds still wait loses the oldest
      for lane in lanes {
        lane.withLock {
          while now >= lane.nextDue {
            lane.nextDue += lane.model.interval
            lane.wanted += lane.model.perRound
            lane.pending += lane.model.perRound
          }
          let cap = 2 * lane.model.perRound
          if lane.pending > cap {
            lane.missed += lane.pending - cap
            lane.pending = cap
          }
        }
      }
      // one job after this reply, the models taking turns
      for k in 0..<lanes.count {
        let lane = lanes[(turn + k) % lanes.count]
        let go = lane.withLock {
          guard lane.pending > 0, !lane.busy else { return false }
          lane.pending -= 1
          lane.busy = true
          lane.sent += 1
          return true
        }
        guard go else { continue }
        turn = (turn + k + 1) % lanes.count
        let submitted = DispatchTime.now().uptimeNanoseconds
        replies.enter()
        let refusal = lane.input.withUnsafeBytes { input in
          hub.submit(id: 0, sha256: lane.model.sha256, outputs: [], input: input, inputU8: true) { json, floats in
            // the reply as the comma takes it: float16
            for values in floats {
              var half = Data(count: values.count * 2)
              half.withUnsafeMutableBytes { out in
                values.withUnsafeBytes { Convert.f32ToF16($0.baseAddress!, out.baseAddress!, count: values.count) }
              }
            }
            let replyMs = Double(DispatchTime.now().uptimeNanoseconds - submitted) / 1e6
            lane.withLock {
              lane.busy = false
              if json["ok"] as? Bool == true {
                lane.runMs.append(Double(json["run_us"] as? Int ?? 0) / 1000)
                lane.waitMs.append(Double(json["queue_us"] as? Int ?? 0) / 1000)
                lane.replyMs.append(replyMs)
              } else {
                lane.failed += 1
              }
            }
            replies.leave()
          }
        }
        if let refusal {
          lane.withLock {
            lane.busy = false
            lane.refused[refusal["error"] as? String ?? "?", default: 0] += 1
          }
          replies.leave()
        }
        break
      }
    }
    _ = replies.wait(timeout: .now() + 5)
    let cpuWith = (processCPUSeconds() - cpuAtStart) / max(report.seconds, 0.001)

    func ms(_ s: BenchmarkStats) -> String { "p50 \(s.p50) p90 \(s.p90) p99 \(s.p99) max \(s.max)" }
    lines.append(
      "aux: driving frames alone \(ms(alone.frame)), over 50 ms \(alone.over50) of \(alone.frames); "
        + "with aux \(ms(report.frame)), over 50 ms \(report.over50) of \(report.frames)")
    lines.append(
      String(format: "aux: process CPU %.0f%% alone, %.0f%% with aux (100%% = one core)", cpuAlone * 100, cpuWith * 100))
    var short: [String] = []
    for lane in lanes {
      lane.withLock {
        let done = lane.runMs.count
        let target = lane.model.perSecond
        let got = Double(done) / max(report.seconds, 0.001)
        let refusals = lane.refused.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", ")
        lines.append(
          "aux: \(lane.model.name) on \(lane.engine.device): \(done) done of \(lane.wanted) wanted "
            + String(format: "(%.1f/s of %.1f/s)", got, target)
            + ", run \(ms(BenchmarkStats.of(lane.runMs))) ms, waited p50 \(BenchmarkStats.of(lane.waitMs).p50) max \(BenchmarkStats.of(lane.waitMs).max) ms"
            + ", reply \(ms(BenchmarkStats.of(lane.replyMs))) ms"
            + (lane.missed > 0 ? ", \(lane.missed) dropped behind" : "")
            + (refusals.isEmpty ? "" : ", refused: \(refusals)")
            + (lane.failed > 0 ? ", \(lane.failed) failed" : ""))
        if got < 0.9 * target { short.append("\(lane.model.name) \(String(format: "%.1f/s of %.1f/s", got, target))") }
        if lane.engine.device == "CPU" { short.append("\(lane.model.name) on the CPU") }
      }
    }
    // The comma's 50 ms: a driving frame later than that is a frame lost.
    var reasons = short
    let lateShare = Double(report.over50) / Double(max(report.frames, 1))
    if lateShare > 0.005 { reasons.insert("\(report.over50) driving frames over 50 ms", at: 0) }
    if report.frame.p99 > 45 { reasons.insert("driving p99 \(report.frame.p99) ms", at: 0) }
    lines.append(reasons.isEmpty ? "aux: verdict: usable, the driving model keeps its pace" : "aux: verdict: not yet, \(reasons.joined(separator: "; "))")
    return lines
  }
}
