import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// Carrot's auxiliary vision over the wire (`AuxHub`): a single-input model
/// (`aux_tiny.onnx`: output0 = images * 2, output1 = images + 1) uploaded,
/// loaded on the CPU provider and run.
@Suite struct AuxTests {
  static let model = Fixture.url("aux_tiny.onnx")
  static let sha = "db0dfb20462dc1df6bd94cc8a60fd2abda116c8f7c4284fc9cf40ceb4d8561c0"

  static func payload(_ json: [String: Any], _ blob: Data = Data()) throws -> Data {
    let body = try JSONSerialization.data(withJSONObject: json)
    var out = Data()
    withUnsafeBytes(of: UInt32(body.count).littleEndian) { out.append(contentsOf: $0) }
    out.append(body)
    out.append(blob)
    return out
  }

  static func decode(_ reply: Reply) throws -> (json: [String: Any], blob: Data) {
    let length = Int(reply.payload.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
    let json = try JSONSerialization.jsonObject(with: reply.payload.subdata(in: 4..<(4 + length))) as! [String: Any]
    return (json, reply.payload.subdata(in: (4 + length)..<reply.payload.count))
  }

  static func floats(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { Data(buffer: $0) }
  }

  /// Uploads the model and waits until it is ready; returns the ready state.
  static func ready(_ client: TestClient) throws -> [String: Any] {
    let bytes = try Data(contentsOf: model)
    try client.send(.auxLoad, payload(["sha256": sha, "nbytes": bytes.count]))
    #expect(try decode(client.recv(.auxState)).json["state"] as? String == "need_upload")
    let half = bytes.count / 2
    try client.send(.auxChunk, payload(["sha256": sha, "offset": 0], bytes.subdata(in: 0..<half)))
    try client.send(.auxChunk, payload(["sha256": sha, "offset": half], bytes.subdata(in: half..<bytes.count)))
    var state = try decode(client.recv(.auxState)).json
    #expect(state["state"] as? String == "loading")
    while state["state"] as? String == "loading" {
      state = try decode(client.recv(.auxState)).json
    }
    return state
  }

  @Test func helloOffersAuxiliaryVision() throws {
    try serve { _, client in
      try client.sendJSON(.helloReq, ["client": ["name": "test", "nonce": "1"]])
      #expect(try client.recv(.helloResp).json["carrot_aux_v1"] as? Bool == true)
    }
  }

  @Test func uploadsLoadsAndRunsAModel() throws {
    try serve { _, client in
      try client.sendJSON(.helloReq, ["client": ["name": "test", "nonce": "1"]])
      _ = try client.recv(.helloResp)
      let state = try Self.ready(client)
      #expect(state["state"] as? String == "ready")
      #expect(state["device"] as? String == "CPU")
      #expect((state["input"] as? [String: Any])?["shape"] as? [Int] == [1, 3, 4, 4])

      let input = (0..<48).map { Float($0) / 4 }
      try client.send(.auxReq, Self.payload(["id": 7, "sha256": Self.sha, "outputs": ["output1"]], Self.floats(input)))
      let (json, blob) = try Self.decode(client.recv(.auxResp))
      #expect(json["ok"] as? Bool == true)
      #expect(json["id"] as? Int == 7)
      #expect((json["outputs"] as? [[String: Any]])?.first?["name"] as? String == "output1")
      let values = blob.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
      #expect(values == input.map { $0 + 1 })

      // Every output, in model order, when none are named.
      try client.send(.auxReq, Self.payload(["id": 8, "sha256": Self.sha], Self.floats(input)))
      let all = try Self.decode(client.recv(.auxResp))
      #expect((all.json["outputs"] as? [[String: Any]])?.map { $0["name"] as? String } == ["output0", "output1"])
      #expect(all.blob.count == 2 * 48 * 4)

      // A model already here is ready at once, with no upload.
      try client.send(.auxLoad, Self.payload(["sha256": Self.sha, "nbytes": try Data(contentsOf: Self.model).count]))
      #expect(try Self.decode(client.recv(.auxState)).json["state"] as? String == "ready")
    }
  }

  @Test func refusesAWrongInputAndAnUnknownModel() throws {
    try serve { _, client in
      try client.sendJSON(.helloReq, ["client": ["name": "test", "nonce": "1"]])
      _ = try client.recv(.helloResp)
      try client.send(.auxReq, Self.payload(["id": 1, "sha256": Self.sha], Self.floats([1, 2, 3])))
      #expect(try Self.decode(client.recv(.auxResp)).json["error"] as? String == "not_loaded")
      _ = try Self.ready(client)
      try client.send(.auxReq, Self.payload(["id": 2, "sha256": Self.sha], Self.floats([1, 2, 3])))
      #expect(try Self.decode(client.recv(.auxResp)).json["error"] as? String == "bad_shape")
    }
  }

  @Test func refusesACorruptUpload() throws {
    try serve { _, client in
      try client.sendJSON(.helloReq, ["client": ["name": "test", "nonce": "1"]])
      _ = try client.recv(.helloResp)
      var bytes = try Data(contentsOf: Self.model)
      try client.send(.auxLoad, Self.payload(["sha256": Self.sha, "nbytes": bytes.count]))
      _ = try client.recv(.auxState)
      bytes[bytes.count - 1] ^= 0xff
      try client.send(.auxChunk, Self.payload(["sha256": Self.sha, "offset": 0], bytes))
      let state = try Self.decode(client.recv(.auxState)).json
      #expect(state["state"] as? String == "failed")
      #expect((state["detail"] as? String)?.contains("sha256") == true)
    }
  }

  @Test func shedsJobsWhileTheDeviceIsHot() throws {
    try serve(hooks: ServerHooks(thermal: { "serious" })) { _, client in
      try client.sendJSON(.helloReq, ["client": ["name": "test", "nonce": "1"]])
      _ = try client.recv(.helloResp)
      _ = try Self.ready(client)
      try client.send(.auxReq, Self.payload(["id": 3, "sha256": Self.sha], Self.floats([Float](repeating: 0, count: 48))))
      let json = try Self.decode(client.recv(.auxResp)).json
      #expect(json["error"] as? String == "shed")
      #expect((json["detail"] as? String)?.contains("serious") == true)
    }
  }

  @Test("A benchmark runs the auxiliary models from the cache beside the driving model, and skips one it cannot get")
  func benchmarksBesideTheDrivingModel() throws {
    let golden = try Golden("tiny_queued")
    let bytes = try Data(contentsOf: Self.model)
    let nowhere = "http://127.0.0.1:9/none.onnx"
    let tiny = AuxBenchmarkModel(name: "tiny", sha256: Self.sha, bytes: Int64(bytes.count), url: nowhere, interval: 0.1, perRound: 2)
    let missing = AuxBenchmarkModel(
      name: "missing", sha256: String(repeating: "0", count: 64), bytes: 10, url: nowhere, interval: 0.1, perRound: 1)
    try serve(hooks: ServerHooks(auxBenchmark: [tiny, missing])) { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      client.close()
      let deadline = Date().addingTimeInterval(5)
      while server.host.lock.withLock({ server.host.session != nil }) && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.02)
      }
      // as a comma's upload leaves it
      let directory = server.host.cache.layout.root.appending(path: "aux", directoryHint: .isDirectory)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      try bytes.write(to: directory.appending(path: "\(Self.sha).onnx"))

      let report = try server.host.benchmark(seconds: 2, run: BenchmarkRun())
      #expect(!report.cancelled)
      // again, to read its lines: the model is loaded now
      let l = try #require(server.host.lock.withLock { server.host.loaded })
      let lines = try server.host.auxBenchmark(l, seconds: 1, alone: report, cpuAlone: 0.5, run: BenchmarkRun())
      #expect(lines.contains("aux: tiny on CPU, loaded"), "\(lines)")
      #expect(lines.contains { $0.hasPrefix("aux: missing not run:") }, "\(lines)")
      let line = try #require(lines.first { $0.hasPrefix("aux: tiny on CPU: ") })
      let done = Int(line.split(separator: " ")[4]) ?? 0
      #expect(done >= 5, "\(line)")
      #expect(!line.contains("refused"), "\(line)")
      #expect(lines.contains { $0.hasPrefix("aux: driving frames alone p50 ") })
      #expect(lines.contains { $0.hasPrefix("aux: process CPU ") })
      #expect(lines.last == "aux: verdict: not yet, tiny on the CPU", "\(lines)")
    }
  }
}
