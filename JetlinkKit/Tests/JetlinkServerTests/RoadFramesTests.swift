import Foundation
import Testing

@testable import JetlinkServer

@Suite struct RoadFramesTests {
  private let spec = ModelSpec(
    sha256: "", nbytes: 0, frameSkip: 4,
    inputShapes: [
      NamedShape("img", [1, 12, 8, 16]), NamedShape("big_img", [1, 12, 8, 16]), NamedShape("desire_pulse", [1, 25, 8]),
      NamedShape("traffic_convention", [1, 2]), NamedShape("action_t", [1, 2]), NamedShape("features_buffer", [1, 32, 32, 4]),
    ], outputShapes: [NamedShape("outputs", [1, 2000])], outputSlices: [], checkpoint: nil)

  private func offer(_ r: RoadFrames, _ sha: String, _ value: UInt8) {
    let warped = [UInt8](repeating: value, count: spec.warpedBytes)
    let packed = [UInt8](repeating: value &+ 1, count: spec.packedBytes)
    warped.withUnsafeBytes { w in
      packed.withUnsafeBytes { p in
        r.offer(sha256: sha, warped: w.baseAddress!, warpedBytes: w.count, packed: p.baseAddress!, packedBytes: p.count)
      }
    }
  }

  @Test("One frame in `every` is kept, the last `count`, written and read back in order")
  func keepsTheLastFrames() throws {
    let dir = FileManager.default.temporaryDirectory.appending(path: "roadframes-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    let sha = String(repeating: "ab", count: 32)
    let r = RoadFrames(directory: dir)
    // too few kept frames: nothing written
    for i in 0..<(RoadFrames.every * 3) { offer(r, sha, UInt8(i / RoadFrames.every)) }
    r.save()
    #expect(RoadFrames.load(in: dir, sha256: sha, spec: spec) == nil)
    // more than `count` kept: the oldest go
    let kept = RoadFrames.count + 5
    for i in (RoadFrames.every * 3)..<(RoadFrames.every * (kept + 3)) { offer(r, sha, UInt8(truncatingIfNeeded: i / RoadFrames.every)) }
    r.save()
    let frames = try #require(RoadFrames.load(in: dir, sha256: sha, spec: spec))
    #expect(frames.count == RoadFrames.count)
    // the frame offered at i = every*k - 1 carries k - 1; the last kept is k = kept + 3
    #expect(frames.last?.warped.first == UInt8(kept + 2))
    #expect(frames.first?.warped.first == UInt8(kept + 3 - RoadFrames.count))
    #expect(frames.last?.packed.first == UInt8(kept + 3))
    #expect(frames.allSatisfy { $0.warped.count == spec.warpedBytes && $0.packed.count == spec.packedBytes })
  }

  @Test("Frames of another size are not read")
  func refusesAnotherModel() throws {
    let dir = FileManager.default.temporaryDirectory.appending(path: "roadframes-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    let sha = String(repeating: "cd", count: 32)
    let r = RoadFrames(directory: dir)
    for i in 0..<(RoadFrames.every * RoadFrames.count) { offer(r, sha, UInt8(truncatingIfNeeded: i)) }
    r.save()
    #expect(RoadFrames.load(in: dir, sha256: sha, spec: spec) != nil)
    let other = ModelSpec(
      sha256: "", nbytes: 0, frameSkip: 4,
      inputShapes: [
        NamedShape("img", [1, 12, 16, 16]), NamedShape("big_img", [1, 12, 16, 16]), NamedShape("desire_pulse", [1, 25, 8]),
        NamedShape("traffic_convention", [1, 2]), NamedShape("action_t", [1, 2]), NamedShape("features_buffer", [1, 32, 32, 4]),
      ], outputShapes: [NamedShape("outputs", [1, 2000])], outputSlices: [], checkpoint: nil)
    #expect(RoadFrames.load(in: dir, sha256: sha, spec: other) == nil)
  }
}
