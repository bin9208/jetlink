import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

@Suite struct NooWatchTests {
  /// N16's inputs as carrot's spec names them, small.
  private let n16 = ModelSpec(
    sha256: String(repeating: "16", count: 32), nbytes: 0, frameSkip: 4,
    inputShapes: [
      NamedShape("new_img", [2, 6, 8, 16]), NamedShape("desire", [1, 8]), NamedShape("traffic_convention", [1, 2]),
      NamedShape("action_t", [1, 2]), NamedShape("navigation", [1, 47]), NamedShape("nav_map", [1, 1, 16, 16]),
      NamedShape("state_feat_q", [4, 1, 8]),
    ], outputShapes: [NamedShape("outputs", [1, 100]), NamedShape("next_state_feat_q", [4, 1, 8])], outputSlices: [], checkpoint: nil)

  private let plain = ModelSpec(
    sha256: String(repeating: "ab", count: 32), nbytes: 0, frameSkip: 4,
    inputShapes: [
      NamedShape("new_img", [2, 6, 8, 16]), NamedShape("desire", [1, 8]), NamedShape("traffic_convention", [1, 2]),
      NamedShape("action_t", [1, 2]), NamedShape("state_feat_q", [4, 1, 8]),
    ], outputShapes: [NamedShape("outputs", [1, 100]), NamedShape("next_state_feat_q", [4, 1, 8])], outputSlices: [], checkpoint: nil)

  /// The packed floats of one frame, the maneuver one-hot at `command` and
  /// `metres` ahead, offset by one byte as the wire leaves them unaligned.
  private func offer(_ watch: NooWatch, _ spec: ModelSpec, command: Int?, metres: Float, at now: TimeInterval) {
    var floats = [Float](repeating: 0, count: spec.packedCount)
    if let range = spec.packedLayout.first(where: { $0.name == "navigation" })?.range, let command {
      floats[range.lowerBound + command] = 1
      floats[range.upperBound - 1] = metres / 100
    }
    var bytes = [UInt8](repeating: 0, count: 1)
    floats.withUnsafeBytes { bytes += $0 }
    bytes.withUnsafeBytes { watch.offer(sha256: spec.sha256, spec: spec, packed: $0.baseAddress! + 1, now: now) }
  }

  @Test("N16 with a route ahead is NOO, and says the maneuver")
  func routeIsActive() throws {
    let watch = NooWatch()
    #expect(watch.status(now: 0) == nil)
    offer(watch, n16, command: 15, metres: 320, at: 10)
    let status = try #require(watch.status(now: 10.05))
    #expect(status.model && status.active && status.command == 15)
    #expect(abs((status.distanceM ?? 0) - 320) < 0.01)
  }

  @Test("N16 without a route waits; a gap of a frame or two does not drop it")
  func noRouteWaits() throws {
    let watch = NooWatch()
    offer(watch, n16, command: nil, metres: 0, at: 1)
    #expect(watch.status(now: 1) == NooStatus(model: true, active: false))
    offer(watch, n16, command: 10, metres: 80, at: 2)
    offer(watch, n16, command: nil, metres: 0, at: 2.1)
    #expect(try #require(watch.status(now: 2.1)).active)
    // the route ended: inactive once the hold has passed
    #expect(try #require(watch.status(now: 2 + NooStatus.hold + 0.1)).active == false)
  }

  @Test("A model without navigation inputs is not NOO; a new comma starts over")
  func otherModel() throws {
    let watch = NooWatch()
    offer(watch, n16, command: 8, metres: 50, at: 1)
    offer(watch, plain, command: nil, metres: 0, at: 1.1)
    #expect(watch.status(now: 1.1) == NooStatus(model: false, active: false))
    watch.reset()
    #expect(watch.status(now: 2) == nil)
  }

  @Test("The one-hot's first set slot, and its distance in hundreds of metres")
  func maneuverOfFloats() {
    var floats = [Float](repeating: 0, count: 47)
    #expect(floats.withUnsafeBufferPointer(NooStatus.maneuver) == nil)
    floats[24] = 1
    floats[46] = 2.5
    let maneuver = floats.withUnsafeBufferPointer(NooStatus.maneuver)
    #expect(maneuver?.command == 24 && maneuver?.distanceM == 250)
  }
}
