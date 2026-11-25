import Foundation
import JetlinkKit

/// The navigation input of the frames the comma sends, kept for the app's
/// NOO badge. Its own lock: frames record under the host's lock, the app
/// reads from its snapshot thread.
final class NooWatch: @unchecked Sendable {
  private let lock = NSLock()
  private var sha256 = ""
  /// Where `navigation` sits in the packed floats; nil for a graph without it.
  private var range: Range<Int>?
  private var command: Int?
  private var distanceM: Double?
  private var activeAt: TimeInterval = -.infinity

  /// A frame the model ran; `packed` is its floats after the image, as arrived.
  func offer(sha256: String, spec: ModelSpec, packed: UnsafeRawPointer, now: TimeInterval) {
    lock.lock()
    defer { lock.unlock() }
    if sha256 != self.sha256 {
      self.sha256 = sha256
      range = spec.packedLayout.first { $0.name == ModelConstants.navigation }?.range
      command = nil
      distanceM = nil
      activeAt = -.infinity
    }
    guard let range, range.count >= 2 else { return }
    // The packed floats arrive unaligned (Session.infer); copy the few out.
    var floats = [Float](repeating: 0, count: range.count)
    for (slot, index) in range.enumerated() {
      floats[slot] = Float(bitPattern: UInt32(littleEndian: packed.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)))
    }
    guard let maneuver = floats.withUnsafeBufferPointer(NooStatus.maneuver) else { return }
    command = maneuver.command
    distanceM = maneuver.distanceM
    activeAt = now
  }

  /// The loaded graph's NOO state; nil before its first frame.
  func status(now: TimeInterval) -> NooStatus? {
    lock.lock()
    defer { lock.unlock() }
    guard !sha256.isEmpty else { return nil }
    guard range != nil else { return NooStatus(model: false, active: false) }
    let active = now - activeAt < NooStatus.hold
    return NooStatus(model: true, active: active, command: active ? command : nil, distanceM: active ? distanceM : nil)
  }

  /// The comma went: the next one starts from nothing.
  func reset() {
    lock.lock()
    sha256 = ""
    range = nil
    command = nil
    distanceM = nil
    activeAt = -.infinity
    lock.unlock()
  }
}
