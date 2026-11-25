import Foundation

/// Whether the connected comma is driving on NOO (carrot's Navigate On
/// Openpilot): a navigation graph such as N16 is loaded and the comma's
/// frames carry a maneuver ahead in its `navigation` input. Without a route
/// the comma sends that input as zeros, and turns are left to carrot's ATC.
public struct NooStatus: Codable, Sendable, Equatable {
  /// The loaded graph takes navigation inputs.
  public let model: Bool
  /// A frame in the last `NooStatus.hold` seconds carried a maneuver.
  public let active: Bool
  /// That maneuver as a Valhalla command (the one-hot's index), and how far.
  public let command: Int?
  public let distanceM: Double?

  /// A route's input can drop out between two maneuvers for a frame or two;
  /// the badge should not blink with it.
  public static let hold: TimeInterval = 3

  public init(model: Bool, active: Bool, command: Int? = nil, distanceM: Double? = nil) {
    self.model = model
    self.active = active
    self.command = command
    self.distanceM = distanceM
  }

  /// The `navigation` floats of one frame: the first slot of the one-hot
  /// that is set, and the last float's distance in hundreds of metres.
  public static func maneuver(_ floats: UnsafeBufferPointer<Float>) -> (command: Int, distanceM: Double)? {
    guard floats.count >= 2 else { return nil }
    let distance = Double(floats[floats.count - 1]) * 100
    for index in 0..<(floats.count - 1) where floats[index] > 0.5 {
      return (index, distance.isFinite ? max(0, distance) : 0)
    }
    return nil
  }
}
