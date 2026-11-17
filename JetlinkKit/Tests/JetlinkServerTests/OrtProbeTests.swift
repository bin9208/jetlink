import Foundation
import JetlinkServer
import Testing

@testable import JetlinkONNX
@testable import JetlinkORT

@Suite("Probing a model against the CPU", .serialized)
struct OrtProbeTests {
  @Test("The same graph on the CPU on both sides differs nowhere")
  func cpuAgainstItself() throws {
    let backend = OrtBackend(profile: .cpu, preparer: ONNXPreparer(), keepAlive: false, keepCPUWarm: false)
    let dir = FileManager.default.temporaryDirectory.appending(path: "probe-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    var seen: [String] = []
    let lines = backend.probe(
      model: Fixture.url("tiny_stateful.onnx"), directory: dir, layout: OrtProfile.cpu.layout, provider: nil, options: [:], label: "cpu",
      progress: { seen.append($0) })
    #expect(lines.count == 3, "\(lines)")
    #expect(lines.first?.hasPrefix("probe: ") == true && lines.first?.contains("against the CPU, 3 frames") == true, "\(lines)")
    #expect(lines.contains("probe: no tensor over 5% from the CPU"), "\(lines)")
    #expect(seen.contains("probe: frame 3/3"))
    #expect(lines.first?.hasPrefix("probe: 0 ") == false, "\(lines)")
    #expect(!FileManager.default.fileExists(atPath: dir.appending(path: "probe-cpu").path))
  }
}
