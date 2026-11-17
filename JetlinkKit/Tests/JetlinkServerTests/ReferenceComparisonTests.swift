import Testing

@testable import JetlinkServer

@Suite struct ReferenceComparisonTests {
  private let spec = ModelSpec(
    sha256: "", nbytes: 0, frameSkip: 4, inputShapes: [], outputShapes: [NamedShape("outputs", [1, 2000])],
    outputSlices: [NamedSlice("lane_lines", start: 0, stop: 528), NamedSlice("plan", start: 917, stop: 1907), NamedSlice("lead", start: 1907, stop: 1950)],
    checkpoint: nil)

  @Test("The accelerator's plan end, lanes and lead are measured against the CPU's")
  func measuresTheDifference() throws {
    var c = try #require(ReferenceComparison(spec: spec))
    var cpu = [Float](repeating: 0, count: 2000)
    var npu = cpu
    let endY = 917 + 32 * 15 + 1
    c.record(frame: 0, accelerator: npu, cpu: cpu)
    npu[endY] = 4.5
    npu[10] = 0.25
    cpu[1907] = 30
    c.record(frame: 1, accelerator: npu, cpu: cpu)
    let line = try #require(c.summary)
    #expect(line.contains("over 2 changing frames"))
    #expect(line.contains("plan end y mean 2.250 max 4.500 m"))
    #expect(line.contains("lane lines mean 0.125 max 0.250"))
    #expect(line.contains("lead x mean 15.000 max 30.000 m"))
    #expect(line.contains("worst plan end y at frame 1"))
  }

  @Test("A model without a single 33x15 plan is not compared")
  func needsAPlan() {
    let other = ModelSpec(
      sha256: "", nbytes: 0, frameSkip: 4, inputShapes: [], outputShapes: [], outputSlices: [NamedSlice("plan", start: 0, stop: 4955)],
      checkpoint: nil)
    #expect(ReferenceComparison(spec: other) == nil)
  }
}
