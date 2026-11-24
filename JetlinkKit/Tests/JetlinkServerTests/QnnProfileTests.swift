import Testing

@testable import JetlinkORT

@Suite struct QnnProfileTests {
  @Test("QNN's profile CSV becomes calls and the slowest nodes")
  func readsTheCSV() {
    let csv = """
      Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier
      1,,20000,US,BACKEND,ROOT,QNN execute
      1,,300,CYCLES,BACKEND,SUB-EVENT,node_conv_1
      1,,900,CYCLES,BACKEND,SUB-EVENT,node_matmul_2
      2,,22000,US,BACKEND,ROOT,QNN execute
      2,,500,CYCLES,BACKEND,SUB-EVENT,node_conv_1
      2,,1100,CYCLES,BACKEND,SUB-EVENT,node_matmul_2
      """
    let p = QnnProfile(csv: csv)
    #expect(p.rows == 6)
    #expect(p.roots["QNN execute"]?.count == 2)
    let lines = p.summary(wallMs: [24, 25, 26])
    #expect(lines[0].contains("run p50 25.00 ms over 3 runs; 6 profile rows"))
    #expect(lines[1] == "profile: calls (mean): QNN execute 21000.0 US")
    #expect(lines[2] == "profile: 2 nodes, 1400 CYCLES a run in all")
    #expect(lines[3] == "profile: top 1-2: node_matmul_2 1000 (71.4%); node_conv_1 400 (28.6%)")
    #expect(lines[4] == "profile: kinds 1-2: matmul 71.4% (1); conv 28.6% (1)")
  }

  @Test("Build steps in microseconds stay out of the run's cycles, and nodes add up by kind")
  func separatesBuildStepsAndGroupsKinds() {
    let csv = """
      Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier
      1,,9000,US,BACKEND,SUB-EVENT,VTCM Allocation
      1,,600,CYCLES,BACKEND,SUB-EVENT,node_gelu_35_12:OpId_7920 (cycles)
      1,,200,CYCLES,BACKEND,SUB-EVENT,node_gelu_7_12:OpId_100 (cycles)
      1,,200,CYCLES,BACKEND,SUB-EVENT,val_545__reshape_out_12:OpId_7898 (cycles)
      """
    let lines = QnnProfile(csv: csv).summary(wallMs: [30])
    #expect(lines.contains("profile: 3 nodes, 1000 CYCLES a run in all"))
    #expect(lines.contains("profile: kinds 1-2: gelu 80.0% (2); reshape_out 20.0% (1)"))
    #expect(lines.contains("profile: build steps (not the run): VTCM Allocation 9000 US"))
    #expect(QnnProfile.kind("node_conv2d_36_token_316_6:OpId_1 (cycles)") == "conv2d_token")
  }
}
