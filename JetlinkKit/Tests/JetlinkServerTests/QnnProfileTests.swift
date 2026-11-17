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
  }
}
