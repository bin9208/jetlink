import Testing

@testable import JetlinkORT

@Suite struct MemoryStatusTests {
  /// /proc/self/status pads its numbers with a tab, /proc/meminfo with
  /// spaces; carrot.57 read only the spaces and logged the app as "?".
  @Test func readsBothPaddings() {
    let status = MemoryStatus.parse("Name:\tjetlink\nVmRSS:\t 1843200 kB\nRssAnon:\t  1740800 kB\n")
    #expect(status["VmRSS"] == 1_843_200)
    #expect(status["RssAnon"] == 1_740_800)
    #expect(status["Name"] == nil)
    let info = MemoryStatus.parse("MemTotal:       11427588 kB\nMemAvailable:    4404020 kB\n")
    #expect(info["MemTotal"] == 11_427_588)
    #expect(info["MemAvailable"] == 4_404_020)
  }

  @Test func describesThisProcessOnLinux() throws {
    #if os(Linux) || os(Android)
      let line = try #require(MemoryStatus.describe())
      #expect(!line.contains("?"))
    #endif
  }
}
