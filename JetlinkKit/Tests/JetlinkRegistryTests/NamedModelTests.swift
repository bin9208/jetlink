import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// Models a comma uploads: named by the comma, or by the built-in names.
struct NamedModelTests {
  @Test func aModelTheCommaUploadedIsNamedByTheBuiltInNames() throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout)
    let n16 = "01effc679eb46d6036ce0302673a87fe2dcfcc95a679daeeb929679e53c3647a"
    try FileManager.default.createDirectory(at: tmp.layout.models, withIntermediateDirectories: true)
    // half sent: the file has the model's 16-hex name only
    try Data(repeating: 1, count: 100).write(to: tmp.layout.models.appending(path: "\(n16.prefix(16)).onnx"))
    let models = registry.inventory(artifactTag: nil, artifactSuffix: ".ortcache", loaded: nil).models
    #expect(models.map(\.sha256) == [n16])
    #expect(models.map(\.name) == ["N16 Nav (carrot)"])
  }

  @Test func theCommasNameIsRecordedOnceAndKept() throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout)
    try registry.nameModel(sha256: RegistryFixture.blobSHA, name: "Comma Model")
    #expect(registry.name(for: RegistryFixture.blobSHA).name == "Comma Model")
    try registry.nameModel(sha256: RegistryFixture.blobSHA, name: "Another")
    #expect(registry.name(for: RegistryFixture.blobSHA).name == "Comma Model")
    try registry.nameModel(sha256: RegistryFixture.blobSHA, name: "Another", replace: true)
    #expect(registry.name(for: RegistryFixture.blobSHA).name == "Another")
    // not a digest: nothing written
    try registry.nameModel(sha256: "abc", name: "x")
    #expect(registry.localModels().count == 1)
  }
}

struct KnownCatalogTests {
  @Test func carrotsN16IsInTheCatalogFirstEvenOffline() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout, session: MockNet([:]).session)
    let catalog = await registry.catalog().withKnownModels()
    #expect(await registry.catalog().models.isEmpty, "the index itself stays as served")
    let n16 = try #require(catalog.models.first)
    #expect(n16.name == "N16 Nav (carrot)" && n16.sha256?.hasPrefix("01effc679eb46d60") == true && n16.bytes == 814_591_107)
    #expect(n16.ref == String(n16.sha256!.prefix(40)))
    let pointer = try await registry.resolve(ref: n16.ref)
    #expect(pointer.oid == n16.sha256 && pointer.size == 814_591_107)
    #expect(try await registry.pointer(for: n16.sha256!) == pointer)
    #expect(registry.size(for: n16.sha256!) == 814_591_107)
  }

  @Test func theNavStudentsFollowN16WithTheirSizes() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout, session: MockNet([:]).session)
    let catalog = await registry.catalog().withKnownModels()
    #expect(catalog.models.map(\.name).prefix(3) == ["N16 Nav (carrot)", "LS2 Nav (carrot)", "BM Nav (carrot)"])
    for (name, prefix, bytes) in [("LS2 Nav (carrot)", "12f8356d0e96f280", Int64(791_023_125)), ("BM Nav (carrot)", "a5ee04acf37a3528", Int64(1_786_430_748))] {
      let row = try #require(catalog.models.first { $0.name == name })
      #expect(row.sha256?.hasPrefix(prefix) == true && row.bytes == bytes && row.folder == "Carrot")
      let pointer = try await registry.resolve(ref: row.ref)
      #expect(pointer.oid == row.sha256 && pointer.size == bytes)
      #expect(KnownModels.bySHA(row.sha256!)?.url.hasPrefix("https://lunabox.myqnapcloud.com/share.cgi?ssid=") == true)
    }
  }
}
