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
