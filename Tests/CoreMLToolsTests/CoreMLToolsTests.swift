import Foundation
import Testing
import CoreMLTools

@Test
func addConstantModelSerialization() async throws {
    let model = try MLProgramBuilder.makeAddConstantModel(shape: [1], constant: 2.0)
    let data = try model.serializedData()
    #expect(!data.isEmpty)
    let decoded = try CoreML_Specification_Model(serializedData: data)
    #expect(decoded.specificationVersion == 9)
    #expect(decoded.description_p.input.first?.name == "x")
}

@Test
func packageCreationWritesManifestAndModel() async throws {
    let packageURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("mlpackage")
    defer { try? FileManager.default.removeItem(at: packageURL) }

    let model = try MLProgramBuilder.makeAddConstantModel(shape: [1], constant: 1.0)
    try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)

    let manifestURL = packageURL.appendingPathComponent("Manifest.json")
    #expect(FileManager.default.fileExists(atPath: manifestURL.path))

    let modelURL = packageURL
        .appendingPathComponent("Data")
        .appendingPathComponent(MLProgramBuilder.defaultAuthor)
        .appendingPathComponent(MLProgramBuilder.defaultModelFileName)
    #expect(FileManager.default.fileExists(atPath: modelURL.path))
}
