import Foundation
import SwiftProtobuf

public struct MLPackageBuilder {
    public static func writeMLPackage(
        model: CoreML_Specification_Model,
        to packageURL: URL,
        includeEmptyWeights: Bool = true
    ) throws {
        if FileManager.default.fileExists(atPath: packageURL.path) {
            try FileManager.default.removeItem(at: packageURL)
        }
        var writer = try ModelPackageWriter(packageURL: packageURL, createIfNecessary: true)

        let modelURL = try writeTemporaryModel(model: model)
        defer { try? FileManager.default.removeItem(at: modelURL) }

        _ = try writer.setRootModel(
            from: modelURL,
            name: MLProgramBuilder.defaultModelFileName,
            author: MLProgramBuilder.defaultAuthor,
            description: "CoreML Model Specification"
        )

        if includeEmptyWeights {
            let weightsDir = try createEmptyWeightsDirectory()
            defer { try? FileManager.default.removeItem(at: weightsDir) }
            _ = try writer.addItem(
                from: weightsDir,
                name: "weights",
                author: MLProgramBuilder.defaultAuthor,
                description: "CoreML Model Weights"
            )
        }

        try writer.save()
    }

    private static func writeTemporaryModel(model: CoreML_Specification_Model) throws -> URL {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("mlmodel")
        let data = try model.serializedData()
        try data.write(to: tempURL, options: [.atomic])
        return tempURL
    }

    private static func createEmptyWeightsDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
