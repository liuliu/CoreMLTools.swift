import Foundation

public final class ModelPackageWriter {
    public struct ItemInfo: Codable, Equatable {
        public var path: String
        public var name: String
        public var author: String
        public var description: String
    }

    private struct Manifest: Codable {
        var fileFormatVersion: String
        var itemInfoEntries: [String: ItemInfo]
        var rootModelIdentifier: String?
    }

    private let packageURL: URL
    private let dataDirURL: URL
    private var manifest: Manifest

    public init(packageURL: URL, createIfNecessary: Bool = true) throws {
        self.packageURL = packageURL
        self.dataDirURL = packageURL.appendingPathComponent("Data", isDirectory: true)

        if FileManager.default.fileExists(atPath: packageURL.path) {
            let manifestURL = packageURL.appendingPathComponent("Manifest.json")
            guard FileManager.default.fileExists(atPath: manifestURL.path) else {
                throw CoreMLToolsError.manifestMissing
            }
            let data = try Data(contentsOf: manifestURL)
            self.manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } else if createIfNecessary {
            try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: dataDirURL, withIntermediateDirectories: true)
            self.manifest = Manifest(fileFormatVersion: "1.0.0", itemInfoEntries: [:], rootModelIdentifier: nil)
        } else {
            throw CoreMLToolsError.invalidPackagePath
        }
    }

    public func addItem(from sourceURL: URL, name: String, author: String, description: String) throws -> String {
        if manifest.itemInfoEntries.values.contains(where: { $0.name == name && $0.author == author }) {
            throw CoreMLToolsError.itemAlreadyExists(name: name, author: author)
        }

        let relativePath = author + "/" + name
        let destinationURL = dataDirURL.appendingPathComponent(relativePath, isDirectory: false)

        try FileManager.default.createDirectory(at: destinationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.copyItem(at: sourceURL, to: destinationURL)

        let identifier = UUID().uuidString
        manifest.itemInfoEntries[identifier] = ItemInfo(
            path: relativePath,
            name: name,
            author: author,
            description: description
        )
        return identifier
    }

    public func setRootModel(from sourceURL: URL, name: String, author: String, description: String) throws -> String {
        if manifest.rootModelIdentifier != nil {
            throw CoreMLToolsError.rootModelAlreadyExists
        }
        let identifier = try addItem(from: sourceURL, name: name, author: author, description: description)
        manifest.rootModelIdentifier = identifier
        return identifier
    }

    public func save() throws {
        let manifestURL = packageURL.appendingPathComponent("Manifest.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: manifestURL, options: [.atomic])
    }
}
