import Foundation
import CoreML
import CoreMLTools

let packageURL = FileManager.default.temporaryDirectory
    .appendingPathComponent("AddConstant", isDirectory: false)
    .appendingPathExtension("mlpackage")

if FileManager.default.fileExists(atPath: packageURL.path) {
    try? FileManager.default.removeItem(at: packageURL)
}

let model = try MLProgramBuilder.makeAddConstantModel(shape: [1], constant: 1.0)
try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
let compiledURL = try MLModel.compileModel(at: packageURL)
let mlModel = try MLModel(contentsOf: compiledURL)

let input = try MLMultiArray(shape: [1], dataType: .float32)
input[0] = 41

let provider = try MLDictionaryFeatureProvider(dictionary: ["x": input])
let output = try mlModel.prediction(from: provider)

if let result = output.featureValue(for: "y")?.multiArrayValue {
    print("y = \(result[0])")
} else {
    print("Failed to read output")
}
