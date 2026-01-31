import Foundation
import CoreML
import CoreMLTools

struct MLTestUtils {
    static func runFloatModelNoInputs(
        model: CoreML_Specification_Model,
        outputName: String
    ) throws -> [Float] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let provider = try MLDictionaryFeatureProvider(dictionary: [:])
        let output = try mlModel.prediction(from: provider)
        guard let outArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        var values: [Float] = []
        values.reserveCapacity(outArray.count)
        for idx in 0..<outArray.count {
            values.append(outArray[idx].floatValue)
        }
        return values
    }

    static func runInt32ModelNoInputs(
        model: CoreML_Specification_Model,
        outputName: String
    ) throws -> [Int32] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let provider = try MLDictionaryFeatureProvider(dictionary: [:])
        let output = try mlModel.prediction(from: provider)
        guard let outArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        var values: [Int32] = []
        values.reserveCapacity(outArray.count)
        for idx in 0..<outArray.count {
            values.append(outArray[idx].int32Value)
        }
        return values
    }

    static func runModelOutputs(
        model: CoreML_Specification_Model,
        inputName: String,
        inputShape: [Int],
        inputValues: [Float],
        outputNames: [String]
    ) throws -> [String: MLMultiArray] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArray = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValues.enumerated() {
            inputArray[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        let output = try mlModel.prediction(from: provider)

        var results: [String: MLMultiArray] = [:]
        for name in outputNames {
            if let outArray = output.featureValue(for: name)?.multiArrayValue {
                results[name] = outArray
            }
        }
        return results
    }

    static func runModelOutputsTwoInputs(
        model: CoreML_Specification_Model,
        inputNameX: String,
        inputShapeX: [Int],
        inputValuesX: [Float],
        inputNameY: String,
        inputShapeY: [Int],
        inputValuesY: [Float],
        outputNames: [String]
    ) throws -> [String: MLMultiArray] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArrayX = try MLMultiArray(shape: inputShapeX.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesX.enumerated() {
            inputArrayX[idx] = NSNumber(value: value)
        }
        let inputArrayY = try MLMultiArray(shape: inputShapeY.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesY.enumerated() {
            inputArrayY[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputNameX: inputArrayX,
            inputNameY: inputArrayY
        ])
        let output = try mlModel.prediction(from: provider)

        var results: [String: MLMultiArray] = [:]
        for name in outputNames {
            if let outArray = output.featureValue(for: name)?.multiArrayValue {
                results[name] = outArray
            }
        }
        return results
    }

    static func runModelOutputsThreeInputs(
        model: CoreML_Specification_Model,
        inputNameX: String,
        inputShapeX: [Int],
        inputValuesX: [Float],
        inputNameY: String,
        inputShapeY: [Int],
        inputValuesY: [Float],
        inputNameZ: String,
        inputShapeZ: [Int],
        inputValuesZ: [Float],
        outputNames: [String]
    ) throws -> [String: MLMultiArray] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArrayX = try MLMultiArray(shape: inputShapeX.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesX.enumerated() {
            inputArrayX[idx] = NSNumber(value: value)
        }
        let inputArrayY = try MLMultiArray(shape: inputShapeY.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesY.enumerated() {
            inputArrayY[idx] = NSNumber(value: value)
        }
        let inputArrayZ = try MLMultiArray(shape: inputShapeZ.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesZ.enumerated() {
            inputArrayZ[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputNameX: inputArrayX,
            inputNameY: inputArrayY,
            inputNameZ: inputArrayZ
        ])
        let output = try mlModel.prediction(from: provider)

        var results: [String: MLMultiArray] = [:]
        for name in outputNames {
            if let outArray = output.featureValue(for: name)?.multiArrayValue {
                results[name] = outArray
            }
        }
        return results
    }
    static func runFloatModel(
        model: CoreML_Specification_Model,
        inputName: String,
        outputName: String,
        inputShape: [Int],
        inputValues: [Float]
    ) throws -> [Float] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArray = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValues.enumerated() {
            inputArray[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        let output = try mlModel.prediction(from: provider)
        guard let outArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        var values: [Float] = []
        values.reserveCapacity(outArray.count)
        for idx in 0..<outArray.count {
            values.append(outArray[idx].floatValue)
        }
        return values
    }

    static func runFloatModelInt32Input(
        model: CoreML_Specification_Model,
        inputName: String,
        outputName: String,
        inputShape: [Int],
        inputValues: [Int32]
    ) throws -> [Float] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArray = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .int32)
        for (idx, value) in inputValues.enumerated() {
            inputArray[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        let output = try mlModel.prediction(from: provider)
        guard let outArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        var values: [Float] = []
        values.reserveCapacity(outArray.count)
        for idx in 0..<outArray.count {
            values.append(outArray[idx].floatValue)
        }
        return values
    }

    static func runFloatModel(
        model: CoreML_Specification_Model,
        inputNameX: String,
        inputValuesX: [Float],
        inputNameY: String,
        inputValuesY: [Float],
        inputShape: [Int],
        outputName: String
    ) throws -> [Float] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArrayX = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesX.enumerated() {
            inputArrayX[idx] = NSNumber(value: value)
        }
        let inputArrayY = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesY.enumerated() {
            inputArrayY[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputNameX: inputArrayX,
            inputNameY: inputArrayY
        ])
        let output = try mlModel.prediction(from: provider)
        guard let outArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        var values: [Float] = []
        values.reserveCapacity(outArray.count)
        for idx in 0..<outArray.count {
            values.append(outArray[idx].floatValue)
        }
        return values
    }

    static func runFloatModelTwoInputs(
        model: CoreML_Specification_Model,
        inputNameX: String,
        inputShapeX: [Int],
        inputValuesX: [Float],
        inputNameY: String,
        inputShapeY: [Int],
        inputValuesY: [Float],
        outputName: String
    ) throws -> [Float] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArrayX = try MLMultiArray(shape: inputShapeX.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesX.enumerated() {
            inputArrayX[idx] = NSNumber(value: value)
        }
        let inputArrayY = try MLMultiArray(shape: inputShapeY.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesY.enumerated() {
            inputArrayY[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputNameX: inputArrayX,
            inputNameY: inputArrayY
        ])
        let output = try mlModel.prediction(from: provider)
        guard let outArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        var values: [Float] = []
        values.reserveCapacity(outArray.count)
        for idx in 0..<outArray.count {
            values.append(outArray[idx].floatValue)
        }
        return values
    }

    static func runInt32Model(
        model: CoreML_Specification_Model,
        inputName: String,
        outputName: String,
        inputShape: [Int],
        inputValues: [Float]
    ) throws -> [Int32] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArray = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValues.enumerated() {
            inputArray[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        let output = try mlModel.prediction(from: provider)
        guard let outArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        var values: [Int32] = []
        values.reserveCapacity(outArray.count)
        for idx in 0..<outArray.count {
            values.append(outArray[idx].int32Value)
        }
        return values
    }

    static func runInt32Model(
        model: CoreML_Specification_Model,
        inputNameX: String,
        inputValuesX: [Float],
        inputNameY: String,
        inputValuesY: [Float],
        inputShape: [Int],
        outputName: String
    ) throws -> [Int32] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArrayX = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesX.enumerated() {
            inputArrayX[idx] = NSNumber(value: value)
        }
        let inputArrayY = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValuesY.enumerated() {
            inputArrayY[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputNameX: inputArrayX,
            inputNameY: inputArrayY
        ])
        let output = try mlModel.prediction(from: provider)
        guard let outArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        var values: [Int32] = []
        values.reserveCapacity(outArray.count)
        for idx in 0..<outArray.count {
            values.append(outArray[idx].int32Value)
        }
        return values
    }

    static func runFloatModelOutputs(
        model: CoreML_Specification_Model,
        inputName: String,
        inputShape: [Int],
        inputValues: [Float],
        outputNames: [String]
    ) throws -> [String: [Float]] {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArray = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValues.enumerated() {
            inputArray[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        let output = try mlModel.prediction(from: provider)

        var results: [String: [Float]] = [:]
        for name in outputNames {
            guard let outArray = output.featureValue(for: name)?.multiArrayValue else {
                results[name] = []
                continue
            }
            var values: [Float] = []
            values.reserveCapacity(outArray.count)
            for idx in 0..<outArray.count {
                values.append(outArray[idx].floatValue)
            }
            results[name] = values
        }
        return results
    }

    static func runFloatIntModel(
        model: CoreML_Specification_Model,
        inputName: String,
        inputShape: [Int],
        inputValues: [Float],
        outputNameFloat: String,
        outputNameInt: String
    ) throws -> ([Float], [Int32]) {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArray = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValues.enumerated() {
            inputArray[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        let output = try mlModel.prediction(from: provider)

        let floatValues: [Float] = {
            guard let outArray = output.featureValue(for: outputNameFloat)?.multiArrayValue else {
                return []
            }
            var values: [Float] = []
            values.reserveCapacity(outArray.count)
            for idx in 0..<outArray.count {
                values.append(outArray[idx].floatValue)
            }
            return values
        }()

        let intValues: [Int32] = {
            guard let outArray = output.featureValue(for: outputNameInt)?.multiArrayValue else {
                return []
            }
            var values: [Int32] = []
            values.reserveCapacity(outArray.count)
            for idx in 0..<outArray.count {
                values.append(outArray[idx].int32Value)
            }
            return values
        }()

        return (floatValues, intValues)
    }

    static func runClassifyModel(
        model: CoreML_Specification_Model,
        inputName: String,
        inputShape: [Int],
        inputValues: [Float],
        labelOutput: String,
        probabilityOutput: String
    ) throws -> (String, [String: Double]) {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArray = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValues.enumerated() {
            inputArray[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        let output = try mlModel.prediction(from: provider)

        let label = output.featureValue(for: labelOutput)?.stringValue ?? ""
        var probabilities: [String: Double] = [:]
        if let dict = output.featureValue(for: probabilityOutput)?.dictionaryValue {
            for (key, value) in dict {
                if let keyString = key as? String {
                    probabilities[keyString] = value.doubleValue
                }
            }
        }

        return (label, probabilities)
    }

    static func runClassifyModelInt64(
        model: CoreML_Specification_Model,
        inputName: String,
        inputShape: [Int],
        inputValues: [Float],
        labelOutput: String,
        probabilityOutput: String
    ) throws -> (Int64, [Int64: Double]) {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try MLPackageBuilder.writeMLPackage(model: model, to: packageURL)
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let mlModel = try MLModel(contentsOf: compiledURL)

        let inputArray = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float32)
        for (idx, value) in inputValues.enumerated() {
            inputArray[idx] = NSNumber(value: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        let output = try mlModel.prediction(from: provider)

        let label = output.featureValue(for: labelOutput)?.int64Value ?? 0
        var probabilities: [Int64: Double] = [:]
        if let dict = output.featureValue(for: probabilityOutput)?.dictionaryValue {
            for (key, value) in dict {
                if let keyNumber = key as? NSNumber {
                    probabilities[keyNumber.int64Value] = value.doubleValue
                }
            }
        }

        return (label, probabilities)
    }
}
