import Foundation
import SwiftProtobuf

public struct MLProgramBuilder {
    public static let defaultAuthor = "com.apple.CoreML"
    public static let defaultModelFileName = "model.mlmodel"

    public static func makeModel(
        program: CoreML_Specification_MILSpec_Program,
        inputs: [(name: String, shape: [Int], dataType: CoreML_Specification_ArrayFeatureType.ArrayDataType)],
        outputs: [(name: String, shape: [Int], dataType: CoreML_Specification_ArrayFeatureType.ArrayDataType)],
        specificationVersion: Int32 = 9,
        metadata: CoreML_Specification_Metadata? = nil
    ) -> CoreML_Specification_Model {
        let inputTypes = inputs.map { item in
            (item.name, makeArrayFeatureType(shape: item.shape, dataType: item.dataType))
        }
        let outputTypes = outputs.map { item in
            (item.name, makeArrayFeatureType(shape: item.shape, dataType: item.dataType))
        }
        return makeModel(
            program: program,
            inputs: inputTypes,
            outputs: outputTypes,
            specificationVersion: specificationVersion,
            metadata: metadata
        )
    }

    public static func makeModel(
        program: CoreML_Specification_MILSpec_Program,
        inputs: [(name: String, type: CoreML_Specification_FeatureType)],
        outputs: [(name: String, type: CoreML_Specification_FeatureType)],
        specificationVersion: Int32 = 9,
        metadata: CoreML_Specification_Metadata? = nil
    ) -> CoreML_Specification_Model {
        var description = CoreML_Specification_ModelDescription()
        description.input = inputs.map { item in
            var feature = CoreML_Specification_FeatureDescription()
            feature.name = item.name
            feature.type = item.type
            return feature
        }
        description.output = outputs.map { item in
            var feature = CoreML_Specification_FeatureDescription()
            feature.name = item.name
            feature.type = item.type
            return feature
        }
        if let metadata = metadata {
            description.metadata = metadata
        }

        var model = CoreML_Specification_Model()
        model.specificationVersion = specificationVersion
        model.description_p = description
        model.mlProgram = program
        return model
    }

    public static func makeAddConstantModel(
        inputName: String = "x",
        outputName: String = "y",
        shape: [Int],
        constant: Float = 1.0
    ) throws -> CoreML_Specification_Model {
        guard !shape.isEmpty else {
            throw CoreMLToolsError.invalidShape
        }

        let tensorType = makeTensorType(dataType: .float32, shape: shape)

        var inputNamed = CoreML_Specification_MILSpec_NamedValueType()
        inputNamed.name = inputName
        inputNamed.type = tensorType

        var outputNamed = CoreML_Specification_MILSpec_NamedValueType()
        outputNamed.name = outputName
        outputNamed.type = tensorType

        let constValue = makeImmediateTensorValue(type: tensorType, shape: shape, fill: constant)

        var argX = CoreML_Specification_MILSpec_Argument()
        var argXBinding = CoreML_Specification_MILSpec_Argument.Binding()
        argXBinding.name = inputName
        argX.arguments = [argXBinding]

        var argConst = CoreML_Specification_MILSpec_Argument()
        var argConstBinding = CoreML_Specification_MILSpec_Argument.Binding()
        argConstBinding.value = constValue
        argConst.arguments = [argConstBinding]

        var op = CoreML_Specification_MILSpec_Operation()
        op.type = "add"
        op.inputs = [
            "x": argX,
            "y": argConst
        ]
        op.outputs = [outputNamed]

        var block = CoreML_Specification_MILSpec_Block()
        block.operations = [op]
        block.outputs = [outputName]

        var function = CoreML_Specification_MILSpec_Function()
        function.inputs = [inputNamed]
        function.opset = "CoreML8"
        function.blockSpecializations = ["CoreML8": block]

        var program = CoreML_Specification_MILSpec_Program()
        program.version = 1
        program.functions = ["main": function]

        var inputFeature = CoreML_Specification_FeatureDescription()
        inputFeature.name = inputName
        inputFeature.type = makeArrayFeatureType(shape: shape, dataType: .float32)

        var outputFeature = CoreML_Specification_FeatureDescription()
        outputFeature.name = outputName
        outputFeature.type = makeArrayFeatureType(shape: shape, dataType: .float32)

        var metadata = CoreML_Specification_Metadata()
        metadata.author = "CoreMLTools.swift"
        metadata.shortDescription = "Simple add-constant ML Program"
        return makeModel(
            program: program,
            inputs: [(inputName, shape, .float32)],
            outputs: [(outputName, shape, .float32)],
            metadata: metadata
        )
    }

    static func makeArrayFeatureType(
        shape: [Int],
        dataType: CoreML_Specification_ArrayFeatureType.ArrayDataType
    ) -> CoreML_Specification_FeatureType {
        var arrayType = CoreML_Specification_ArrayFeatureType()
        arrayType.shape = shape.map { Int64($0) }
        arrayType.dataType = dataType

        var featureType = CoreML_Specification_FeatureType()
        featureType.multiArrayType = arrayType
        return featureType
    }

    private static func makeTensorType(
        dataType: CoreML_Specification_MILSpec_DataType,
        shape: [Int]
    ) -> CoreML_Specification_MILSpec_ValueType {
        var tensorType = CoreML_Specification_MILSpec_TensorType()
        tensorType.dataType = dataType
        tensorType.rank = Int64(shape.count)
        tensorType.dimensions = shape.map { size in
            var dim = CoreML_Specification_MILSpec_Dimension()
            var constant = CoreML_Specification_MILSpec_Dimension.ConstantDimension()
            constant.size = UInt64(size)
            dim.dimension = .constant(constant)
            return dim
        }

        var valueType = CoreML_Specification_MILSpec_ValueType()
        valueType.tensorType = tensorType
        return valueType
    }

    private static func makeImmediateTensorValue(
        type: CoreML_Specification_MILSpec_ValueType,
        shape: [Int],
        fill: Float
    ) -> CoreML_Specification_MILSpec_Value {
        let count = shape.reduce(1) { $0 * $1 }
        var tensorValue = CoreML_Specification_MILSpec_TensorValue()
        var floatValues = CoreML_Specification_MILSpec_TensorValue.RepeatedFloats()
        floatValues.values = Array(repeating: fill, count: count)
        tensorValue.value = .floats(floatValues)

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .tensor(tensorValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = type
        value.value = .immediateValue(immediate)
        return value
    }
}
