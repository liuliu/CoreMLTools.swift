import Foundation

public struct MILBuilder {
    public static func operation(
        type: String,
        inputs: [String: MILArgument],
        outputs: [CoreML_Specification_MILSpec_NamedValueType]
    ) -> CoreML_Specification_MILSpec_Operation {
        var op = CoreML_Specification_MILSpec_Operation()
        op.type = type
        op.inputs = inputs.mapValues { $0.toProto() }
        op.outputs = outputs
        return op
    }

    public static func namedValue(
        name: String,
        type: CoreML_Specification_MILSpec_ValueType
    ) -> CoreML_Specification_MILSpec_NamedValueType {
        var named = CoreML_Specification_MILSpec_NamedValueType()
        named.name = name
        named.type = type
        return named
    }

    public static func block(
        operations: [CoreML_Specification_MILSpec_Operation],
        outputs: [String]
    ) -> CoreML_Specification_MILSpec_Block {
        var block = CoreML_Specification_MILSpec_Block()
        block.operations = operations
        block.outputs = outputs
        return block
    }

    public static func block(
        operations: [CoreML_Specification_MILSpec_Operation],
        outputs: [String],
        inputs: [CoreML_Specification_MILSpec_NamedValueType],
        attributes: [String: CoreML_Specification_MILSpec_Value] = [:]
    ) -> CoreML_Specification_MILSpec_Block {
        var block = CoreML_Specification_MILSpec_Block()
        block.inputs = inputs
        block.operations = operations
        block.outputs = outputs
        block.attributes = attributes
        return block
    }

    public static func function(
        inputs: [CoreML_Specification_MILSpec_NamedValueType],
        opset: String,
        block: CoreML_Specification_MILSpec_Block
    ) -> CoreML_Specification_MILSpec_Function {
        var function = CoreML_Specification_MILSpec_Function()
        function.inputs = inputs
        function.opset = opset
        function.blockSpecializations = [opset: block]
        return function
    }

    public static func program(
        functions: [String: CoreML_Specification_MILSpec_Function]
    ) -> CoreML_Specification_MILSpec_Program {
        var program = CoreML_Specification_MILSpec_Program()
        program.version = 1
        program.functions = functions
        return program
    }
}
