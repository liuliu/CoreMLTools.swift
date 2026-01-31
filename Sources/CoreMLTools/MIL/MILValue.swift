import Foundation

public enum MILValue {
    public static func tensorFloat(
        shape: [Int],
        values: [Float]
    ) -> CoreML_Specification_MILSpec_Value {
        let valueType = MILType.tensor(dataType: .float32, shape: shape)
        var tensorValue = CoreML_Specification_MILSpec_TensorValue()
        var floats = CoreML_Specification_MILSpec_TensorValue.RepeatedFloats()
        floats.values = values
        tensorValue.value = .floats(floats)

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .tensor(tensorValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = valueType
        value.value = .immediateValue(immediate)
        return value
    }

    public static func tensorInt32(
        shape: [Int],
        values: [Int32]
    ) -> CoreML_Specification_MILSpec_Value {
        let valueType = MILType.tensor(dataType: .int32, shape: shape)
        var tensorValue = CoreML_Specification_MILSpec_TensorValue()
        var ints = CoreML_Specification_MILSpec_TensorValue.RepeatedInts()
        ints.values = values
        tensorValue.value = .ints(ints)

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .tensor(tensorValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = valueType
        value.value = .immediateValue(immediate)
        return value
    }

    public static func tensorInt64(
        shape: [Int],
        values: [Int64]
    ) -> CoreML_Specification_MILSpec_Value {
        let valueType = MILType.tensor(dataType: .int64, shape: shape)
        var tensorValue = CoreML_Specification_MILSpec_TensorValue()
        var ints = CoreML_Specification_MILSpec_TensorValue.RepeatedLongInts()
        ints.values = values
        tensorValue.value = .longInts(ints)

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .tensor(tensorValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = valueType
        value.value = .immediateValue(immediate)
        return value
    }

    public static func tensorBool(
        shape: [Int],
        values: [Bool]
    ) -> CoreML_Specification_MILSpec_Value {
        let valueType = MILType.tensor(dataType: .bool, shape: shape)
        var tensorValue = CoreML_Specification_MILSpec_TensorValue()
        var bools = CoreML_Specification_MILSpec_TensorValue.RepeatedBools()
        bools.values = values
        tensorValue.value = .bools(bools)

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .tensor(tensorValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = valueType
        value.value = .immediateValue(immediate)
        return value
    }

    public static func tensorUInt8(
        shape: [Int],
        values: [UInt8]
    ) -> CoreML_Specification_MILSpec_Value {
        let valueType = MILType.tensor(dataType: .uint8, shape: shape)
        var tensorValue = CoreML_Specification_MILSpec_TensorValue()
        var bytes = CoreML_Specification_MILSpec_TensorValue.RepeatedBytes()
        bytes.values = Data(values)
        tensorValue.value = .bytes(bytes)

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .tensor(tensorValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = valueType
        value.value = .immediateValue(immediate)
        return value
    }

    public static func scalarFloat(_ value: Float) -> CoreML_Specification_MILSpec_Value {
        return tensorFloat(shape: [], values: [value])
    }

    public static func scalarInt32(_ value: Int32) -> CoreML_Specification_MILSpec_Value {
        return tensorInt32(shape: [], values: [value])
    }

    public static func scalarBool(_ value: Bool) -> CoreML_Specification_MILSpec_Value {
        return tensorBool(shape: [], values: [value])
    }

    public static func tensorString(
        shape: [Int],
        values: [String]
    ) -> CoreML_Specification_MILSpec_Value {
        let valueType = MILType.tensor(dataType: .string, shape: shape)
        var tensorValue = CoreML_Specification_MILSpec_TensorValue()
        var strings = CoreML_Specification_MILSpec_TensorValue.RepeatedStrings()
        strings.values = values
        tensorValue.value = .strings(strings)

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .tensor(tensorValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = valueType
        value.value = .immediateValue(immediate)
        return value
    }

    public static func scalarString(_ value: String) -> CoreML_Specification_MILSpec_Value {
        return tensorString(shape: [], values: [value])
    }

    public static func listString(_ values: [String]) -> CoreML_Specification_MILSpec_Value {
        let elementType = MILType.tensor(dataType: .string, shape: [])
        let listType = MILType.list(elementType: elementType, length: values.count)

        let elementValues = values.map { scalarString($0) }
        var listValue = CoreML_Specification_MILSpec_ListValue()
        listValue.values = elementValues

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .list(listValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = listType
        value.value = .immediateValue(immediate)
        return value
    }

    public static func listInt64(_ values: [Int64]) -> CoreML_Specification_MILSpec_Value {
        let elementType = MILType.tensor(dataType: .int64, shape: [])
        let listType = MILType.list(elementType: elementType, length: values.count)

        let elementValues = values.map { tensorInt64(shape: [], values: [$0]) }
        var listValue = CoreML_Specification_MILSpec_ListValue()
        listValue.values = elementValues

        var immediate = CoreML_Specification_MILSpec_Value.ImmediateValue()
        immediate.value = .list(listValue)

        var value = CoreML_Specification_MILSpec_Value()
        value.type = listType
        value.value = .immediateValue(immediate)
        return value
    }
}

public enum MILType {
    public static func tensor(
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

    public static func list(
        elementType: CoreML_Specification_MILSpec_ValueType,
        length: Int?
    ) -> CoreML_Specification_MILSpec_ValueType {
        var listType = CoreML_Specification_MILSpec_ListType()
        listType.type = elementType

        var dim = CoreML_Specification_MILSpec_Dimension()
        if let length = length {
            var constant = CoreML_Specification_MILSpec_Dimension.ConstantDimension()
            constant.size = UInt64(length)
            dim.dimension = .constant(constant)
        } else {
            var unknown = CoreML_Specification_MILSpec_Dimension.UnknownDimension()
            unknown.variadic = false
            dim.dimension = .unknown(unknown)
        }
        listType.length = dim

        var valueType = CoreML_Specification_MILSpec_ValueType()
        valueType.listType = listType
        return valueType
    }

    public static func dictionary(
        keyType: CoreML_Specification_MILSpec_ValueType,
        valueType: CoreML_Specification_MILSpec_ValueType
    ) -> CoreML_Specification_MILSpec_ValueType {
        var dictionaryType = CoreML_Specification_MILSpec_DictionaryType()
        dictionaryType.keyType = keyType
        dictionaryType.valueType = valueType

        var valueTypeWrapper = CoreML_Specification_MILSpec_ValueType()
        valueTypeWrapper.dictionaryType = dictionaryType
        return valueTypeWrapper
    }
}
