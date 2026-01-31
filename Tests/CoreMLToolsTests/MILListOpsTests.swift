import Foundation
import Testing
import CoreMLTools

@Test
func testMakeListWriteRead() async throws {
    let inputShape = [2]
    let elemType = MILType.tensor(dataType: .float32, shape: inputShape)
    let listType = MILType.list(elementType: elemType, length: 1)

    let inputNamed = MILBuilder.namedValue(name: "x", type: elemType)
    let listNamed = MILBuilder.namedValue(name: "ls", type: listType)
    let listNamed2 = MILBuilder.namedValue(name: "ls2", type: listType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: elemType)

    let makeList = MILBuilder.operation(
        type: "make_list",
        inputs: [
            "init_length": MILArgument(.value(MILValue.scalarInt32(1))),
            "dynamic_length": MILArgument(.value(MILValue.scalarBool(false))),
            "elem_shape": MILArgument([.value(MILValue.scalarInt32(2))]),
            "dtype": MILArgument(.value(MILValue.scalarString("fp32")))
        ],
        outputs: [listNamed]
    )

    let listWrite = MILBuilder.operation(
        type: "list_write",
        inputs: [
            "ls": MILArgument(.name("ls")),
            "index": MILArgument(.value(MILValue.scalarInt32(0))),
            "value": MILArgument(.name("x"))
        ],
        outputs: [listNamed2]
    )

    let listRead = MILBuilder.operation(
        type: "list_read",
        inputs: [
            "ls": MILArgument(.name("ls2")),
            "index": MILArgument(.value(MILValue.scalarInt32(0)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [makeList, listWrite, listRead], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", inputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2]
    )

    let expected: [Float] = [1, 2]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testListGather() async throws {
    let elemShape = [2]
    let elemType = MILType.tensor(dataType: .float32, shape: elemShape)
    let listType = MILType.list(elementType: elemType, length: 2)
    let outputShape = [2, 2]
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamedA = MILBuilder.namedValue(name: "a", type: elemType)
    let inputNamedB = MILBuilder.namedValue(name: "b", type: elemType)
    let listNamed = MILBuilder.namedValue(name: "ls", type: listType)
    let listNamed2 = MILBuilder.namedValue(name: "ls2", type: listType)
    let listNamed3 = MILBuilder.namedValue(name: "ls3", type: listType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let makeList = MILBuilder.operation(
        type: "make_list",
        inputs: [
            "init_length": MILArgument(.value(MILValue.scalarInt32(2))),
            "dynamic_length": MILArgument(.value(MILValue.scalarBool(false))),
            "elem_shape": MILArgument([.value(MILValue.scalarInt32(2))]),
            "dtype": MILArgument(.value(MILValue.scalarString("fp32")))
        ],
        outputs: [listNamed]
    )

    let listWrite0 = MILBuilder.operation(
        type: "list_write",
        inputs: [
            "ls": MILArgument(.name("ls")),
            "index": MILArgument(.value(MILValue.scalarInt32(0))),
            "value": MILArgument(.name("a"))
        ],
        outputs: [listNamed2]
    )

    let listWrite1 = MILBuilder.operation(
        type: "list_write",
        inputs: [
            "ls": MILArgument(.name("ls2")),
            "index": MILArgument(.value(MILValue.scalarInt32(1))),
            "value": MILArgument(.name("b"))
        ],
        outputs: [listNamed3]
    )

    let indicesValue = MILValue.tensorInt32(shape: [2], values: [0, 1])
    let listGather = MILBuilder.operation(
        type: "list_gather",
        inputs: [
            "ls": MILArgument(.name("ls3")),
            "indices": MILArgument(.value(indicesValue))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [makeList, listWrite0, listWrite1, listGather], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamedA, inputNamedB], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("a", elemShape, .float32), ("b", elemShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelTwoInputs(
        model: model,
        inputNameX: "a",
        inputShapeX: elemShape,
        inputValuesX: [1, 2],
        inputNameY: "b",
        inputShapeY: elemShape,
        inputValuesY: [3, 4],
        outputName: "y"
    )

    let expected: [Float] = [1, 2, 3, 4]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testListScatter() async throws {
    let elemShape = [2]
    let elemType = MILType.tensor(dataType: .float32, shape: elemShape)
    let listType = MILType.list(elementType: elemType, length: 2)
    let valueShape = [2, 2]
    let outputShape = [2, 2]
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "values", type: MILType.tensor(dataType: .float32, shape: valueShape))
    let listNamed = MILBuilder.namedValue(name: "ls", type: listType)
    let listNamed2 = MILBuilder.namedValue(name: "ls2", type: listType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let makeList = MILBuilder.operation(
        type: "make_list",
        inputs: [
            "init_length": MILArgument(.value(MILValue.scalarInt32(2))),
            "dynamic_length": MILArgument(.value(MILValue.scalarBool(false))),
            "elem_shape": MILArgument([.value(MILValue.scalarInt32(2))]),
            "dtype": MILArgument(.value(MILValue.scalarString("fp32")))
        ],
        outputs: [listNamed]
    )

    let indicesValue = MILValue.tensorInt32(shape: [2], values: [0, 1])
    let listScatter = MILBuilder.operation(
        type: "list_scatter",
        inputs: [
            "ls": MILArgument(.name("ls")),
            "indices": MILArgument(.value(indicesValue)),
            "value": MILArgument(.name("values"))
        ],
        outputs: [listNamed2]
    )

    let listGather = MILBuilder.operation(
        type: "list_gather",
        inputs: [
            "ls": MILArgument(.name("ls2")),
            "indices": MILArgument(.value(indicesValue))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [makeList, listScatter, listGather], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("values", valueShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "values",
        outputName: "y",
        inputShape: valueShape,
        inputValues: [1, 2, 3, 4]
    )

    let expected: [Float] = [1, 2, 3, 4]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testListLength() async throws {
    let outputType = MILType.tensor(dataType: .int32, shape: [])
    let outputNamed = MILBuilder.namedValue(name: "len", type: outputType)
    let listType = MILType.list(elementType: MILType.tensor(dataType: .float32, shape: [1]), length: 3)
    let listNamed = MILBuilder.namedValue(name: "ls", type: listType)

    let makeList = MILBuilder.operation(
        type: "make_list",
        inputs: [
            "init_length": MILArgument(.value(MILValue.scalarInt32(3))),
            "dynamic_length": MILArgument(.value(MILValue.scalarBool(false))),
            "elem_shape": MILArgument([.value(MILValue.scalarInt32(1))]),
            "dtype": MILArgument(.value(MILValue.scalarString("fp32")))
        ],
        outputs: [listNamed]
    )

    let listLength = MILBuilder.operation(
        type: "list_length",
        inputs: [
            "ls": MILArgument(.name("ls"))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [makeList, listLength], outputs: ["len"])
    let function = MILBuilder.function(inputs: [], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [],
        outputs: [("len", [], .int32)]
    )

    let outputs = try MLTestUtils.runInt32ModelNoInputs(
        model: model,
        outputName: "len"
    )

    #expect(outputs == [3])
}
