import Foundation
import Testing
import CoreMLTools

@Test
func testScatter() async throws {
    let inputShape = [3]
    let outputShape = [3]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "data", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let indicesValue = MILValue.tensorInt32(shape: [1], values: [1])
    let updatesValue = MILValue.tensorFloat(shape: [1], values: [9])

    let op = MILOps.scatter(
        inputs: [
            "data": MILArgument(.name("data")),
            "indices": MILArgument(.value(indicesValue)),
            "updates": MILArgument(.value(updatesValue)),
            "axis": MILArgument(.value(MILValue.scalarInt32(0))),
            "mode": MILArgument(.value(MILValue.scalarString("update"))),
            "validate_indices": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("data", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "data",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3]
    )

    let expected: [Float] = [1, 9, 3]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testScatterAlongAxis() async throws {
    let inputShape = [2, 2]
    let outputShape = [2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "data", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let indicesValue = MILValue.tensorInt32(shape: [2, 2], values: [0, 1, 1, 0])
    let updatesValue = MILValue.tensorFloat(shape: [2, 2], values: [9, 8, 7, 6])

    let op = MILOps.scatter_along_axis(
        inputs: [
            "data": MILArgument(.name("data")),
            "indices": MILArgument(.value(indicesValue)),
            "updates": MILArgument(.value(updatesValue)),
            "axis": MILArgument(.value(MILValue.scalarInt32(1))),
            "mode": MILArgument(.value(MILValue.scalarString("update"))),
            "validate_indices": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("data", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "data",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3, 4]
    )

    let expected: [Float] = [9, 8, 6, 7]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testScatterNd() async throws {
    let inputShape = [3]
    let outputShape = [3]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "data", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let indicesValue = MILValue.tensorInt32(shape: [2, 1], values: [0, 2])
    let updatesValue = MILValue.tensorFloat(shape: [2], values: [9, 8])

    let op = MILOps.scatter_nd(
        inputs: [
            "data": MILArgument(.name("data")),
            "indices": MILArgument(.value(indicesValue)),
            "updates": MILArgument(.value(updatesValue)),
            "mode": MILArgument(.value(MILValue.scalarString("update"))),
            "validate_indices": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("data", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "data",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3]
    )

    let expected: [Float] = [9, 2, 8]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}
