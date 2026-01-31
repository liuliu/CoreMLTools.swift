import Foundation
import Testing
import CoreMLTools

@Test
func testArgsort() async throws {
    let inputShape = [3]
    let outputShape = [3]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .int32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "argsort",
        inputs: [
            "x": MILArgument(.name("x")),
            "axis": MILArgument(.value(MILValue.scalarInt32(0))),
            "ascending": MILArgument(.value(MILValue.scalarBool(true)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .int32)]
    )

    let outputs = try MLTestUtils.runInt32Model(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [3, 1, 2]
    )

    #expect(outputs == [1, 2, 0])
}

@Test
func testCumsum() async throws {
    let inputShape = [3]
    let outputShape = [3]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "cumsum",
        inputs: [
            "x": MILArgument(.name("x")),
            "axis": MILArgument(.value(MILValue.scalarInt32(0))),
            "exclusive": MILArgument(.value(MILValue.scalarBool(false))),
            "reverse": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3]
    )

    let expected: [Float] = [1, 3, 6]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testGather() async throws {
    let inputShape = [3]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let indicesValue = MILValue.tensorInt32(shape: [2], values: [2, 0])

    let op = MILBuilder.operation(
        type: "gather",
        inputs: [
            "x": MILArgument(.name("x")),
            "indices": MILArgument(.value(indicesValue)),
            "axis": MILArgument(.value(MILValue.scalarInt32(0))),
            "validate_indices": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3]
    )

    let expected: [Float] = [3, 1]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testGatherAlongAxis() async throws {
    let inputShape = [2, 2]
    let outputShape = [2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let indicesValue = MILValue.tensorInt32(shape: [2, 2], values: [1, 0, 0, 1])

    let op = MILBuilder.operation(
        type: "gather_along_axis",
        inputs: [
            "x": MILArgument(.name("x")),
            "indices": MILArgument(.value(indicesValue)),
            "axis": MILArgument(.value(MILValue.scalarInt32(1))),
            "validate_indices": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3, 4]
    )

    let expected: [Float] = [2, 1, 3, 4]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testGatherNd() async throws {
    let inputShape = [4]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let indicesValue = MILValue.tensorInt32(shape: [2, 1], values: [0, 2])

    let op = MILBuilder.operation(
        type: "gather_nd",
        inputs: [
            "x": MILArgument(.name("x")),
            "indices": MILArgument(.value(indicesValue)),
            "validate_indices": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3, 4]
    )

    let expected: [Float] = [1, 3]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testSliceByIndex() async throws {
    let inputShape = [4]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let beginValue = MILValue.tensorInt32(shape: [1], values: [1])
    let endValue = MILValue.tensorInt32(shape: [1], values: [3])
    let strideValue = MILValue.tensorInt32(shape: [1], values: [1])

    let op = MILBuilder.operation(
        type: "slice_by_index",
        inputs: [
            "x": MILArgument(.name("x")),
            "begin": MILArgument(.value(beginValue)),
            "end": MILArgument(.value(endValue)),
            "stride": MILArgument(.value(strideValue))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3, 4]
    )

    let expected: [Float] = [2, 3]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testSliceBySize() async throws {
    let inputShape = [4]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let beginValue = MILValue.tensorInt32(shape: [1], values: [1])
    let sizeValue = MILValue.tensorInt32(shape: [1], values: [2])

    let op = MILBuilder.operation(
        type: "slice_by_size",
        inputs: [
            "x": MILArgument(.name("x")),
            "begin": MILArgument(.value(beginValue)),
            "size": MILArgument(.value(sizeValue))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3, 4]
    )

    let expected: [Float] = [2, 3]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testReverse() async throws {
    let inputShape = [3]
    let outputShape = [3]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let axesValue = MILValue.tensorInt32(shape: [1], values: [0])

    let op = MILBuilder.operation(
        type: "reverse",
        inputs: [
            "x": MILArgument(.name("x")),
            "axes": MILArgument(.value(axesValue))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3]
    )

    let expected: [Float] = [3, 2, 1]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testShape() async throws {
    let inputShape = [2, 3]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .int32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "shape",
        inputs: [
            "x": MILArgument(.name("x"))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y", outputShape, .int32)]
    )

    let outputs = try MLTestUtils.runInt32Model(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1, 2, 3, 4, 5, 6]
    )

    #expect(outputs == [2, 3])
}

@Test
func testOneHot() async throws {
    let inputShape = [2]
    let outputShape = [2, 3]
    let inputType = MILType.tensor(dataType: .int32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "indices", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let depthValue = MILValue.scalarInt32(3)

    let op = MILBuilder.operation(
        type: "one_hot",
        inputs: [
            "indices": MILArgument(.name("indices")),
            "one_hot_vector_size": MILArgument(.value(depthValue)),
            "axis": MILArgument(.value(MILValue.scalarInt32(-1))),
            "on_value": MILArgument(.value(MILValue.scalarFloat(1.0))),
            "off_value": MILArgument(.value(MILValue.scalarFloat(0.0)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("indices", inputShape, .int32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelInt32Input(
        model: model,
        inputName: "indices",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [0, 2]
    )

    let expected: [Float] = [1, 0, 0, 0, 0, 1]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testTopk() async throws {
    let inputShape = [3]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)
    let indexType = MILType.tensor(dataType: .int32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputValues = MILBuilder.namedValue(name: "values", type: outputType)
    let outputIndices = MILBuilder.namedValue(name: "indices", type: indexType)

    let op = MILBuilder.operation(
        type: "topk",
        inputs: [
            "x": MILArgument(.name("x")),
            "k": MILArgument(.value(MILValue.scalarInt32(2))),
            "axis": MILArgument(.value(MILValue.scalarInt32(0))),
            "ascending": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputValues, outputIndices]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["values", "indices"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [
            ("values", outputShape, .float32),
            ("indices", outputShape, .int32)
        ]
    )

    let (values, indices) = try MLTestUtils.runFloatIntModel(
        model: model,
        inputName: "x",
        inputShape: inputShape,
        inputValues: [1, 3, 2],
        outputNameFloat: "values",
        outputNameInt: "indices"
    )

    let expectedValues: [Float] = [3, 2]
    let expectedIndices: [Int32] = [1, 2]
    for (out, exp) in zip(values, expectedValues) {
        #expect(abs(out - exp) < 1e-4)
    }
    for (out, exp) in zip(indices, expectedIndices) {
        #expect(out == exp)
    }
}
