import Foundation
import Testing
import CoreMLTools

@Test
func testReshape() async throws {
    let inputShape = [2]
    let outputShape = [1, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let shapeValue = MILValue.tensorInt32(shape: [2], values: [1, 2])
    let op = MILBuilder.operation(
        type: "reshape",
        inputs: [
            "x": MILArgument(.name("x")),
            "shape": MILArgument(.value(shapeValue))
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
        inputValues: [1.0, 2.0]
    )

    #expect(outputs.count == 2)
    #expect(abs(outputs[0] - 1.0) < 1e-4)
    #expect(abs(outputs[1] - 2.0) < 1e-4)
}

@Test
func testSqueeze() async throws {
    let inputShape = [1, 2]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let axesValue = MILValue.tensorInt32(shape: [1], values: [0])
    let op = MILBuilder.operation(
        type: "squeeze",
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
        inputValues: [1.0, 2.0]
    )

    #expect(outputs.count == 2)
    #expect(abs(outputs[0] - 1.0) < 1e-4)
    #expect(abs(outputs[1] - 2.0) < 1e-4)
}

@Test
func testExpandDims() async throws {
    let inputShape = [2]
    let outputShape = [1, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let axesValue = MILValue.tensorInt32(shape: [1], values: [0])
    let op = MILBuilder.operation(
        type: "expand_dims",
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
        inputValues: [1.0, 2.0]
    )

    #expect(outputs.count == 2)
    #expect(abs(outputs[0] - 1.0) < 1e-4)
    #expect(abs(outputs[1] - 2.0) < 1e-4)
}

@Test
func testTranspose() async throws {
    let inputShape = [2, 3]
    let outputShape = [3, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let permValue = MILValue.tensorInt32(shape: [2], values: [1, 0])
    let op = MILBuilder.operation(
        type: "transpose",
        inputs: [
            "x": MILArgument(.name("x")),
            "perm": MILArgument(.value(permValue))
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
        inputValues: [1, 2, 3, 4, 5, 6]
    )

    #expect(outputs.count == 6)
    // Expected transpose of [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    let expected: [Float] = [1, 4, 2, 5, 3, 6]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testConcat() async throws {
    let inputShape = [1, 2]
    let outputShape = [2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamedA = MILBuilder.namedValue(name: "a", type: inputType)
    let inputNamedB = MILBuilder.namedValue(name: "b", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let valuesArg = MILArgument([
        .name("a"),
        .name("b")
    ])
    let axisValue = MILValue.scalarInt32(0)

    let op = MILBuilder.operation(
        type: "concat",
        inputs: [
            "values": valuesArg,
            "axis": MILArgument(.value(axisValue)),
            "interleave": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamedA, inputNamedB], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("a", inputShape, .float32), ("b", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputNameX: "a",
        inputValuesX: [1, 2],
        inputNameY: "b",
        inputValuesY: [3, 4],
        inputShape: inputShape,
        outputName: "y"
    )

    let expected: [Float] = [1, 2, 3, 4]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testStack() async throws {
    let inputShape = [2]
    let outputShape = [2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamedA = MILBuilder.namedValue(name: "a", type: inputType)
    let inputNamedB = MILBuilder.namedValue(name: "b", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let valuesArg = MILArgument([
        .name("a"),
        .name("b")
    ])
    let axisValue = MILValue.scalarInt32(0)

    let op = MILBuilder.operation(
        type: "stack",
        inputs: [
            "values": valuesArg,
            "axis": MILArgument(.value(axisValue))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamedA, inputNamedB], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("a", inputShape, .float32), ("b", inputShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputNameX: "a",
        inputValuesX: [1, 2],
        inputNameY: "b",
        inputValuesY: [3, 4],
        inputShape: inputShape,
        outputName: "y"
    )

    let expected: [Float] = [1, 2, 3, 4]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testTile() async throws {
    let inputShape = [2]
    let outputShape = [4]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let repsValue = MILValue.tensorInt32(shape: [1], values: [2])
    let op = MILBuilder.operation(
        type: "tile",
        inputs: [
            "x": MILArgument(.name("x")),
            "reps": MILArgument(.value(repsValue))
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
        inputValues: [1, 2]
    )

    let expected: [Float] = [1, 2, 1, 2]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testPad() async throws {
    let inputShape = [2]
    let outputShape = [4]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let padValue = MILValue.tensorInt32(shape: [2], values: [1, 1])
    let op = MILBuilder.operation(
        type: "pad",
        inputs: [
            "x": MILArgument(.name("x")),
            "pad": MILArgument(.value(padValue)),
            "mode": MILArgument(.value(MILValue.scalarString("constant"))),
            "constant_val": MILArgument(.value(MILValue.scalarFloat(0.0)))
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
        inputValues: [1, 2]
    )

    let expected: [Float] = [0, 1, 2, 0]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testSlidingWindows() async throws {
    let inputShape = [1, 4, 1, 1]
    let outputShape = [1, 3, 2, 1, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "sliding_windows",
        inputs: [
            "x": MILArgument(.name("x")),
            "axis": MILArgument(.value(MILValue.scalarInt32(1))),
            "size": MILArgument(.value(MILValue.scalarInt32(2))),
            "stride": MILArgument(.value(MILValue.scalarInt32(1)))
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
        inputValues: [9.0, 5.0, 1.0, 3.0]
    )

    let expected: [Float] = [9.0, 5.0, 5.0, 1.0, 1.0, 3.0]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testReverseSequence() async throws {
    let inputShape = [4, 8]
    let outputShape = [4, 8]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let lengths = MILValue.tensorInt32(shape: [4], values: [7, 2, 3, 5])
    let op = MILBuilder.operation(
        type: "reverse_sequence",
        inputs: [
            "x": MILArgument(.name("x")),
            "lengths": MILArgument(.value(lengths)),
            "seq_axis": MILArgument(.value(MILValue.scalarInt32(1))),
            "batch_axis": MILArgument(.value(MILValue.scalarInt32(0)))
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

    let inputValues: [Float] = [
        1, 2, 3, 4, 5, 0, 0, 0,
        1, 2, 0, 0, 0, 0, 0, 0,
        1, 2, 3, 4, 0, 0, 0, 0,
        1, 2, 3, 4, 5, 6, 7, 8
    ]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputValues
    )

    let expected: [Float] = [
        0, 0, 5, 4, 3, 2, 1, 0,
        2, 1, 0, 0, 0, 0, 0, 0,
        3, 2, 1, 4, 0, 0, 0, 0,
        5, 4, 3, 2, 1, 6, 7, 8
    ]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}
