import Foundation
import Testing
import CoreMLTools

@Test
func testFlatten2d() async throws {
    let inputShape = [2, 2, 2]
    let outputShape = [2, 4]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "flatten2d",
        inputs: [
            "x": MILArgument(.name("x")),
            "axis": MILArgument(.value(MILValue.scalarInt32(1)))
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
        inputValues: [1, 2, 3, 4, 5, 6, 7, 8]
    )

    let expected: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testBandPart() async throws {
    let inputShape = [2, 2]
    let outputShape = [2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "band_part",
        inputs: [
            "x": MILArgument(.name("x")),
            "lower": MILArgument(.value(MILValue.scalarInt32(0))),
            "upper": MILArgument(.value(MILValue.scalarInt32(0)))
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

    let expected: [Float] = [1, 0, 0, 4]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testRange1D() async throws {
    let dummyShape = [1]
    let outputShape = [3]
    let dummyType = MILType.tensor(dataType: .float32, shape: dummyShape)
    let outputType = MILType.tensor(dataType: .int32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "dummy", type: dummyType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "range_1d",
        inputs: [
            "start": MILArgument(.value(MILValue.scalarInt32(0))),
            "end": MILArgument(.value(MILValue.scalarInt32(3))),
            "step": MILArgument(.value(MILValue.scalarInt32(1)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("dummy", dummyShape, .float32)],
        outputs: [("y", outputShape, .int32)]
    )

    let outputs = try MLTestUtils.runInt32Model(
        model: model,
        inputName: "dummy",
        outputName: "y",
        inputShape: dummyShape,
        inputValues: [0]
    )

    #expect(outputs == [0, 1, 2])
}

@Test
func testFill() async throws {
    let dummyShape = [1]
    let outputShape = [2]
    let dummyType = MILType.tensor(dataType: .float32, shape: dummyShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "dummy", type: dummyType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let shapeValue = MILValue.tensorInt32(shape: [1], values: [2])

    let op = MILBuilder.operation(
        type: "fill",
        inputs: [
            "shape": MILArgument(.value(shapeValue)),
            "value": MILArgument(.value(MILValue.scalarFloat(3.0)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("dummy", dummyShape, .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "dummy",
        outputName: "y",
        inputShape: dummyShape,
        inputValues: [0]
    )

    let expected: [Float] = [3, 3]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testSelect() async throws {
    let inputShape = [2]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamedA = MILBuilder.namedValue(name: "a", type: inputType)
    let inputNamedB = MILBuilder.namedValue(name: "b", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let condValue = MILValue.tensorBool(shape: [2], values: [true, false])

    let op = MILBuilder.operation(
        type: "select",
        inputs: [
            "cond": MILArgument(.value(condValue)),
            "a": MILArgument(.name("a")),
            "b": MILArgument(.name("b"))
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

    let expected: [Float] = [1, 4]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testSplit() async throws {
    let inputShape = [4]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let output0 = MILBuilder.namedValue(name: "y0", type: outputType)
    let output1 = MILBuilder.namedValue(name: "y1", type: outputType)

    let op = MILBuilder.operation(
        type: "split",
        inputs: [
            "x": MILArgument(.name("x")),
            "num_splits": MILArgument(.value(MILValue.scalarInt32(2))),
            "axis": MILArgument(.value(MILValue.scalarInt32(0)))
        ],
        outputs: [output0, output1]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y0", "y1"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32)],
        outputs: [("y0", outputShape, .float32), ("y1", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelOutputs(
        model: model,
        inputName: "x",
        inputShape: inputShape,
        inputValues: [1, 2, 3, 4],
        outputNames: ["y0", "y1"]
    )

    let expected0: [Float] = [1, 2]
    let expected1: [Float] = [3, 4]
    for (out, exp) in zip(outputs["y0"] ?? [], expected0) {
        #expect(abs(out - exp) < 1e-4)
    }
    for (out, exp) in zip(outputs["y1"] ?? [], expected1) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testNonZero() async throws {
    let inputShape = [4]
    let outputShape = [2, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .int32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "non_zero",
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
        inputValues: [0, 1, 0, 2]
    )

    #expect(outputs == [1, 3])
}
