import Foundation
import Testing
import CoreMLTools

@Test
func testMatmul() async throws {
    let inputShape = [2, 2]
    let outputShape = [2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamedX = MILBuilder.namedValue(name: "x", type: inputType)
    let inputNamedY = MILBuilder.namedValue(name: "y", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "z", type: outputType)

    let op = MILBuilder.operation(
        type: "matmul",
        inputs: [
            "x": MILArgument(.name("x")),
            "y": MILArgument(.name("y")),
            "transpose_x": MILArgument(.value(MILValue.scalarBool(false))),
            "transpose_y": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["z"])
    let function = MILBuilder.function(inputs: [inputNamedX, inputNamedY], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32), ("y", inputShape, .float32)],
        outputs: [("z", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputNameX: "x",
        inputValuesX: [1, 2, 3, 4],
        inputNameY: "y",
        inputValuesY: [5, 6, 7, 8],
        inputShape: inputShape,
        outputName: "z"
    )

    let expected: [Float] = [19, 22, 43, 50]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testSoftmax() async throws {
    let inputShape = [2]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "softmax",
        inputs: [
            "x": MILArgument(.name("x")),
            "axis": MILArgument(.value(MILValue.scalarInt32(0)))
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

    let exp1 = exp(1.0)
    let exp2 = exp(2.0)
    let denom = exp1 + exp2
    let expected: [Float] = [Float(exp1 / denom), Float(exp2 / denom)]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testGelu() async throws {
    let inputShape = [2]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "gelu",
        inputs: [
            "x": MILArgument(.name("x")),
            "mode": MILArgument(.value(MILValue.scalarString("EXACT")))
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
        inputValues: [0, 1]
    )

    let expected0 = Float(0.0)
    let expected1 = Float(0.5 * (1.0 + erf(1.0 / sqrt(2.0))))
    let expected: [Float] = [expected0, expected1]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testLinearActivation() async throws {
    let inputShape = [2]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "linear_activation",
        inputs: [
            "x": MILArgument(.name("x")),
            "alpha": MILArgument(.value(MILValue.scalarFloat(2.0))),
            "beta": MILArgument(.value(MILValue.scalarFloat(1.0)))
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

    let expected: [Float] = [3, 5]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testPrelu() async throws {
    let inputShape = [1, 2, 1]
    let outputShape = [1, 2, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let alphaValue = MILValue.tensorFloat(shape: [2], values: [0.1, 0.2])

    let op = MILBuilder.operation(
        type: "prelu",
        inputs: [
            "x": MILArgument(.name("x")),
            "alpha": MILArgument(.value(alphaValue))
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
        inputValues: [-1, 2]
    )

    let expected: [Float] = [-0.1, 2.0]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testIdentity() async throws {
    let inputShape = [2]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "identity",
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
        outputs: [("y", outputShape, .float32)]
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
