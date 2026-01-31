import Foundation
import Testing
import CoreMLTools

@Test
func testAvgPool() async throws {
    let inputShape = [1, 1, 2, 2]
    let outputShape = [1, 1, 1, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "avg_pool",
        inputs: [
            "x": MILArgument(.name("x")),
            "kernel_sizes": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [2, 2]))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid"))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [4], values: [0, 0, 0, 0]))),
            "ceil_mode": MILArgument(.value(MILValue.scalarBool(false))),
            "exclude_padding_from_average": MILArgument(.value(MILValue.scalarBool(false)))
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

    #expect(outputs.count == 1)
    #expect(abs(outputs[0] - 2.5) < 1e-4)
}

@Test
func testMaxPool() async throws {
    let inputShape = [1, 1, 2, 2]
    let outputShape = [1, 1, 1, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "max_pool",
        inputs: [
            "x": MILArgument(.name("x")),
            "kernel_sizes": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [2, 2]))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid"))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [4], values: [0, 0, 0, 0]))),
            "ceil_mode": MILArgument(.value(MILValue.scalarBool(false)))
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

    #expect(outputs.count == 1)
    #expect(abs(outputs[0] - 4.0) < 1e-4)
}

@Test
func testL2Pool() async throws {
    let inputShape = [1, 1, 2, 2]
    let outputShape = [1, 1, 1, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "l2_pool",
        inputs: [
            "x": MILArgument(.name("x")),
            "kernel_sizes": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [2, 2]))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid"))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [4], values: [0, 0, 0, 0]))),
            "ceil_mode": MILArgument(.value(MILValue.scalarBool(false)))
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

    #expect(outputs.count == 1)
    #expect(abs(outputs[0] - Float(sqrt(30.0))) < 1e-4)
}

@Test
func testBatchNorm() async throws {
    let inputShape = [1, 1, 1, 2]
    let outputShape = [1, 1, 1, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "batch_norm",
        inputs: [
            "x": MILArgument(.name("x")),
            "mean": MILArgument(.value(MILValue.tensorFloat(shape: [1], values: [1.0]))),
            "variance": MILArgument(.value(MILValue.tensorFloat(shape: [1], values: [1.0]))),
            "gamma": MILArgument(.value(MILValue.tensorFloat(shape: [1], values: [1.0]))),
            "beta": MILArgument(.value(MILValue.tensorFloat(shape: [1], values: [0.0]))),
            "epsilon": MILArgument(.value(MILValue.scalarFloat(0.0)))
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
        inputValues: [1, 3]
    )

    let expected: [Float] = [0, 2]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testInstanceNorm() async throws {
    let inputShape = [1, 1, 1, 2]
    let outputShape = [1, 1, 1, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "instance_norm",
        inputs: [
            "x": MILArgument(.name("x")),
            "gamma": MILArgument(.value(MILValue.tensorFloat(shape: [1], values: [1.0]))),
            "beta": MILArgument(.value(MILValue.tensorFloat(shape: [1], values: [0.0]))),
            "epsilon": MILArgument(.value(MILValue.scalarFloat(0.0)))
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
        inputValues: [1, 3]
    )

    let expected: [Float] = [-1, 1]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testLayerNorm() async throws {
    let inputShape = [1, 2]
    let outputShape = [1, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "layer_norm",
        inputs: [
            "x": MILArgument(.name("x")),
            "axes": MILArgument(.value(MILValue.tensorInt32(shape: [1], values: [1]))),
            "gamma": MILArgument(.value(MILValue.tensorFloat(shape: [2], values: [1.0, 1.0]))),
            "beta": MILArgument(.value(MILValue.tensorFloat(shape: [2], values: [0.0, 0.0]))),
            "epsilon": MILArgument(.value(MILValue.scalarFloat(0.0)))
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
        inputValues: [1, 3]
    )

    let expected: [Float] = [-1, 1]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testL2Norm() async throws {
    let inputShape = [1, 1, 1]
    let outputShape = [1, 1, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "l2_norm",
        inputs: [
            "x": MILArgument(.name("x")),
            "epsilon": MILArgument(.value(MILValue.scalarFloat(0.0)))
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
        inputValues: [2]
    )

    #expect(outputs.count == 1)
    #expect(abs(outputs[0] - 1.0) < 1e-4)
}

@Test
func testLocalResponseNorm() async throws {
    let inputShape = [1, 2, 1]
    let outputShape = [1, 2, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "local_response_norm",
        inputs: [
            "x": MILArgument(.name("x")),
            "size": MILArgument(.value(MILValue.scalarInt32(1))),
            "alpha": MILArgument(.value(MILValue.scalarFloat(1.0))),
            "beta": MILArgument(.value(MILValue.scalarFloat(1.0))),
            "k": MILArgument(.value(MILValue.scalarFloat(1.0)))
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
        inputValues: [2, 4]
    )

    let expected: [Float] = [0.4, 4.0 / 17.0]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testLinear() async throws {
    let inputShape = [1, 2]
    let outputShape = [1, 3]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let weightValue = MILValue.tensorFloat(shape: [3, 2], values: [1, 0, 0, 1, 1, 1])
    let biasValue = MILValue.tensorFloat(shape: [3], values: [0, 0, 0])

    let op = MILBuilder.operation(
        type: "linear",
        inputs: [
            "x": MILArgument(.name("x")),
            "weight": MILArgument(.value(weightValue)),
            "bias": MILArgument(.value(biasValue))
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

    let expected: [Float] = [1, 2, 3]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}
