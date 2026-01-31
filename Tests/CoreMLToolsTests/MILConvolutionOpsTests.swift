import Foundation
import Testing
import CoreMLTools

@Test
func testConv1D() async throws {
    let inputShape = [1, 1, 4]
    let outputShape = [1, 1, 4]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let weightValue = MILValue.tensorFloat(shape: [1, 1, 1], values: [1])

    let op = MILBuilder.operation(
        type: "conv",
        inputs: [
            "x": MILArgument(.name("x")),
            "weight": MILArgument(.value(weightValue)),
            "groups": MILArgument(.value(MILValue.scalarInt32(1))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [1], values: [1]))),
            "dilations": MILArgument(.value(MILValue.tensorInt32(shape: [1], values: [1]))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [0, 0]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid")))
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

    let inputs: [Float] = [1, 2, 3, 4]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputs
    )

    for (out, exp) in zip(outputs, inputs) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testConv2D() async throws {
    let inputShape = [1, 1, 2, 2]
    let outputShape = [1, 1, 2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let weightValue = MILValue.tensorFloat(shape: [1, 1, 1, 1], values: [1])

    let op = MILBuilder.operation(
        type: "conv",
        inputs: [
            "x": MILArgument(.name("x")),
            "weight": MILArgument(.value(weightValue)),
            "groups": MILArgument(.value(MILValue.scalarInt32(1))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "dilations": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [4], values: [0, 0, 0, 0]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid")))
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

    let inputs: [Float] = [1, 2, 3, 4]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputs
    )

    for (out, exp) in zip(outputs, inputs) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testConv3D() async throws {
    let inputShape = [1, 1, 2, 2, 2]
    let outputShape = [1, 1, 2, 2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let weightValue = MILValue.tensorFloat(shape: [1, 1, 1, 1, 1], values: [1])

    let op = MILBuilder.operation(
        type: "conv",
        inputs: [
            "x": MILArgument(.name("x")),
            "weight": MILArgument(.value(weightValue)),
            "groups": MILArgument(.value(MILValue.scalarInt32(1))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [3], values: [1, 1, 1]))),
            "dilations": MILArgument(.value(MILValue.tensorInt32(shape: [3], values: [1, 1, 1]))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [6], values: [0, 0, 0, 0, 0, 0]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid")))
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

    let inputs: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputs
    )

    for (out, exp) in zip(outputs, inputs) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testConvTranspose1D() async throws {
    let inputShape = [1, 1, 4]
    let outputShape = [1, 1, 4]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let weightValue = MILValue.tensorFloat(shape: [1, 1, 1], values: [1])

    let op = MILBuilder.operation(
        type: "conv_transpose",
        inputs: [
            "x": MILArgument(.name("x")),
            "weight": MILArgument(.value(weightValue)),
            "groups": MILArgument(.value(MILValue.scalarInt32(1))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [1], values: [1]))),
            "dilations": MILArgument(.value(MILValue.tensorInt32(shape: [1], values: [1]))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [0, 0]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid")))
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

    let inputs: [Float] = [1, 2, 3, 4]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputs
    )

    for (out, exp) in zip(outputs, inputs) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testConvTranspose2D() async throws {
    let inputShape = [1, 1, 2, 2]
    let outputShape = [1, 1, 2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let weightValue = MILValue.tensorFloat(shape: [1, 1, 1, 1], values: [1])

    let op = MILBuilder.operation(
        type: "conv_transpose",
        inputs: [
            "x": MILArgument(.name("x")),
            "weight": MILArgument(.value(weightValue)),
            "groups": MILArgument(.value(MILValue.scalarInt32(1))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "dilations": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [4], values: [0, 0, 0, 0]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid")))
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

    let inputs: [Float] = [1, 2, 3, 4]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputs
    )

    for (out, exp) in zip(outputs, inputs) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testConvTranspose3D() async throws {
    let inputShape = [1, 1, 2, 2, 2]
    let outputShape = [1, 1, 2, 2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let weightValue = MILValue.tensorFloat(shape: [1, 1, 1, 1, 1], values: [1])

    let op = MILBuilder.operation(
        type: "conv_transpose",
        inputs: [
            "x": MILArgument(.name("x")),
            "weight": MILArgument(.value(weightValue)),
            "groups": MILArgument(.value(MILValue.scalarInt32(1))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [3], values: [1, 1, 1]))),
            "dilations": MILArgument(.value(MILValue.tensorInt32(shape: [3], values: [1, 1, 1]))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [6], values: [0, 0, 0, 0, 0, 0]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid")))
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

    let inputs: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputs
    )

    for (out, exp) in zip(outputs, inputs) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testConvQuantized2D() async throws {
    if ProcessInfo.processInfo.environment["COREMLTOOLS_ENABLE_CONV_QUANTIZED"] != "1" {
        return
    }
    let inputShape = [1, 1, 2, 2]
    let outputShape = [1, 1, 2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let weightValue = MILValue.tensorUInt8(shape: [1, 1, 1, 1], values: [1])

    let op = MILBuilder.operation(
        type: "conv_quantized",
        inputs: [
            "x": MILArgument(.name("x")),
            "weight": MILArgument(.value(weightValue)),
            "quantization_type": MILArgument(.value(MILValue.scalarString("linear"))),
            "quant_scale": MILArgument(.value(MILValue.scalarFloat(1.0))),
            "quant_bias": MILArgument(.value(MILValue.scalarFloat(0.0))),
            "groups": MILArgument(.value(MILValue.scalarInt32(1))),
            "strides": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "dilations": MILArgument(.value(MILValue.tensorInt32(shape: [2], values: [1, 1]))),
            "pad": MILArgument(.value(MILValue.tensorInt32(shape: [4], values: [0, 0, 0, 0]))),
            "pad_type": MILArgument(.value(MILValue.scalarString("valid")))
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

    let inputs: [Float] = [1, 2, 3, 4]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputs
    )

    for (out, exp) in zip(outputs, inputs) {
        #expect(abs(out - exp) < 1e-4)
    }
}
