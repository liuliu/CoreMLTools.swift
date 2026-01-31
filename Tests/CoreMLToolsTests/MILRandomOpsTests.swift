import Foundation
import Testing
import CoreMLTools

@Test
func testRandomUniform() async throws {
    let shapeInputShape = [2]
    let outputShape = [2, 2]
    let shapeType = MILType.tensor(dataType: .int32, shape: shapeInputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "shape", type: shapeType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "random_uniform",
        inputs: [
            "shape": MILArgument(.name("shape")),
            "low": MILArgument(.value(MILValue.scalarFloat(0.0))),
            "high": MILArgument(.value(MILValue.scalarFloat(1.0))),
            "seed": MILArgument(.value(MILValue.scalarInt32(-1)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("shape", shapeInputShape, .int32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelInt32Input(
        model: model,
        inputName: "shape",
        outputName: "y",
        inputShape: shapeInputShape,
        inputValues: [2, 2]
    )

    #expect(outputs.count == 4)
    #expect(outputs.allSatisfy { $0 >= 0.0 && $0 <= 1.0 })
}

@Test
func testRandomNormal() async throws {
    let shapeInputShape = [2]
    let outputShape = [2, 2]
    let shapeType = MILType.tensor(dataType: .int32, shape: shapeInputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "shape", type: shapeType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "random_normal",
        inputs: [
            "shape": MILArgument(.name("shape")),
            "mean": MILArgument(.value(MILValue.scalarFloat(0.0))),
            "stddev": MILArgument(.value(MILValue.scalarFloat(1.0))),
            "seed": MILArgument(.value(MILValue.scalarInt32(-1)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("shape", shapeInputShape, .int32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelInt32Input(
        model: model,
        inputName: "shape",
        outputName: "y",
        inputShape: shapeInputShape,
        inputValues: [2, 2]
    )

    #expect(outputs.count == 4)
    #expect(outputs.allSatisfy { $0.isFinite })
}

@Test
func testRandomBernoulli() async throws {
    let shapeInputShape = [1]
    let outputShape = [4]
    let shapeType = MILType.tensor(dataType: .int32, shape: shapeInputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "shape", type: shapeType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "random_bernoulli",
        inputs: [
            "shape": MILArgument(.name("shape")),
            "prob": MILArgument(.value(MILValue.scalarFloat(0.5))),
            "seed": MILArgument(.value(MILValue.scalarInt32(-1)))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("shape", shapeInputShape, .int32)],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelInt32Input(
        model: model,
        inputName: "shape",
        outputName: "y",
        inputShape: shapeInputShape,
        inputValues: [4]
    )

    #expect(outputs.count == 4)
    #expect(outputs.allSatisfy { abs($0 - 0.0) < 1e-4 || abs($0 - 1.0) < 1e-4 })
}

@Test
func testRandomCategorical() async throws {
    let inputShape = [1, 2]
    let outputShape = [1, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "random_categorical",
        inputs: [
            "x": MILArgument(.name("x")),
            "mode": MILArgument(.value(MILValue.scalarString("logits"))),
            "size": MILArgument(.value(MILValue.scalarInt32(1))),
            "seed": MILArgument(.value(MILValue.scalarInt32(-1)))
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
        inputValues: [0.0, 1.0]
    )

    #expect(outputs.count == 1)
    if let value = outputs.first {
        #expect(value >= 0.0 && value < 2.0)
    }
}
