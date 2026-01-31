import Foundation
import CoreML
import Testing
import CoreMLTools

private func floatValues(_ array: MLMultiArray) -> [Float] {
    var values: [Float] = []
    values.reserveCapacity(array.count)
    for idx in 0..<array.count {
        values.append(array[idx].floatValue)
    }
    return values
}

@Test
func testRnnZeroWeights() async throws {
    let xShape = [2, 1, 1]
    let hShape = [1, 1]
    let yShape = [1, 1, 1]

    let xType = MILType.tensor(dataType: .float32, shape: xShape)
    let hType = MILType.tensor(dataType: .float32, shape: hShape)
    let yType = MILType.tensor(dataType: .float32, shape: yShape)

    let inputX = MILBuilder.namedValue(name: "x", type: xType)
    let inputH = MILBuilder.namedValue(name: "initial_h", type: hType)
    let outputY = MILBuilder.namedValue(name: "y", type: yType)
    let outputH = MILBuilder.namedValue(name: "h", type: hType)

    let op = MILBuilder.operation(
        type: "rnn",
        inputs: [
            "x": MILArgument(.name("x")),
            "initial_h": MILArgument(.name("initial_h")),
            "weight_ih": MILArgument(.value(MILValue.tensorFloat(shape: [1, 1], values: [0.0]))),
            "weight_hh": MILArgument(.value(MILValue.tensorFloat(shape: [1, 1], values: [0.0]))),
            "output_sequence": MILArgument(.value(MILValue.scalarBool(false))),
            "direction": MILArgument(.value(MILValue.scalarString("forward"))),
            "activation": MILArgument(.value(MILValue.scalarString("tanh")))
        ],
        outputs: [outputY, outputH]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y", "h"])
    let function = MILBuilder.function(inputs: [inputX, inputH], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", xShape, .float32), ("initial_h", hShape, .float32)],
        outputs: [("y", yShape, .float32), ("h", hShape, .float32)]
    )

    let outputs = try MLTestUtils.runModelOutputsTwoInputs(
        model: model,
        inputNameX: "x",
        inputShapeX: xShape,
        inputValuesX: [1.0, 2.0],
        inputNameY: "initial_h",
        inputShapeY: hShape,
        inputValuesY: [0.0],
        outputNames: ["y", "h"]
    )

    if let y = outputs["y"] {
        #expect(floatValues(y) == [0.0])
    } else {
        #expect(Bool(false))
    }

    if let h = outputs["h"] {
        #expect(floatValues(h) == [0.0])
    } else {
        #expect(Bool(false))
    }
}

@Test
func testGruZeroWeights() async throws {
    let xShape = [2, 1, 1]
    let hShape = [1, 1]
    let yShape = [1, 1, 1]

    let xType = MILType.tensor(dataType: .float32, shape: xShape)
    let hType = MILType.tensor(dataType: .float32, shape: hShape)
    let yType = MILType.tensor(dataType: .float32, shape: yShape)

    let inputX = MILBuilder.namedValue(name: "x", type: xType)
    let inputH = MILBuilder.namedValue(name: "initial_h", type: hType)
    let outputY = MILBuilder.namedValue(name: "y", type: yType)
    let outputH = MILBuilder.namedValue(name: "h", type: hType)

    let op = MILBuilder.operation(
        type: "gru",
        inputs: [
            "x": MILArgument(.name("x")),
            "initial_h": MILArgument(.name("initial_h")),
            "weight_ih": MILArgument(.value(MILValue.tensorFloat(shape: [3, 1], values: [0.0, 0.0, 0.0]))),
            "weight_hh": MILArgument(.value(MILValue.tensorFloat(shape: [3, 1], values: [0.0, 0.0, 0.0]))),
            "output_sequence": MILArgument(.value(MILValue.scalarBool(false))),
            "direction": MILArgument(.value(MILValue.scalarString("forward"))),
            "recurrent_activation": MILArgument(.value(MILValue.scalarString("sigmoid"))),
            "activation": MILArgument(.value(MILValue.scalarString("tanh")))
        ],
        outputs: [outputY, outputH]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y", "h"])
    let function = MILBuilder.function(inputs: [inputX, inputH], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", xShape, .float32), ("initial_h", hShape, .float32)],
        outputs: [("y", yShape, .float32), ("h", hShape, .float32)]
    )

    let outputs = try MLTestUtils.runModelOutputsTwoInputs(
        model: model,
        inputNameX: "x",
        inputShapeX: xShape,
        inputValuesX: [1.0, 2.0],
        inputNameY: "initial_h",
        inputShapeY: hShape,
        inputValuesY: [0.0],
        outputNames: ["y", "h"]
    )

    if let y = outputs["y"] {
        #expect(floatValues(y) == [0.0])
    } else {
        #expect(Bool(false))
    }

    if let h = outputs["h"] {
        #expect(floatValues(h) == [0.0])
    } else {
        #expect(Bool(false))
    }
}

@Test
func testLstmZeroWeights() async throws {
    let xShape = [2, 1, 1]
    let hShape = [1, 1]
    let yShape = [1, 1, 1]

    let xType = MILType.tensor(dataType: .float32, shape: xShape)
    let hType = MILType.tensor(dataType: .float32, shape: hShape)
    let yType = MILType.tensor(dataType: .float32, shape: yShape)

    let inputX = MILBuilder.namedValue(name: "x", type: xType)
    let inputH = MILBuilder.namedValue(name: "initial_h", type: hType)
    let inputC = MILBuilder.namedValue(name: "initial_c", type: hType)
    let outputY = MILBuilder.namedValue(name: "y", type: yType)
    let outputH = MILBuilder.namedValue(name: "h", type: hType)
    let outputC = MILBuilder.namedValue(name: "c", type: hType)

    let op = MILBuilder.operation(
        type: "lstm",
        inputs: [
            "x": MILArgument(.name("x")),
            "initial_h": MILArgument(.name("initial_h")),
            "initial_c": MILArgument(.name("initial_c")),
            "weight_ih": MILArgument(.value(MILValue.tensorFloat(shape: [4, 1], values: [0.0, 0.0, 0.0, 0.0]))),
            "weight_hh": MILArgument(.value(MILValue.tensorFloat(shape: [4, 1], values: [0.0, 0.0, 0.0, 0.0]))),
            "output_sequence": MILArgument(.value(MILValue.scalarBool(false))),
            "direction": MILArgument(.value(MILValue.scalarString("forward"))),
            "recurrent_activation": MILArgument(.value(MILValue.scalarString("sigmoid"))),
            "cell_activation": MILArgument(.value(MILValue.scalarString("tanh"))),
            "activation": MILArgument(.value(MILValue.scalarString("tanh")))
        ],
        outputs: [outputY, outputH, outputC]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y", "h", "c"])
    let function = MILBuilder.function(inputs: [inputX, inputH, inputC], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", xShape, .float32), ("initial_h", hShape, .float32), ("initial_c", hShape, .float32)],
        outputs: [("y", yShape, .float32), ("h", hShape, .float32), ("c", hShape, .float32)]
    )

    let outputs = try MLTestUtils.runModelOutputsThreeInputs(
        model: model,
        inputNameX: "x",
        inputShapeX: xShape,
        inputValuesX: [1.0, 2.0],
        inputNameY: "initial_h",
        inputShapeY: hShape,
        inputValuesY: [0.0],
        inputNameZ: "initial_c",
        inputShapeZ: hShape,
        inputValuesZ: [0.0],
        outputNames: ["y", "h", "c"]
    )

    if let y = outputs["y"] {
        #expect(floatValues(y) == [0.0])
    } else {
        #expect(Bool(false))
    }

    if let h = outputs["h"] {
        #expect(floatValues(h) == [0.0])
    } else {
        #expect(Bool(false))
    }

    if let c = outputs["c"] {
        #expect(floatValues(c) == [0.0])
    } else {
        #expect(Bool(false))
    }
}
