import Foundation
import Testing
import CoreMLTools

struct ReduceAxesCase {
    let op: String
    let expected: [Float]
}

struct ReduceArgCase {
    let op: String
    let expected: [Int32]
}

private let reduceAxesCases: [ReduceAxesCase] = {
    let logSumExpRow0 = log(exp(1.0) + exp(2.0))
    let logSumExpRow1 = log(exp(3.0) + exp(4.0))
    return [
        ReduceAxesCase(op: "reduce_sum", expected: [3.0, 7.0]),
        ReduceAxesCase(op: "reduce_mean", expected: [1.5, 3.5]),
        ReduceAxesCase(op: "reduce_max", expected: [2.0, 4.0]),
        ReduceAxesCase(op: "reduce_min", expected: [1.0, 3.0]),
        ReduceAxesCase(op: "reduce_prod", expected: [2.0, 12.0]),
        ReduceAxesCase(op: "reduce_sum_square", expected: [5.0, 25.0]),
        ReduceAxesCase(op: "reduce_l1_norm", expected: [3.0, 7.0]),
        ReduceAxesCase(op: "reduce_l2_norm", expected: [Float(sqrt(5.0)), 5.0]),
        ReduceAxesCase(op: "reduce_log_sum", expected: [Float(log(3.0)), Float(log(7.0))]),
        ReduceAxesCase(op: "reduce_log_sum_exp", expected: [Float(logSumExpRow0), Float(logSumExpRow1)])
    ]
}()

private let reduceArgCases: [ReduceArgCase] = [
    ReduceArgCase(op: "reduce_argmax", expected: [1, 1]),
    ReduceArgCase(op: "reduce_argmin", expected: [0, 0])
]

@Test(arguments: reduceAxesCases)
func testReduceAxesOps(caseItem: ReduceAxesCase) async throws {
    let inputShape = [2, 2]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let axesValue = MILValue.tensorInt32(shape: [1], values: [1])

    let op = MILBuilder.operation(
        type: caseItem.op,
        inputs: [
            "x": MILArgument(.name("x")),
            "axes": MILArgument(.value(axesValue)),
            "keep_dims": MILArgument(.value(MILValue.scalarBool(false)))
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

    for (out, exp) in zip(outputs, caseItem.expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test(arguments: reduceArgCases)
func testReduceArgOps(caseItem: ReduceArgCase) async throws {
    let inputShape = [2, 2]
    let outputShape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .int32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: caseItem.op,
        inputs: [
            "x": MILArgument(.name("x")),
            "axis": MILArgument(.value(MILValue.scalarInt32(1))),
            "keep_dims": MILArgument(.value(MILValue.scalarBool(false)))
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
        inputValues: [1, 2, 3, 4]
    )

    for (out, exp) in zip(outputs, caseItem.expected) {
        #expect(out == exp)
    }
}
