import Foundation
import Testing
import CoreMLTools

struct ComparisonCase: Sendable {
    let op: String
    let x: [Float]
    let y: [Float]
    let expected: [Int32]
}

private let comparisonCases: [ComparisonCase] = [
    ComparisonCase(op: "equal", x: [1.0, 2.0], y: [1.0, 3.0], expected: [1, 0]),
    ComparisonCase(op: "not_equal", x: [1.0, 2.0], y: [1.0, 3.0], expected: [0, 1]),
    ComparisonCase(op: "greater", x: [1.0, 2.0], y: [1.0, 3.0], expected: [0, 0]),
    ComparisonCase(op: "greater_equal", x: [1.0, 2.0], y: [1.0, 3.0], expected: [1, 0]),
    ComparisonCase(op: "less", x: [1.0, 2.0], y: [1.0, 3.0], expected: [0, 1]),
    ComparisonCase(op: "less_equal", x: [1.0, 2.0], y: [1.0, 3.0], expected: [1, 1])
]

private func makeComparisonOp(
    op: String,
    inputs: [String: MILArgument],
    outputs: [CoreML_Specification_MILSpec_NamedValueType]
) -> CoreML_Specification_MILSpec_Operation {
    switch op {
    case "equal":
        return MILOps.equal(inputs: inputs, outputs: outputs)
    case "not_equal":
        return MILOps.not_equal(inputs: inputs, outputs: outputs)
    case "greater":
        return MILOps.greater(inputs: inputs, outputs: outputs)
    case "greater_equal":
        return MILOps.greater_equal(inputs: inputs, outputs: outputs)
    case "less":
        return MILOps.less(inputs: inputs, outputs: outputs)
    case "less_equal":
        return MILOps.less_equal(inputs: inputs, outputs: outputs)
    default:
        preconditionFailure("Unsupported comparison op: \(op)")
    }
}

@Test(arguments: comparisonCases)
func testComparisonOps(caseItem: ComparisonCase) async throws {
    let shape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: shape)
    let boolType = MILType.tensor(dataType: .bool, shape: shape)
    let intType = MILType.tensor(dataType: .int32, shape: shape)

    let inputNamedX = MILBuilder.namedValue(name: "x", type: inputType)
    let inputNamedY = MILBuilder.namedValue(name: "y", type: inputType)

    let compareOutput = MILBuilder.namedValue(name: "c", type: boolType)
    let compareOp = makeComparisonOp(
        op: caseItem.op,
        inputs: [
            "x": MILArgument(.name("x")),
            "y": MILArgument(.name("y"))
        ],
        outputs: [compareOutput]
    )

    let castOutput = MILBuilder.namedValue(name: "z", type: intType)
    let castOp = MILOps.cast(
        inputs: [
            "x": MILArgument(.name("c")),
            "dtype": MILArgument(.value(MILValue.scalarString("int32")))
        ],
        outputs: [castOutput]
    )

    let block = MILBuilder.block(operations: [compareOp, castOp], outputs: ["z"])
    let function = MILBuilder.function(inputs: [inputNamedX, inputNamedY], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", shape, .float32), ("y", shape, .float32)],
        outputs: [("z", shape, .int32)]
    )

    let outputs = try MLTestUtils.runInt32Model(
        model: model,
        inputNameX: "x",
        inputValuesX: caseItem.x,
        inputNameY: "y",
        inputValuesY: caseItem.y,
        inputShape: shape,
        outputName: "z"
    )

    for (out, expected) in zip(outputs, caseItem.expected) {
        #expect(out == expected)
    }
}
