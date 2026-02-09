import Foundation
import Testing
import CoreMLTools

struct ActivationCase: Sendable {
    let op: String
    let input: [Float]
    let expected: [Float]
    let extraInputs: [String: CoreML_Specification_MILSpec_Value]

    init(op: String, input: [Float], expected: [Float], extraInputs: [String: CoreML_Specification_MILSpec_Value] = [:]) {
        self.op = op
        self.input = input
        self.expected = expected
        self.extraInputs = extraInputs
    }
}

private let activationCases: [ActivationCase] = [
    ActivationCase(op: "relu", input: [-1.0, 0.5], expected: [0.0, 0.5]),
    ActivationCase(op: "relu6", input: [-1.0, 7.0], expected: [0.0, 6.0]),
    ActivationCase(op: "leaky_relu", input: [-1.0, 0.5], expected: [-0.1, 0.5], extraInputs: ["alpha": MILValue.scalarFloat(0.1)]),
    ActivationCase(op: "elu", input: [-1.0, 0.5], expected: [expf(-1.0) - 1.0, 0.5], extraInputs: ["alpha": MILValue.scalarFloat(1.0)]),
    ActivationCase(op: "sigmoid", input: [-1.0, 0.5], expected: [1.0 / (1.0 + expf(1.0)), 1.0 / (1.0 + expf(-0.5))]),
    ActivationCase(op: "silu", input: [-1.0, 0.5], expected: [(-1.0) * (1.0 / (1.0 + expf(1.0))), 0.5 * (1.0 / (1.0 + expf(-0.5)))]),
    ActivationCase(op: "softplus", input: [-1.0, 0.5], expected: [logf(1 + expf(-abs(-1.0))) + max(-1.0, 0.0), logf(1 + expf(-abs(0.5))) + max(0.5, 0.0)]),
    ActivationCase(op: "softsign", input: [-1.0, 0.5], expected: [-1.0 / (1.0 + 1.0), 0.5 / (1.0 + 0.5)]),
    ActivationCase(op: "thresholded_relu", input: [-1.0, 0.5], expected: [0.0, 0.5], extraInputs: ["alpha": MILValue.scalarFloat(0.2)]),
    ActivationCase(op: "clamped_relu", input: [-2.0, 2.0], expected: [-0.2, 1.0], extraInputs: ["alpha": MILValue.scalarFloat(0.1), "beta": MILValue.scalarFloat(1.0)]),
    ActivationCase(op: "sigmoid_hard", input: [-2.0, 2.0], expected: [0.1, 0.9], extraInputs: ["alpha": MILValue.scalarFloat(0.2), "beta": MILValue.scalarFloat(0.5)]),
    ActivationCase(op: "scaled_tanh", input: [-1.0, 1.0], expected: [1.5 * tanhf(-0.5), 1.5 * tanhf(0.5)], extraInputs: ["alpha": MILValue.scalarFloat(1.5), "beta": MILValue.scalarFloat(0.5)])
]

private func makeActivationOp(
    op: String,
    inputs: [String: MILArgument],
    outputs: [CoreML_Specification_MILSpec_NamedValueType]
) -> CoreML_Specification_MILSpec_Operation {
    switch op {
    case "relu":
        return MILOps.relu(inputs: inputs, outputs: outputs)
    case "relu6":
        return MILOps.relu6(inputs: inputs, outputs: outputs)
    case "leaky_relu":
        return MILOps.leaky_relu(inputs: inputs, outputs: outputs)
    case "elu":
        return MILOps.elu(inputs: inputs, outputs: outputs)
    case "sigmoid":
        return MILOps.sigmoid(inputs: inputs, outputs: outputs)
    case "silu":
        return MILOps.silu(inputs: inputs, outputs: outputs)
    case "softplus":
        return MILOps.softplus(inputs: inputs, outputs: outputs)
    case "softsign":
        return MILOps.softsign(inputs: inputs, outputs: outputs)
    case "thresholded_relu":
        return MILOps.thresholded_relu(inputs: inputs, outputs: outputs)
    case "clamped_relu":
        return MILOps.clamped_relu(inputs: inputs, outputs: outputs)
    case "sigmoid_hard":
        return MILOps.sigmoid_hard(inputs: inputs, outputs: outputs)
    case "scaled_tanh":
        return MILOps.scaled_tanh(inputs: inputs, outputs: outputs)
    default:
        preconditionFailure("Unsupported activation op: \(op)")
    }
}

@Test(arguments: activationCases)
func testActivationOps(caseItem: ActivationCase) async throws {
    let shape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: shape)
    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: inputType)

    var inputs: [String: MILArgument] = ["x": MILArgument(.name("x"))]
    for (key, value) in caseItem.extraInputs {
        inputs[key] = MILArgument(.value(value))
    }

    let op = makeActivationOp(
        op: caseItem.op,
        inputs: inputs,
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", shape, .float32)],
        outputs: [("y", shape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: shape,
        inputValues: caseItem.input
    )

    for (out, expected) in zip(outputs, caseItem.expected) {
        #expect(abs(out - expected) < 1e-4)
    }
}
