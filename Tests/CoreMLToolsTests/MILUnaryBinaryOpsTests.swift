import Foundation
import Testing
import CoreMLTools

struct UnaryCase: Sendable {
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

struct BinaryCase: Sendable {
    let op: String
    let x: [Float]
    let y: [Float]
    let expected: [Float]
}

private let unaryCases: [UnaryCase] = [
    UnaryCase(op: "abs", input: [-1.5, 2.0], expected: [1.5, 2.0]),
    UnaryCase(op: "acos", input: [-0.5, 0.5], expected: [acosf(-0.5), acosf(0.5)]),
    UnaryCase(op: "asin", input: [-0.5, 0.5], expected: [asinf(-0.5), asinf(0.5)]),
    UnaryCase(op: "atan", input: [-1.0, 1.0], expected: [atanf(-1.0), atanf(1.0)]),
    UnaryCase(op: "atanh", input: [-0.5, 0.5], expected: [atanhf(-0.5), atanhf(0.5)]),
    UnaryCase(op: "ceil", input: [1.2, -1.2], expected: [ceilf(1.2), ceilf(-1.2)]),
    UnaryCase(op: "clip", input: [-1.0, 2.0], expected: [0.0, 1.0], extraInputs: [
        "alpha": MILValue.scalarFloat(0.0),
        "beta": MILValue.scalarFloat(1.0)
    ]),
    UnaryCase(op: "cos", input: [0.25, -0.75], expected: [cosf(0.25), cosf(-0.75)]),
    UnaryCase(op: "cosh", input: [0.25, -0.75], expected: [coshf(0.25), coshf(-0.75)]),
    UnaryCase(op: "erf", input: [0.25, -0.75], expected: [erff(0.25), erff(-0.75)]),
    UnaryCase(op: "exp", input: [0.25, -0.75], expected: [expf(0.25), expf(-0.75)]),
    UnaryCase(op: "exp2", input: [0.25, -0.75], expected: [exp2f(0.25), exp2f(-0.75)]),
    UnaryCase(op: "floor", input: [1.2, -1.2], expected: [floorf(1.2), floorf(-1.2)]),
    UnaryCase(op: "inverse", input: [0.5, -2.0], expected: [1.0 / 0.5, 1.0 / -2.0], extraInputs: ["epsilon": MILValue.scalarFloat(1e-6)]),
    UnaryCase(op: "log", input: [0.5, 2.0], expected: [logf(0.5), logf(2.0)], extraInputs: ["epsilon": MILValue.scalarFloat(1e-6)]),
    UnaryCase(op: "round", input: [1.4, 1.6], expected: [roundf(1.4), roundf(1.6)]),
    UnaryCase(op: "rsqrt", input: [0.5, 4.0], expected: [1.0 / sqrtf(0.5), 1.0 / sqrtf(4.0)], extraInputs: ["epsilon": MILValue.scalarFloat(1e-6)]),
    UnaryCase(op: "sign", input: [-2.0, 0.0], expected: [-1.0, 0.0]),
    UnaryCase(op: "sin", input: [0.25, -0.75], expected: [sinf(0.25), sinf(-0.75)]),
    UnaryCase(op: "sinh", input: [0.25, -0.75], expected: [sinhf(0.25), sinhf(-0.75)]),
    UnaryCase(op: "sqrt", input: [0.5, 4.0], expected: [sqrtf(0.5), sqrtf(4.0)]),
    UnaryCase(op: "square", input: [-1.5, 2.0], expected: [2.25, 4.0]),
    UnaryCase(op: "tan", input: [0.25, -0.75], expected: [tanf(0.25), tanf(-0.75)]),
    UnaryCase(op: "tanh", input: [0.25, -0.75], expected: [tanhf(0.25), tanhf(-0.75)]),
    UnaryCase(op: "threshold", input: [-1.0, 2.0], expected: [0.0, 2.0], extraInputs: [
        "alpha": MILValue.scalarFloat(0.0)
    ])
]

private let binaryCases: [BinaryCase] = [
    BinaryCase(op: "add", x: [1.0, -2.0], y: [3.0, 4.0], expected: [4.0, 2.0]),
    BinaryCase(op: "floor_div", x: [7.0, -7.0], y: [2.0, 2.0], expected: [floorf(7.0/2.0), floorf(-7.0/2.0)]),
    BinaryCase(op: "maximum", x: [1.0, -2.0], y: [3.0, -3.0], expected: [3.0, -2.0]),
    BinaryCase(op: "minimum", x: [1.0, -2.0], y: [3.0, -3.0], expected: [1.0, -3.0]),
    BinaryCase(op: "mod", x: [7.0, -7.0], y: [2.0, 2.0], expected: [fmodf(7.0, 2.0), fmodf(-7.0, 2.0)]),
    BinaryCase(op: "mul", x: [1.0, -2.0], y: [3.0, 4.0], expected: [3.0, -8.0]),
    BinaryCase(op: "real_div", x: [7.0, -7.0], y: [2.0, 2.0], expected: [3.5, -3.5]),
    BinaryCase(op: "pow", x: [2.0, 9.0], y: [3.0, 0.5], expected: [powf(2.0, 3.0), powf(9.0, 0.5)]),
    BinaryCase(op: "sub", x: [1.0, -2.0], y: [3.0, 4.0], expected: [-2.0, -6.0])
]

@Test(arguments: unaryCases)
func testUnaryOps(caseItem: UnaryCase) async throws {
    let shape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: shape)
    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: inputType)

    var inputs: [String: MILArgument] = ["x": MILArgument(.name("x"))]
    for (key, value) in caseItem.extraInputs {
        inputs[key] = MILArgument(.value(value))
    }
    let op = MILBuilder.operation(
        type: caseItem.op,
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

@Test(arguments: binaryCases)
func testBinaryOps(caseItem: BinaryCase) async throws {
    let shape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: shape)
    // Build model with y bound as a constant via immediate value
    let inputNamedX2 = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed2 = MILBuilder.namedValue(name: "z", type: inputType)

    let yValue = MILValue.tensorFloat(shape: shape, values: caseItem.y)
    let op2 = MILBuilder.operation(
        type: caseItem.op,
        inputs: [
            "x": MILArgument(.name("x")),
            "y": MILArgument(.value(yValue))
        ],
        outputs: [outputNamed2]
    )

    let block2 = MILBuilder.block(operations: [op2], outputs: ["z"])
    let function2 = MILBuilder.function(inputs: [inputNamedX2], opset: "CoreML8", block: block2)
    let program2 = MILBuilder.program(functions: ["main": function2])
    let model2 = MLProgramBuilder.makeModel(
        program: program2,
        inputs: [("x", shape, .float32)],
        outputs: [("z", shape, .float32)]
    )

    let outputs2 = try MLTestUtils.runFloatModel(
        model: model2,
        inputName: "x",
        outputName: "z",
        inputShape: shape,
        inputValues: caseItem.x
    )

    for (out, expected) in zip(outputs2, caseItem.expected) {
        #expect(abs(out - expected) < 1e-4)
    }
}
