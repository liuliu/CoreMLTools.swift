import Foundation
import Testing
import CoreMLTools

@Test
func testSoftplusParametric() async throws {
    let inputShape = [1, 2, 1, 1]
    let outputShape = [1, 2, 1, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let alphaValue = MILValue.tensorFloat(shape: [2], values: [1.0, 2.0])
    let betaValue = MILValue.tensorFloat(shape: [2], values: [1.0, 0.5])

    let op = MILOps.softplus_parametric(
        inputs: [
            "x": MILArgument(.name("x")),
            "alpha": MILArgument(.value(alphaValue)),
            "beta": MILArgument(.value(betaValue))
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

    let inputs: [Float] = [0.0, 1.0]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputs
    )

    let expected0 = Float(log(1.0 + exp(1.0 * 0.0)))
    let expected1 = Float(2.0 * log(1.0 + exp(0.5 * 1.0)))
    let expected: [Float] = [expected0, expected1]
    for (out, expVal) in zip(outputs, expected) {
        #expect(abs(out - expVal) < 1e-4)
    }
}
