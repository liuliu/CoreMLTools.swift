import Foundation
import Testing
import CoreMLTools

@Test
func testConstSymbolic() async throws {
    if ProcessInfo.processInfo.environment["COREMLTOOLS_ENABLE_CONST_SYMBOLIC"] != "1" {
        return
    }

    let outputShape = [2]
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)
    let constValue = MILValue.tensorFloat(shape: outputShape, values: [3, 4])

    var constOp = MILOps._const_symbolic(
        inputs: [:],
        outputs: [outputNamed]
    )
    constOp.attributes = ["val": constValue]

    let block = MILBuilder.block(operations: [constOp], outputs: ["y"])
    let function = MILBuilder.function(inputs: [], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [],
        outputs: [("y", outputShape, .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelNoInputs(
        model: model,
        outputName: "y"
    )

    let expected: [Float] = [3, 4]
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}
