import Foundation
import Testing
import CoreMLTools

@Test
func testEinsumChwWhr() async throws {
    let xShape = [1, 2, 3] // C,H,W1
    let yShape = [3, 2, 2] // W1,H,W2
    let outputShape = [1, 2, 2] // C,H,W2

    let xType = MILType.tensor(dataType: .float32, shape: xShape)
    let yType = MILType.tensor(dataType: .float32, shape: yShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamedX = MILBuilder.namedValue(name: "x", type: xType)
    let inputNamedY = MILBuilder.namedValue(name: "y", type: yType)
    let outputNamed = MILBuilder.namedValue(name: "z", type: outputType)

    let einsumOp = MILOps.einsum(
        inputs: [
            "values": MILArgument([.name("x"), .name("y")]),
            "equation": MILArgument(.value(MILValue.scalarString("chw,whr->chr")))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [einsumOp], outputs: ["z"])
    let function = MILBuilder.function(inputs: [inputNamedX, inputNamedY], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", xShape, .float32), ("y", yShape, .float32)],
        outputs: [("z", outputShape, .float32)]
    )

    let xValues: [Float] = [1, 2, 3, 4, 5, 6] // shape [1,2,3]
    let yValues: [Float] = [
        1, 0,  // w1=0, h=0..1, w2=0
        0, 1,  // w1=0, h=0..1, w2=1
        1, 0,  // w1=1
        0, 1,
        1, 0,  // w1=2
        0, 1
    ]

    let outputs = try MLTestUtils.runFloatModelTwoInputs(
        model: model,
        inputNameX: "x",
        inputShapeX: xShape,
        inputValuesX: xValues,
        inputNameY: "y",
        inputShapeY: yShape,
        inputValuesY: yValues,
        outputName: "z"
    )

    var expected: [Float] = []
    expected.reserveCapacity(4)
    for c in 0..<1 {
        for h in 0..<2 {
            for w2 in 0..<2 {
                var sum: Float = 0
                for w1 in 0..<3 {
                    let xIndex = c * 2 * 3 + h * 3 + w1
                    let yIndex = w1 * 2 * 2 + h * 2 + w2
                    sum += xValues[xIndex] * yValues[yIndex]
                }
                expected.append(sum)
            }
        }
    }

    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}
