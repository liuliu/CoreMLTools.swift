import Foundation
import CoreML
import Testing
import CoreMLTools

@Test
func testNonMaximumSuppression() async throws {
    let boxesShape = [1, 4, 2]
    let scoresShape = [1, 1, 2]
    let boxesType = MILType.tensor(dataType: .float32, shape: boxesShape)
    let scoresType = MILType.tensor(dataType: .float32, shape: scoresShape)

    let outBoxesShape = [1, 4, 1]
    let outScoresShape = [1, 1, 1]
    let outIndicesShape = [1, 1]

    let inputNamedBoxes = MILBuilder.namedValue(name: "boxes", type: boxesType)
    let inputNamedScores = MILBuilder.namedValue(name: "scores", type: scoresType)

    let outBoxesNamed = MILBuilder.namedValue(name: "out_boxes", type: MILType.tensor(dataType: .float32, shape: outBoxesShape))
    let outScoresNamed = MILBuilder.namedValue(name: "out_scores", type: MILType.tensor(dataType: .float32, shape: outScoresShape))
    let outIndicesNamed = MILBuilder.namedValue(name: "out_indices", type: MILType.tensor(dataType: .int32, shape: outIndicesShape))

    let op = MILOps.non_maximum_suppression(
        inputs: [
            "boxes": MILArgument(.name("boxes")),
            "scores": MILArgument(.name("scores")),
            "iou_threshold": MILArgument(.value(MILValue.scalarFloat(0.5))),
            "max_boxes": MILArgument(.value(MILValue.scalarInt32(1))),
            "per_class_suppression": MILArgument(.value(MILValue.scalarBool(false)))
        ],
        outputs: [outBoxesNamed, outScoresNamed, outIndicesNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["out_boxes", "out_scores", "out_indices"])
    let function = MILBuilder.function(inputs: [inputNamedBoxes, inputNamedScores], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("boxes", boxesShape, .float32), ("scores", scoresShape, .float32)],
        outputs: [
            ("out_boxes", outBoxesShape, .float32),
            ("out_scores", outScoresShape, .float32),
            ("out_indices", outIndicesShape, .int32)
        ]
    )

    let boxesValues: [Float] = [
        0.0, 0.0,
        0.0, 0.0,
        1.0, 1.0,
        1.0, 1.0
    ]
    let scoresValues: [Float] = [
        0.9, 0.5
    ]

    let outputs = try MLTestUtils.runModelOutputsTwoInputs(
        model: model,
        inputNameX: "boxes",
        inputShapeX: boxesShape,
        inputValuesX: boxesValues,
        inputNameY: "scores",
        inputShapeY: scoresShape,
        inputValuesY: scoresValues,
        outputNames: ["out_boxes", "out_scores", "out_indices"]
    )

    func floatValues(_ array: MLMultiArray) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(array.count)
        for idx in 0..<array.count {
            values.append(array[idx].floatValue)
        }
        return values
    }

    func intValues(_ array: MLMultiArray) -> [Int32] {
        var values: [Int32] = []
        values.reserveCapacity(array.count)
        for idx in 0..<array.count {
            values.append(array[idx].int32Value)
        }
        return values
    }

    if let outBoxes = outputs["out_boxes"], let outScores = outputs["out_scores"] {
        let boxVals = floatValues(outBoxes)
        let scoreVals = floatValues(outScores)
        #expect(boxVals == [0.0, 0.0, 1.0, 1.0])
        #expect(scoreVals == [0.9])
    } else {
        #expect(Bool(false))
    }

    if let outIndices = outputs["out_indices"] {
        let indicesVals = intValues(outIndices)
        #expect(indicesVals == [0])
    } else {
        #expect(Bool(false))
    }
}
