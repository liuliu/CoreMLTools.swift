import Foundation
import CoreML
import Testing
import CoreMLTools

@Test
func testClassifyInt64Output() async throws {
    if ProcessInfo.processInfo.environment["COREMLTOOLS_ENABLE_CLASSIFY_TEST"] != "1" {
        return
    }

    let probsShape = [2]
    let probsType = MILType.tensor(dataType: .float32, shape: probsShape)

    let inputNamed = MILBuilder.namedValue(name: "probs", type: probsType)
    let labelType = MILType.tensor(dataType: .int64, shape: [])
    let dictType = MILType.dictionary(
        keyType: MILType.tensor(dataType: .int64, shape: []),
        valueType: MILType.tensor(dataType: .float64, shape: [])
    )

    let labelNamed = MILBuilder.namedValue(name: "classLabel", type: labelType)
    let probsNamed = MILBuilder.namedValue(name: "classLabel_probs", type: dictType)

    let op = MILBuilder.operation(
        type: "classify",
        inputs: [
            "probabilities": MILArgument(.name("probs")),
            "classes": MILArgument(.value(MILValue.listInt64([0, 1])))
        ],
        outputs: [labelNamed, probsNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["classLabel", "classLabel_probs"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    var inputArrayType = CoreML_Specification_ArrayFeatureType()
    inputArrayType.shape = probsShape.map { Int64($0) }
    inputArrayType.dataType = .float32
    var inputFeatureType = CoreML_Specification_FeatureType()
    inputFeatureType.multiArrayType = inputArrayType

    var labelFeatureType = CoreML_Specification_FeatureType()
    labelFeatureType.int64Type = CoreML_Specification_Int64FeatureType()

    var dictTypeFeature = CoreML_Specification_DictionaryFeatureType()
    dictTypeFeature.int64KeyType = CoreML_Specification_Int64FeatureType()
    var probsFeatureType = CoreML_Specification_FeatureType()
    probsFeatureType.dictionaryType = dictTypeFeature

    var model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("probs", inputFeatureType)],
        outputs: [("classLabel", labelFeatureType), ("classLabel_probs", probsFeatureType)]
    )
    model.description_p.predictedFeatureName = "classLabel"
    model.description_p.predictedProbabilitiesName = "classLabel_probs"

    let (label, probabilities) = try MLTestUtils.runClassifyModelInt64(
        model: model,
        inputName: "probs",
        inputShape: probsShape,
        inputValues: [0.2, 0.8],
        labelOutput: "classLabel",
        probabilityOutput: "classLabel_probs"
    )

    #expect(label == 1)
    #expect(probabilities.count == 2)
    if let oneProb = probabilities[1] {
        #expect(abs(oneProb - 0.8) < 1e-6)
    } else {
        #expect(Bool(false))
    }
}
