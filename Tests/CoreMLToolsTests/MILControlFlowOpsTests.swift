import Foundation
import Testing
import CoreMLTools

private func makeConstOp(
    name: String,
    value: CoreML_Specification_MILSpec_Value,
    outputType: CoreML_Specification_MILSpec_ValueType
) -> CoreML_Specification_MILSpec_Operation {
    var op = MILOps.const(
        inputs: [:],
        outputs: [MILBuilder.namedValue(name: name, type: outputType)]
    )
    op.attributes = ["val": value]
    return op
}

@Test
func testCond() async throws {
    let boolType = MILType.tensor(dataType: .bool, shape: [])
    let floatType = MILType.tensor(dataType: .float32, shape: [])

    let predOp = makeConstOp(
        name: "p",
        value: MILValue.scalarBool(true),
        outputType: boolType
    )

    let trueConst = makeConstOp(
        name: "t",
        value: MILValue.scalarFloat(1.0),
        outputType: floatType
    )
    let falseConst = makeConstOp(
        name: "f",
        value: MILValue.scalarFloat(2.0),
        outputType: floatType
    )

    let trueBlock = MILBuilder.block(
        operations: [trueConst],
        outputs: ["t"],
        inputs: []
    )
    let falseBlock = MILBuilder.block(
        operations: [falseConst],
        outputs: ["f"],
        inputs: []
    )

    var condOp = MILOps.cond(
        inputs: [
            "pred": MILArgument(.name("p"))
        ],
        outputs: [MILBuilder.namedValue(name: "y", type: floatType)]
    )
    condOp.blocks = [trueBlock, falseBlock]

    let block = MILBuilder.block(operations: [predOp, condOp], outputs: ["y"])
    let function = MILBuilder.function(inputs: [], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [],
        outputs: [("y", [], .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelNoInputs(
        model: model,
        outputName: "y"
    )

    #expect(outputs == [1.0])
}

@Test
func testWhileLoop() async throws {
    let floatType = MILType.tensor(dataType: .float32, shape: [])
    let boolType = MILType.tensor(dataType: .bool, shape: [])

    let initConst = makeConstOp(
        name: "x0",
        value: MILValue.scalarFloat(0.0),
        outputType: floatType
    )

    let condInput = MILBuilder.namedValue(name: "x", type: floatType)
    let bodyInput = MILBuilder.namedValue(name: "x", type: floatType)

    let condOpInner = MILOps.less(
        inputs: [
            "x": MILArgument(.name("x")),
            "y": MILArgument(.value(MILValue.scalarFloat(3.0)))
        ],
        outputs: [MILBuilder.namedValue(name: "cond", type: boolType)]
    )
    let condBlock = MILBuilder.block(
        operations: [condOpInner],
        outputs: ["cond"],
        inputs: [condInput]
    )

    let addOpInner = MILOps.add(
        inputs: [
            "x": MILArgument(.name("x")),
            "y": MILArgument(.value(MILValue.scalarFloat(1.0)))
        ],
        outputs: [MILBuilder.namedValue(name: "x1", type: floatType)]
    )
    let bodyBlock = MILBuilder.block(
        operations: [addOpInner],
        outputs: ["x1"],
        inputs: [bodyInput]
    )

    var whileOp = MILOps.while_loop(
        inputs: [
            "loop_vars": MILArgument([.name("x0")])
        ],
        outputs: [MILBuilder.namedValue(name: "y", type: floatType)]
    )
    whileOp.blocks = [condBlock, bodyBlock]

    let block = MILBuilder.block(operations: [initConst, whileOp], outputs: ["y"])
    let function = MILBuilder.function(inputs: [], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [],
        outputs: [("y", [], .float32)]
    )

    let outputs = try MLTestUtils.runFloatModelNoInputs(
        model: model,
        outputName: "y"
    )

    #expect(outputs == [3.0])
}
