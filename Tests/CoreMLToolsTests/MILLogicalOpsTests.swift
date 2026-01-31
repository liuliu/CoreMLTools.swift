import Foundation
import Testing
import CoreMLTools

struct LogicalUnaryCase: Sendable {
    let op: String
    let x: [Float]
    let expected: [Int32]
}

struct LogicalBinaryCase: Sendable {
    let op: String
    let x: [Float]
    let y: [Float]
    let expected: [Int32]
}

private let logicalUnaryCases: [LogicalUnaryCase] = [
    LogicalUnaryCase(op: "logical_not", x: [0.0, 1.0], expected: [1, 0])
]

private let logicalBinaryCases: [LogicalBinaryCase] = [
    LogicalBinaryCase(op: "logical_and", x: [0.0, 1.0], y: [1.0, 1.0], expected: [0, 1]),
    LogicalBinaryCase(op: "logical_or", x: [0.0, 1.0], y: [0.0, 1.0], expected: [0, 1]),
    LogicalBinaryCase(op: "logical_xor", x: [0.0, 1.0], y: [1.0, 1.0], expected: [1, 0])
]

@Test(arguments: logicalUnaryCases)
func testLogicalUnaryOps(caseItem: LogicalUnaryCase) async throws {
    let shape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: shape)
    let boolType = MILType.tensor(dataType: .bool, shape: shape)
    let intType = MILType.tensor(dataType: .int32, shape: shape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)

    let castToBool = MILBuilder.operation(
        type: "cast",
        inputs: [
            "x": MILArgument(.name("x")),
            "dtype": MILArgument(.value(MILValue.scalarString("bool")))
        ],
        outputs: [MILBuilder.namedValue(name: "bx", type: boolType)]
    )

    let logicalOp = MILBuilder.operation(
        type: caseItem.op,
        inputs: ["x": MILArgument(.name("bx"))],
        outputs: [MILBuilder.namedValue(name: "lb", type: boolType)]
    )

    let castToInt = MILBuilder.operation(
        type: "cast",
        inputs: [
            "x": MILArgument(.name("lb")),
            "dtype": MILArgument(.value(MILValue.scalarString("int32")))
        ],
        outputs: [MILBuilder.namedValue(name: "z", type: intType)]
    )

    let block = MILBuilder.block(operations: [castToBool, logicalOp, castToInt], outputs: ["z"])
    let function = MILBuilder.function(inputs: [inputNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", shape, .float32)],
        outputs: [("z", shape, .int32)]
    )

    let outputs = try MLTestUtils.runInt32Model(
        model: model,
        inputName: "x",
        outputName: "z",
        inputShape: shape,
        inputValues: caseItem.x
    )

    for (out, expected) in zip(outputs, caseItem.expected) {
        #expect(out == expected)
    }
}

@Test(arguments: logicalBinaryCases)
func testLogicalBinaryOps(caseItem: LogicalBinaryCase) async throws {
    let shape = [2]
    let inputType = MILType.tensor(dataType: .float32, shape: shape)
    let boolType = MILType.tensor(dataType: .bool, shape: shape)
    let intType = MILType.tensor(dataType: .int32, shape: shape)

    let inputNamedX = MILBuilder.namedValue(name: "x", type: inputType)
    let inputNamedY = MILBuilder.namedValue(name: "y", type: inputType)

    let castX = MILBuilder.operation(
        type: "cast",
        inputs: [
            "x": MILArgument(.name("x")),
            "dtype": MILArgument(.value(MILValue.scalarString("bool")))
        ],
        outputs: [MILBuilder.namedValue(name: "bx", type: boolType)]
    )

    let castY = MILBuilder.operation(
        type: "cast",
        inputs: [
            "x": MILArgument(.name("y")),
            "dtype": MILArgument(.value(MILValue.scalarString("bool")))
        ],
        outputs: [MILBuilder.namedValue(name: "by", type: boolType)]
    )

    let logicalOp = MILBuilder.operation(
        type: caseItem.op,
        inputs: [
            "x": MILArgument(.name("bx")),
            "y": MILArgument(.name("by"))
        ],
        outputs: [MILBuilder.namedValue(name: "lb", type: boolType)]
    )

    let castToInt = MILBuilder.operation(
        type: "cast",
        inputs: [
            "x": MILArgument(.name("lb")),
            "dtype": MILArgument(.value(MILValue.scalarString("int32")))
        ],
        outputs: [MILBuilder.namedValue(name: "z", type: intType)]
    )

    let block = MILBuilder.block(operations: [castX, castY, logicalOp, castToInt], outputs: ["z"])
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
