import Foundation
import Testing
import CoreMLTools

@Test
func testDepthToSpace() async throws {
    let inputShape = [1, 4, 1, 1]
    let outputShape = [1, 1, 2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "depth_to_space",
        inputs: [
            "x": MILArgument(.name("x")),
            "block_size": MILArgument(.value(MILValue.scalarInt32(2)))
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

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [9.0, 5.0, 1.0, 3.0]
    )

    let expected: [Float] = [9.0, 5.0, 1.0, 3.0]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testSpaceToDepth() async throws {
    let inputShape = [1, 1, 2, 2]
    let outputShape = [1, 4, 1, 1]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "space_to_depth",
        inputs: [
            "x": MILArgument(.name("x")),
            "block_size": MILArgument(.value(MILValue.scalarInt32(2)))
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

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [7.0, 9.0, 4.0, 6.0]
    )

    let expected: [Float] = [7.0, 9.0, 4.0, 6.0]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testPixelShuffle() async throws {
    let inputShape = [1, 4, 1, 1]
    let outputShape = [1, 1, 2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "pixel_shuffle",
        inputs: [
            "x": MILArgument(.name("x")),
            "upscale_factor": MILArgument(.value(MILValue.scalarInt32(2)))
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

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [9.0, 5.0, 1.0, 3.0]
    )

    let expected: [Float] = [9.0, 5.0, 1.0, 3.0]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testSpaceToBatch() async throws {
    let inputShape = [2, 1, 2, 4]
    let outputShape = [8, 1, 1, 3]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let blockShape = MILValue.tensorInt32(shape: [2], values: [2, 2])
    let paddings = MILValue.tensorInt32(shape: [2, 2], values: [0, 0, 2, 0])
    let op = MILBuilder.operation(
        type: "space_to_batch",
        inputs: [
            "x": MILArgument(.name("x")),
            "block_shape": MILArgument(.value(blockShape)),
            "paddings": MILArgument(.value(paddings))
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

    let inputValues: [Float] = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputValues
    )

    let expected: [Float] = [
        0, 1, 3,
        0, 9, 11,
        0, 2, 4,
        0, 10, 12,
        0, 5, 7,
        0, 13, 15,
        0, 6, 8,
        0, 14, 16
    ]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testBatchToSpace() async throws {
    let inputShape = [8, 1, 1, 3]
    let outputShape = [2, 1, 2, 4]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let blockShape = MILValue.tensorInt32(shape: [2], values: [2, 2])
    let crops = MILValue.tensorInt32(shape: [2, 2], values: [0, 0, 2, 0])
    let op = MILBuilder.operation(
        type: "batch_to_space",
        inputs: [
            "x": MILArgument(.name("x")),
            "block_shape": MILArgument(.value(blockShape)),
            "crops": MILArgument(.value(crops))
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

    let inputValues: [Float] = [
        0, 1, 3,
        0, 9, 11,
        0, 2, 4,
        0, 10, 12,
        0, 5, 7,
        0, 13, 15,
        0, 6, 8,
        0, 14, 16
    ]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputValues
    )

    let expected: [Float] = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testResizeNearestNeighbor() async throws {
    let inputShape = [1, 1, 2, 1]
    let outputShape = [1, 1, 2, 3]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "resize_nearest_neighbor",
        inputs: [
            "x": MILArgument(.name("x")),
            "target_size_height": MILArgument(.value(MILValue.scalarInt32(2))),
            "target_size_width": MILArgument(.value(MILValue.scalarInt32(3)))
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

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [0.37, 6.17]
    )

    let expected: [Float] = [0.37, 0.37, 0.37, 6.17, 6.17, 6.17]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testResizeBilinear() async throws {
    let inputShape = [1, 1, 2]
    let outputShape = [1, 1, 5]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "resize_bilinear",
        inputs: [
            "x": MILArgument(.name("x")),
            "target_size_height": MILArgument(.value(MILValue.scalarInt32(1))),
            "target_size_width": MILArgument(.value(MILValue.scalarInt32(5))),
            "sampling_mode": MILArgument(.value(MILValue.scalarString("UNALIGN_CORNERS")))
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

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [0.0, 1.0]
    )

    let expected: [Float] = [0.0, 0.1, 0.5, 0.9, 1.0]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testUpsampleNearestNeighbor() async throws {
    let inputShape = [1, 1, 1, 3]
    let outputShape = [1, 1, 1, 6]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "upsample_nearest_neighbor",
        inputs: [
            "x": MILArgument(.name("x")),
            "scale_factor_height": MILArgument(.value(MILValue.scalarInt32(1))),
            "scale_factor_width": MILArgument(.value(MILValue.scalarInt32(2)))
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

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [1.5, 2.5, 3.5]
    )

    let expected: [Float] = [1.5, 1.5, 2.5, 2.5, 3.5, 3.5]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testUpsampleBilinear() async throws {
    let inputShape = [1, 1, 2]
    let outputShape = [1, 1, 6]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "upsample_bilinear",
        inputs: [
            "x": MILArgument(.name("x")),
            "scale_factor_height": MILArgument(.value(MILValue.scalarInt32(1))),
            "scale_factor_width": MILArgument(.value(MILValue.scalarInt32(3))),
            "align_corners": MILArgument(.value(MILValue.scalarBool(true)))
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

    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: [0.0, 1.0]
    )

    let expected: [Float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testCrop() async throws {
    let inputShape = [1, 1, 4, 4]
    let outputShape = [1, 1, 3, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let cropHeight = MILValue.tensorInt32(shape: [2], values: [0, 1])
    let cropWidth = MILValue.tensorInt32(shape: [2], values: [1, 1])
    let op = MILBuilder.operation(
        type: "crop",
        inputs: [
            "x": MILArgument(.name("x")),
            "crop_height": MILArgument(.value(cropHeight)),
            "crop_width": MILArgument(.value(cropWidth))
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

    let inputValues: [Float] = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ]
    let outputs = try MLTestUtils.runFloatModel(
        model: model,
        inputName: "x",
        outputName: "y",
        inputShape: inputShape,
        inputValues: inputValues
    )

    let expected: [Float] = [2, 3, 6, 7, 10, 11]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}

@Test
func testCropResize() async throws {
    if ProcessInfo.processInfo.environment["COREMLTOOLS_ENABLE_CROP_RESIZE"] != "1" {
        return
    }
    let inputShape = [1, 1, 4, 4]
    let outputShape = [1, 1, 2, 2]
    let inputType = MILType.tensor(dataType: .float32, shape: inputShape)
    let outputType = MILType.tensor(dataType: .float32, shape: outputShape)

    let inputNamed = MILBuilder.namedValue(name: "x", type: inputType)
    let roiType = MILType.tensor(dataType: .float32, shape: [1, 4])
    let roiNamed = MILBuilder.namedValue(name: "boxes", type: roiType)
    let outputNamed = MILBuilder.namedValue(name: "y", type: outputType)

    let op = MILBuilder.operation(
        type: "crop_resize",
        inputs: [
            "x": MILArgument(.name("x")),
            "boxes": MILArgument(.name("boxes")),
            "target_width": MILArgument(.value(MILValue.scalarInt32(2))),
            "target_height": MILArgument(.value(MILValue.scalarInt32(2))),
            "normalized_coordinates": MILArgument(.value(MILValue.scalarBool(true))),
            "spatial_scale": MILArgument(.value(MILValue.scalarFloat(1.0))),
            "box_indices": MILArgument(.value(MILValue.tensorInt32(shape: [1], values: [0]))),
            "box_coordinate_mode": MILArgument(.value(MILValue.scalarString("CORNERS_HEIGHT_FIRST"))),
            "sampling_mode": MILArgument(.value(MILValue.scalarString("STRICT_ALIGN_CORNERS")))
        ],
        outputs: [outputNamed]
    )

    let block = MILBuilder.block(operations: [op], outputs: ["y"])
    let function = MILBuilder.function(inputs: [inputNamed, roiNamed], opset: "CoreML8", block: block)
    let program = MILBuilder.program(functions: ["main": function])

    let model = MLProgramBuilder.makeModel(
        program: program,
        inputs: [("x", inputShape, .float32), ("boxes", [1, 4], .float32)],
        outputs: [("y", outputShape, .float32)]
    )

    let inputValues: [Float] = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ]
    let outputs = try MLTestUtils.runFloatModelTwoInputs(
        model: model,
        inputNameX: "x",
        inputShapeX: inputShape,
        inputValuesX: inputValues,
        inputNameY: "boxes",
        inputShapeY: [1, 4],
        inputValuesY: [0, 0, 1, 1],
        outputName: "y"
    )

    let expected: [Float] = [1, 4, 13, 16]
    #expect(outputs.count == expected.count)
    for (out, exp) in zip(outputs, expected) {
        #expect(abs(out - exp) < 1e-4)
    }
}
