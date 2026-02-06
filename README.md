# CoreMLTools.swift

`CoreMLTools.swift` is a SwiftPM project for building Core ML `mlprogram` models (`.mlpackage`) directly from Swift code (protobuf-backed), then loading and running them with the Core ML framework.

## Scope

- Target platforms: iOS 18+ and macOS 15+
- Package type: SwiftPM library + executable example
- Runtime dependency goal: minimal (`SwiftProtobuf` only for protobuf model/spec handling)
- Reference project: `./coremltools` (Python)

## What Is Implemented So Far

- Core ML protobuf bindings are generated and checked in under `Sources/CoreMLTools/Generated/`.
- `MLProgramBuilder` can build a `CoreML_Specification_Model` from MIL programs, including typed input/output feature descriptions.
- `MLPackageBuilder` writes `.mlpackage` directories that Core ML can compile and load.
- MIL authoring helpers are available:
  - `MILBuilder` and generated op builders (`OpBuilders.generated.swift`)
  - value/type helpers in `MILValue`/`MILType` (tensor/list/dictionary support)
- Example executable (`CoreMLToolsExample`) does end-to-end:
  - build model in Swift
  - write `.mlpackage`
  - compile model with Core ML
  - run inference (`y = 42`)
- Test suite is broad and currently passing:
  - `swift test` passes locally
  - 93 Swift Testing tests
  - all op names in `ops_iOS15_18.json` are referenced by tests (including internal/guarded cases)

## Quick Start

```bash
swift build
.build/arm64-apple-macosx/debug/CoreMLToolsExample
swift test
```

Expected example output:

```text
y = 42
```

## Optional / Guarded Test Paths

Some tests are intentionally guarded behind environment variables because behavior can vary by Core ML runtime/compiler:

- `COREMLTOOLS_ENABLE_CONV_QUANTIZED=1`
- `COREMLTOOLS_ENABLE_CROP_RESIZE=1`
- `COREMLTOOLS_ENABLE_CLASSIFY_TEST=1`
- `COREMLTOOLS_ENABLE_CONST_SYMBOLIC=1`

By default these tests return early to keep normal local test runs stable.

## Repository Layout

- `Sources/CoreMLTools/` - library source
- `Sources/CoreMLToolsExample/` - executable end-to-end example
- `Tests/CoreMLToolsTests/` - Swift Testing suites
- `Protos/` - Core ML protobuf schema files
- `coremltools/` - Python reference implementation (not built by SwiftPM)

## Known Notes

- In this environment, running the built binary directly is the most reliable workflow (`swift build` then run `.build/.../CoreMLToolsExample`).
- Generated protobuf sources may produce deprecation warnings with newer SwiftProtobuf toolchains; warnings are currently non-blocking.
