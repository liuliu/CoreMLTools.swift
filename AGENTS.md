# AGENTS.md

## Canonical Workflow

1. Build first: `swift build`
2. Run example binary directly: `.build/arm64-apple-macosx/debug/CoreMLToolsExample`
3. Run tests: `swift test`

Do not use `swift run` inside Codex unless explicitly required by the user. The direct binary path has been the stable path in this repo.

## Project Intent

- Build Core ML `mlprogram` packages (`.mlpackage`) from Swift protobuf/MIL code.
- Keep dependency surface minimal (library dependency is `SwiftProtobuf`).
- Use SwiftPM and Swift Testing.
- Target iOS 18+ and macOS 15+.

## Current State Snapshot

- End-to-end example is working (`y = 42`).
- `swift test` currently passes (93 tests).
- Tests cover all op names listed in `ops_iOS15_18.json` (with some guarded runtime-specific tests).
- Guarded test flags:
  - `COREMLTOOLS_ENABLE_CONV_QUANTIZED`
  - `COREMLTOOLS_ENABLE_CROP_RESIZE`
  - `COREMLTOOLS_ENABLE_CLASSIFY_TEST`
  - `COREMLTOOLS_ENABLE_CONST_SYMBOLIC`

## Editing Expectations

- Prefer Swift Testing over XCTest for new tests.
- If MIL builders/types are changed, add or update corresponding tests under `Tests/CoreMLToolsTests/`.
- Keep `README.md` and `AGENTS.md` aligned with actual verified commands and current status.
