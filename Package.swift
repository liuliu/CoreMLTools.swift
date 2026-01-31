// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CoreMLTools",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(name: "CoreMLTools", targets: ["CoreMLTools"]),
        .executable(name: "CoreMLToolsExample", targets: ["CoreMLToolsExample"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.28.0"),
        .package(url: "https://github.com/apple/swift-testing.git", from: "0.5.0")
    ],
    targets: [
        .target(
            name: "CoreMLTools",
            dependencies: [
                .product(name: "SwiftProtobuf", package: "swift-protobuf")
            ],
            path: "Sources/CoreMLTools"
        ),
        .executableTarget(
            name: "CoreMLToolsExample",
            dependencies: ["CoreMLTools"],
            path: "Sources/CoreMLToolsExample"
        ),
        .testTarget(
            name: "CoreMLToolsTests",
            dependencies: [
                "CoreMLTools",
                .product(name: "Testing", package: "swift-testing")
            ],
            path: "Tests/CoreMLToolsTests"
        )
    ]
)
