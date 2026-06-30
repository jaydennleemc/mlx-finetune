// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "SwiftFinetuneGemma2b",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", from: "2.29.1")
    ],
    targets: [
        .executableTarget(
            name: "SwiftFinetuneGemma2b",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples")
            ],
            path: ".",
            sources: ["main.swift"]
        )
    ]
)
