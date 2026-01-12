// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NumericSwift",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "NumericSwift",
            targets: ["NumericSwift"]
        ),
    ],
    targets: [
        .target(
            name: "NumericSwift",
            path: "Sources/NumericSwift"
        ),
        .testTarget(
            name: "NumericSwiftTests",
            dependencies: ["NumericSwift"]
        ),
    ]
)
