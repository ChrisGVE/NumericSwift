// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

// Check for optional ArraySwift dependency
let includeArraySwift = ProcessInfo.processInfo.environment["NUMERICSWIFT_INCLUDE_ARRAYSWIFT"] == "1"

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
    dependencies: {
        var deps: [Package.Dependency] = []
        if includeArraySwift {
            deps.append(.package(path: "../ArraySwift"))
        }
        return deps
    }(),
    targets: [
        .target(
            name: "NumericSwift",
            dependencies: {
                var deps: [Target.Dependency] = []
                if includeArraySwift {
                    deps.append(.product(name: "ArraySwift", package: "ArraySwift"))
                }
                return deps
            }(),
            path: "Sources/NumericSwift",
            swiftSettings: {
                var settings: [SwiftSetting] = []
                if includeArraySwift {
                    settings.append(.define("NUMERICSWIFT_ARRAYSWIFT"))
                }
                return settings
            }()
        ),
        .testTarget(
            name: "NumericSwiftTests",
            dependencies: ["NumericSwift"]
        ),
    ]
)
