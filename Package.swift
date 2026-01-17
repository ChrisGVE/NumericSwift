// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

// Check for optional dependencies
let includeArraySwift = ProcessInfo.processInfo.environment["NUMERICSWIFT_INCLUDE_ARRAYSWIFT"] == "1"
let includePlotSwift = ProcessInfo.processInfo.environment["NUMERICSWIFT_INCLUDE_PLOTSWIFT"] == "1"

let package = Package(
    name: "NumericSwift",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .visionOS(.v1),
        .watchOS(.v8),
        .tvOS(.v15)
    ],
    products: [
        .library(
            name: "NumericSwift",
            targets: ["NumericSwift"]
        ),
    ],
    dependencies: {
        var deps: [Package.Dependency] = [
            .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0"),
        ]
        if includeArraySwift {
            deps.append(.package(path: "../ArraySwift"))
        }
        if includePlotSwift {
            deps.append(.package(path: "../PlotSwift"))
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
                if includePlotSwift {
                    deps.append(.product(name: "PlotSwift", package: "PlotSwift"))
                }
                return deps
            }(),
            path: "Sources/NumericSwift",
            swiftSettings: {
                var settings: [SwiftSetting] = []
                if includeArraySwift {
                    settings.append(.define("NUMERICSWIFT_ARRAYSWIFT"))
                }
                if includePlotSwift {
                    settings.append(.define("NUMERICSWIFT_PLOTSWIFT"))
                }
                return settings
            }()
        ),
        .testTarget(
            name: "NumericSwiftTests",
            dependencies: {
                var deps: [Target.Dependency] = ["NumericSwift"]
                if includeArraySwift {
                    deps.append(.product(name: "ArraySwift", package: "ArraySwift"))
                }
                if includePlotSwift {
                    deps.append(.product(name: "PlotSwift", package: "PlotSwift"))
                }
                return deps
            }(),
            swiftSettings: {
                var settings: [SwiftSetting] = []
                if includeArraySwift {
                    settings.append(.define("NUMERICSWIFT_ARRAYSWIFT"))
                }
                if includePlotSwift {
                    settings.append(.define("NUMERICSWIFT_PLOTSWIFT"))
                }
                return settings
            }()
        ),
    ]
)
