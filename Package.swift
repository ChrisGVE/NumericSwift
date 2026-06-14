// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

// Check for optional dependencies
let includeArraySwift =
  ProcessInfo.processInfo.environment["NUMERICSWIFT_INCLUDE_ARRAYSWIFT"] == "1"
let includePlotSwift = ProcessInfo.processInfo.environment["NUMERICSWIFT_INCLUDE_PLOTSWIFT"] == "1"
// MathLex Rust crate is opt-in (default OFF) so the package resolves without a local
// ../mathlex checkout and is consumable as a remote SPM dependency.
// Set NUMERICSWIFT_INCLUDE_MATHLEX=1 to activate the Rust-backed parser.
let includeMathLex = ProcessInfo.processInfo.environment["NUMERICSWIFT_INCLUDE_MATHLEX"] == "1"

let package = Package(
  name: "NumericSwift",
  platforms: [
    .iOS(.v15),
    .macOS(.v12),
    .visionOS(.v1),
    .watchOS(.v8),
    .tvOS(.v15),
  ],
  products: [
    .library(
      name: "NumericSwift",
      targets: ["NumericSwift"]
    )
  ],
  dependencies: {
    var deps: [Package.Dependency] = [
      .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0")
    ]
    if includeMathLex {
      deps.append(.package(path: "../mathlex"))
    }
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
        if includeMathLex {
          deps.append(.product(name: "MathLex", package: "mathlex"))
        }
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
        if includeMathLex {
          settings.append(.define("NUMERICSWIFT_MATHLEX"))
        }
        if includeArraySwift {
          settings.append(.define("NUMERICSWIFT_ARRAYSWIFT"))
        }
        if includePlotSwift {
          settings.append(.define("NUMERICSWIFT_PLOTSWIFT"))
        }
        return settings
      }(),
      linkerSettings: {
        var settings: [LinkerSetting] = []
        if includeMathLex {
          settings.append(.unsafeFlags(["-L../mathlex/target/release"]))
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
      // Frozen parity fixtures are read by #file-relative filesystem path
      // (see ParityCorpusTests), not as SPM bundle resources, so they are
      // excluded from the target to avoid the unhandled-files build warning.
      exclude: ["Fixtures"],
      swiftSettings: {
        var settings: [SwiftSetting] = []
        if includeMathLex {
          settings.append(.define("NUMERICSWIFT_MATHLEX"))
        }
        if includeArraySwift {
          settings.append(.define("NUMERICSWIFT_ARRAYSWIFT"))
        }
        if includePlotSwift {
          settings.append(.define("NUMERICSWIFT_PLOTSWIFT"))
        }
        return settings
      }()
    ),
    // Performance benchmark harness — NOT part of the NumericSwift library
    // product and therefore invisible to remote SPM consumers (e.g. LuaSwift).
    // Build explicitly: swift build --product NumericSwiftBench
    .executableTarget(
      name: "NumericSwiftBench",
      dependencies: ["NumericSwift"],
      path: "Sources/NumericSwiftBench"
    ),
  ]
)
