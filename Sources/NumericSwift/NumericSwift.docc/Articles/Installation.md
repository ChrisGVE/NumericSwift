# Installation

Add NumericSwift to your Swift project.

## Overview

NumericSwift is distributed as a Swift Package and can be added to your project using Swift Package Manager.

## Swift Package Manager

Add NumericSwift to your `Package.swift` dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/ChrisGVE/NumericSwift.git", from: "0.1.0")
]
```

Then add it to your target dependencies:

```swift
.target(
    name: "YourTarget",
    dependencies: ["NumericSwift"]
)
```

## Xcode Project

1. In Xcode, select **File > Add Package Dependencies...**
2. Enter the repository URL: `https://github.com/ChrisGVE/NumericSwift.git`
3. Select the version rule (e.g., "Up to Next Major Version" from 0.1.0)
4. Click **Add Package**

## Requirements

- Swift 5.9 or later
- iOS 15+ or macOS 12+
- Accelerate framework (included with Apple platforms)

## Optional Dependencies

NumericSwift can optionally integrate with companion libraries:

### ArraySwift

To enable ArraySwift integration for enhanced array operations:

```bash
NUMERICSWIFT_INCLUDE_ARRAYSWIFT=1 swift build
```

This enables additional vDSP-optimized array utilities and serialization support.
