//
//  SendableConformanceTests.swift
//  NumericSwift
//
//  Compile-time verification that the types underpinning the NumericValue tower
//  conform to Sendable. Each call to requireSendable(_:) is a constraint-check:
//  if any conformance is missing the call fails to compile, surfacing the gap
//  before it can block NumericValue: Sendable.
//
//  Storage rationale (all structural Sendable — no locks or @unchecked needed):
//    Complex         — two Double fields                        → Sendable
//    LinAlg.Matrix   — Int × 2, [Double]                       → Sendable
//    LinAlg.ComplexMatrix — Int × 2, [Double] × 2              → Sendable
//    LinAlg.LinAlgError — enum; associated values are String/Int → Sendable
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - Compile-time Sendable constraint checker

/// Accepts any Sendable value. If T does not conform to Sendable, the call
/// site is a compile error — which is the intended signal.
private func requireSendable<T: Sendable>(_ value: T) {}

// MARK: - Sendable conformance tests

final class SendableConformanceTests: XCTestCase {

    /// Complex stores two Double fields — structurally Sendable.
    func testComplexIsSendable() {
        requireSendable(Complex(re: 1.0, im: 2.0))
    }

    /// LinAlg.Matrix stores Int rows/cols and a [Double] data array — structurally Sendable.
    func testMatrixIsSendable() {
        requireSendable(LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]]))
    }

    /// LinAlg.ComplexMatrix stores Int rows/cols and two [Double] arrays — structurally Sendable.
    func testComplexMatrixIsSendable() {
        requireSendable(LinAlg.ComplexMatrix(LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])))
    }

    /// LinAlg.LinAlgError carries only String and Int associated values — structurally Sendable.
    func testLinAlgErrorIsSendable() {
        requireSendable(LinAlg.LinAlgError.notSquare(rows: 2, cols: 3))
        requireSendable(LinAlg.LinAlgError.dimensionMismatch("test"))
        requireSendable(LinAlg.LinAlgError.invalidParameter("step"))
    }
}
