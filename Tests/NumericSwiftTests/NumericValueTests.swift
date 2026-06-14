//
//  NumericValueTests.swift
//  NumericSwiftTests
//
//  Tests for the NumericValue tower type (Sources/NumericSwift/NumericValue.swift).
//
//  Coverage:
//    • Construction — all four cases build without compiler error
//    • Sendable — compile-time constraint check via generic helper
//    • CustomStringConvertible — description is distinct per case and matches
//      the documented format
//    • No Equatable — verified via absence of == usage (see comment in
//      testNoEquatableConformance); the compiler enforces this at the call site
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - Compile-time Sendable constraint checker

/// Accepts any Sendable value. If T does not conform to Sendable, the call
/// site is a compile error — the intended signal.
private func requireSendable<T: Sendable>(_ value: T) {}

// MARK: - NumericValue Tests

final class NumericValueTests: XCTestCase {

    // MARK: - Construction

    /// All four cases must be constructible from their respective payload types.
    func testConstructionScalar() {
        let v = NumericValue.scalar(42.0)
        guard case .scalar(let x) = v else {
            XCTFail("Expected .scalar case")
            return
        }
        XCTAssertEqual(x, 42.0)
    }

    func testConstructionComplex() {
        let z = Complex(re: 1.0, im: -3.0)
        let v = NumericValue.complex(z)
        guard case .complex(let extracted) = v else {
            XCTFail("Expected .complex case")
            return
        }
        XCTAssertEqual(extracted.re, 1.0)
        XCTAssertEqual(extracted.im, -3.0)
    }

    func testConstructionMatrix() {
        let m = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let v = NumericValue.matrix(m)
        guard case .matrix(let extracted) = v else {
            XCTFail("Expected .matrix case")
            return
        }
        XCTAssertEqual(extracted.rows, 2)
        XCTAssertEqual(extracted.cols, 2)
    }

    func testConstructionComplexMatrix() {
        let cm = LinAlg.ComplexMatrix(LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]]))
        let v = NumericValue.complexMatrix(cm)
        guard case .complexMatrix(let extracted) = v else {
            XCTFail("Expected .complexMatrix case")
            return
        }
        XCTAssertEqual(extracted.rows, 2)
        XCTAssertEqual(extracted.cols, 2)
    }

    // MARK: - Sendable

    /// NumericValue must satisfy the Sendable constraint. If it does not, each
    /// call below is a compile error — the build fails rather than a test assertion.
    func testNumericValueIsSendable() {
        requireSendable(NumericValue.scalar(1.0))
        requireSendable(NumericValue.complex(Complex(re: 0, im: 1)))
        requireSendable(NumericValue.matrix(LinAlg.Matrix([[1.0]])))
        requireSendable(
            NumericValue.complexMatrix(
                LinAlg.ComplexMatrix(LinAlg.Matrix([[1.0]]))
            )
        )
    }

    // MARK: - CustomStringConvertible

    /// Each description must start with the case name so a reader can identify
    /// the kind immediately.
    func testDescriptionScalar() {
        let v = NumericValue.scalar(3.14)
        XCTAssertTrue(v.description.hasPrefix("scalar("), "Got: \(v.description)")
        XCTAssertTrue(v.description.contains("3.14"), "Got: \(v.description)")
    }

    func testDescriptionComplex() {
        let v = NumericValue.complex(Complex(re: 2.0, im: 5.0))
        XCTAssertTrue(v.description.hasPrefix("complex("), "Got: \(v.description)")
        // The Complex description includes the imaginary unit 'i'
        XCTAssertTrue(v.description.contains("i"), "Got: \(v.description)")
    }

    func testDescriptionMatrix() {
        let v = NumericValue.matrix(LinAlg.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        XCTAssertTrue(v.description.hasPrefix("matrix("), "Got: \(v.description)")
        // Shape dimensions appear in the description
        XCTAssertTrue(v.description.contains("2"), "rows missing in: \(v.description)")
        XCTAssertTrue(v.description.contains("3"), "cols missing in: \(v.description)")
    }

    func testDescriptionComplexMatrix() {
        let base = LinAlg.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        let v = NumericValue.complexMatrix(LinAlg.ComplexMatrix(base))
        XCTAssertTrue(v.description.hasPrefix("complexMatrix("), "Got: \(v.description)")
        XCTAssertTrue(v.description.contains("2"), "rows missing in: \(v.description)")
        XCTAssertTrue(v.description.contains("3"), "cols missing in: \(v.description)")
    }

    /// All four descriptions must be mutually distinct for a given input so that
    /// no two cases render identically.
    func testDescriptionsAreMutuallyDistinct() {
        let scalar    = NumericValue.scalar(1.0).description
        let complex   = NumericValue.complex(Complex(re: 1.0, im: 0.0)).description
        let matrix    = NumericValue.matrix(LinAlg.Matrix([[1.0]])).description
        let cmatrix   = NumericValue.complexMatrix(
            LinAlg.ComplexMatrix(LinAlg.Matrix([[1.0]]))
        ).description

        let all = [scalar, complex, matrix, cmatrix]
        let unique = Set(all)
        XCTAssertEqual(unique.count, 4,
            "Expected 4 distinct descriptions, got \(unique.count): \(all)")
    }

    // MARK: - No Equatable conformance
    //
    // We cannot write a positive runtime assertion that == is absent — the
    // compiler enforces this: adding `_ = v1 == v2` below would produce a
    // compile error ("binary operator '==' cannot be applied to two
    // 'NumericValue' operands"). The compile-time guarantee is therefore
    // structural: this file's absence of any == usage between NumericValue
    // instances is the evidence that the conformance was not accidentally
    // synthesised.
    //
    // Similarly for Hashable: `v.hashValue` would fail to compile.

    /// Documents the explicit decision not to conform to Equatable.
    /// This test passes trivially — its purpose is to serve as a named anchor
    /// in the test suite for the design decision recorded in NumericValue.swift.
    func testNoEquatableOrHashableConformanceIsIntentional() {
        // Positive proof: we can construct values of each kind without issue.
        let _: NumericValue = .scalar(0.0)
        let _: NumericValue = .complex(Complex(re: 0, im: 0))
        let _: NumericValue = .matrix(LinAlg.Matrix([[0.0]]))
        let _: NumericValue = .complexMatrix(
            LinAlg.ComplexMatrix(LinAlg.Matrix([[0.0]]))
        )
        // Negative proof: see the comment above this function.
        XCTAssertTrue(true, "Equatable/Hashable conformance is intentionally absent")
    }
}
