//
//  NumericValueEqualityTests.swift
//  NumericSwiftTests
//
//  Tests for NumericValue+Equality.swift:
//    • isExactlyEqual — NaN non-reflexivity, signed zero, LinAlg tolerant-==
//      bypass, cross-kind false, shape-mismatch false, complex real+imag
//    • isApproximatelyEqual — boundary (just inside / just outside tolerance),
//      cross-kind false, non-transitivity demonstration, NaN non-reflexivity
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - isExactlyEqual tests

final class NumericValueExactEqualityTests: XCTestCase {

    // MARK: Scalar — basic equality

    func testScalarExactEqual() {
        XCTAssertTrue(NumericValue.scalar(3.14).isExactlyEqual(to: .scalar(3.14)))
    }

    func testScalarExactNotEqual() {
        XCTAssertFalse(NumericValue.scalar(1.0).isExactlyEqual(to: .scalar(1.0 + 1e-12)))
    }

    // MARK: NaN non-reflexivity (scalars)

    func testScalarNaNNotEqualToItself() {
        let nan = NumericValue.scalar(Double.nan)
        XCTAssertFalse(nan.isExactlyEqual(to: nan),
                       "NaN must not equal itself (IEEE 754 non-reflexivity)")
    }

    func testScalarNaNNotEqualToNaN() {
        XCTAssertFalse(NumericValue.scalar(Double.nan)
                        .isExactlyEqual(to: .scalar(Double.nan)))
    }

    func testScalarNaNNotEqualToZero() {
        XCTAssertFalse(NumericValue.scalar(Double.nan).isExactlyEqual(to: .scalar(0.0)))
    }

    // MARK: Signed zero (scalars)

    func testSignedZeroTreatedAsEqual() {
        // IEEE 754 value equality: +0.0 == -0.0
        XCTAssertTrue(NumericValue.scalar(0.0).isExactlyEqual(to: .scalar(-0.0)),
                      "+0.0 and -0.0 must compare equal under IEEE 754 value equality")
    }

    // MARK: Complex exact equality

    func testComplexExactEqual() {
        let a = NumericValue.complex(Complex(re: 1.0, im: 2.0))
        let b = NumericValue.complex(Complex(re: 1.0, im: 2.0))
        XCTAssertTrue(a.isExactlyEqual(to: b))
    }

    func testComplexRealPartDiffers() {
        let a = NumericValue.complex(Complex(re: 1.0, im: 2.0))
        let b = NumericValue.complex(Complex(re: 1.0 + 1e-15, im: 2.0))
        XCTAssertFalse(a.isExactlyEqual(to: b))
    }

    func testComplexImagPartDiffers() {
        let a = NumericValue.complex(Complex(re: 1.0, im: 2.0))
        let b = NumericValue.complex(Complex(re: 1.0, im: 2.0 + 1e-15))
        XCTAssertFalse(a.isExactlyEqual(to: b))
    }

    func testComplexNaNRealPartNonReflexive() {
        let nan = NumericValue.complex(Complex(re: Double.nan, im: 0.0))
        XCTAssertFalse(nan.isExactlyEqual(to: nan))
    }

    func testComplexNaNImagPartNonReflexive() {
        let nan = NumericValue.complex(Complex(re: 0.0, im: Double.nan))
        XCTAssertFalse(nan.isExactlyEqual(to: nan))
    }

    // MARK: Matrix exact equality — LinAlg tolerant-== bypass

    /// Two matrices that differ by 1e-12 in one element are equal under
    /// LinAlg.Matrix's tolerant == (threshold 1e-10) but must be UNEQUAL
    /// under isExactlyEqual — this is the bypass guarantee.
    func testMatrixExactBypassesLinAlgTolerance() {
        let a = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        var bData = a.data
        bData[0] += 1e-12          // well within LinAlg's 1e-10 threshold
        let b = LinAlg.Matrix(rows: 2, cols: 2, data: bData)

        // Confirm LinAlg itself treats them as equal
        XCTAssertTrue(a == b, "LinAlg.Matrix.== should ignore a 1e-12 difference")

        // isExactlyEqual must return false
        let va = NumericValue.matrix(a)
        let vb = NumericValue.matrix(b)
        XCTAssertFalse(va.isExactlyEqual(to: vb),
                       "isExactlyEqual must bypass LinAlg tolerance and see 1e-12 difference")
    }

    func testMatrixExactEqual() {
        let m = LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])
        XCTAssertTrue(NumericValue.matrix(m).isExactlyEqual(to: .matrix(m)))
    }

    func testMatrixShapeMismatchFalse() {
        let a = NumericValue.matrix(LinAlg.Matrix([[1.0, 2.0]]))
        let b = NumericValue.matrix(LinAlg.Matrix([[1.0], [2.0]]))
        XCTAssertFalse(a.isExactlyEqual(to: b))
    }

    func testMatrixNaNNonReflexive() {
        let m = LinAlg.Matrix(rows: 1, cols: 2, data: [Double.nan, 1.0])
        let v = NumericValue.matrix(m)
        XCTAssertFalse(v.isExactlyEqual(to: v),
                       "Matrix with NaN element must not equal itself")
    }

    // MARK: ComplexMatrix exact equality — tolerant-== bypass

    func testComplexMatrixExactBypassesLinAlgTolerance() {
        let ca = LinAlg.ComplexMatrix(rows: 2, cols: 2,
                                      real: [1.0, 2.0, 3.0, 4.0],
                                      imag: [0.0, 0.0, 0.0, 0.0])
        var realB = ca.real
        realB[2] += 1e-12          // 1e-12 < 1e-10 threshold
        let cb = LinAlg.ComplexMatrix(rows: 2, cols: 2, real: realB, imag: ca.imag)

        // Confirm LinAlg itself treats them as equal
        XCTAssertTrue(ca == cb, "LinAlg.ComplexMatrix.== should ignore a 1e-12 difference")

        let va = NumericValue.complexMatrix(ca)
        let vb = NumericValue.complexMatrix(cb)
        XCTAssertFalse(va.isExactlyEqual(to: vb),
                       "isExactlyEqual must bypass LinAlg tolerance for ComplexMatrix")
    }

    func testComplexMatrixExactEqual() {
        let c = LinAlg.ComplexMatrix(rows: 1, cols: 2,
                                     real: [3.0, -1.0],
                                     imag: [0.5,  2.0])
        XCTAssertTrue(NumericValue.complexMatrix(c).isExactlyEqual(to: .complexMatrix(c)))
    }

    func testComplexMatrixShapeMismatchFalse() {
        let a = LinAlg.ComplexMatrix(rows: 1, cols: 2, real: [1.0, 2.0], imag: [0.0, 0.0])
        let b = LinAlg.ComplexMatrix(rows: 2, cols: 1, real: [1.0, 2.0], imag: [0.0, 0.0])
        XCTAssertFalse(NumericValue.complexMatrix(a).isExactlyEqual(to: .complexMatrix(b)))
    }

    func testComplexMatrixNaNNonReflexive() {
        let c = LinAlg.ComplexMatrix(rows: 1, cols: 1,
                                     real: [Double.nan], imag: [0.0])
        let v = NumericValue.complexMatrix(c)
        XCTAssertFalse(v.isExactlyEqual(to: v))
    }

    // MARK: Cross-kind comparisons always false

    func testScalarVsComplexFalse() {
        XCTAssertFalse(NumericValue.scalar(1.0)
                        .isExactlyEqual(to: .complex(Complex(re: 1.0, im: 0.0))))
    }

    func testScalarVsMatrixFalse() {
        let m = LinAlg.Matrix([[1.0]])
        XCTAssertFalse(NumericValue.scalar(1.0).isExactlyEqual(to: .matrix(m)))
    }

    func testMatrixVsComplexMatrixFalse() {
        let m  = LinAlg.Matrix([[1.0, 2.0]])
        let cm = LinAlg.ComplexMatrix(rows: 1, cols: 2, real: [1.0, 2.0], imag: [0.0, 0.0])
        XCTAssertFalse(NumericValue.matrix(m).isExactlyEqual(to: .complexMatrix(cm)))
    }

    func testComplexVsMatrixFalse() {
        let z = NumericValue.complex(Complex(re: 0.0, im: 0.0))
        let m = NumericValue.matrix(LinAlg.Matrix([[0.0]]))
        XCTAssertFalse(z.isExactlyEqual(to: m))
    }
}

// MARK: - isApproximatelyEqual tests

final class NumericValueApproximateEqualityTests: XCTestCase {

    // MARK: Scalar boundary tests

    func testScalarWithinTolerance() {
        let a = NumericValue.scalar(1.0)
        let b = NumericValue.scalar(1.0 + 9e-11)   // < 1e-10
        XCTAssertTrue(a.isApproximatelyEqual(to: b))
    }

    func testScalarExactlyAtTolerance() {
        let a = NumericValue.scalar(0.0)
        let b = NumericValue.scalar(1e-10)   // == tolerance, should be equal (<=)
        XCTAssertTrue(a.isApproximatelyEqual(to: b))
    }

    func testScalarJustOutsideTolerance() {
        let a = NumericValue.scalar(0.0)
        let b = NumericValue.scalar(1e-10 + 1e-20)   // > 1e-10
        XCTAssertFalse(a.isApproximatelyEqual(to: b))
    }

    func testScalarCustomTolerance() {
        let a = NumericValue.scalar(0.0)
        let b = NumericValue.scalar(0.05)
        XCTAssertTrue(a.isApproximatelyEqual(to: b, tolerance: 0.1))
        XCTAssertFalse(a.isApproximatelyEqual(to: b, tolerance: 0.01))
    }

    // MARK: NaN non-reflexivity (approximate)

    func testScalarNaNApproximateNonReflexive() {
        let nan = NumericValue.scalar(Double.nan)
        XCTAssertFalse(nan.isApproximatelyEqual(to: nan),
                       "NaN must not approximately equal itself")
    }

    func testScalarNaNNotApproximatelyEqualToZero() {
        XCTAssertFalse(NumericValue.scalar(Double.nan)
                        .isApproximatelyEqual(to: .scalar(0.0)))
    }

    // MARK: Signed zero (approximate)

    func testSignedZeroApproximatelyEqual() {
        XCTAssertTrue(NumericValue.scalar(0.0).isApproximatelyEqual(to: .scalar(-0.0)))
    }

    // MARK: Complex approximate equality

    func testComplexBothPartsWithinTolerance() {
        let a = NumericValue.complex(Complex(re: 1.0, im: 2.0))
        let b = NumericValue.complex(Complex(re: 1.0 + 5e-11, im: 2.0 - 5e-11))
        XCTAssertTrue(a.isApproximatelyEqual(to: b))
    }

    func testComplexRealPartOutsideTolerance() {
        let a = NumericValue.complex(Complex(re: 1.0, im: 2.0))
        let b = NumericValue.complex(Complex(re: 1.0 + 2e-10, im: 2.0))
        XCTAssertFalse(a.isApproximatelyEqual(to: b))
    }

    func testComplexImagPartOutsideTolerance() {
        let a = NumericValue.complex(Complex(re: 1.0, im: 2.0))
        let b = NumericValue.complex(Complex(re: 1.0, im: 2.0 + 2e-10))
        XCTAssertFalse(a.isApproximatelyEqual(to: b))
    }

    func testComplexNaNApproximateNonReflexive() {
        let nan = NumericValue.complex(Complex(re: Double.nan, im: 0.0))
        XCTAssertFalse(nan.isApproximatelyEqual(to: nan))
    }

    // MARK: Matrix approximate equality

    func testMatrixApproximateEqual() {
        let a = NumericValue.matrix(LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]]))
        var bData = [1.0, 2.0, 3.0, 4.0]
        bData[0] += 9e-11
        let b = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 2, data: bData))
        XCTAssertTrue(a.isApproximatelyEqual(to: b))
    }

    func testMatrixApproximateNotEqual() {
        let a = NumericValue.matrix(LinAlg.Matrix([[1.0, 2.0]]))
        let b = NumericValue.matrix(LinAlg.Matrix([[1.0, 2.0 + 2e-10]]))
        XCTAssertFalse(a.isApproximatelyEqual(to: b))
    }

    func testMatrixShapeMismatchFalse() {
        let a = NumericValue.matrix(LinAlg.Matrix([[1.0, 2.0]]))
        let b = NumericValue.matrix(LinAlg.Matrix([[1.0], [2.0]]))
        XCTAssertFalse(a.isApproximatelyEqual(to: b))
    }

    // MARK: ComplexMatrix approximate equality

    func testComplexMatrixApproximateEqual() {
        let ca = LinAlg.ComplexMatrix(rows: 1, cols: 2,
                                      real: [1.0, 2.0], imag: [0.0, 0.0])
        let cb = LinAlg.ComplexMatrix(rows: 1, cols: 2,
                                      real: [1.0 + 5e-11, 2.0], imag: [0.0, 5e-11])
        XCTAssertTrue(NumericValue.complexMatrix(ca).isApproximatelyEqual(to: .complexMatrix(cb)))
    }

    func testComplexMatrixApproximateNotEqual() {
        let ca = LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [0.0], imag: [0.0])
        let cb = LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [0.0], imag: [2e-10])
        XCTAssertFalse(NumericValue.complexMatrix(ca).isApproximatelyEqual(to: .complexMatrix(cb)))
    }

    // MARK: Cross-kind always false

    func testScalarVsComplexApproximateFalse() {
        XCTAssertFalse(NumericValue.scalar(0.0)
                        .isApproximatelyEqual(to: .complex(Complex(re: 0.0, im: 0.0))))
    }

    func testScalarVsMatrixApproximateFalse() {
        XCTAssertFalse(NumericValue.scalar(1.0)
                        .isApproximatelyEqual(to: .matrix(LinAlg.Matrix([[1.0]]))))
    }

    func testMatrixVsComplexMatrixApproximateFalse() {
        let m  = LinAlg.Matrix([[0.0]])
        let cm = LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [0.0], imag: [0.0])
        XCTAssertFalse(NumericValue.matrix(m).isApproximatelyEqual(to: .complexMatrix(cm)))
    }

    // MARK: Non-transitivity demonstration

    /// Demonstrates that isApproximatelyEqual is non-transitive.
    ///
    /// With tolerance t, we can have a ≈ b and b ≈ c yet a ≉ c.
    /// Here t = 1e-10, a = 0, b = 0.7·t, c = 1.4·t.
    /// |a−b| = 0.7t ≤ t  ✓
    /// |b−c| = 0.7t ≤ t  ✓
    /// |a−c| = 1.4t > t  ✗
    func testNonTransitivity() {
        let t = 1e-10
        let a = NumericValue.scalar(0.0)
        let b = NumericValue.scalar(0.7 * t)
        let c = NumericValue.scalar(1.4 * t)

        XCTAssertTrue(a.isApproximatelyEqual(to: b), "a ≈ b must hold")
        XCTAssertTrue(b.isApproximatelyEqual(to: c), "b ≈ c must hold")
        XCTAssertFalse(a.isApproximatelyEqual(to: c),
                       "a ≉ c demonstrates non-transitivity of approximate equality")
    }
}
