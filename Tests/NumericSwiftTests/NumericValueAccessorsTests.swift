//
//  NumericValueAccessorsTests.swift
//  NumericSwiftTests
//
//  Tests for NumericValue+Accessors.swift.
//
//  Coverage:
//    • NumericValue.Kind — all four cases, Equatable, Hashable, CaseIterable,
//      CustomStringConvertible, Sendable
//    • kind accessor — correct Kind for each NumericValue case
//    • isScalar / isComplex / isMatrix / isComplexMatrix — mutually exclusive
//      and exhaustive across all four cases
//    • isMatrixLike / isComplexLike — correct grouping
//    • asScalar / asComplex / asMatrix / asComplexMatrix — value returned for
//      matching case, nil for every non-matching case
//    • asScalarThrowing / asComplexThrowing / asMatrixThrowing /
//      asComplexMatrixThrowing — value returned on match; kindMismatch thrown
//      with correct expected/actual on mismatch
//    • rows / cols — correct Int? for all four cases
//    • shape — correct (rows, cols)? for all four cases
//    • elementCount — 1 for scalars/complex; rows*cols for matrices
//    • is1x1 — true only for 1×1 matrix/complexMatrix; false for all others
//    • typeAndShapeDescription — label matches kind; shape embedded for matrices
//    • §15 truth-table property: Kind.allCases covers all four; flags are
//      mutually exclusive and exhaustive
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - Fixtures

private let scalarValue   = NumericValue.scalar(3.14)
private let complexValue  = NumericValue.complex(Complex(re: 1.0, im: -2.0))
private let mat2x3        = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))
private let mat1x1        = NumericValue.matrix(LinAlg.Matrix([[9.0]]))
private let cmat3x2       = NumericValue.complexMatrix(
    LinAlg.ComplexMatrix(rows: 3, cols: 2,
                         real: [1, 2, 3, 4, 5, 6],
                         imag: [0, 0, 0, 0, 0, 0])
)
private let cmat1x1       = NumericValue.complexMatrix(
    LinAlg.ComplexMatrix(LinAlg.Matrix([[7.0]]))
)

// MARK: - NumericValueKindTests

final class NumericValueKindTests: XCTestCase {

    // MARK: Kind mapping (§15 truth-table rows)

    func testKindScalar() {
        XCTAssertEqual(scalarValue.kind, .scalar)
    }

    func testKindComplex() {
        XCTAssertEqual(complexValue.kind, .complex)
    }

    func testKindMatrix() {
        XCTAssertEqual(mat2x3.kind, .matrix)
    }

    func testKindComplexMatrix() {
        XCTAssertEqual(cmat3x2.kind, .complexMatrix)
    }

    // MARK: Kind type properties

    func testKindAllCasesExhaustive() {
        // CaseIterable must cover exactly the four documented kinds.
        XCTAssertEqual(NumericValue.Kind.allCases.count, 4)
        XCTAssertTrue(NumericValue.Kind.allCases.contains(.scalar))
        XCTAssertTrue(NumericValue.Kind.allCases.contains(.complex))
        XCTAssertTrue(NumericValue.Kind.allCases.contains(.matrix))
        XCTAssertTrue(NumericValue.Kind.allCases.contains(.complexMatrix))
    }

    func testKindDescriptions() {
        XCTAssertEqual(NumericValue.Kind.scalar.description,        "scalar")
        XCTAssertEqual(NumericValue.Kind.complex.description,       "complex")
        XCTAssertEqual(NumericValue.Kind.matrix.description,        "matrix")
        XCTAssertEqual(NumericValue.Kind.complexMatrix.description, "complexMatrix")
    }

    func testKindEquatable() {
        XCTAssertEqual(NumericValue.Kind.scalar, NumericValue.Kind.scalar)
        XCTAssertNotEqual(NumericValue.Kind.scalar, NumericValue.Kind.complex)
    }

    func testKindHashable() {
        var set = Set<NumericValue.Kind>()
        set.insert(.scalar)
        set.insert(.complex)
        set.insert(.matrix)
        set.insert(.complexMatrix)
        XCTAssertEqual(set.count, 4)
    }

    func testKindSendable() {
        // Compile-time check: Kind must satisfy Sendable.
        func requireSendable<T: Sendable>(_: T) {}
        requireSendable(NumericValue.Kind.scalar)
        requireSendable(NumericValue.Kind.complexMatrix)
    }
}

// MARK: - NumericValueFlagTests

final class NumericValueFlagTests: XCTestCase {

    // MARK: isScalar

    func testIsScalarTrueForScalar()          { XCTAssertTrue(scalarValue.isScalar) }
    func testIsScalarFalseForComplex()        { XCTAssertFalse(complexValue.isScalar) }
    func testIsScalarFalseForMatrix()         { XCTAssertFalse(mat2x3.isScalar) }
    func testIsScalarFalseForComplexMatrix()  { XCTAssertFalse(cmat3x2.isScalar) }

    // MARK: isComplex

    func testIsComplexFalseForScalar()        { XCTAssertFalse(scalarValue.isComplex) }
    func testIsComplexTrueForComplex()        { XCTAssertTrue(complexValue.isComplex) }
    func testIsComplexFalseForMatrix()        { XCTAssertFalse(mat2x3.isComplex) }
    func testIsComplexFalseForComplexMatrix() { XCTAssertFalse(cmat3x2.isComplex) }

    // MARK: isMatrix

    func testIsMatrixFalseForScalar()         { XCTAssertFalse(scalarValue.isMatrix) }
    func testIsMatrixFalseForComplex()        { XCTAssertFalse(complexValue.isMatrix) }
    func testIsMatrixTrueForMatrix()          { XCTAssertTrue(mat2x3.isMatrix) }
    func testIsMatrixFalseForComplexMatrix()  { XCTAssertFalse(cmat3x2.isMatrix) }

    // MARK: isComplexMatrix

    func testIsComplexMatrixFalseForScalar()         { XCTAssertFalse(scalarValue.isComplexMatrix) }
    func testIsComplexMatrixFalseForComplex()        { XCTAssertFalse(complexValue.isComplexMatrix) }
    func testIsComplexMatrixFalseForMatrix()         { XCTAssertFalse(mat2x3.isComplexMatrix) }
    func testIsComplexMatrixTrueForComplexMatrix()   { XCTAssertTrue(cmat3x2.isComplexMatrix) }

    // MARK: Mutual exclusion (§15 invariant)

    /// Exactly one `isX` flag must be true for every value.
    func testFlagsMutuallyExclusiveForAllFour() {
        let values: [NumericValue] = [scalarValue, complexValue, mat2x3, cmat3x2]
        for v in values {
            let trueCount = [v.isScalar, v.isComplex, v.isMatrix, v.isComplexMatrix]
                .filter { $0 }.count
            XCTAssertEqual(trueCount, 1,
                "Expected exactly 1 true flag for \(v.description), got \(trueCount)")
        }
    }

    // MARK: isMatrixLike

    func testIsMatrixLikeTrueForScalar()         { XCTAssertTrue(scalarValue.isMatrixLike) }
    func testIsMatrixLikeFalseForComplex()       { XCTAssertFalse(complexValue.isMatrixLike) }
    func testIsMatrixLikeTrueForMatrix()         { XCTAssertTrue(mat2x3.isMatrixLike) }
    func testIsMatrixLikeFalseForComplexMatrix() { XCTAssertFalse(cmat3x2.isMatrixLike) }

    // MARK: isComplexLike

    func testIsComplexLikeFalseForScalar()         { XCTAssertFalse(scalarValue.isComplexLike) }
    func testIsComplexLikeTrueForComplex()         { XCTAssertTrue(complexValue.isComplexLike) }
    func testIsComplexLikeFalseForMatrix()         { XCTAssertFalse(mat2x3.isComplexLike) }
    func testIsComplexLikeTrueForComplexMatrix()   { XCTAssertTrue(cmat3x2.isComplexLike) }
}

// MARK: - NumericValueOptionalAccessorTests

final class NumericValueOptionalAccessorTests: XCTestCase {

    // MARK: asScalar

    func testAsScalarReturnsValueForScalar() {
        XCTAssertEqual(scalarValue.asScalar, 3.14)
    }

    func testAsScalarNilForComplex()        { XCTAssertNil(complexValue.asScalar) }
    func testAsScalarNilForMatrix()         { XCTAssertNil(mat2x3.asScalar) }
    func testAsScalarNilForComplexMatrix()  { XCTAssertNil(cmat3x2.asScalar) }

    // MARK: asComplex

    func testAsComplexReturnsValueForComplex() {
        let z = complexValue.asComplex
        XCTAssertNotNil(z)
        XCTAssertEqual(z?.re, 1.0)
        XCTAssertEqual(z?.im, -2.0)
    }

    func testAsComplexNilForScalar()        { XCTAssertNil(scalarValue.asComplex) }
    func testAsComplexNilForMatrix()        { XCTAssertNil(mat2x3.asComplex) }
    func testAsComplexNilForComplexMatrix() { XCTAssertNil(cmat3x2.asComplex) }

    // MARK: asMatrix

    func testAsMatrixReturnsValueForMatrix() {
        let m = mat2x3.asMatrix
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.rows, 2)
        XCTAssertEqual(m?.cols, 3)
    }

    func testAsMatrixNilForScalar()         { XCTAssertNil(scalarValue.asMatrix) }
    func testAsMatrixNilForComplex()        { XCTAssertNil(complexValue.asMatrix) }
    func testAsMatrixNilForComplexMatrix()  { XCTAssertNil(cmat3x2.asMatrix) }

    // MARK: asComplexMatrix

    func testAsComplexMatrixReturnsValueForComplexMatrix() {
        let cm = cmat3x2.asComplexMatrix
        XCTAssertNotNil(cm)
        XCTAssertEqual(cm?.rows, 3)
        XCTAssertEqual(cm?.cols, 2)
    }

    func testAsComplexMatrixNilForScalar()  { XCTAssertNil(scalarValue.asComplexMatrix) }
    func testAsComplexMatrixNilForComplex() { XCTAssertNil(complexValue.asComplexMatrix) }
    func testAsComplexMatrixNilForMatrix()  { XCTAssertNil(mat2x3.asComplexMatrix) }
}

// MARK: - NumericValueThrowingAccessorTests

final class NumericValueThrowingAccessorTests: XCTestCase {

    // MARK: asScalarThrowing

    func testAsScalarThrowingSucceedsForScalar() throws {
        let x = try scalarValue.asScalarThrowing()
        XCTAssertEqual(x, 3.14)
    }

    func testAsScalarThrowingKindMismatchForComplex() {
        XCTAssertThrowsError(try complexValue.asScalarThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch, got \(error)")
                return
            }
            XCTAssertEqual(expected, .scalar)
            XCTAssertEqual(actual,   .complex)
        }
    }

    func testAsScalarThrowingKindMismatchForMatrix() {
        XCTAssertThrowsError(try mat2x3.asScalarThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch"); return
            }
            XCTAssertEqual(expected, .scalar)
            XCTAssertEqual(actual,   .matrix)
        }
    }

    func testAsScalarThrowingKindMismatchForComplexMatrix() {
        XCTAssertThrowsError(try cmat3x2.asScalarThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch"); return
            }
            XCTAssertEqual(expected, .scalar)
            XCTAssertEqual(actual,   .complexMatrix)
        }
    }

    // MARK: asComplexThrowing

    func testAsComplexThrowingSucceedsForComplex() throws {
        let z = try complexValue.asComplexThrowing()
        XCTAssertEqual(z.re, 1.0)
        XCTAssertEqual(z.im, -2.0)
    }

    func testAsComplexThrowingKindMismatchForScalar() {
        XCTAssertThrowsError(try scalarValue.asComplexThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch"); return
            }
            XCTAssertEqual(expected, .complex)
            XCTAssertEqual(actual,   .scalar)
        }
    }

    func testAsComplexThrowingKindMismatchForMatrix() {
        XCTAssertThrowsError(try mat2x3.asComplexThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch"); return
            }
            XCTAssertEqual(expected, .complex)
            XCTAssertEqual(actual,   .matrix)
        }
    }

    // MARK: asMatrixThrowing

    func testAsMatrixThrowingSucceedsForMatrix() throws {
        let m = try mat2x3.asMatrixThrowing()
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 3)
    }

    func testAsMatrixThrowingKindMismatchForScalar() {
        XCTAssertThrowsError(try scalarValue.asMatrixThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch"); return
            }
            XCTAssertEqual(expected, .matrix)
            XCTAssertEqual(actual,   .scalar)
        }
    }

    func testAsMatrixThrowingKindMismatchForComplexMatrix() {
        XCTAssertThrowsError(try cmat3x2.asMatrixThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch"); return
            }
            XCTAssertEqual(expected, .matrix)
            XCTAssertEqual(actual,   .complexMatrix)
        }
    }

    // MARK: asComplexMatrixThrowing

    func testAsComplexMatrixThrowingSucceedsForComplexMatrix() throws {
        let cm = try cmat3x2.asComplexMatrixThrowing()
        XCTAssertEqual(cm.rows, 3)
        XCTAssertEqual(cm.cols, 2)
    }

    func testAsComplexMatrixThrowingKindMismatchForScalar() {
        XCTAssertThrowsError(try scalarValue.asComplexMatrixThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch"); return
            }
            XCTAssertEqual(expected, .complexMatrix)
            XCTAssertEqual(actual,   .scalar)
        }
    }

    func testAsComplexMatrixThrowingKindMismatchForMatrix() {
        XCTAssertThrowsError(try mat2x3.asComplexMatrixThrowing()) { error in
            guard case NumericValue.AccessorError.kindMismatch(let expected, let actual) = error else {
                XCTFail("Expected kindMismatch"); return
            }
            XCTAssertEqual(expected, .complexMatrix)
            XCTAssertEqual(actual,   .matrix)
        }
    }

    // MARK: AccessorError description

    func testAccessorErrorDescription() {
        let err = NumericValue.AccessorError.kindMismatch(expected: .scalar, actual: .matrix)
        XCTAssertTrue(err.description.contains("scalar"), "Got: \(err.description)")
        XCTAssertTrue(err.description.contains("matrix"), "Got: \(err.description)")
    }
}

// MARK: - NumericValueShapeTests

final class NumericValueShapeTests: XCTestCase {

    // MARK: rows

    func testRowsNilForScalar()           { XCTAssertNil(scalarValue.rows) }
    func testRowsNilForComplex()          { XCTAssertNil(complexValue.rows) }
    func testRowsCorrectForMatrix()       { XCTAssertEqual(mat2x3.rows, 2) }
    func testRowsCorrectForComplexMatrix(){ XCTAssertEqual(cmat3x2.rows, 3) }

    // MARK: cols

    func testColsNilForScalar()           { XCTAssertNil(scalarValue.cols) }
    func testColsNilForComplex()          { XCTAssertNil(complexValue.cols) }
    func testColsCorrectForMatrix()       { XCTAssertEqual(mat2x3.cols, 3) }
    func testColsCorrectForComplexMatrix(){ XCTAssertEqual(cmat3x2.cols, 2) }

    // MARK: shape

    func testShapeNilForScalar()  { XCTAssertNil(scalarValue.shape) }
    func testShapeNilForComplex() { XCTAssertNil(complexValue.shape) }

    func testShapeCorrectForMatrix() {
        let s = mat2x3.shape
        XCTAssertNotNil(s)
        XCTAssertEqual(s?.rows, 2)
        XCTAssertEqual(s?.cols, 3)
    }

    func testShapeCorrectForComplexMatrix() {
        let s = cmat3x2.shape
        XCTAssertNotNil(s)
        XCTAssertEqual(s?.rows, 3)
        XCTAssertEqual(s?.cols, 2)
    }

    // MARK: elementCount

    func testElementCountOneForScalar()         { XCTAssertEqual(scalarValue.elementCount, 1) }
    func testElementCountOneForComplex()        { XCTAssertEqual(complexValue.elementCount, 1) }
    func testElementCountRowsTimesColsForMatrix() {
        XCTAssertEqual(mat2x3.elementCount, 6)  // 2 × 3
    }
    func testElementCountRowsTimesColsForComplexMatrix() {
        XCTAssertEqual(cmat3x2.elementCount, 6)  // 3 × 2
    }

    // MARK: is1x1 — §4.3a coercion gate

    func testIs1x1FalseForScalar()        { XCTAssertFalse(scalarValue.is1x1) }
    func testIs1x1FalseForComplex()       { XCTAssertFalse(complexValue.is1x1) }
    func testIs1x1FalseForLargerMatrix()  { XCTAssertFalse(mat2x3.is1x1) }
    func testIs1x1TrueFor1x1Matrix()      { XCTAssertTrue(mat1x1.is1x1) }
    func testIs1x1TrueFor1x1ComplexMatrix(){ XCTAssertTrue(cmat1x1.is1x1) }
    func testIs1x1FalseForLargerComplexMatrix() { XCTAssertFalse(cmat3x2.is1x1) }

    /// A 1×n (n>1) matrix is NOT 1×1.
    func testIs1x1FalseFor1xNMatrix() {
        let v = NumericValue.matrix(LinAlg.Matrix([[1.0, 2.0, 3.0]]))
        XCTAssertFalse(v.is1x1)
    }

    /// An m×1 (m>1) matrix is NOT 1×1.
    func testIs1x1FalseForMx1Matrix() {
        let v = NumericValue.matrix(LinAlg.Matrix([[1.0], [2.0]]))
        XCTAssertFalse(v.is1x1)
    }
}

// MARK: - NumericValueDiagnosticDescriptionTests

final class NumericValueDiagnosticDescriptionTests: XCTestCase {

    func testTypeAndShapeDescriptionScalar() {
        XCTAssertEqual(scalarValue.typeAndShapeDescription, "scalar")
    }

    func testTypeAndShapeDescriptionComplex() {
        XCTAssertEqual(complexValue.typeAndShapeDescription, "complex")
    }

    func testTypeAndShapeDescriptionMatrix() {
        XCTAssertEqual(mat2x3.typeAndShapeDescription, "matrix(2x3)")
    }

    func testTypeAndShapeDescriptionComplexMatrix() {
        XCTAssertEqual(cmat3x2.typeAndShapeDescription, "complexMatrix(3x2)")
    }

    func testTypeAndShapeDescriptionFor1x1Matrix() {
        XCTAssertEqual(mat1x1.typeAndShapeDescription, "matrix(1x1)")
    }

    func testTypeAndShapeDescriptionFor1x1ComplexMatrix() {
        XCTAssertEqual(cmat1x1.typeAndShapeDescription, "complexMatrix(1x1)")
    }
}
