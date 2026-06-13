//
//  LinAlgErrorTests.swift
//  NumericSwift
//
//  Verifies the M14 recoverable-error API: fallible named operations throw
//  `LinAlg.LinAlgError` on bad runtime shapes/parameters, while singular but
//  well-shaped problems still return `nil`, and valid inputs never throw.
//  Constructors and operators are deliberately NOT covered here — they remain
//  `precondition` (programmer-error) sites by design (policy A), matching the
//  Swift standard library's treatment of `Array`/`SIMD`.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class LinAlgErrorTests: XCTestCase {

    typealias E = LinAlg.LinAlgError

    // MARK: - notSquare on named operations

    func testTraceRectangularThrowsNotSquare() {
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try LinAlg.trace(m)) { error in
            XCTAssertEqual(error as? E, E.notSquare(rows: 2, cols: 3))
        }
    }

    func testDetRectangularThrowsNotSquare() {
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try LinAlg.det(m)) { error in
            XCTAssertEqual(error as? E, E.notSquare(rows: 2, cols: 3))
        }
    }

    func testInvRectangularThrowsNotSquare() {
        let m = LinAlg.Matrix(rows: 3, cols: 2, data: [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try LinAlg.inv(m)) { error in
            XCTAssertEqual(error as? E, E.notSquare(rows: 3, cols: 2))
        }
    }

    func testEigRectangularThrowsNotSquare() {
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try LinAlg.eig(m)) { error in
            XCTAssertEqual(error as? E, E.notSquare(rows: 2, cols: 3))
        }
    }

    func testExpmRectangularThrowsNotSquare() {
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try LinAlg.expm(m)) { error in
            XCTAssertEqual(error as? E, E.notSquare(rows: 2, cols: 3))
        }
    }

    func testInstanceInverseRectangularThrowsNotSquare() {
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try m.inverse()) { error in
            XCTAssertEqual(error as? E, E.notSquare(rows: 2, cols: 3))
        }
    }

    // MARK: - dimensionMismatch / notSquare on solvers

    func testSolveDimensionMismatchThrows() {
        let A = LinAlg.Matrix([[1, 2], [3, 4]])     // 2x2
        let b = LinAlg.Matrix([1, 2, 3])            // 3x1
        XCTAssertThrowsError(try LinAlg.solve(A, b)) { error in
            guard case .dimensionMismatch? = error as? E else {
                return XCTFail("expected dimensionMismatch, got \(error)")
            }
        }
    }

    func testSolveNonSquareThrowsNotSquare() {
        let A = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let b = LinAlg.Matrix([1, 2])
        XCTAssertThrowsError(try LinAlg.solve(A, b)) { error in
            XCTAssertEqual(error as? E, E.notSquare(rows: 2, cols: 3))
        }
    }

    // MARK: - invalidParameter on factories

    func testArangeZeroStepThrows() {
        XCTAssertThrowsError(try LinAlg.arange(0, 5, 0)) { error in
            guard case .invalidParameter? = error as? E else {
                return XCTFail("expected invalidParameter, got \(error)")
            }
        }
    }

    func testLinspaceCountBelowTwoThrows() {
        XCTAssertThrowsError(try LinAlg.linspace(0, 1, 1)) { error in
            guard case .invalidParameter? = error as? E else {
                return XCTFail("expected invalidParameter, got \(error)")
            }
        }
    }

    // MARK: - Singular but well-shaped stays nil (NOT a throw)

    func testSingularInvReturnsNilNotThrow() throws {
        let singular = LinAlg.Matrix([[1, 2], [2, 4]])
        XCTAssertNil(try LinAlg.inv(singular),
                     "singular but square inv must return nil, not throw")
    }

    func testSingularSolveReturnsNilNotThrow() throws {
        let A = LinAlg.Matrix([[1, 2], [2, 4]])     // singular
        let b = LinAlg.Matrix([1, 2])
        XCTAssertNil(try LinAlg.solve(A, b),
                     "singular but well-shaped solve must return nil, not throw")
    }

    // MARK: - Valid inputs do not throw

    func testValidSquareOpsDoNotThrow() throws {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        XCTAssertEqual(try LinAlg.trace(a), 5, accuracy: 1e-12)
        XCTAssertEqual(try LinAlg.det(a), -2, accuracy: 1e-10)
        XCTAssertNotNil(try LinAlg.inv(a))
    }

    func testValidFactoriesDoNotThrow() throws {
        XCTAssertEqual(try LinAlg.arange(0, 3, 1).data, [0, 1, 2])
        XCTAssertEqual(try LinAlg.linspace(0, 1, 3).data, [0, 0.5, 1])
    }
}
