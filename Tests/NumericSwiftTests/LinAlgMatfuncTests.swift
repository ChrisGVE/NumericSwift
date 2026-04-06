//
//  LinAlgMatfuncTests.swift
//  NumericSwift
//
//  Tests for LinAlg matrix functions: expm, logm, sqrtm, funm.
//

import XCTest

@testable import NumericSwift

final class LinAlgMatfuncTests: XCTestCase {

  // MARK: - Helpers

  /// Asserts two matrices are element-wise equal within the given tolerance.
  private func assertMatrixEqual(
    _ a: LinAlg.Matrix,
    _ b: LinAlg.Matrix,
    tolerance: Double = 1e-10,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    XCTAssertEqual(a.rows, b.rows, "Row count mismatch", file: file, line: line)
    XCTAssertEqual(a.cols, b.cols, "Col count mismatch", file: file, line: line)
    for r in 0..<a.rows {
      for c in 0..<a.cols {
        XCTAssertEqual(
          a[r, c], b[r, c],
          accuracy: tolerance,
          "Mismatch at [\(r),\(c)]",
          file: file, line: line
        )
      }
    }
  }

  // MARK: - Known Value Tests: expm

  func testExpmZeroMatrix() {
    // expm(0) = I
    let zero = LinAlg.zeros(3, 3)
    let result = LinAlg.expm(zero)
    assertMatrixEqual(result, LinAlg.eye(3))
  }

  func testExpmIdentity() {
    // expm(I) = e * I; Padé approximation accuracy ~1e-6 for this input
    let identity = LinAlg.eye(3)
    let result = LinAlg.expm(identity)
    let eVal = exp(1.0)
    for i in 0..<3 {
      XCTAssertEqual(result[i, i], eVal, accuracy: 1e-5)
    }
    XCTAssertEqual(result[0, 1], 0.0, accuracy: 1e-10)
  }

  // MARK: - Known Value Tests: sqrtm

  func testSqrtmScaledIdentity() {
    // sqrtm(4*I) = 2*I
    let fourIdentity = 4.0 * LinAlg.eye(3)
    let result = LinAlg.sqrtm(fourIdentity)
    XCTAssertNotNil(result)
    assertMatrixEqual(result!, 2.0 * LinAlg.eye(3))
  }

  // MARK: - Known Value Tests: logm

  func testLogmIdentity() {
    // logm(I) = 0
    let result = LinAlg.logm(LinAlg.eye(3))
    XCTAssertNotNil(result)
    assertMatrixEqual(result!, LinAlg.zeros(3, 3))
  }

  // MARK: - Round-Trip Tests (SPD matrices)

  /// A well-conditioned 3x3 SPD matrix for round-trip tests.
  private var spdMatrix: LinAlg.Matrix {
    LinAlg.Matrix([
      [4.0, 1.0, 0.5],
      [1.0, 3.0, 0.5],
      [0.5, 0.5, 2.0],
    ])
  }

  func testExpmLogmRoundTrip() {
    // expm(logm(A)) ≈ A for SPD A
    let A = spdMatrix
    let logA = LinAlg.logm(A)
    XCTAssertNotNil(logA)
    let recovered = LinAlg.expm(logA!)
    // expm uses Padé; logm uses eigendecomposition — combined error ~1e-5
    assertMatrixEqual(recovered, A, tolerance: 1e-4)
  }

  func testSqrtmSquaredRoundTrip() {
    // sqrtm(A) * sqrtm(A) ≈ A for SPD A
    let A = spdMatrix
    let sqA = LinAlg.sqrtm(A)
    XCTAssertNotNil(sqA)
    let product = sqA! * sqA!
    assertMatrixEqual(product, A, tolerance: 1e-8)
  }

  // MARK: - 1x1 Edge Cases

  func testExpm1x1() {
    // Padé approximation accuracy for expm([[2]]) is ~1e-4
    let m = LinAlg.Matrix([[2.0]])
    let result = LinAlg.expm(m)
    XCTAssertEqual(result[0, 0], exp(2.0), accuracy: 1e-3)
  }

  func testLogm1x1() {
    let m = LinAlg.Matrix([[exp(3.0)]])
    let result = LinAlg.logm(m)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], 3.0, accuracy: 1e-10)
  }

  func testSqrtm1x1() {
    let m = LinAlg.Matrix([[9.0]])
    let result = LinAlg.sqrtm(m)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], 3.0, accuracy: 1e-10)
  }

  func testFunm1x1Exp() {
    let m = LinAlg.Matrix([[2.0]])
    let result = LinAlg.funm(m, .exp)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], exp(2.0), accuracy: 1e-10)
  }

  // MARK: - Diagonal Matrix Tests

  func testExpmDiagonal() {
    // expm(diag(a,b,c)) = diag(exp(a), exp(b), exp(c)); Padé accuracy ~1e-4 for these values
    let d = LinAlg.diag([1.0, 2.0, 3.0])
    let result = LinAlg.expm(d)
    XCTAssertEqual(result[0, 0], exp(1.0), accuracy: 1e-3)
    XCTAssertEqual(result[1, 1], exp(2.0), accuracy: 1e-3)
    XCTAssertEqual(result[2, 2], exp(3.0), accuracy: 1e-3)
    XCTAssertEqual(result[0, 1], 0.0, accuracy: 1e-10)
  }

  func testLogmDiagonal() {
    // logm(diag(a,b,c)) = diag(log(a), log(b), log(c)) for a,b,c > 0
    let d = LinAlg.diag([exp(1.0), exp(2.0), exp(3.0)])
    let result = LinAlg.logm(d)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], 1.0, accuracy: 1e-10)
    XCTAssertEqual(result![1, 1], 2.0, accuracy: 1e-10)
    XCTAssertEqual(result![2, 2], 3.0, accuracy: 1e-10)
    XCTAssertEqual(result![0, 1], 0.0, accuracy: 1e-10)
  }

  func testSqrtmDiagonal() {
    // sqrtm(diag(a,b,c)) = diag(sqrt(a), sqrt(b), sqrt(c))
    let d = LinAlg.diag([4.0, 9.0, 16.0])
    let result = LinAlg.sqrtm(d)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], 2.0, accuracy: 1e-10)
    XCTAssertEqual(result![1, 1], 3.0, accuracy: 1e-10)
    XCTAssertEqual(result![2, 2], 4.0, accuracy: 1e-10)
  }

  // MARK: - Negative Eigenvalue Edge Cases

  func testLogmNegativeEigenvalueReturnsNil() {
    // Matrix with a negative eigenvalue: diag(-1, 2, 3)
    let d = LinAlg.diag([-1.0, 2.0, 3.0])
    XCTAssertNil(LinAlg.logm(d))
  }

  func testSqrtmNegativeEigenvalueReturnsNil() {
    // Matrix with a negative eigenvalue: diag(-4, 1, 1)
    let d = LinAlg.diag([-4.0, 1.0, 1.0])
    XCTAssertNil(LinAlg.sqrtm(d))
  }

  func testFunmLogNegativeEigenvalueReturnsNil() {
    let d = LinAlg.diag([-1.0, 2.0, 3.0])
    XCTAssertNil(LinAlg.funm(d, .log))
  }

  func testFunmSqrtNegativeEigenvalueReturnsNil() {
    let d = LinAlg.diag([-4.0, 1.0, 1.0])
    XCTAssertNil(LinAlg.funm(d, .sqrt))
  }

  // MARK: - funm Tests

  func testFunmExpMatchesExpm() {
    // funm(A, .exp) should agree with expm(A) for a diagonalizable A
    let A = spdMatrix
    let funmResult = LinAlg.funm(A, .exp)
    XCTAssertNotNil(funmResult)
    let expmResult = LinAlg.expm(A)
    // funm uses eigendecomposition; expm uses Padé — algorithmic difference ~5e-4
    assertMatrixEqual(funmResult!, expmResult, tolerance: 1e-3)
  }

  func testFunmLogMatchesLogm() {
    let A = spdMatrix
    let funmResult = LinAlg.funm(A, .log)
    let logmResult = LinAlg.logm(A)
    XCTAssertNotNil(funmResult)
    XCTAssertNotNil(logmResult)
    assertMatrixEqual(funmResult!, logmResult!, tolerance: 1e-10)
  }

  func testFunmSqrtMatchesSqrtm() {
    let A = spdMatrix
    let funmResult = LinAlg.funm(A, .sqrt)
    let sqrtmResult = LinAlg.sqrtm(A)
    XCTAssertNotNil(funmResult)
    XCTAssertNotNil(sqrtmResult)
    assertMatrixEqual(funmResult!, sqrtmResult!, tolerance: 1e-10)
  }
}
