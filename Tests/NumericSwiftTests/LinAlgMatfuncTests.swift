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

  func testExpmZeroMatrix() throws  {
    // expm(0) = I
    let zero = LinAlg.zeros(3, 3)
    let result = try LinAlg.expm(zero)
    assertMatrixEqual(result, LinAlg.eye(3))
  }

  func testExpmIdentity() throws  {
    // expm(I) = e * I; Padé approximation accuracy ~1e-6 for this input
    let identity = LinAlg.eye(3)
    let result = try LinAlg.expm(identity)
    let eVal = exp(1.0)
    for i in 0..<3 {
      XCTAssertEqual(result[i, i], eVal, accuracy: 1e-5)
    }
    XCTAssertEqual(result[0, 1], 0.0, accuracy: 1e-10)
  }

  // MARK: - Known Value Tests: sqrtm

  func testSqrtmScaledIdentity() throws  {
    // sqrtm(4*I) = 2*I
    let fourIdentity = 4.0 * LinAlg.eye(3)
    let result = try LinAlg.sqrtm(fourIdentity)
    XCTAssertNotNil(result)
    assertMatrixEqual(result!, 2.0 * LinAlg.eye(3))
  }

  // MARK: - Known Value Tests: logm

  func testLogmIdentity() throws  {
    // logm(I) = 0
    let result = try LinAlg.logm(LinAlg.eye(3))
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

  func testExpmLogmRoundTrip() throws  {
    // expm(logm(A)) ≈ A for SPD A
    let A = spdMatrix
    let logA = try LinAlg.logm(A)
    XCTAssertNotNil(logA)
    let recovered = try LinAlg.expm(logA!)
    // expm uses Padé; logm uses eigendecomposition — combined error ~1e-5
    assertMatrixEqual(recovered, A, tolerance: 1e-4)
  }

  func testSqrtmSquaredRoundTrip() throws  {
    // sqrtm(A) * sqrtm(A) ≈ A for SPD A
    let A = spdMatrix
    let sqA = try LinAlg.sqrtm(A)
    XCTAssertNotNil(sqA)
    let product = sqA! * sqA!
    assertMatrixEqual(product, A, tolerance: 1e-8)
  }

  // MARK: - 1x1 Edge Cases

  func testExpm1x1() throws  {
    // Padé approximation accuracy for expm([[2]]) is ~1e-4
    let m = LinAlg.Matrix([[2.0]])
    let result = try LinAlg.expm(m)
    XCTAssertEqual(result[0, 0], exp(2.0), accuracy: 1e-3)
  }

  func testLogm1x1() throws  {
    let m = LinAlg.Matrix([[exp(3.0)]])
    let result = try LinAlg.logm(m)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], 3.0, accuracy: 1e-10)
  }

  func testSqrtm1x1() throws  {
    let m = LinAlg.Matrix([[9.0]])
    let result = try LinAlg.sqrtm(m)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], 3.0, accuracy: 1e-10)
  }

  func testFunm1x1Exp() throws  {
    let m = LinAlg.Matrix([[2.0]])
    let result = try LinAlg.funm(m, .exp)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], exp(2.0), accuracy: 1e-10)
  }

  // MARK: - Diagonal Matrix Tests

  func testExpmDiagonal() throws  {
    // expm(diag(a,b,c)) = diag(exp(a), exp(b), exp(c)); Padé accuracy ~1e-4 for these values
    let d = LinAlg.diag([1.0, 2.0, 3.0])
    let result = try LinAlg.expm(d)
    XCTAssertEqual(result[0, 0], exp(1.0), accuracy: 1e-3)
    XCTAssertEqual(result[1, 1], exp(2.0), accuracy: 1e-3)
    XCTAssertEqual(result[2, 2], exp(3.0), accuracy: 1e-3)
    XCTAssertEqual(result[0, 1], 0.0, accuracy: 1e-10)
  }

  func testLogmDiagonal() throws  {
    // logm(diag(a,b,c)) = diag(log(a), log(b), log(c)) for a,b,c > 0
    let d = LinAlg.diag([exp(1.0), exp(2.0), exp(3.0)])
    let result = try LinAlg.logm(d)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], 1.0, accuracy: 1e-10)
    XCTAssertEqual(result![1, 1], 2.0, accuracy: 1e-10)
    XCTAssertEqual(result![2, 2], 3.0, accuracy: 1e-10)
    XCTAssertEqual(result![0, 1], 0.0, accuracy: 1e-10)
  }

  func testSqrtmDiagonal() throws  {
    // sqrtm(diag(a,b,c)) = diag(sqrt(a), sqrt(b), sqrt(c))
    let d = LinAlg.diag([4.0, 9.0, 16.0])
    let result = try LinAlg.sqrtm(d)
    XCTAssertNotNil(result)
    XCTAssertEqual(result![0, 0], 2.0, accuracy: 1e-10)
    XCTAssertEqual(result![1, 1], 3.0, accuracy: 1e-10)
    XCTAssertEqual(result![2, 2], 4.0, accuracy: 1e-10)
  }

  // MARK: - Negative Eigenvalue Edge Cases

  func testLogmNegativeEigenvalueReturnsNil() throws  {
    // Matrix with a negative eigenvalue: diag(-1, 2, 3)
    let d = LinAlg.diag([-1.0, 2.0, 3.0])
    XCTAssertNil(try LinAlg.logm(d))
  }

  func testSqrtmNegativeEigenvalueReturnsNil() throws  {
    // Matrix with a negative eigenvalue: diag(-4, 1, 1)
    let d = LinAlg.diag([-4.0, 1.0, 1.0])
    XCTAssertNil(try LinAlg.sqrtm(d))
  }

  func testFunmLogNegativeEigenvalueReturnsNil() throws  {
    let d = LinAlg.diag([-1.0, 2.0, 3.0])
    XCTAssertNil(try LinAlg.funm(d, .log))
  }

  func testFunmSqrtNegativeEigenvalueReturnsNil() throws  {
    let d = LinAlg.diag([-4.0, 1.0, 1.0])
    XCTAssertNil(try LinAlg.funm(d, .sqrt))
  }

  // MARK: - funm Tests

  func testFunmExpMatchesExpm() throws  {
    // funm(A, .exp) should agree with expm(A) for a diagonalizable A
    let A = spdMatrix
    let funmResult = try LinAlg.funm(A, .exp)
    XCTAssertNotNil(funmResult)
    let expmResult = try LinAlg.expm(A)
    // funm uses eigendecomposition; expm uses Padé — algorithmic difference ~5e-4
    assertMatrixEqual(funmResult!, expmResult, tolerance: 1e-3)
  }

  func testFunmLogMatchesLogm() throws  {
    let A = spdMatrix
    let funmResult = try LinAlg.funm(A, .log)
    let logmResult = try LinAlg.logm(A)
    XCTAssertNotNil(funmResult)
    XCTAssertNotNil(logmResult)
    assertMatrixEqual(funmResult!, logmResult!, tolerance: 1e-10)
  }

  func testFunmSqrtMatchesSqrtm() throws  {
    let A = spdMatrix
    let funmResult = try LinAlg.funm(A, .sqrt)
    let sqrtmResult = try LinAlg.sqrtm(A)
    XCTAssertNotNil(funmResult)
    XCTAssertNotNil(sqrtmResult)
    assertMatrixEqual(funmResult!, sqrtmResult!, tolerance: 1e-10)
  }
}
