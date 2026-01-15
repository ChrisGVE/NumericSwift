//
//  LinAlgTests.swift
//  NumericSwift
//
//  Tests for LinAlg namespace (NumericSwift.LinAlg)
//

import XCTest
@testable import NumericSwift

final class LinAlgTests: XCTestCase {

    // MARK: - Matrix Creation Tests

    func testMatrixCreation() {
        let m = LinAlg.Matrix([[1, 2], [3, 4]])
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
        XCTAssertEqual(m[0, 0], 1)
        XCTAssertEqual(m[1, 1], 4)
    }

    func testVectorCreation() {
        let v = LinAlg.Matrix([1, 2, 3])
        XCTAssertEqual(v.rows, 3)
        XCTAssertEqual(v.cols, 1)
        XCTAssertTrue(v.isVector)
    }

    func testZerosAndOnes() {
        let z = LinAlg.zeros(3, 2)
        XCTAssertEqual(z.rows, 3)
        XCTAssertEqual(z.cols, 2)
        XCTAssertEqual(z[1, 1], 0)

        let o = LinAlg.ones(2, 3)
        XCTAssertEqual(o.rows, 2)
        XCTAssertEqual(o.cols, 3)
        XCTAssertEqual(o[0, 2], 1)
    }

    func testEye() {
        let I = LinAlg.eye(3)
        XCTAssertEqual(I[0, 0], 1)
        XCTAssertEqual(I[1, 1], 1)
        XCTAssertEqual(I[2, 2], 1)
        XCTAssertEqual(I[0, 1], 0)
    }

    func testDiag() {
        let d = LinAlg.diag([1, 2, 3])
        XCTAssertEqual(d[0, 0], 1)
        XCTAssertEqual(d[1, 1], 2)
        XCTAssertEqual(d[2, 2], 3)
        XCTAssertEqual(d[0, 1], 0)
    }

    func testDiagExtract() {
        let m = LinAlg.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let d = LinAlg.diag(m)
        XCTAssertEqual(d, [1, 5, 9])
    }

    func testArange() {
        let v = LinAlg.arange(0, 5, 1)
        XCTAssertEqual(v.rows, 5)
        XCTAssertEqual(v[0], 0)
        XCTAssertEqual(v[4], 4)
    }

    func testLinspace() {
        let v = LinAlg.linspace(0, 1, 5)
        XCTAssertEqual(v.rows, 5)
        XCTAssertEqual(v[0], 0, accuracy: 1e-10)
        XCTAssertEqual(v[2], 0.5, accuracy: 1e-10)
        XCTAssertEqual(v[4], 1, accuracy: 1e-10)
    }

    // MARK: - Arithmetic Tests

    func testMatrixAddSub() {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let b = LinAlg.Matrix([[5, 6], [7, 8]])

        let sum = a + b
        XCTAssertEqual(sum[0, 0], 6)
        XCTAssertEqual(sum[1, 1], 12)

        let diff = b - a
        XCTAssertEqual(diff[0, 0], 4)
        XCTAssertEqual(diff[1, 1], 4)
    }

    func testScalarMultiply() {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let scaled = 2.0 * a
        XCTAssertEqual(scaled[0, 0], 2)
        XCTAssertEqual(scaled[1, 1], 8)
    }

    func testMatrixMultiply() {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let b = LinAlg.Matrix([[5, 6], [7, 8]])
        let c = a * b  // [[19, 22], [43, 50]]
        XCTAssertEqual(c[0, 0], 19)
        XCTAssertEqual(c[0, 1], 22)
        XCTAssertEqual(c[1, 0], 43)
        XCTAssertEqual(c[1, 1], 50)
    }

    func testDotProduct() {
        let u = LinAlg.Matrix([1, 2, 3])
        let v = LinAlg.Matrix([4, 5, 6])
        let d = LinAlg.dot(u, v)  // 1*4 + 2*5 + 3*6 = 32
        XCTAssertEqual(d[0], 32)
    }

    func testHadamard() {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let b = LinAlg.Matrix([[5, 6], [7, 8]])
        let h = LinAlg.hadamard(a, b)
        XCTAssertEqual(h[0, 0], 5)
        XCTAssertEqual(h[0, 1], 12)
        XCTAssertEqual(h[1, 0], 21)
        XCTAssertEqual(h[1, 1], 32)
    }

    func testNegation() {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let neg = -a
        XCTAssertEqual(neg[0, 0], -1)
        XCTAssertEqual(neg[1, 1], -4)
    }

    func testDivide() {
        let a = LinAlg.Matrix([[2, 4], [6, 8]])
        let div = a / 2.0
        XCTAssertEqual(div[0, 0], 1)
        XCTAssertEqual(div[1, 1], 4)
    }

    // MARK: - Matrix Properties

    func testTranspose() {
        let a = LinAlg.Matrix([[1, 2, 3], [4, 5, 6]])
        let t = a.T
        XCTAssertEqual(t.rows, 3)
        XCTAssertEqual(t.cols, 2)
        XCTAssertEqual(t[0, 1], 4)
        XCTAssertEqual(t[2, 0], 3)
    }

    func testTrace() {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        XCTAssertEqual(LinAlg.trace(a), 5)
    }

    func testDeterminant() {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        XCTAssertEqual(LinAlg.det(a), -2, accuracy: 1e-10)

        let b = LinAlg.Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        XCTAssertEqual(LinAlg.det(b), -306, accuracy: 1e-10)
    }

    func testInverse() {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        guard let aInv = LinAlg.inv(a) else {
            XCTFail("Inverse should exist")
            return
        }

        // A * A^(-1) should be identity
        let product = a * aInv
        XCTAssertEqual(product[0, 0], 1, accuracy: 1e-10)
        XCTAssertEqual(product[0, 1], 0, accuracy: 1e-10)
        XCTAssertEqual(product[1, 0], 0, accuracy: 1e-10)
        XCTAssertEqual(product[1, 1], 1, accuracy: 1e-10)
    }

    func testNorm() {
        let v = LinAlg.Matrix([3, 4])
        XCTAssertEqual(LinAlg.norm(v, 2), 5)  // sqrt(9 + 16) = 5
        XCTAssertEqual(LinAlg.norm(v, 1), 7)  // |3| + |4| = 7
    }

    func testRank() {
        let a = LinAlg.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // Rank 2
        XCTAssertEqual(LinAlg.rank(a), 2)

        let b = LinAlg.eye(3)  // Rank 3
        XCTAssertEqual(LinAlg.rank(b), 3)
    }

    func testCond() {
        let I = LinAlg.eye(3)
        XCTAssertEqual(LinAlg.cond(I), 1, accuracy: 1e-10)
    }

    // MARK: - Decompositions

    func testLU() {
        let a = LinAlg.Matrix([[2, 1], [1, 3]])
        let (L, U, P) = LinAlg.lu(a)

        // P @ L @ U should equal A
        let reconstructed = P * L * U
        XCTAssertEqual(reconstructed[0, 0], a[0, 0], accuracy: 1e-10)
        XCTAssertEqual(reconstructed[1, 1], a[1, 1], accuracy: 1e-10)
    }

    func testQR() {
        let a = LinAlg.Matrix([[1, 2], [3, 4], [5, 6]])
        let (Q, R) = LinAlg.qr(a)

        // Q @ R should equal A
        let reconstructed = Q * R
        for i in 0..<a.rows {
            for j in 0..<a.cols {
                XCTAssertEqual(reconstructed[i, j], a[i, j], accuracy: 1e-10)
            }
        }
    }

    func testSVD() {
        let a = LinAlg.Matrix([[1, 2], [3, 4], [5, 6]])
        let (s, U, Vt) = LinAlg.svd(a)

        XCTAssertEqual(s.count, 2)
        XCTAssertTrue(s[0] >= s[1])  // Singular values in descending order
    }

    func testEigenvalues() {
        // Symmetric matrix has real eigenvalues
        let a = LinAlg.Matrix([[2, 1], [1, 2]])
        let (real, imag) = LinAlg.eigvals(a)

        // Eigenvalues should be 3 and 1
        XCTAssertTrue(imag.allSatisfy { abs($0) < 1e-10 })  // All real
        let sorted = real.sorted()
        XCTAssertEqual(sorted[0], 1, accuracy: 1e-10)
        XCTAssertEqual(sorted[1], 3, accuracy: 1e-10)
    }

    func testEig() {
        let a = LinAlg.Matrix([[2, 1], [1, 2]])
        let (values, imag, vectors) = LinAlg.eig(a)

        XCTAssertEqual(values.count, 2)
        XCTAssertEqual(vectors.rows, 2)
        XCTAssertEqual(vectors.cols, 2)
    }

    func testCholesky() {
        // Symmetric positive definite matrix
        let a = LinAlg.Matrix([[4, 2], [2, 5]])
        guard let L = LinAlg.cholesky(a) else {
            XCTFail("Cholesky should succeed")
            return
        }

        // L @ L^T should equal A
        let reconstructed = L * L.T
        XCTAssertEqual(reconstructed[0, 0], a[0, 0], accuracy: 1e-10)
        XCTAssertEqual(reconstructed[1, 1], a[1, 1], accuracy: 1e-10)
    }

    // MARK: - Linear System Solvers

    func testSolve() {
        // Solve [[2, 1], [1, 3]] * x = [1, 2]
        let A = LinAlg.Matrix([[2, 1], [1, 3]])
        let b = LinAlg.Matrix([1, 2])

        guard let x = LinAlg.solve(A, b) else {
            XCTFail("Solve should succeed")
            return
        }

        // A @ x should equal b
        let result = A * x
        XCTAssertEqual(result[0], b[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], b[1], accuracy: 1e-10)
    }

    func testLstSq() {
        // Overdetermined system: fit y = ax + b to points
        let A = LinAlg.Matrix([[1, 1], [2, 1], [3, 1]])  // [x, 1]
        let b = LinAlg.Matrix([2, 3, 4])  // y values for y = x + 1

        guard let x = LinAlg.lstsq(A, b) else {
            XCTFail("Lstsq should succeed")
            return
        }

        XCTAssertEqual(x[0], 1, accuracy: 1e-10)  // slope = 1
        XCTAssertEqual(x[1], 1, accuracy: 1e-10)  // intercept = 1
    }

    func testSolveTriangular() {
        // Lower triangular system
        let L = LinAlg.Matrix([[2, 0], [1, 3]])
        let b = LinAlg.Matrix([4, 7])

        guard let x = LinAlg.solveTriangular(L, b, lower: true, trans: false) else {
            XCTFail("Solve triangular should succeed")
            return
        }

        // L @ x should equal b
        let result = L * x
        XCTAssertEqual(result[0], b[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], b[1], accuracy: 1e-10)
    }

    func testChoSolve() {
        // Solve using Cholesky factor
        let A = LinAlg.Matrix([[4, 2], [2, 5]])
        let b = LinAlg.Matrix([1, 2])

        guard let L = LinAlg.cholesky(A) else {
            XCTFail("Cholesky should succeed")
            return
        }

        guard let x = LinAlg.choSolve(L, b) else {
            XCTFail("choSolve should succeed")
            return
        }

        // A @ x should equal b
        let result = A * x
        XCTAssertEqual(result[0], b[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], b[1], accuracy: 1e-10)
    }

    func testLuSolve() {
        let A = LinAlg.Matrix([[2, 1], [1, 3]])
        let b = LinAlg.Matrix([1, 2])

        let (L, U, P) = LinAlg.lu(A)
        let x = LinAlg.luSolve(L, U, P, b)

        // A @ x should equal b
        let result = A * x
        XCTAssertEqual(result[0], b[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], b[1], accuracy: 1e-10)
    }

    // MARK: - Matrix Functions

    func testExpm() {
        // exp(0) = I
        let Z = LinAlg.zeros(2, 2)
        let expZ = LinAlg.expm(Z)
        XCTAssertEqual(expZ[0, 0], 1, accuracy: 1e-10)
        XCTAssertEqual(expZ[1, 1], 1, accuracy: 1e-10)
        XCTAssertEqual(expZ[0, 1], 0, accuracy: 1e-10)

        // exp(diag(a, b)) = diag(exp(a), exp(b))
        let D = LinAlg.Matrix([[1, 0], [0, 2]])
        let expD = LinAlg.expm(D)
        XCTAssertEqual(expD[0, 0], exp(1.0), accuracy: 1e-5)
        XCTAssertEqual(expD[1, 1], exp(2.0), accuracy: 1e-5)
    }

    func testLogm() {
        // log(I) = 0
        let I = LinAlg.eye(2)
        guard let logI = LinAlg.logm(I) else {
            XCTFail("logm should succeed for identity")
            return
        }
        XCTAssertEqual(logI[0, 0], 0, accuracy: 1e-10)
        XCTAssertEqual(logI[1, 1], 0, accuracy: 1e-10)

        // log(exp(A)) ≈ A for diagonal matrices
        let D = LinAlg.Matrix([[2, 0], [0, 3]])
        let expD = LinAlg.expm(D)
        guard let logExpD = LinAlg.logm(expD) else {
            XCTFail("logm should succeed")
            return
        }
        XCTAssertEqual(logExpD[0, 0], 2, accuracy: 1e-4)
        XCTAssertEqual(logExpD[1, 1], 3, accuracy: 1e-4)
    }

    func testSqrtm() {
        // sqrt(I) = I
        let I = LinAlg.eye(2)
        guard let sqrtI = LinAlg.sqrtm(I) else {
            XCTFail("sqrtm should succeed for identity")
            return
        }
        XCTAssertEqual(sqrtI[0, 0], 1, accuracy: 1e-10)
        XCTAssertEqual(sqrtI[1, 1], 1, accuracy: 1e-10)

        // sqrt(diag(4, 9)) = diag(2, 3)
        let D = LinAlg.Matrix([[4, 0], [0, 9]])
        guard let sqrtD = LinAlg.sqrtm(D) else {
            XCTFail("sqrtm should succeed")
            return
        }
        XCTAssertEqual(sqrtD[0, 0], 2, accuracy: 1e-10)
        XCTAssertEqual(sqrtD[1, 1], 3, accuracy: 1e-10)
    }

    func testFunm() {
        // sin(0) = 0
        let Z = LinAlg.zeros(2, 2)
        guard let sinZ = LinAlg.funm(Z, .sin) else {
            XCTFail("funm should succeed")
            return
        }
        XCTAssertEqual(sinZ[0, 0], 0, accuracy: 1e-10)

        // cos(0) = I
        guard let cosZ = LinAlg.funm(Z, .cos) else {
            XCTFail("funm should succeed")
            return
        }
        XCTAssertEqual(cosZ[0, 0], 1, accuracy: 1e-10)
        XCTAssertEqual(cosZ[1, 1], 1, accuracy: 1e-10)
    }

    // MARK: - Complex Matrix Tests

    func testComplexMatrixCreation() {
        let cm = LinAlg.ComplexMatrix(
            rows: 2, cols: 2,
            real: [1, 2, 3, 4],
            imag: [0.1, 0.2, 0.3, 0.4]
        )
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 2)
        XCTAssertEqual(cm[0, 0].re, 1)
        XCTAssertEqual(cm[0, 0].im, 0.1)
    }

    func testComplexFromReal() {
        let m = LinAlg.Matrix([[1, 2], [3, 4]])
        let cm = LinAlg.ComplexMatrix(m)
        XCTAssertEqual(cm[0, 0].re, 1)
        XCTAssertEqual(cm[0, 0].im, 0)
    }

    func testCSolve() {
        // Identity system: I * x = b => x = b
        let I = LinAlg.ComplexMatrix(LinAlg.eye(2))
        let b = LinAlg.ComplexMatrix(
            rows: 2, cols: 1,
            real: [1, 2],
            imag: [0.5, 1.5]
        )

        guard let x = LinAlg.csolve(I, b) else {
            XCTFail("csolve should succeed")
            return
        }

        XCTAssertEqual(x.rows, 2)
        XCTAssertEqual(x.cols, 1)
        // x should equal b
        XCTAssertEqual(x.real[0], 1, accuracy: 1e-10)
        XCTAssertEqual(x.real[1], 2, accuracy: 1e-10)
        XCTAssertEqual(x.imag[0], 0.5, accuracy: 1e-10)
        XCTAssertEqual(x.imag[1], 1.5, accuracy: 1e-10)
    }

    func testCDet() {
        // Determinant of identity should be 1
        let I = LinAlg.ComplexMatrix(LinAlg.eye(2))
        guard let det = LinAlg.cdet(I) else {
            XCTFail("cdet should succeed")
            return
        }
        XCTAssertEqual(det.re, 1, accuracy: 1e-10)
        XCTAssertEqual(det.im, 0, accuracy: 1e-10)
    }

    func testCInv() {
        // Inverse of identity should be identity
        let I = LinAlg.ComplexMatrix(LinAlg.eye(2))
        guard let invI = LinAlg.cinv(I) else {
            XCTFail("cinv should succeed")
            return
        }
        XCTAssertEqual(invI[0, 0].re, 1, accuracy: 1e-10)
        XCTAssertEqual(invI[0, 0].im, 0, accuracy: 1e-10)
    }

    func testCEigvals() {
        let m = LinAlg.ComplexMatrix(LinAlg.eye(2))
        guard let eigvals = LinAlg.ceigvals(m) else {
            XCTFail("ceigvals should succeed")
            return
        }
        XCTAssertEqual(eigvals.rows, 2)
        // Eigenvalues of identity are all 1
        XCTAssertEqual(eigvals.real[0], 1, accuracy: 1e-10)
        XCTAssertEqual(eigvals.real[1], 1, accuracy: 1e-10)
    }

    func testCSVD() {
        let m = LinAlg.ComplexMatrix(LinAlg.Matrix([[1, 0], [0, 2]]))
        guard let (s, U, Vt) = LinAlg.csvd(m) else {
            XCTFail("csvd should succeed")
            return
        }
        // Singular values of diagonal matrix
        XCTAssertEqual(s.count, 2)
        let sortedS = s.sorted(by: >)
        XCTAssertEqual(sortedS[0], 2, accuracy: 1e-10)
        XCTAssertEqual(sortedS[1], 1, accuracy: 1e-10)
    }

    // MARK: - Pinv Tests

    func testPinv() {
        // For invertible matrix, pinv = inv
        let A = LinAlg.Matrix([[1, 2], [3, 4]])
        let pA = LinAlg.pinv(A)
        guard let invA = LinAlg.inv(A) else {
            XCTFail("inv should succeed")
            return
        }

        for i in 0..<2 {
            for j in 0..<2 {
                XCTAssertEqual(pA[i, j], invA[i, j], accuracy: 1e-10)
            }
        }
    }

    // MARK: - Matrix Row/Col Access

    func testRowAccess() {
        let m = LinAlg.Matrix([[1, 2, 3], [4, 5, 6]])
        let r = m.row(1)
        XCTAssertEqual(r.rows, 1)
        XCTAssertEqual(r.cols, 3)
        XCTAssertEqual(r[0, 0], 4)
        XCTAssertEqual(r[0, 2], 6)
    }

    func testColAccess() {
        let m = LinAlg.Matrix([[1, 2, 3], [4, 5, 6]])
        let c = m.col(1)
        XCTAssertEqual(c.rows, 2)
        XCTAssertEqual(c.cols, 1)
        XCTAssertEqual(c[0], 2)
        XCTAssertEqual(c[1], 5)
    }

    func testToArray() {
        let m = LinAlg.Matrix([[1, 2], [3, 4]])
        let arr = m.toArray()
        XCTAssertEqual(arr.count, 2)
        XCTAssertEqual(arr[0], [1, 2])
        XCTAssertEqual(arr[1], [3, 4])
    }
}
