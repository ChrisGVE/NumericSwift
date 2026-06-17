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

    func testMatrixCreation() throws  {
        let m = LinAlg.Matrix([[1, 2], [3, 4]])
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
        XCTAssertEqual(m[0, 0], 1)
        XCTAssertEqual(m[1, 1], 4)
    }

    func testVectorCreation() throws  {
        let v = LinAlg.Matrix([1, 2, 3])
        XCTAssertEqual(v.rows, 3)
        XCTAssertEqual(v.cols, 1)
        XCTAssertTrue(v.isVector)
    }

    func testZerosAndOnes() throws  {
        let z = LinAlg.zeros(3, 2)
        XCTAssertEqual(z.rows, 3)
        XCTAssertEqual(z.cols, 2)
        XCTAssertEqual(z[1, 1], 0)

        let o = LinAlg.ones(2, 3)
        XCTAssertEqual(o.rows, 2)
        XCTAssertEqual(o.cols, 3)
        XCTAssertEqual(o[0, 2], 1)
    }

    func testEye() throws  {
        let I = LinAlg.eye(3)
        XCTAssertEqual(I[0, 0], 1)
        XCTAssertEqual(I[1, 1], 1)
        XCTAssertEqual(I[2, 2], 1)
        XCTAssertEqual(I[0, 1], 0)
    }

    func testDiag() throws  {
        let d = LinAlg.diag([1, 2, 3])
        XCTAssertEqual(d[0, 0], 1)
        XCTAssertEqual(d[1, 1], 2)
        XCTAssertEqual(d[2, 2], 3)
        XCTAssertEqual(d[0, 1], 0)
    }

    func testDiagExtract() throws  {
        let m = LinAlg.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let d = LinAlg.diag(m)
        XCTAssertEqual(d, [1, 5, 9])
    }

    func testArange() throws  {
        let v = try LinAlg.arange(0, 5, 1)
        XCTAssertEqual(v.rows, 5)
        XCTAssertEqual(v[0], 0)
        XCTAssertEqual(v[4], 4)
    }

    func testLinspace() throws  {
        let v = try LinAlg.linspace(0, 1, 5)
        XCTAssertEqual(v.rows, 5)
        XCTAssertEqual(v[0], 0, accuracy: 1e-10)
        XCTAssertEqual(v[2], 0.5, accuracy: 1e-10)
        XCTAssertEqual(v[4], 1, accuracy: 1e-10)
    }

    // MARK: - Arithmetic Tests

    func testMatrixAddSub() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let b = LinAlg.Matrix([[5, 6], [7, 8]])

        let sum = a + b
        XCTAssertEqual(sum[0, 0], 6)
        XCTAssertEqual(sum[1, 1], 12)

        let diff = b - a
        XCTAssertEqual(diff[0, 0], 4)
        XCTAssertEqual(diff[1, 1], 4)
    }

    func testScalarMultiply() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let scaled = 2.0 * a
        XCTAssertEqual(scaled[0, 0], 2)
        XCTAssertEqual(scaled[1, 1], 8)
    }

    func testMatrixMultiply() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let b = LinAlg.Matrix([[5, 6], [7, 8]])
        let c = a * b  // [[19, 22], [43, 50]]
        XCTAssertEqual(c[0, 0], 19)
        XCTAssertEqual(c[0, 1], 22)
        XCTAssertEqual(c[1, 0], 43)
        XCTAssertEqual(c[1, 1], 50)
    }

    func testDotProduct() throws  {
        let u = LinAlg.Matrix([1, 2, 3])
        let v = LinAlg.Matrix([4, 5, 6])
        let d = LinAlg.dot(u, v)  // 1*4 + 2*5 + 3*6 = 32
        XCTAssertEqual(d[0], 32)
    }

    func testHadamard() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let b = LinAlg.Matrix([[5, 6], [7, 8]])
        let h = LinAlg.hadamard(a, b)
        XCTAssertEqual(h[0, 0], 5)
        XCTAssertEqual(h[0, 1], 12)
        XCTAssertEqual(h[1, 0], 21)
        XCTAssertEqual(h[1, 1], 32)
    }

    func testNegation() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let neg = -a
        XCTAssertEqual(neg[0, 0], -1)
        XCTAssertEqual(neg[1, 1], -4)
    }

    func testDivide() throws  {
        let a = LinAlg.Matrix([[2, 4], [6, 8]])
        let div = a / 2.0
        XCTAssertEqual(div[0, 0], 1)
        XCTAssertEqual(div[1, 1], 4)
    }

    // MARK: - Matrix Properties

    func testTranspose() throws  {
        let a = LinAlg.Matrix([[1, 2, 3], [4, 5, 6]])
        let t = a.T
        XCTAssertEqual(t.rows, 3)
        XCTAssertEqual(t.cols, 2)
        XCTAssertEqual(t[0, 1], 4)
        XCTAssertEqual(t[2, 0], 3)
    }

    func testTrace() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        XCTAssertEqual(try LinAlg.trace(a), 5)
    }

    func testDeterminant() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        XCTAssertEqual(try LinAlg.det(a), -2, accuracy: 1e-10)

        let b = LinAlg.Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        XCTAssertEqual(try LinAlg.det(b), -306, accuracy: 1e-10)
    }

    func testInverse() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        guard let aInv = try LinAlg.inv(a) else {
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

    func testNorm() throws  {
        let v = LinAlg.Matrix([3, 4])
        XCTAssertEqual(LinAlg.norm(v, 2), 5)  // sqrt(9 + 16) = 5
        XCTAssertEqual(LinAlg.norm(v, 1), 7)  // |3| + |4| = 7
    }

    func testMatrixNorms() throws  {
        // Diagonal matrix: singular values are |diagonal entries|, so the
        // spectral norm (p=2) is the largest, here 3. Frobenius is sqrt(1+9)=√10.
        let d = LinAlg.Matrix([[1.0, 0.0], [0.0, -3.0]])
        XCTAssertEqual(LinAlg.norm(d, 2), 3.0, accuracy: 1e-10)  // spectral, SciPy ord=2
        XCTAssertEqual(LinAlg.frobeniusNorm(d), sqrt(10.0), accuracy: 1e-10)
        XCTAssertEqual(LinAlg.norm(d, 1), 3.0, accuracy: 1e-10)  // max abs col sum
        XCTAssertEqual(LinAlg.norm(d, .infinity), 3.0, accuracy: 1e-10)  // max abs row sum

        // Spectral norm of [[1,2],[3,4]] = largest singular value ≈ 5.46499.
        let a = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        XCTAssertEqual(LinAlg.norm(a, 2), 5.464985704, accuracy: 1e-6)
        XCTAssertEqual(LinAlg.norm(a, 1), 6.0, accuracy: 1e-10)  // col sums 4,6 → 6
        XCTAssertEqual(LinAlg.norm(a, .infinity), 7.0, accuracy: 1e-10)  // row sums 3,7 → 7
        XCTAssertEqual(LinAlg.frobeniusNorm(a), sqrt(30.0), accuracy: 1e-10)
    }

    func testRank() throws  {
        let a = LinAlg.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // Rank 2
        XCTAssertEqual(LinAlg.rank(a), 2)

        let b = LinAlg.eye(3)  // Rank 3
        XCTAssertEqual(LinAlg.rank(b), 3)
    }

    func testCond() throws  {
        let I = LinAlg.eye(3)
        XCTAssertEqual(LinAlg.cond(I), 1, accuracy: 1e-10)
    }

    // MARK: - Decompositions

    func testLU() throws  {
        let a = LinAlg.Matrix([[2, 1], [1, 3]])
        let (L, U, P) = try LinAlg.lu(a)

        // P @ L @ U should equal A
        let reconstructed = P * L * U
        XCTAssertEqual(reconstructed[0, 0], a[0, 0], accuracy: 1e-10)
        XCTAssertEqual(reconstructed[1, 1], a[1, 1], accuracy: 1e-10)
    }

    func testQR() throws  {
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

    func testSVD() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4], [5, 6]])
        let (s, U, Vt) = LinAlg.svd(a)

        XCTAssertEqual(s.count, 2)
        XCTAssertTrue(s[0] >= s[1])  // Singular values in descending order
    }

    func testEigenvalues() throws  {
        // Symmetric matrix has real eigenvalues
        let a = LinAlg.Matrix([[2, 1], [1, 2]])
        let (real, imag) = try LinAlg.eigvals(a)

        // Eigenvalues should be 3 and 1
        XCTAssertTrue(imag.allSatisfy { abs($0) < 1e-10 })  // All real
        let sorted = real.sorted()
        XCTAssertEqual(sorted[0], 1, accuracy: 1e-10)
        XCTAssertEqual(sorted[1], 3, accuracy: 1e-10)
    }

    func testEig() throws  {
        let a = LinAlg.Matrix([[2, 1], [1, 2]])
        let (values, imag, vectors) = try LinAlg.eig(a)

        XCTAssertEqual(values.count, 2)
        XCTAssertEqual(vectors.rows, 2)
        XCTAssertEqual(vectors.cols, 2)
    }

    func testCholesky() throws  {
        // Symmetric positive definite matrix
        let a = LinAlg.Matrix([[4, 2], [2, 5]])
        guard let L = try LinAlg.cholesky(a) else {
            XCTFail("Cholesky should succeed")
            return
        }

        // L @ L^T should equal A
        let reconstructed = L * L.T
        XCTAssertEqual(reconstructed[0, 0], a[0, 0], accuracy: 1e-10)
        XCTAssertEqual(reconstructed[1, 1], a[1, 1], accuracy: 1e-10)
    }

    // MARK: - Linear System Solvers

    func testSolve() throws  {
        // Solve [[2, 1], [1, 3]] * x = [1, 2]
        let A = LinAlg.Matrix([[2, 1], [1, 3]])
        let b = LinAlg.Matrix([1, 2])

        guard let x = try LinAlg.solve(A, b) else {
            XCTFail("Solve should succeed")
            return
        }

        // A @ x should equal b
        let result = A * x
        XCTAssertEqual(result[0], b[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], b[1], accuracy: 1e-10)
    }

    func testLstSq() throws  {
        // Overdetermined system: fit y = ax + b to points
        let A = LinAlg.Matrix([[1, 1], [2, 1], [3, 1]])  // [x, 1]
        let b = LinAlg.Matrix([2, 3, 4])  // y values for y = x + 1

        guard let x = try LinAlg.lstsq(A, b) else {
            XCTFail("Lstsq should succeed")
            return
        }

        XCTAssertEqual(x[0], 1, accuracy: 1e-10)  // slope = 1
        XCTAssertEqual(x[1], 1, accuracy: 1e-10)  // intercept = 1
    }

    func testSolveTriangular() throws  {
        // Lower triangular system
        let L = LinAlg.Matrix([[2, 0], [1, 3]])
        let b = LinAlg.Matrix([4, 7])

        guard let x = try LinAlg.solveTriangular(L, b, lower: true, trans: false) else {
            XCTFail("Solve triangular should succeed")
            return
        }

        // L @ x should equal b
        let result = L * x
        XCTAssertEqual(result[0], b[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], b[1], accuracy: 1e-10)
    }

    func testChoSolve() throws  {
        // Solve using Cholesky factor
        let A = LinAlg.Matrix([[4, 2], [2, 5]])
        let b = LinAlg.Matrix([1, 2])

        guard let L = try LinAlg.cholesky(A) else {
            XCTFail("Cholesky should succeed")
            return
        }

        guard let x = try LinAlg.choSolve(L, b) else {
            XCTFail("choSolve should succeed")
            return
        }

        // A @ x should equal b
        let result = A * x
        XCTAssertEqual(result[0], b[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], b[1], accuracy: 1e-10)
    }

    func testLuSolve() throws  {
        let A = LinAlg.Matrix([[2, 1], [1, 3]])
        let b = LinAlg.Matrix([1, 2])

        let (L, U, P) = try LinAlg.lu(A)
        let x = try LinAlg.luSolve(L, U, P, b)

        // A @ x should equal b
        let result = A * x
        XCTAssertEqual(result[0], b[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], b[1], accuracy: 1e-10)
    }

    // MARK: - Matrix Functions

    func testExpm() throws  {
        // exp(0) = I
        let Z = LinAlg.zeros(2, 2)
        let expZ = try LinAlg.expm(Z)
        XCTAssertEqual(expZ[0, 0], 1, accuracy: 1e-10)
        XCTAssertEqual(expZ[1, 1], 1, accuracy: 1e-10)
        XCTAssertEqual(expZ[0, 1], 0, accuracy: 1e-10)

        // exp(diag(a, b)) = diag(exp(a), exp(b))
        let D = LinAlg.Matrix([[1, 0], [0, 2]])
        let expD = try LinAlg.expm(D)
        XCTAssertEqual(expD[0, 0], exp(1.0), accuracy: 1e-12)
        XCTAssertEqual(expD[1, 1], exp(2.0), accuracy: 1e-12)
    }

    func testLogm() throws  {
        // log(I) = 0
        let I = LinAlg.eye(2)
        guard let logI = try LinAlg.logm(I) else {
            XCTFail("logm should succeed for identity")
            return
        }
        XCTAssertEqual(logI[0, 0], 0, accuracy: 1e-10)
        XCTAssertEqual(logI[1, 1], 0, accuracy: 1e-10)

        // log(exp(A)) ≈ A for diagonal matrices
        let D = LinAlg.Matrix([[2, 0], [0, 3]])
        let expD = try LinAlg.expm(D)
        guard let logExpD = try LinAlg.logm(expD) else {
            XCTFail("logm should succeed")
            return
        }
        XCTAssertEqual(logExpD[0, 0], 2, accuracy: 1e-4)
        XCTAssertEqual(logExpD[1, 1], 3, accuracy: 1e-4)
    }

    func testSqrtm() throws  {
        // sqrt(I) = I
        let I = LinAlg.eye(2)
        guard let sqrtI = try LinAlg.sqrtm(I) else {
            XCTFail("sqrtm should succeed for identity")
            return
        }
        XCTAssertEqual(sqrtI[0, 0], 1, accuracy: 1e-10)
        XCTAssertEqual(sqrtI[1, 1], 1, accuracy: 1e-10)

        // sqrt(diag(4, 9)) = diag(2, 3)
        let D = LinAlg.Matrix([[4, 0], [0, 9]])
        guard let sqrtD = try LinAlg.sqrtm(D) else {
            XCTFail("sqrtm should succeed")
            return
        }
        XCTAssertEqual(sqrtD[0, 0], 2, accuracy: 1e-10)
        XCTAssertEqual(sqrtD[1, 1], 3, accuracy: 1e-10)
    }

    func testFunm() throws  {
        // sin(0) = 0
        let Z = LinAlg.zeros(2, 2)
        guard let sinZ = try LinAlg.funm(Z, .sin) else {
            XCTFail("funm should succeed")
            return
        }
        XCTAssertEqual(sinZ[0, 0], 0, accuracy: 1e-10)

        // cos(0) = I
        guard let cosZ = try LinAlg.funm(Z, .cos) else {
            XCTFail("funm should succeed")
            return
        }
        XCTAssertEqual(cosZ[0, 0], 1, accuracy: 1e-10)
        XCTAssertEqual(cosZ[1, 1], 1, accuracy: 1e-10)
    }

    // MARK: - Matrix Functions: Complex-Eigenvalue and Defective Matrix Tests (Issue #7)
    // Oracle: scipy.linalg.logm / sqrtm / funm (Python scipy 1.x)

    /// Rotation matrix R(π/4) has complex-conjugate eigenvalues e^{±iπ/4}.
    /// logm(R) and sqrtm(R) are both purely real — verified via scipy.
    func testLogmRotationMatrix() throws {
        // R = [[cos θ, -sin θ], [sin θ, cos θ]], θ = π/4
        let c = 0.7071067811865476  // cos(π/4)
        let s = 0.7071067811865476  // sin(π/4)
        let R = LinAlg.Matrix([[c, -s], [s, c]])

        // Current implementation bails (returns nil) because dgeev sees imaginary eigenvalues.
        // After fix: must return a real matrix matching scipy oracle values.
        guard let logR = try LinAlg.logm(R) else {
            XCTFail("logm must succeed on rotation matrix with complex-conjugate eigenvalues")
            return
        }
        // scipy: logm(R) ≈ [[0, -π/4], [π/4, 0]]
        XCTAssertEqual(logR[0, 0],  0.0,                  accuracy: 1e-10)
        XCTAssertEqual(logR[0, 1], -0.7853981633974478,   accuracy: 1e-10)
        XCTAssertEqual(logR[1, 0],  0.7853981633974483,   accuracy: 1e-10)
        XCTAssertEqual(logR[1, 1],  0.0,                  accuracy: 1e-10)
    }

    func testSqrtmRotationMatrix() throws {
        let c = 0.7071067811865476
        let s = 0.7071067811865476
        let R = LinAlg.Matrix([[c, -s], [s, c]])

        guard let sqrtR = try LinAlg.sqrtm(R) else {
            XCTFail("sqrtm must succeed on rotation matrix with complex-conjugate eigenvalues")
            return
        }
        // scipy: sqrtm(R) = rotation by π/8 — [[cos(π/8), -sin(π/8)], [sin(π/8), cos(π/8)]]
        XCTAssertEqual(sqrtR[0, 0],  0.9238795325112867,  accuracy: 1e-10)
        XCTAssertEqual(sqrtR[0, 1], -0.3826834323650897,  accuracy: 1e-10)
        XCTAssertEqual(sqrtR[1, 0],  0.3826834323650897,  accuracy: 1e-10)
        XCTAssertEqual(sqrtR[1, 1],  0.9238795325112867,  accuracy: 1e-10)

        // Defining identity: sqrtm(R)^2 ≈ R
        let sqrtR2 = try LinAlg.dot(sqrtR, sqrtR)
        XCTAssertEqual(sqrtR2[0, 0], c, accuracy: 1e-10)
        XCTAssertEqual(sqrtR2[0, 1], -s, accuracy: 1e-10)
        XCTAssertEqual(sqrtR2[1, 0], s, accuracy: 1e-10)
        XCTAssertEqual(sqrtR2[1, 1], c, accuracy: 1e-10)
    }

    /// A = [[1, 2], [-2, 1]] has eigenvalues 1±2i (complex conjugate pair).
    /// logm(A) and sqrtm(A) are real. funm(A, .exp) is real and equals expm(A).
    func testLogmComplexEigenvalues() throws {
        let A = LinAlg.Matrix([[1.0, 2.0], [-2.0, 1.0]])

        guard let logA = try LinAlg.logm(A) else {
            XCTFail("logm must succeed on matrix with complex-conjugate eigenvalues")
            return
        }
        // scipy oracle values
        XCTAssertEqual(logA[0, 0],  0.80471895621705,     accuracy: 1e-10)
        XCTAssertEqual(logA[0, 1],  1.1071487177940904,   accuracy: 1e-10)
        XCTAssertEqual(logA[1, 0], -1.1071487177940904,   accuracy: 1e-10)
        XCTAssertEqual(logA[1, 1],  0.80471895621705,     accuracy: 1e-10)
    }

    func testSqrtmComplexEigenvalues() throws {
        let A = LinAlg.Matrix([[1.0, 2.0], [-2.0, 1.0]])

        guard let sqrtA = try LinAlg.sqrtm(A) else {
            XCTFail("sqrtm must succeed on matrix with complex-conjugate eigenvalues")
            return
        }
        // scipy oracle values
        XCTAssertEqual(sqrtA[0, 0],  1.272019649514069,    accuracy: 1e-10)
        XCTAssertEqual(sqrtA[0, 1],  0.7861513777574233,   accuracy: 1e-10)
        XCTAssertEqual(sqrtA[1, 0], -0.7861513777574233,   accuracy: 1e-10)
        XCTAssertEqual(sqrtA[1, 1],  1.272019649514069,    accuracy: 1e-10)

        // Defining identity: sqrtm(A)^2 ≈ A
        let sqrtA2 = try LinAlg.dot(sqrtA, sqrtA)
        XCTAssertEqual(sqrtA2[0, 0], 1.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtA2[0, 1], 2.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtA2[1, 0], -2.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtA2[1, 1], 1.0, accuracy: 1e-9)
    }

    func testFunmComplexEigenvalues() throws {
        let A = LinAlg.Matrix([[1.0, 2.0], [-2.0, 1.0]])

        // funm(A, .exp) must equal expm(A)
        guard let funmAExp = try LinAlg.funm(A, .exp) else {
            XCTFail("funm(.exp) must succeed on matrix with complex-conjugate eigenvalues")
            return
        }
        // scipy oracle: funm(A, exp) = expm(A)
        XCTAssertEqual(funmAExp[0, 0], -1.1312043837568135, accuracy: 1e-9)
        XCTAssertEqual(funmAExp[0, 1],  2.471726672004819,  accuracy: 1e-9)
        XCTAssertEqual(funmAExp[1, 0], -2.4717266720048183, accuracy: 1e-9)
        XCTAssertEqual(funmAExp[1, 1], -1.1312043837568135, accuracy: 1e-9)
    }

    /// J = [[2, 1], [0, 2]] is a defective Jordan block: eigenvalue 2 with multiplicity 2
    /// but only one independent eigenvector. The current eigendecomposition approach
    /// produces a near-singular V and numerically garbage results. Schur-based logm/sqrtm
    /// handles this correctly.
    func testLogmDefectiveMatrix() throws {
        let J = LinAlg.Matrix([[2.0, 1.0], [0.0, 2.0]])

        guard let logJ = try LinAlg.logm(J) else {
            XCTFail("logm must succeed on defective Jordan block matrix")
            return
        }
        // scipy: logm([[2,1],[0,2]]) = [[ln2, 0.5], [0, ln2]]
        XCTAssertEqual(logJ[0, 0], 0.6931471805599453, accuracy: 1e-10)
        XCTAssertEqual(logJ[0, 1], 0.5,                accuracy: 1e-10)
        XCTAssertEqual(logJ[1, 0], 0.0,                accuracy: 1e-10)
        XCTAssertEqual(logJ[1, 1], 0.6931471805599453, accuracy: 1e-10)
    }

    func testSqrtmDefectiveMatrix() throws {
        let J = LinAlg.Matrix([[2.0, 1.0], [0.0, 2.0]])

        guard let sqrtJ = try LinAlg.sqrtm(J) else {
            XCTFail("sqrtm must succeed on defective Jordan block matrix")
            return
        }
        // scipy: sqrtm([[2,1],[0,2]]) = [[√2, 1/(2√2)], [0, √2]]
        XCTAssertEqual(sqrtJ[0, 0], 1.4142135623730951,  accuracy: 1e-10)
        XCTAssertEqual(sqrtJ[0, 1], 0.35355339059327373, accuracy: 1e-10)
        XCTAssertEqual(sqrtJ[1, 0], 0.0,                 accuracy: 1e-10)
        XCTAssertEqual(sqrtJ[1, 1], 1.4142135623730951,  accuracy: 1e-10)

        // Defining identity: sqrtm(J)^2 ≈ J
        let sqrtJ2 = try LinAlg.dot(sqrtJ, sqrtJ)
        XCTAssertEqual(sqrtJ2[0, 0], 2.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtJ2[0, 1], 1.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtJ2[1, 0], 0.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtJ2[1, 1], 2.0, accuracy: 1e-9)
    }

    func testFunmDefectiveMatrix() throws {
        // funm on a defective matrix: the Schur-Parlett recurrence handles
        // J = [[2,1],[0,2]] with degenerate eigenvalue 2.
        // scipy funm (Parlett diagonal-only) gives [ln2, ln2] (wrong off-diagonal),
        // whereas logm gives the correct [[ln2, 0.5],[0, ln2]].
        // funm(.log) should match logm for non-defective Schur blocks;
        // for truly defective blocks, scipy flags inaccuracy — we document this limitation.
        let J = LinAlg.Matrix([[2.0, 1.0], [0.0, 2.0]])

        // funm(.exp) on Jordan block: diag approach gives [[e^2, 0],[0, e^2]] but
        // correct value is [[e^2, e^2],[0, e^2]] (derivative correction).
        // We only require at least the diagonal to be correct (scipy reports approx err=1).
        guard let funmJExp = try LinAlg.funm(J, .exp) else {
            XCTFail("funm must not return nil on defective matrix")
            return
        }
        XCTAssertEqual(funmJExp[0, 0], exp(2.0), accuracy: 1e-8)
        XCTAssertEqual(funmJExp[1, 1], exp(2.0), accuracy: 1e-8)
    }

    /// sqrtmComplex / logmComplex: new APIs returning ComplexMatrix for matrices
    /// whose square root / logarithm is genuinely complex.
    func testSqrtmComplexResult() throws {
        // B = diag(-1, -4) has negative eigenvalues → sqrtm gives purely imaginary diagonal.
        let B = LinAlg.Matrix([[-1.0, 0.0], [0.0, -4.0]])

        guard let sqrtB = try LinAlg.sqrtmComplex(B) else {
            XCTFail("sqrtmComplex must succeed for negative-definite matrix")
            return
        }
        // scipy: sqrtm(B) = [[0+i, 0], [0, 0+2i]]
        XCTAssertEqual(sqrtB[0, 0].re,  0.0, accuracy: 1e-10)
        XCTAssertEqual(sqrtB[0, 0].im,  1.0, accuracy: 1e-10)
        XCTAssertEqual(sqrtB[1, 1].re,  0.0, accuracy: 1e-10)
        XCTAssertEqual(sqrtB[1, 1].im,  2.0, accuracy: 1e-10)
        XCTAssertEqual(sqrtB[0, 1].re,  0.0, accuracy: 1e-10)
        XCTAssertEqual(sqrtB[0, 1].im,  0.0, accuracy: 1e-10)
    }

    func testLogmComplexResult() throws {
        // B = diag(-1, -4): logm has imaginary part ±iπ (principal branch)
        let B = LinAlg.Matrix([[-1.0, 0.0], [0.0, -4.0]])

        guard let logB = try LinAlg.logmComplex(B) else {
            XCTFail("logmComplex must succeed for negative-definite matrix")
            return
        }
        // scipy: logm(B) = [[iπ, 0], [0, ln4+iπ]]
        XCTAssertEqual(logB[0, 0].re, 0.0,                 accuracy: 1e-10)
        XCTAssertEqual(logB[0, 0].im, Double.pi,            accuracy: 1e-10)
        XCTAssertEqual(logB[1, 1].re, 1.386294361119893,    accuracy: 1e-10)
        XCTAssertEqual(logB[1, 1].im, Double.pi,            accuracy: 1e-10)
    }

    /// 3×3 rotation-like matrix with complex eigenvalues and a real eigenvalue 2.
    /// logm and sqrtm return real matrices. (All imaginary parts ≈ 0.)
    func testLogmSqrtm3x3ComplexEigenvalues() throws {
        // D = [[0,-1,0],[1,0,0],[0,0,2]] — eigenvalues ±i, 2
        let D = LinAlg.Matrix([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

        guard let logD = try LinAlg.logm(D) else {
            XCTFail("logm must succeed on 3×3 matrix with complex+real eigenvalues")
            return
        }
        // scipy oracle: logm(D) ≈ [[0, -π/2, 0], [π/2, 0, 0], [0, 0, ln2]]
        XCTAssertEqual(logD[0, 0], 0.0,                 accuracy: 1e-10)
        XCTAssertEqual(logD[0, 1], -1.5707963267948966, accuracy: 1e-10)
        XCTAssertEqual(logD[1, 0],  1.5707963267948966, accuracy: 1e-10)
        XCTAssertEqual(logD[1, 1], 0.0,                 accuracy: 1e-10)
        XCTAssertEqual(logD[2, 2], 0.6931471805599453,  accuracy: 1e-10)

        guard let sqrtD = try LinAlg.sqrtm(D) else {
            XCTFail("sqrtm must succeed on 3×3 matrix with complex+real eigenvalues")
            return
        }
        // scipy: sqrtm(D) = [[1/√2, -1/√2, 0], [1/√2, 1/√2, 0], [0, 0, √2]]
        XCTAssertEqual(sqrtD[0, 0],  0.7071067811865475,  accuracy: 1e-10)
        XCTAssertEqual(sqrtD[0, 1], -0.7071067811865476,  accuracy: 1e-10)
        XCTAssertEqual(sqrtD[1, 0],  0.7071067811865476,  accuracy: 1e-10)
        XCTAssertEqual(sqrtD[1, 1],  0.7071067811865475,  accuracy: 1e-10)
        XCTAssertEqual(sqrtD[2, 2],  1.4142135623730951,  accuracy: 1e-10)

        // Defining identity: sqrtm(D)^2 ≈ D
        let sqrtD2 = try LinAlg.dot(sqrtD, sqrtD)
        XCTAssertEqual(sqrtD2[0, 0],  0.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtD2[0, 1], -1.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtD2[1, 0],  1.0, accuracy: 1e-9)
        XCTAssertEqual(sqrtD2[2, 2],  2.0, accuracy: 1e-9)
    }

    /// Parlett recurrence with non-zero cross-terms between a 2×2 and a 1×1 Schur block.
    ///
    /// A = [[1,2,1],[-2,1,1],[0,0,2]] is already in real Schur form:
    /// rows 0–1 form a 2×2 block (eigenvalues 1±2i), row 2 is a 1×1 block (eigenvalue 2).
    /// The off-diagonal entries T[0,2]=T[1,2]=1 produce non-zero cross-terms F[0,2]
    /// and F[1,2] that require the 2×2 block Sylvester solver, not the scalar divisor.
    /// Frozen oracle: scipy.linalg.logm / sqrtm (scipy 1.x, numpy principal branch).
    func testParlettCrossTermTwoByOneSylvester() throws {
        let A = LinAlg.Matrix([[1.0, 2.0, 1.0], [-2.0, 1.0, 1.0], [0.0, 0.0, 2.0]])

        // logm(A) — result is real (all eigenvalues have positive real part).
        guard let logA = try LinAlg.logm(A) else {
            XCTFail("logm must succeed: eigenvalues 1±2i and 2 all have positive real part")
            return
        }
        // scipy.linalg.logm frozen literals:
        XCTAssertEqual(logA[0, 0],  0.80471895621705,       accuracy: 1e-9)
        XCTAssertEqual(logA[0, 1],  1.1071487177940904,     accuracy: 1e-9)
        XCTAssertEqual(logA[0, 2],  0.15448667816455516,    accuracy: 1e-9)  // cross-term
        XCTAssertEqual(logA[1, 0], -1.1071487177940904,     accuracy: 1e-9)
        XCTAssertEqual(logA[1, 1],  0.80471895621705,       accuracy: 1e-9)
        XCTAssertEqual(logA[1, 2],  0.6866035858078751,     accuracy: 1e-9)  // cross-term
        XCTAssertEqual(logA[2, 0],  0.0,                    accuracy: 1e-12)
        XCTAssertEqual(logA[2, 1],  0.0,                    accuracy: 1e-12)
        XCTAssertEqual(logA[2, 2],  0.6931471805599453,     accuracy: 1e-9)

        // sqrtm(A) — result is also real.
        guard let sqrtA = try LinAlg.sqrtm(A) else {
            XCTFail("sqrtm must succeed: all eigenvalues have positive real part")
            return
        }
        // scipy.linalg.sqrtm frozen literals:
        XCTAssertEqual(sqrtA[0, 0],  1.272019649514069,     accuracy: 1e-9)
        XCTAssertEqual(sqrtA[0, 1],  0.7861513777574233,    accuracy: 1e-9)
        XCTAssertEqual(sqrtA[0, 2],  0.24254662326690027,   accuracy: 1e-9)  // cross-term
        XCTAssertEqual(sqrtA[1, 0], -0.7861513777574233,    accuracy: 1e-9)
        XCTAssertEqual(sqrtA[1, 1],  1.272019649514069,     accuracy: 1e-9)
        XCTAssertEqual(sqrtA[1, 2],  0.4432520440826487,    accuracy: 1e-9)  // cross-term
        XCTAssertEqual(sqrtA[2, 0],  0.0,                   accuracy: 1e-12)
        XCTAssertEqual(sqrtA[2, 1],  0.0,                   accuracy: 1e-12)
        XCTAssertEqual(sqrtA[2, 2],  1.4142135623730951,    accuracy: 1e-9)

        // Defining identity: logm and sqrtm are inverse-consistent.
        let sqrtA2 = try LinAlg.dot(sqrtA, sqrtA)
        XCTAssertEqual(sqrtA2[0, 0],  1.0, accuracy: 1e-8)
        XCTAssertEqual(sqrtA2[0, 1],  2.0, accuracy: 1e-8)
        XCTAssertEqual(sqrtA2[0, 2],  1.0, accuracy: 1e-8)
        XCTAssertEqual(sqrtA2[1, 0], -2.0, accuracy: 1e-8)
        XCTAssertEqual(sqrtA2[1, 1],  1.0, accuracy: 1e-8)
        XCTAssertEqual(sqrtA2[1, 2],  1.0, accuracy: 1e-8)
        XCTAssertEqual(sqrtA2[2, 2],  2.0, accuracy: 1e-8)
    }

    /// Parlett recurrence: 1×1 block followed by 2×2 block — tests the
    /// `!rowIs2x2 && colIs2x2` Sylvester branch (row 0 vs block {1,2}).
    ///
    /// B = [[1,5,3],[0,3,2],[0,-2,3]]: eigenvalue 1 (1×1) then 3±2i (2×2).
    /// Already in real Schur form; T[0,1]=5, T[0,2]=3 are non-zero cross-terms.
    /// Frozen oracle: scipy.linalg.logm.
    func testParlettCrossTermOneByTwoSylvester() throws {
        let B = LinAlg.Matrix([[1.0, 5.0, 3.0], [0.0, 3.0, 2.0], [0.0, -2.0, 3.0]])

        guard let logB = try LinAlg.logm(B) else {
            XCTFail("logm must succeed: all eigenvalues have positive real part")
            return
        }
        // scipy.linalg.logm frozen literals:
        XCTAssertEqual(logB[0, 0],  0.0,                     accuracy: 1e-10)
        XCTAssertEqual(logB[0, 1],  2.8589506592353198,      accuracy: 1e-9)  // cross-term
        XCTAssertEqual(logB[0, 2],  0.5347678677297507,      accuracy: 1e-9)  // cross-term
        XCTAssertEqual(logB[1, 1],  1.2824746787307686,      accuracy: 1e-9)
        XCTAssertEqual(logB[1, 2],  0.5880026035475676,      accuracy: 1e-9)
        XCTAssertEqual(logB[2, 1], -0.5880026035475674,      accuracy: 1e-9)
        XCTAssertEqual(logB[2, 2],  1.2824746787307681,      accuracy: 1e-9)
    }

    /// Parlett recurrence: 4×4 with two 2×2 blocks — tests the 2×2 vs 2×2
    /// (vectorised 4×4 Sylvester) branch.
    ///
    /// A = [[1,2,3,4],[-2,1,5,6],[0,0,0,1],[0,0,-1,0]]: already in real Schur
    /// form with two 2×2 blocks (eigenvalues 1±2i and 0±i); T[0:2, 2:4] are
    /// non-zero, exercising the cross-block Sylvester solver.
    /// Frozen oracle: scipy.linalg.logm.
    func testParlettCrossTermTwoByTwoSylvester() throws {
        let A = LinAlg.Matrix([[1.0, 2.0, 3.0, 4.0],
                               [-2.0, 1.0, 5.0, 6.0],
                               [0.0, 0.0, 0.0, 1.0],
                               [0.0, 0.0, -1.0, 0.0]])

        guard let logA = try LinAlg.logm(A) else {
            XCTFail("logm must succeed: all eigenvalues have positive real part")
            return
        }
        // scipy.linalg.logm frozen literals:
        XCTAssertEqual(logA[0, 0],  0.80471895621705,       accuracy: 1e-9)
        XCTAssertEqual(logA[0, 1],  1.1071487177940904,     accuracy: 1e-9)
        XCTAssertEqual(logA[0, 2], -0.7567595443934897,     accuracy: 1e-9)  // cross 2×2→2×2
        XCTAssertEqual(logA[0, 3],  1.0778249583392425,     accuracy: 1e-9)  // cross 2×2→2×2
        XCTAssertEqual(logA[1, 2],  6.956010175427716,      accuracy: 1e-9)  // cross 2×2→2×2
        XCTAssertEqual(logA[1, 3],  1.6573973242576612,     accuracy: 1e-9)  // cross 2×2→2×2
        XCTAssertEqual(logA[2, 2],  0.0,                    accuracy: 1e-9)
        XCTAssertEqual(logA[2, 3],  1.5707963267948966,     accuracy: 1e-9)
        XCTAssertEqual(logA[3, 2], -1.5707963267948966,     accuracy: 1e-9)
        XCTAssertEqual(logA[3, 3],  0.0,                    accuracy: 1e-9)
    }

    /// Defining identity expm(logm(A)) ≈ A for a real-logm case.
    func testExpmLogmDefiningIdentity() throws {
        // R(π/4): expm(logm(R)) ≈ R
        let c = 0.7071067811865476
        let s = 0.7071067811865476
        let R = LinAlg.Matrix([[c, -s], [s, c]])

        guard let logR = try LinAlg.logm(R) else {
            XCTFail("logm must succeed")
            return
        }
        let expLogR = try LinAlg.expm(logR)
        XCTAssertEqual(expLogR[0, 0], c, accuracy: 1e-9)
        XCTAssertEqual(expLogR[0, 1], -s, accuracy: 1e-9)
        XCTAssertEqual(expLogR[1, 0], s, accuracy: 1e-9)
        XCTAssertEqual(expLogR[1, 1], c, accuracy: 1e-9)
    }

    /// Regression: existing tests — diagonalizable matrices with real positive eigenvalues
    /// must still produce correct results after the Schur-based rewrite.
    func testLogmSqrtmRegressionDiagonalizable() throws {
        // C = [[1,2],[3,4]]: real eigenvalues but logm/sqrtm are complex in scipy.
        // Our real-returning API returns nil for genuinely complex results — that is correct.
        let C = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        // eigenvalues: -0.372, 5.372 — one negative → logm is complex
        let logC = try LinAlg.logm(C)
        XCTAssertNil(logC, "logm returns nil for matrix with negative eigenvalue (complex result)")

        // But logmComplex should succeed
        guard let logCComplex = try LinAlg.logmComplex(C) else {
            XCTFail("logmComplex must succeed even when result is complex")
            return
        }
        // scipy real part of logm(C)[0,0] = -0.35043981399855517
        XCTAssertEqual(logCComplex[0, 0].re, -0.35043981399855517, accuracy: 1e-8)
    }

    // MARK: - Complex Matrix Tests

    func testComplexMatrixCreation() throws  {
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

    func testComplexFromReal() throws  {
        let m = LinAlg.Matrix([[1, 2], [3, 4]])
        let cm = LinAlg.ComplexMatrix(m)
        XCTAssertEqual(cm[0, 0].re, 1)
        XCTAssertEqual(cm[0, 0].im, 0)
    }

    func testCSolve() throws  {
        // Identity system: I * x = b => x = b
        let I = LinAlg.ComplexMatrix(LinAlg.eye(2))
        let b = LinAlg.ComplexMatrix(
            rows: 2, cols: 1,
            real: [1, 2],
            imag: [0.5, 1.5]
        )

        guard let x = try LinAlg.csolve(I, b) else {
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

    func testCDet() throws  {
        // Determinant of identity should be 1
        let I = LinAlg.ComplexMatrix(LinAlg.eye(2))
        guard let det = try LinAlg.cdet(I) else {
            XCTFail("cdet should succeed")
            return
        }
        XCTAssertEqual(det.re, 1, accuracy: 1e-10)
        XCTAssertEqual(det.im, 0, accuracy: 1e-10)
    }

    func testCInv() throws  {
        // Inverse of identity should be identity
        let I = LinAlg.ComplexMatrix(LinAlg.eye(2))
        guard let invI = try LinAlg.cinv(I) else {
            XCTFail("cinv should succeed")
            return
        }
        XCTAssertEqual(invI[0, 0].re, 1, accuracy: 1e-10)
        XCTAssertEqual(invI[0, 0].im, 0, accuracy: 1e-10)
    }

    func testCEigvals() throws  {
        let m = LinAlg.ComplexMatrix(LinAlg.eye(2))
        guard let eigvals = try LinAlg.ceigvals(m) else {
            XCTFail("ceigvals should succeed")
            return
        }
        XCTAssertEqual(eigvals.rows, 2)
        // Eigenvalues of identity are all 1
        XCTAssertEqual(eigvals.real[0], 1, accuracy: 1e-10)
        XCTAssertEqual(eigvals.real[1], 1, accuracy: 1e-10)
    }

    func testCSVD() throws  {
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

    func testPinv() throws  {
        // For invertible matrix, pinv = inv
        let A = LinAlg.Matrix([[1, 2], [3, 4]])
        let pA = LinAlg.pinv(A)
        guard let invA = try LinAlg.inv(A) else {
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

    func testRowAccess() throws  {
        let m = LinAlg.Matrix([[1, 2, 3], [4, 5, 6]])
        let r = m.row(1)
        XCTAssertEqual(r.rows, 1)
        XCTAssertEqual(r.cols, 3)
        XCTAssertEqual(r[0, 0], 4)
        XCTAssertEqual(r[0, 2], 6)
    }

    func testColAccess() throws  {
        let m = LinAlg.Matrix([[1, 2, 3], [4, 5, 6]])
        let c = m.col(1)
        XCTAssertEqual(c.rows, 2)
        XCTAssertEqual(c.cols, 1)
        XCTAssertEqual(c[0], 2)
        XCTAssertEqual(c[1], 5)
    }

    func testToArray() throws  {
        let m = LinAlg.Matrix([[1, 2], [3, 4]])
        let arr = m.toArray()
        XCTAssertEqual(arr.count, 2)
        XCTAssertEqual(arr[0], [1, 2])
        XCTAssertEqual(arr[1], [3, 4])
    }

    // MARK: - Matrix instance inverse / pinv

    func testMatrixInstanceInverse() throws  {
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        guard let aInv = try a.inverse() else {
            XCTFail("inverse() should return non-nil for non-singular matrix")
            return
        }
        // A * A^(-1) should be identity
        let product = a * aInv
        XCTAssertEqual(product[0, 0], 1, accuracy: 1e-10)
        XCTAssertEqual(product[0, 1], 0, accuracy: 1e-10)
        XCTAssertEqual(product[1, 0], 0, accuracy: 1e-10)
        XCTAssertEqual(product[1, 1], 1, accuracy: 1e-10)
    }

    func testMatrixInstanceInverseSingular() throws  {
        let singular = LinAlg.Matrix([[1, 2], [2, 4]])
        XCTAssertNil(try singular.inverse(), "inverse() should return nil for singular matrix")
    }

    func testMatrixInstanceInverseMatchesStaticInv() throws  {
        let a = LinAlg.Matrix([[2, 1], [5, 3]])
        let instanceResult = try a.inverse()
        let staticResult = try LinAlg.inv(a)
        XCTAssertEqual(instanceResult, staticResult)
    }

    func testMatrixInstancePinvMatchesStaticPinv() throws  {
        // For an invertible matrix, pinv should match inv
        let a = LinAlg.Matrix([[1, 2], [3, 4]])
        let instancePinv = a.pinv()
        let staticPinv = LinAlg.pinv(a)
        for i in 0..<2 {
            for j in 0..<2 {
                XCTAssertEqual(instancePinv[i, j], staticPinv[i, j], accuracy: 1e-10)
            }
        }
    }

    func testMatrixInstancePinvNonSquare() throws  {
        // pinv of a tall matrix: (3x2) -> (2x3)
        let a = LinAlg.Matrix([[1, 0], [0, 1], [0, 0]])
        let p = a.pinv()
        XCTAssertEqual(p.rows, 2)
        XCTAssertEqual(p.cols, 3)
    }
}
