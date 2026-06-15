//
//  LinAlg+Decompositions.swift
//  NumericSwift
//
//  Matrix decompositions: LU, QR, SVD, eigendecomposition, Cholesky.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Decompositions

    /// LU decomposition with partial pivoting.
    ///
    /// - Parameter m: Square matrix
    /// - Returns: (L, U, P) where P @ L @ U = m
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func lu(_ m: Matrix) throws -> (L: Matrix, U: Matrix, P: Matrix) {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = toColumnMajor(m)
        var n1 = __CLPK_integer(n)
        var n2 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&n1, &n2, &a, &lda, &ipiv, &info)

        var L = [Double](repeating: 0, count: n * n)
        var U = [Double](repeating: 0, count: n * n)

        for i in 0..<n {
            for j in 0..<n {
                let colMajorIdx = j * n + i
                if i > j {
                    L[i * n + j] = a[colMajorIdx]
                } else if i == j {
                    L[i * n + j] = 1.0
                    U[i * n + j] = a[colMajorIdx]
                } else {
                    U[i * n + j] = a[colMajorIdx]
                }
            }
        }

        var P = [Double](repeating: 0, count: n * n)
        var perm = Array(0..<n)
        for i in 0..<n {
            let pivot = Int(ipiv[i]) - 1
            if pivot != i {
                perm.swapAt(i, pivot)
            }
        }
        for i in 0..<n {
            P[i * n + perm[i]] = 1.0
        }

        return (Matrix(rows: n, cols: n, data: L),
                Matrix(rows: n, cols: n, data: U),
                Matrix(rows: n, cols: n, data: P))
    }

    /// QR decomposition.
    ///
    /// - Parameter m: Matrix (m×n)
    /// - Returns: (Q, R) where Q @ R = m
    public static func qr(_ m: Matrix) -> (Q: Matrix, R: Matrix) {
        let minDim = min(m.rows, m.cols)

        var a = toColumnMajor(m)

        var m1 = __CLPK_integer(m.rows)
        var n1 = __CLPK_integer(m.cols)
        var lda1 = __CLPK_integer(m.rows)
        var tau = [Double](repeating: 0, count: minDim)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        dgeqrf_(&m1, &n1, &a, &lda1, &tau, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        var m2 = __CLPK_integer(m.rows)
        var n2 = __CLPK_integer(m.cols)
        var lda2 = __CLPK_integer(m.rows)
        dgeqrf_(&m2, &n2, &a, &lda2, &tau, &work, &lwork, &info)

        var R = [Double](repeating: 0, count: minDim * m.cols)
        for i in 0..<minDim {
            for j in i..<m.cols {
                R[i * m.cols + j] = a[j * m.rows + i]
            }
        }

        let (Q, Rfinal) = extractQFromQR(a: a, tau: tau, m: m, minDim: minDim, R: R, info: &info)
        return (Q, Rfinal)
    }

    /// Singular Value Decomposition.
    ///
    /// - Parameter m: Matrix (m×n)
    /// - Returns: (s, U, Vt) where U @ diag(s) @ Vt = m
    public static func svd(_ m: Matrix) -> (s: [Double], U: Matrix, Vt: Matrix) {
        let minDim = min(m.rows, m.cols)

        var a = toColumnMajor(m)

        var m1 = __CLPK_integer(m.rows)
        var n1 = __CLPK_integer(m.cols)
        var lda1 = __CLPK_integer(m.rows)
        var s = [Double](repeating: 0, count: minDim)
        var u = [Double](repeating: 0, count: m.rows * m.rows)
        var vt = [Double](repeating: 0, count: m.cols * m.cols)
        var ldu = __CLPK_integer(m.rows)
        var ldvt = __CLPK_integer(m.cols)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var jobu = Int8(UInt8(ascii: "A"))
        var jobvt = Int8(UInt8(ascii: "A"))

        dgesvd_(&jobu, &jobvt, &m1, &n1, &a, &lda1, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(m)
        var m2 = __CLPK_integer(m.rows)
        var n2 = __CLPK_integer(m.cols)
        var lda2 = __CLPK_integer(m.rows)
        dgesvd_(&jobu, &jobvt, &m2, &n2, &a, &lda2, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)

        var U = [Double](repeating: 0, count: m.rows * m.rows)
        for i in 0..<m.rows {
            for j in 0..<m.rows {
                U[i * m.rows + j] = u[j * m.rows + i]
            }
        }

        var Vt = [Double](repeating: 0, count: m.cols * m.cols)
        for i in 0..<m.cols {
            for j in 0..<m.cols {
                Vt[i * m.cols + j] = vt[j * m.cols + i]
            }
        }

        return (s, Matrix(rows: m.rows, cols: m.rows, data: U),
                Matrix(rows: m.cols, cols: m.cols, data: Vt))
    }

    /// Eigenvalue decomposition.
    ///
    /// - Parameter m: Square matrix
    /// - Returns: (eigenvalues, imagParts, eigenvectors)
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func eig(_ m: Matrix) throws -> (values: [Double], imagParts: [Double], vectors: Matrix) {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = toColumnMajor(m)

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var wr = [Double](repeating: 0, count: n)
        var wi = [Double](repeating: 0, count: n)
        var vl = [Double](repeating: 0, count: 1)
        var vr = [Double](repeating: 0, count: n * n)
        var ldvl: __CLPK_integer = 1
        var ldvr = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var jobvl = Int8(UInt8(ascii: "N"))
        var jobvr = Int8(UInt8(ascii: "V"))

        dgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(m)
        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)
        dgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        var vecs = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                vecs[i * n + j] = vr[j * n + i]
            }
        }

        return (wr, wi, Matrix(rows: n, cols: n, data: vecs))
    }

    /// Eigenvalues only (more efficient than full decomposition).
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func eigvals(_ m: Matrix) throws -> (real: [Double], imag: [Double]) {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = toColumnMajor(m)

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var wr = [Double](repeating: 0, count: n)
        var wi = [Double](repeating: 0, count: n)
        var vl = [Double](repeating: 0, count: 1)
        var vr = [Double](repeating: 0, count: 1)
        var ldvl: __CLPK_integer = 1
        var ldvr: __CLPK_integer = 1
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var jobvl = Int8(UInt8(ascii: "N"))
        var jobvr = Int8(UInt8(ascii: "N"))

        dgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(m)
        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)
        dgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        return (wr, wi)
    }

    /// Cholesky decomposition (for positive definite matrices).
    ///
    /// - Parameter m: Symmetric positive definite matrix
    /// - Returns: Lower triangular L where L @ L^T = m, or nil if not positive definite
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    ///   Returns `nil` (does not throw) when `m` is square but not positive definite.
    public static func cholesky(_ m: Matrix) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = toColumnMajor(m)

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var info: __CLPK_integer = 0
        var uplo = Int8(UInt8(ascii: "L"))

        dpotrf_(&uplo, &n1, &a, &lda1, &info)

        if info != 0 { return nil }

        var L = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0...i {
                L[i * n + j] = a[j * n + i]
            }
        }

        return Matrix(rows: n, cols: n, data: L)
    }

    // MARK: - QR helper

    /// Generate the explicit Q matrix from a packed LAPACK QR result.
    private static func extractQFromQR(
        a: [Double],
        tau: [Double],
        m: Matrix,
        minDim: Int,
        R: [Double],
        info: inout __CLPK_integer
    ) -> (Q: Matrix, R: Matrix) {
        var aCopy = a
        var m3 = __CLPK_integer(m.rows)
        var k1 = __CLPK_integer(minDim)
        var k2 = __CLPK_integer(minDim)
        var lda3 = __CLPK_integer(m.rows)
        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var tauCopy = tau

        dorgqr_(&m3, &k1, &k2, &aCopy, &lda3, &tauCopy, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))
        var m4 = __CLPK_integer(m.rows)
        var k3 = __CLPK_integer(minDim)
        var k4 = __CLPK_integer(minDim)
        var lda4 = __CLPK_integer(m.rows)
        dorgqr_(&m4, &k3, &k4, &aCopy, &lda4, &tauCopy, &work, &lwork, &info)

        var Q = [Double](repeating: 0, count: m.rows * minDim)
        for i in 0..<m.rows {
            for j in 0..<minDim {
                Q[i * minDim + j] = aCopy[j * m.rows + i]
            }
        }

        return (Matrix(rows: m.rows, cols: minDim, data: Q),
                Matrix(rows: minDim, cols: m.cols, data: R))
    }
}
