//
//  LinAlg+Internal.swift
//  NumericSwift
//
//  Internal (cross-file) helpers for LinAlg.
//  These were formerly private to LinAlg.swift; promoting them to internal
//  is required so that LinAlg+Decompositions, LinAlg+Solvers,
//  LinAlg+Properties, LinAlg+MatrixFunctions, and LinAlg+Complex can all
//  reach them after the file is split.  They are NOT public — the public API
//  surface is unchanged.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Layout Conversion Helpers

    /// Convert row-major Matrix to column-major flat array (LAPACK input format).
    static func toColumnMajor(_ m: Matrix) -> [Double] {
        var result = [Double](repeating: 0, count: m.size)
        for i in 0..<m.rows {
            for j in 0..<m.cols {
                result[j * m.rows + i] = m.data[i * m.cols + j]
            }
        }
        return result
    }

    /// Convert column-major flat array back to a row-major Matrix.
    static func fromColumnMajor(_ data: [Double], rows: Int, cols: Int) -> Matrix {
        var result = [Double](repeating: 0, count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                result[i * cols + j] = data[j * rows + i]
            }
        }
        return Matrix(rows: rows, cols: cols, data: result)
    }

    // MARK: - Low-Level Matrix Arithmetic

    /// Internal matrix multiplication for two n×n flat row-major arrays via BLAS.
    static func matmulInternal(_ A: [Double], _ B: [Double], _ n: Int) -> [Double] {
        var C = [Double](repeating: 0, count: n * n)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(n), Int32(n), Int32(n),
                    1.0, A, Int32(n), B, Int32(n),
                    0.0, &C, Int32(n))
        return C
    }

    /// Internal linear system solver for two n×n row-major arrays via LAPACK dgesv.
    ///
    /// Solves A * X = B and returns X (or the input B unchanged on failure).
    static func solveLinearSystemInternal(_ A: [Double], _ B: [Double], _ n: Int) -> [Double] {
        var a = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                a[j * n + i] = A[i * n + j]
            }
        }

        var b = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                b[j * n + i] = B[i * n + j]
            }
        }

        var n1 = __CLPK_integer(n)
        var nrhs = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var ldb = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        dgesv_(&n1, &nrhs, &a, &lda, &ipiv, &b, &ldb, &info)

        var result = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                result[i * n + j] = b[j * n + i]
            }
        }

        return result
    }

    // MARK: - Eigendecomposition Helpers

    /// Compute a real eigendecomposition via LAPACK dgeev.
    ///
    /// Returns `(eigenvalues, eigenvectors)` in row-major order, or `nil` when
    /// LAPACK fails or the matrix has non-real eigenvalues (imaginary part > 1e-10).
    static func computeRealEigendecomposition(
        _ data: [Double], _ n: Int
    ) -> (eigenvalues: [Double], eigenvectors: [Double])? {
        // Convert row-major input to column-major for LAPACK.
        var a = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n { a[j * n + i] = data[i * n + j] }
        }

        guard let (wr, wi, vr) = dgeevCall(&a, n: n) else { return nil }

        // Reject matrices with non-real eigenvalues (imaginary part > 1e-10).
        for im in wi { if abs(im) > 1e-10 { return nil } }

        // Transpose eigenvectors from LAPACK column-major to row-major.
        var vecs = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n { vecs[i * n + j] = vr[j * n + i] }
        }

        return (wr, vecs)
    }

    /// Call LAPACK `dgeev` with a two-phase workspace query then compute.
    ///
    /// Phase 1 queries the optimal LWORK size; phase 2 performs the actual
    /// eigendecomposition.  `a` is a column-major copy of the input matrix — dgeev_
    /// may overwrite it during the workspace query, so we snapshot and restore it
    /// before the actual compute call (same defensive pattern as the original code).
    /// Returns `(wr, wi, vr)` — real eigenvalues, imaginary parts, and right
    /// eigenvectors in column-major order — or `nil` on LAPACK failure.
    private static func dgeevCall(
        _ a: inout [Double], n: Int
    ) -> (wr: [Double], wi: [Double], vr: [Double])? {
        let snapshot = a                                // save before workspace query
        var jobvl = Int8(UInt8(ascii: "N"))
        var jobvr = Int8(UInt8(ascii: "V"))
        var n1 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var wr = [Double](repeating: 0, count: n)
        var wi = [Double](repeating: 0, count: n)
        var vl = [Double](repeating: 0, count: 1)
        var vr = [Double](repeating: 0, count: n * n)
        var ldvl: __CLPK_integer = 1
        var ldvr = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        // Phase 1: workspace size query (lwork = -1).
        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        dgeev_(&jobvl, &jobvr, &n1, &a, &lda,
               &wr, &wi, &vl, &ldvl, &vr, &ldvr,
               &work, &lwork, &info)
        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        // Phase 2: actual eigendecomposition — restore the matrix first.
        a = snapshot
        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)
        dgeev_(&jobvl, &jobvr, &n2, &a, &lda2,
               &wr, &wi, &vl, &ldvl, &vr, &ldvr,
               &work, &lwork, &info)

        if info != 0 { return nil }
        return (wr, wi, vr)
    }

    /// Reconstruct a matrix from eigendecomposition: result = V * diag(λ) * V⁻¹.
    static func reconstructFromEigen(
        _ V: [Double], _ eigenvalues: [Double], _ n: Int
    ) -> [Double] {
        let Vinv = invertMatrixInternal(V, n)

        var VD = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                VD[i * n + j] = V[i * n + j] * eigenvalues[j]
            }
        }

        return matmulInternal(VD, Vinv, n)
    }

    /// Invert an n×n row-major matrix using LAPACK LU decomposition.
    ///
    /// Returns the original matrix unchanged on failure (singular input).
    static func invertMatrixInternal(_ M: [Double], _ n: Int) -> [Double] {
        var a = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                a[j * n + i] = M[i * n + j]
            }
        }

        var n1 = __CLPK_integer(n)
        var n1b = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&n1, &n1b, &a, &lda, &ipiv, &info)

        if info != 0 { return M }

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)

        dgetri_(&n2, &a, &lda2, &ipiv, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        var n3 = __CLPK_integer(n)
        var lda3 = __CLPK_integer(n)
        dgetri_(&n3, &a, &lda3, &ipiv, &work, &lwork, &info)

        var result = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                result[i * n + j] = a[j * n + i]
            }
        }

        return result
    }
}
