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
    /// Solves `A · X = B` and returns `X`, or `nil` when LAPACK reports a failure
    /// (`info != 0` — a singular `A` or an illegal argument). Returning `nil`
    /// rather than the half-overwritten right-hand side lets the caller surface a
    /// real error (CLAUDE.md rule 10: treat failures as errors, never as garbage).
    static func solveLinearSystemInternal(_ A: [Double], _ B: [Double], _ n: Int) -> [Double]? {
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

        // info > 0: U(info,info) is exactly zero → A is singular. info < 0: the
        // info-th argument was illegal. Either way the solve did not produce a
        // valid X; signal failure instead of returning the clobbered buffer.
        guard info == 0 else { return nil }

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

    // MARK: - Schur Decomposition (real, via LAPACK dgees)

    /// Real Schur decomposition A = Q T Q^T via LAPACK dgees.
    ///
    /// Returns `(Q, T)` — orthogonal Q and quasi-upper-triangular real Schur form T —
    /// where columns of Q are the Schur vectors and diagonal blocks of T are either
    /// 1×1 (real eigenvalue) or 2×2 (complex-conjugate pair). Both Q and T are
    /// row-major n×n arrays. Returns `nil` when LAPACK reports failure.
    ///
    /// Reference: LAPACK Users' Guide, 3rd ed., §2.4 "Real Schur Form".
    static func realSchurDecomposition(
        _ data: [Double], _ n: Int
    ) -> (Q: [Double], T: [Double])? {
        // dgees expects column-major input.
        var a = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n { a[j * n + i] = data[i * n + j] }
        }

        var jobvs = Int8(UInt8(ascii: "V"))  // compute Schur vectors
        var sort  = Int8(UInt8(ascii: "N"))  // no eigenvalue ordering
        var n1    = __CLPK_integer(n)
        var lda   = __CLPK_integer(n)
        var sdim: __CLPK_integer = 0
        var wr    = [Double](repeating: 0, count: n)
        var wi    = [Double](repeating: 0, count: n)
        var ldvs  = __CLPK_integer(n)
        var vs    = [Double](repeating: 0, count: n * n)  // Schur vectors (column-major)
        var info: __CLPK_integer = 0

        // Phase 1: workspace size query.
        var lwork: __CLPK_integer = -1
        var work  = [Double](repeating: 0, count: 1)
        // selectf is not used (sort = N) — pass nil.
        dgees_(&jobvs, &sort, nil, &n1, &a, &lda, &sdim,
               &wr, &wi, &vs, &ldvs, &work, &lwork, nil, &info)
        lwork = __CLPK_integer(work[0])
        work  = [Double](repeating: 0, count: Int(lwork))

        // Phase 2: actual decomposition — restore column-major a first.
        for i in 0..<n {
            for j in 0..<n { a[j * n + i] = data[i * n + j] }
        }
        var n2  = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)
        dgees_(&jobvs, &sort, nil, &n2, &a, &lda2, &sdim,
               &wr, &wi, &vs, &ldvs, &work, &lwork, nil, &info)

        if info != 0 { return nil }

        // Convert Q (column-major vs → row-major Q) and T (column-major a → row-major T).
        var Q = [Double](repeating: 0, count: n * n)
        var T = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                Q[i * n + j] = vs[j * n + i]
                T[i * n + j] =  a[j * n + i]
            }
        }
        return (Q, T)
    }

    // MARK: - Complex arithmetic helpers for Schur-Parlett

    // Swift tuples (re, im): Double complex numbers used in Schur-Parlett recurrence.
    // These are module-internal helpers (no explicit access modifier → `internal`):
    // LinAlg+MatrixFunctions.swift calls them across file boundaries, so `fileprivate`
    // would not compile.  They are intentionally unexported — callers outside this
    // module should use the public Complex type.

    typealias C2 = (re: Double, im: Double)

    @inline(__always) static func cAdd(_ a: C2, _ b: C2) -> C2 { (a.re+b.re, a.im+b.im) }
    @inline(__always) static func cSub(_ a: C2, _ b: C2) -> C2 { (a.re-b.re, a.im-b.im) }
    @inline(__always) static func cMul(_ a: C2, _ b: C2) -> C2 {
        (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)
    }
    @inline(__always) static func cDiv(_ a: C2, _ b: C2) -> C2 {
        let denom = b.re*b.re + b.im*b.im
        return ((a.re*b.re + a.im*b.im) / denom, (a.im*b.re - a.re*b.im) / denom)
    }
    @inline(__always) static func cAbs(_ a: C2) -> Double { sqrt(a.re*a.re + a.im*a.im) }

    /// Complex logarithm (principal branch): log(re + i·im).
    @inline(__always) static func cLog(_ a: C2) -> C2 {
        (log(cAbs(a)), atan2(a.im, a.re))
    }

    /// Complex square root (principal branch): sqrt(re + i·im).
    @inline(__always) static func cSqrt(_ a: C2) -> C2 {
        let r     = cAbs(a)
        let theta = atan2(a.im, a.re)
        let sr    = sqrt(r)
        return (sr * cos(theta / 2), sr * sin(theta / 2))
    }

    /// Apply a scalar complex function to a complex number.
    @inline(__always) static func applyComplexFunction(
        _ fn: LinAlg.MatrixFunction, _ z: C2
    ) -> C2? {
        switch fn {
        case .exp:  return (exp(z.re) * cos(z.im), exp(z.re) * sin(z.im))
        case .log:  return cLog(z)
        case .sqrt: return cSqrt(z)
        case .sin:
            // sin(x+iy) = sin(x)cosh(y) + i cos(x)sinh(y)
            return (sin(z.re)*cosh(z.im), cos(z.re)*sinh(z.im))
        case .cos:
            return (cos(z.re)*cosh(z.im), -sin(z.re)*sinh(z.im))
        case .sinh:
            return (sinh(z.re)*cos(z.im), cosh(z.re)*sin(z.im))
        case .cosh:
            return (cosh(z.re)*cos(z.im), sinh(z.re)*sin(z.im))
        case .tanh:
            let num = (sinh(z.re)*cos(z.im), cosh(z.re)*sin(z.im))
            let den = (cosh(z.re)*cos(z.im), sinh(z.re)*sin(z.im))
            return cDiv(num, den)
        case .abs:
            return (cAbs(z), 0.0)
        }
    }

    /// First derivative f'(z) for the Parlett repeated-eigenvalue limit case.
    ///
    /// When two diagonal Schur eigenvalues coincide, the divided-difference in
    /// the Parlett recurrence has a 0/0 form that resolves to the limit f'(λ)
    /// (Higham 2008, "Functions of Matrices" SIAM, §4.2, eq. 4.6). Exact analytic
    /// derivatives avoid the cancellation errors of finite-difference approximations.
    @inline(__always) static func applyComplexDerivative(
        _ fn: LinAlg.MatrixFunction, _ z: C2
    ) -> C2? {
        switch fn {
        case .exp:  return applyComplexFunction(.exp, z)           // (e^z)' = e^z
        case .log:  return cDiv((1.0, 0.0), z)                     // (ln z)' = 1/z
        case .sqrt:
            // (√z)' = 1/(2√z)
            let sq = cSqrt(z)
            return cDiv((1.0, 0.0), cMul((2.0, 0.0), sq))
        case .sin:
            return applyComplexFunction(.cos, z)                   // (sin z)' = cos z
        case .cos:
            guard let sv = applyComplexFunction(.sin, z) else { return nil }
            return (-sv.re, -sv.im)                                // (cos z)' = -sin z
        case .sinh:
            return applyComplexFunction(.cosh, z)                  // (sinh z)' = cosh z
        case .cosh:
            return applyComplexFunction(.sinh, z)                  // (cosh z)' = sinh z
        case .tanh:
            // (tanh z)' = sech²z = 1 - tanh²z
            guard let tv = applyComplexFunction(.tanh, z) else { return nil }
            return cSub((1.0, 0.0), cMul(tv, tv))
        case .abs:
            // (|z|)' = z/|z| for non-zero z (not holomorphic; real sign for real axis)
            let r = cAbs(z)
            guard r > 1e-300 else { return nil }
            return (z.re / r, z.im / r)
        }
    }

    /// Invert an n×n row-major matrix using LAPACK LU decomposition (dgetrf + dgetri).
    ///
    /// Returns the original matrix unchanged on failure (singular input).
    ///
    /// LAPACK two-phase mutable-pointer workaround: LAPACK routines require separate
    /// `__CLPK_integer` variables even when the same logical value (e.g. `n`) appears
    /// in multiple positions, because each parameter is passed as a distinct `inout`
    /// pointer.  Swift will not pass the same var twice as distinct `inout` bindings,
    /// so we declare separate variables (`rowsLU`, `colsLU`, `ldaLU`, `rowsInv`,
    /// `ldaInv`, `rowsInvFinal`, `ldaInvFinal`) mirroring the same pattern used in
    /// `dgeevCall` above.
    static func invertMatrixInternal(_ M: [Double], _ n: Int) -> [Double] {
        // Convert row-major input to the column-major layout expected by LAPACK.
        var a = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n { a[j * n + i] = M[i * n + j] }
        }

        // Phase 1: LU factorisation via dgetrf.
        var rowsLU = __CLPK_integer(n)
        var colsLU = __CLPK_integer(n)
        var ldaLU  = __CLPK_integer(n)
        var ipiv   = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&rowsLU, &colsLU, &a, &ldaLU, &ipiv, &info)
        if info != 0 { return M }

        // Phase 2a: workspace query for dgetri (lwork = -1).
        var lwork: __CLPK_integer = -1
        var work   = [Double](repeating: 0, count: 1)
        var rowsInv = __CLPK_integer(n)
        var ldaInv  = __CLPK_integer(n)
        dgetri_(&rowsInv, &a, &ldaInv, &ipiv, &work, &lwork, &info)
        lwork = __CLPK_integer(work[0])
        work  = [Double](repeating: 0, count: Int(lwork))

        // Phase 2b: actual inversion.
        var rowsInvFinal = __CLPK_integer(n)
        var ldaInvFinal  = __CLPK_integer(n)
        dgetri_(&rowsInvFinal, &a, &ldaInvFinal, &ipiv, &work, &lwork, &info)

        // Convert column-major result back to row-major.
        var result = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n { result[i * n + j] = a[j * n + i] }
        }
        return result
    }
}
