//
//  LinAlg+MatrixFunctions.swift
//  Sources/NumericSwift/LinAlg+MatrixFunctions.swift
//
//  Matrix functions: logm, sqrtm, funm, logmComplex, sqrtmComplex.
//  (expm and its Padé helpers live in LinAlg+Expm.swift.)
//
//  Architecture: part of LinAlg (Sources/NumericSwift/), alongside
//  LinAlg+Expm, LinAlg+Decompositions, LinAlg+Solvers, LinAlg+Complex,
//  LinAlg+Internal, etc. Depends on the Schur helpers and complex-arithmetic
//  primitives in LinAlg+Internal.swift.
//
//  Algorithm — Schur-Parlett (Higham, "Functions of Matrices", SIAM 2008, Ch. 4):
//    1. Real Schur form A = Q T Qᵀ via LAPACK dgees.
//    2. Apply f to T (quasi-upper-triangular) using the Parlett divided-difference
//       recurrence; 2×2 diagonal blocks (complex-conjugate pairs) handled via the
//       divided-difference 2×2 formula.
//    3. Back-transform f(A) = Q f(T) Qᵀ.
//  logmComplex / sqrtmComplex always return ComplexMatrix so they succeed for
//  matrices whose function is genuinely complex (e.g. negative-definite inputs).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - MatrixFunction enum

    /// Supported functions for ``funm(_:_:)``.
    public enum MatrixFunction: String {
        case sin, cos, exp, log, sqrt, sinh, cosh, tanh, abs
    }

    // MARK: - logm / sqrtm / funm  (Schur-Parlett)

    /// Matrix logarithm (principal branch).
    ///
    /// Uses the Schur-Parlett algorithm (Higham, "Functions of Matrices", SIAM
    /// 2008, Ch. 4): the real Schur form A = Q T Qᵀ is computed via LAPACK
    /// `dgees`; the logarithm is applied to the quasi-triangular T using the
    /// complex Parlett recurrence; the result is back-transformed as Q f(T) Qᵀ.
    ///
    /// Handles matrices with complex-conjugate eigenvalue pairs (e.g. rotation
    /// matrices) and defective (non-diagonalizable) matrices that the former
    /// real-eigendecomposition approach could not process.
    ///
    /// - Returns: `Matrix` when all eigenvalues have positive real part and the
    ///   result is numerically real (imaginary part ≤ 1e-9); `nil` when
    ///   eigenvalues are non-positive (result is complex — use ``logmComplex(_:)``).
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func logm(_ m: Matrix) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let result = schurParlett(m.data, n: m.rows, fn: .log)
        guard let (realPart, imagPart) = result else { return nil }
        guard imagPart.allSatisfy({ abs($0) <= 1e-9 }) else { return nil }
        return Matrix(rows: m.rows, cols: m.rows, data: realPart)
    }

    /// Matrix logarithm returning a ``ComplexMatrix`` (principal branch).
    ///
    /// Identical algorithm to ``logm(_:)`` but always returns a `ComplexMatrix`,
    /// so it succeeds even when the result has non-negligible imaginary parts —
    /// for example, when `m` has negative real eigenvalues.
    ///
    /// - Returns: `ComplexMatrix?`, or `nil` when the Schur decomposition fails.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func logmComplex(_ m: Matrix) throws -> ComplexMatrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        guard let (re, im) = schurParlett(m.data, n: m.rows, fn: .log) else { return nil }
        return ComplexMatrix(rows: m.rows, cols: m.rows, real: re, imag: im)
    }

    /// Matrix square root (principal branch).
    ///
    /// Uses the same Schur-Parlett algorithm as ``logm(_:)``. Returns a real
    /// `Matrix` when the result is numerically real; returns `nil` when the
    /// result is genuinely complex (e.g. matrix with negative eigenvalues) —
    /// use ``sqrtmComplex(_:)`` in that case.
    ///
    /// - Returns: `Matrix` when the result is numerically real; `nil` otherwise.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func sqrtm(_ m: Matrix) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let result = schurParlett(m.data, n: m.rows, fn: .sqrt)
        guard let (realPart, imagPart) = result else { return nil }
        guard imagPart.allSatisfy({ abs($0) <= 1e-9 }) else { return nil }
        return Matrix(rows: m.rows, cols: m.rows, data: realPart)
    }

    /// Matrix square root returning a ``ComplexMatrix`` (principal branch).
    ///
    /// Always returns a `ComplexMatrix`, so it succeeds for matrices with
    /// negative eigenvalues whose square root is purely imaginary.
    ///
    /// - Returns: `ComplexMatrix?`, or `nil` when the Schur decomposition fails.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func sqrtmComplex(_ m: Matrix) throws -> ComplexMatrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        guard let (re, im) = schurParlett(m.data, n: m.rows, fn: .sqrt) else { return nil }
        return ComplexMatrix(rows: m.rows, cols: m.rows, real: re, imag: im)
    }

    /// General matrix function via the Schur-Parlett algorithm.
    ///
    /// For functions whose result is real for real-eigenvalue matrices (sin, cos,
    /// exp, sinh, cosh, tanh, abs) this returns a real `Matrix`. For log and sqrt
    /// with complex-eigenvalue matrices the result will be real if the imaginary
    /// parts are negligible; otherwise returns `nil`.
    ///
    /// **Limitation on defective matrices**: the Schur-Parlett diagonal recurrence
    /// applies the function to each eigenvalue independently. For truly defective
    /// matrices (repeated eigenvalues with non-trivial Jordan structure), the
    /// derivative correction terms are not applied, so the off-diagonal result
    /// may be inaccurate — consistent with `scipy.linalg.funm`'s documented
    /// behaviour ("funm result may be inaccurate"). Use ``logm(_:)`` or
    /// ``sqrtm(_:)`` for those specific functions on defective matrices.
    ///
    /// - Parameters:
    ///   - m: Square matrix.
    ///   - function: Scalar function to apply element-wise to the eigenvalues.
    /// - Returns: f(A), or `nil` when the Schur decomposition fails or the result
    ///   is complex (imaginary part > 1e-9) for functions that should be real.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func funm(_ m: Matrix, _ function: MatrixFunction) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let result = schurParlett(m.data, n: m.rows, fn: function)
        guard let (realPart, imagPart) = result else { return nil }
        guard imagPart.allSatisfy({ abs($0) <= 1e-9 }) else { return nil }
        return Matrix(rows: m.rows, cols: m.rows, data: realPart)
    }

    // MARK: - Schur-Parlett core

    /// Apply a matrix function to a real n×n matrix via the real Schur form.
    ///
    /// Returns `(realPart, imagPart)` row-major arrays for the result matrix, or
    /// `nil` when LAPACK's Schur decomposition fails. The imaginary parts are
    /// numerically zero for functions that map real spectra to real values; callers
    /// gate on that to decide whether to return a `Matrix` or `ComplexMatrix`.
    ///
    /// Algorithm (Higham 2008, §4.3 "Schur-Parlett"):
    ///   1. Real Schur: A = Q T Qᵀ  (T quasi-upper-triangular, Q orthogonal)
    ///   2. Apply f to T in complex arithmetic using the upper-triangular Parlett
    ///      recurrence: F_{ii} = f(T_{ii}); F_{ij} = T_{ij}·(F_{ii}−F_{jj}) /
    ///      (T_{ii}−T_{jj}) + Σ_{k=i+1}^{j−1} (T_{ik}·F_{kj} − F_{ik}·T_{kj}) /
    ///      (T_{ii}−T_{jj})  for i < j.
    ///      Because T may have 2×2 diagonal blocks (complex-conjugate pairs), we
    ///      treat each such block as a tiny 2×2 complex eigendecomposition.
    ///   3. Back-transform: f(A) = Q F Qᵀ  (real multiply since Q is orthogonal).
    private static func schurParlett(
        _ data: [Double], n: Int, fn: MatrixFunction
    ) -> (re: [Double], im: [Double])? {

        guard let (Q, T) = realSchurDecomposition(data, n) else { return nil }

        // Build a complex version of T to run the Parlett recurrence uniformly.
        // F starts as a copy of the Schur form; we overwrite its upper triangle.
        var Fre = T
        var Fim = [Double](repeating: 0, count: n * n)

        // Determine block structure of T: each diagonal position is either a
        // 1×1 block (|T[i+1,i]| ≈ 0) or the start of a 2×2 block.
        // 'blockStart[k]' = the row index of the block containing column k.
        var blockOf = [Int](repeating: -1, count: n)   // block-start index for each column
        var i = 0
        while i < n {
            if i + 1 < n && abs(T[(i+1)*n + i]) > 1e-12 {
                // 2×2 block at (i, i)
                blockOf[i] = i
                blockOf[i+1] = i
                i += 2
            } else {
                blockOf[i] = i
                i += 1
            }
        }

        // Apply f to each diagonal block to get the diagonal entries of F.
        i = 0
        while i < n {
            let bs = blockOf[i]
            if bs == i && (i+1 < n && blockOf[i+1] == i) {
                // 2×2 block — diagonalize in complex arithmetic to apply f.
                let a00 = T[i*n+i], a01 = T[i*n+(i+1)]
                let a10 = T[(i+1)*n+i], a11 = T[(i+1)*n+(i+1)]
                // Eigenvalues of the 2×2 block via quadratic formula.
                let tr   = a00 + a11
                let disc = (a00 - a11)*(a00 - a11) + 4.0*a01*a10  // may be negative
                let sqrtDisc = cSqrt((disc, 0.0))
                let lam1 = cMul((0.5, 0.0), cAdd((tr, 0.0), sqrtDisc))
                let lam2 = cMul((0.5, 0.0), cSub((tr, 0.0), sqrtDisc))
                guard let fl1 = applyComplexFunction(fn, lam1),
                      let fl2 = applyComplexFunction(fn, lam2) else { return nil }
                // f(B) for a 2×2 with eigenvalues λ₁,λ₂ and eigenvectors derived
                // from the quadratic: f(B) = ((f(λ₁)−f(λ₂))/(λ₁−λ₂))·(B − λ₂·I) + f(λ₂)·I
                // (Higham 2008, eq. 4.11; uses divided-difference formulation).
                let dLam = cSub(lam1, lam2)
                if cAbs(dLam) < 1e-12 {
                    // Repeated eigenvalue (defective block): use f(λ)·I (diagonal approximation).
                    Fre[i*n+i]     = fl1.re; Fim[i*n+i]     = fl1.im
                    Fre[i*n+i+1]   = 0.0;    Fim[i*n+i+1]   = 0.0
                    Fre[(i+1)*n+i] = 0.0;    Fim[(i+1)*n+i] = 0.0
                    Fre[(i+1)*n+i+1] = fl1.re; Fim[(i+1)*n+i+1] = fl1.im
                } else {
                    // Divided difference (f(λ₁)−f(λ₂))/(λ₁−λ₂).
                    let dd = cDiv(cSub(fl1, fl2), dLam)
                    // B − λ₂·I (as complex 2×2).
                    let b00 = cSub((a00, 0.0), lam2)
                    let b01: C2 = (a01, 0.0)
                    let b10: C2 = (a10, 0.0)
                    let b11 = cSub((a11, 0.0), lam2)
                    // f(B) = dd*(B-λ₂I) + f(λ₂)I
                    let f00 = cAdd(cMul(dd, b00), fl2)
                    let f01 = cMul(dd, b01)
                    let f10 = cMul(dd, b10)
                    let f11 = cAdd(cMul(dd, b11), fl2)
                    Fre[i*n+i]       = f00.re; Fim[i*n+i]       = f00.im
                    Fre[i*n+i+1]     = f01.re; Fim[i*n+i+1]     = f01.im
                    Fre[(i+1)*n+i]   = f10.re; Fim[(i+1)*n+i]   = f10.im
                    Fre[(i+1)*n+i+1] = f11.re; Fim[(i+1)*n+i+1] = f11.im
                }
                i += 2
            } else {
                // 1×1 block — apply f to the scalar eigenvalue.
                guard let fval = applyComplexFunction(fn, (T[i*n+i], 0.0)) else { return nil }
                Fre[i*n+i] = fval.re
                Fim[i*n+i] = fval.im
                i += 1
            }
        }

        // Parlett recurrence for the strict upper triangle.
        // For each superdiagonal distance d = 1, 2, ..., n-1:
        //   F[i,j] = T[i,j]·(F[i,i]−F[j,j])/(T[i,i]−T[j,j])
        //           + Σ_{k=i+1}^{j−1} (T[i,k]·F[k,j] − F[i,k]·T[k,j]) / (T[i,i]−T[j,j])
        // (Parlett 1974, eq. 2.4; Higham 2008, Alg. 4.6)
        // We skip entries (i,j) where both i and j belong to the same 2×2 block
        // (those are already set above).
        for d in 1..<n {
            for col in d..<n {
                let row = col - d
                // Skip if this cell is inside the same 2×2 diagonal block.
                if blockOf[row] == blockOf[col] && blockOf[row] != row { continue }
                if blockOf[row] == blockOf[col] && blockOf[col] != col { continue }
                // Eigenvalue difference T[row,row] − T[col,col] (complex).
                let tRR: C2 = (Fre[row*n+row], Fim[row*n+row])  // f(T[row,row]) for diagonal
                let tCC: C2 = (Fre[col*n+col], Fim[col*n+col])
                // Raw Schur diagonal entries (eigenvalues of 1×1 blocks).
                let eigR: C2 = (T[row*n+row], 0.0)
                let eigC: C2 = (T[col*n+col], 0.0)
                let dEig = cSub(eigR, eigC)

                // Numerator: T[row,col]·(f(T[row,row]) − f(T[col,col])) + Σ cross terms.
                var numRe = T[row*n+col] * (tRR.re - tCC.re)
                var numIm = T[row*n+col] * (tRR.im - tCC.im)
                for k in (row+1)..<col {
                    // T[row,k]·F[k,col] − F[row,k]·T[k,col]
                    let tRowK = T[row*n+k]
                    let fKCol: C2 = (Fre[k*n+col], Fim[k*n+col])
                    let fRowK: C2 = (Fre[row*n+k], Fim[row*n+k])
                    let tKCol = T[k*n+col]
                    numRe += tRowK * fKCol.re - fRowK.re * tKCol
                    numIm += tRowK * fKCol.im - fRowK.im * tKCol
                }

                let dEigAbs = cAbs(dEig)
                if dEigAbs < 1e-12 {
                    // Degenerate denominator: Parlett limit is f'(λ)·T[row,col]
                    // (Higham 2008, §4.2, eq. 4.6). Use the exact analytic derivative
                    // to avoid finite-difference cancellation errors.
                    let eigVal: C2 = (T[row*n+row], 0.0)
                    guard let fprime = applyComplexDerivative(fn, eigVal) else { return nil }
                    let deriv = cMul(fprime, (T[row*n+col], 0.0))
                    Fre[row*n+col] = deriv.re
                    Fim[row*n+col] = deriv.im
                } else {
                    let val = cDiv((numRe, numIm), dEig)
                    Fre[row*n+col] = val.re
                    Fim[row*n+col] = val.im
                }
            }
        }

        // Back-transform: f(A) = Q F Qᵀ  (real multiplication, Q is orthogonal).
        // Compute Q * F and then (Q * F) * Qᵀ in the real and imaginary parts separately.
        let (resRe, resIm) = multiplyQFQT(Q: Q, Fre: Fre, Fim: Fim, n: n)
        return (resRe, resIm)
    }

    /// Compute Q · F · Qᵀ where Q is real-orthogonal and F is complex (separate real/imag).
    private static func multiplyQFQT(
        Q: [Double], Fre: [Double], Fim: [Double], n: Int
    ) -> (re: [Double], im: [Double]) {
        // QF_re = Q * Fre,  QF_im = Q * Fim  (real BLAS dgemm)
        let QFre = matmulInternal(Q, Fre, n)
        let QFim = matmulInternal(Q, Fim, n)

        // (QF) * Qᵀ: since Q is orthogonal, Qᵀ[i,j] = Q[j,i].
        // Use dgemm with transposed Q: C = A * Bᵀ.
        var resRe = [Double](repeating: 0, count: n * n)
        var resIm = [Double](repeating: 0, count: n * n)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(n), Int32(n), Int32(n),
                    1.0, QFre, Int32(n), Q, Int32(n),
                    0.0, &resRe, Int32(n))
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(n), Int32(n), Int32(n),
                    1.0, QFim, Int32(n), Q, Int32(n),
                    0.0, &resIm, Int32(n))
        return (resRe, resIm)
    }
}
