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
    /// Uses the block Schur-Parlett algorithm (Higham, "Functions of Matrices",
    /// SIAM 2008, Ch. 4): the real Schur form A = Q T Qᵀ is computed via LAPACK
    /// `dgees`; the logarithm is applied to T block-by-block using the complex
    /// Parlett recurrence; the result is back-transformed as Q f(T) Qᵀ.
    ///
    /// Handles matrices with complex-conjugate eigenvalue pairs (e.g. rotation
    /// matrices) and non-diagonalizable matrices.
    ///
    /// - Returns: `Matrix` when the result is numerically real (imaginary parts
    ///   ≤ 1e-9 element-wise); `nil` for **either** of these two reasons:
    ///   1. The Schur decomposition failed (LAPACK reported an error).
    ///   2. The result is genuinely complex (e.g. matrix has eigenvalues with
    ///      non-positive real part) — call ``logmComplex(_:)`` instead.
    ///   The two cases are indistinguishable from a `nil` return; if the matrix
    ///   is expected to have complex logarithm, always prefer `logmComplex`.
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
    /// for example, when `m` has eigenvalues with non-positive real part.
    ///
    /// - Returns: `ComplexMatrix` on success; `nil` **only** when the Schur
    ///   decomposition itself fails (LAPACK error, e.g. non-convergence).
    ///   A complex result is never a reason for `nil` here — use this overload
    ///   whenever ``logm(_:)`` returns `nil` and the matrix is well-conditioned.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func logmComplex(_ m: Matrix) throws -> ComplexMatrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        guard let (re, im) = schurParlett(m.data, n: m.rows, fn: .log) else { return nil }
        return ComplexMatrix(rows: m.rows, cols: m.rows, real: re, imag: im)
    }

    /// Matrix square root (principal branch).
    ///
    /// Uses the same block Schur-Parlett algorithm as ``logm(_:)``. Returns a
    /// real `Matrix` when the result is numerically real.
    ///
    /// - Returns: `Matrix` when the result is numerically real (imaginary parts
    ///   ≤ 1e-9); `nil` for **either** of these two reasons:
    ///   1. The Schur decomposition failed (LAPACK error).
    ///   2. The result is genuinely complex (e.g. matrix has negative
    ///      eigenvalues) — call ``sqrtmComplex(_:)`` instead.
    ///   Callers that expect a complex result should always use `sqrtmComplex`
    ///   directly rather than interpreting a `nil` return.
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
    /// - Returns: `ComplexMatrix` on success; `nil` **only** when the Schur
    ///   decomposition itself fails (LAPACK error).  A complex result is never
    ///   a reason for `nil` here — prefer this overload over ``sqrtm(_:)`` when
    ///   negative or complex eigenvalues are possible.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func sqrtmComplex(_ m: Matrix) throws -> ComplexMatrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        guard let (re, im) = schurParlett(m.data, n: m.rows, fn: .sqrt) else { return nil }
        return ComplexMatrix(rows: m.rows, cols: m.rows, real: re, imag: im)
    }

    /// General matrix function via the block Schur-Parlett algorithm.
    ///
    /// For functions whose result is real on real spectra (sin, cos, exp, sinh,
    /// cosh, tanh, abs) this returns a real `Matrix`.  For log and sqrt, the
    /// result is real when imaginary parts are negligible.
    ///
    /// **Limitation on defective matrices**: repeated eigenvalues with non-trivial
    /// Jordan structure are approximated by the diagonal term only (derivative
    /// correction not applied) — consistent with `scipy.linalg.funm`'s documented
    /// behaviour.  Use ``logm(_:)`` or ``sqrtm(_:)`` for those specific functions.
    ///
    /// - Parameters:
    ///   - m: Square matrix.
    ///   - function: Scalar function to apply via the Schur-Parlett recurrence.
    /// - Returns: f(A), or `nil` for **either** of these two reasons:
    ///   1. The Schur decomposition failed (LAPACK error).
    ///   2. The result is genuinely complex (imaginary part > 1e-9) — for log
    ///      or sqrt, consider ``logmComplex(_:)`` / ``sqrtmComplex(_:)``.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func funm(_ m: Matrix, _ function: MatrixFunction) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let result = schurParlett(m.data, n: m.rows, fn: function)
        guard let (realPart, imagPart) = result else { return nil }
        guard imagPart.allSatisfy({ abs($0) <= 1e-9 }) else { return nil }
        return Matrix(rows: m.rows, cols: m.rows, data: realPart)
    }

    // MARK: - Schur-Parlett core

    /// Apply a matrix function to a real n×n matrix via the block Schur-Parlett algorithm.
    ///
    /// Returns `(realPart, imagPart)` row-major flat arrays for the result matrix, or
    /// `nil` when LAPACK's Schur decomposition fails.  The imaginary parts are
    /// numerically zero for functions that map real spectra to real values (e.g. sin,
    /// exp); callers gate on the imaginary magnitude to decide whether to return a
    /// `Matrix` or `ComplexMatrix`.
    ///
    /// Algorithm (Higham 2008 "Functions of Matrices", SIAM, §4.3 "Schur-Parlett"):
    ///   1. Real Schur form A = Q T Qᵀ via LAPACK `dgees`
    ///      (T quasi-upper-triangular; diagonal blocks are 1×1 (real eigenvalue)
    ///      or 2×2 (complex-conjugate pair)).
    ///   2. Detect block structure (``schurBlockStructure``).
    ///   3. Apply f to each diagonal block in complex arithmetic
    ///      (``applyFunctionToDiagonalBlocks``).
    ///   4. Fill the strict upper triangle block-by-block via the Parlett recurrence,
    ///      solving a 1×1 scalar equation, a 2×1 or 1×2 Sylvester equation, or a
    ///      4×4 vectorised Sylvester depending on the involved block sizes
    ///      (``parlettUpperTriangle``).
    ///   5. Back-transform f(A) = Q F Qᵀ (``multiplyQFQT``).
    private static func schurParlett(
        _ data: [Double], n: Int, fn: MatrixFunction
    ) -> (re: [Double], im: [Double])? {
        guard let (Q, T) = realSchurDecomposition(data, n) else { return nil }
        let blockOf = schurBlockStructure(T: T, n: n)
        var Fre = T
        var Fim = [Double](repeating: 0, count: n * n)
        guard applyFunctionToDiagonalBlocks(fn: fn, T: T, n: n, blockOf: blockOf,
                                            Fre: &Fre, Fim: &Fim) else { return nil }
        guard parlettUpperTriangle(fn: fn, T: T, n: n, blockOf: blockOf,
                                   Fre: &Fre, Fim: &Fim) else { return nil }
        return multiplyQFQT(Q: Q, Fre: Fre, Fim: Fim, n: n)
    }

    /// Map each column index to the row-index of its diagonal block start.
    ///
    /// A sub-diagonal entry |T[i+1,i]| > 1e-12 marks a 2×2 block at rows i,i+1;
    /// otherwise the block at column i is the 1×1 scalar T[i,i].
    private static func schurBlockStructure(T: [Double], n: Int) -> [Int] {
        var blockOf = [Int](repeating: -1, count: n)
        var i = 0
        while i < n {
            if i + 1 < n && abs(T[(i+1)*n + i]) > 1e-12 {
                blockOf[i] = i; blockOf[i+1] = i; i += 2
            } else {
                blockOf[i] = i; i += 1
            }
        }
        return blockOf
    }

    /// Apply f to each diagonal block of the Schur form T, writing results into F.
    ///
    /// - 1×1 block: F[i,i] = f(T[i,i]) directly.
    /// - 2×2 block: eigenvalues computed via the quadratic formula; then
    ///   f(B) = dd·(B−λ₂I) + f(λ₂)I  where dd = (f(λ₁)−f(λ₂))/(λ₁−λ₂)
    ///   (Higham 2008, eq. 4.11). Returns `false` when f is undefined on an eigenvalue.
    private static func applyFunctionToDiagonalBlocks(
        fn: MatrixFunction, T: [Double], n: Int, blockOf: [Int],
        Fre: inout [Double], Fim: inout [Double]
    ) -> Bool {
        var i = 0
        while i < n {
            if i+1 < n && blockOf[i+1] == i {
                let a00 = T[i*n+i], a01 = T[i*n+i+1]
                let a10 = T[(i+1)*n+i], a11 = T[(i+1)*n+i+1]
                let disc = (a00-a11)*(a00-a11) + 4.0*a01*a10
                let sqrtDisc = cSqrt((disc, 0.0))
                let lam1 = cMul((0.5, 0.0), cAdd((a00+a11, 0.0), sqrtDisc))
                let lam2 = cMul((0.5, 0.0), cSub((a00+a11, 0.0), sqrtDisc))
                guard let fl1 = applyComplexFunction(fn, lam1),
                      let fl2 = applyComplexFunction(fn, lam2) else { return false }
                let dLam = cSub(lam1, lam2)
                if cAbs(dLam) < 1e-12 {
                    // Repeated (defective) block: diagonal approximation f(λ)·I.
                    Fre[i*n+i] = fl1.re;   Fim[i*n+i] = fl1.im
                    Fre[i*n+i+1] = 0.0;   Fim[i*n+i+1] = 0.0
                    Fre[(i+1)*n+i] = 0.0; Fim[(i+1)*n+i] = 0.0
                    Fre[(i+1)*n+i+1] = fl1.re; Fim[(i+1)*n+i+1] = fl1.im
                } else {
                    let dd = cDiv(cSub(fl1, fl2), dLam)
                    let f00 = cAdd(cMul(dd, cSub((a00, 0.0), lam2)), fl2)
                    let f01 = cMul(dd, (a01, 0.0))
                    let f10 = cMul(dd, (a10, 0.0))
                    let f11 = cAdd(cMul(dd, cSub((a11, 0.0), lam2)), fl2)
                    Fre[i*n+i] = f00.re;         Fim[i*n+i] = f00.im
                    Fre[i*n+i+1] = f01.re;       Fim[i*n+i+1] = f01.im
                    Fre[(i+1)*n+i] = f10.re;     Fim[(i+1)*n+i] = f10.im
                    Fre[(i+1)*n+i+1] = f11.re;   Fim[(i+1)*n+i+1] = f11.im
                }
                i += 2
            } else {
                guard let fval = applyComplexFunction(fn, (T[i*n+i], 0.0)) else { return false }
                Fre[i*n+i] = fval.re; Fim[i*n+i] = fval.im
                i += 1
            }
        }
        return true
    }

    /// Parlett recurrence for the strict upper triangle of F.
    ///
    /// Implements Higham (2008) "Functions of Matrices" SIAM, Algorithm 4.6
    /// (block Schur-Parlett).  Because the real Schur form T may contain 2×2
    /// diagonal blocks (complex-conjugate eigenvalue pairs a±bi), entries must
    /// be processed in block units:
    ///
    ///   • 1×1 vs 1×1 cross-term: scalar Parlett formula
    ///       F[i,j] = num / (λ_i − λ_j)
    ///     where λ_i = T[i,i] (a true scalar eigenvalue).
    ///
    ///   • 2×2 block_i vs 1×1 block_j (or vice-versa): solve the Sylvester equation
    ///       (T_ii − λ_j·I) · F_ij = RHS           (2×1 linear system)
    ///     where RHS = F_ii·T_ij − T_ij·F_jj + accumulated cross-block terms.
    ///     Using the scalar eigenvalue λ_j (real, from the 1×1 block) and the full
    ///     complex 2×2 F_ii (already computed in the diagonal pass), this avoids
    ///     the bug of treating T[row,row] (real part only) as the eigenvalue of a
    ///     2×2 block whose true eigenvalue is a±bi.
    ///
    ///   • 2×2 vs 2×2 cross-term: general 2×2 Sylvester equation
    ///       T_ii · F_ij − F_ij · T_jj = RHS
    ///     Vectorised via Kronecker product into a 4×4 complex system and solved
    ///     by Gaussian elimination with partial pivoting (``solve4x4Complex``).
    ///     Both blocks have complex-conjugate eigenvalue pairs so neither diagonal
    ///     entry alone represents a true eigenvalue.
    ///
    /// When eigenvalues coincide (|λ_i − λ_j| < 1e-12, 1×1 case only), the
    /// divided-difference limit f'(λ)·T[i,j] applies (Higham 2008, §4.2 eq. 4.6).
    ///
    /// - Returns: `false` when a required derivative is undefined (e.g., f'(0) for
    ///   log, used in the repeated-eigenvalue fallback).
    private static func parlettUpperTriangle(
        fn: MatrixFunction, T: [Double], n: Int, blockOf: [Int],
        Fre: inout [Double], Fim: inout [Double]
    ) -> Bool {
        for d in 1..<n {
            for col in d..<n {
                let row = col - d
                if blockOf[row] == blockOf[col] { continue }  // same 2×2 block — already set

                let rowBlock = blockOf[row]   // start index of row's diagonal block
                let colBlock = blockOf[col]   // start index of col's diagonal block

                let rowIs2x2 = (rowBlock + 1 < n && blockOf[rowBlock + 1] == rowBlock)
                let colIs2x2 = (colBlock + 1 < n && blockOf[colBlock + 1] == colBlock)

                // If row is the SECOND row of a 2×2 block, both F[rowBlock, col] and
                // F[row, col] were (or will be) handled together when the loop reached
                // the first row of that block.  Skip to avoid double-writing.
                if rowIs2x2 && row != rowBlock { continue }

                // Similarly, if col is the SECOND column of a 2×2 block, its pair was
                // handled when col was the block start.
                if colIs2x2 && col != colBlock { continue }

                if !rowIs2x2 && !colIs2x2 {
                    // ── 1×1 vs 1×1 ─────────────────────────────────────────────
                    // Standard scalar Parlett recurrence.
                    let fii: C2 = (Fre[row*n+row], Fim[row*n+row])
                    let fjj: C2 = (Fre[col*n+col], Fim[col*n+col])
                    let dEig   = cSub((T[row*n+row], 0.0), (T[col*n+col], 0.0))
                    var numRe  = T[row*n+col] * (fii.re - fjj.re)
                    var numIm  = T[row*n+col] * (fii.im - fjj.im)
                    for k in (row+1)..<col {
                        numRe += T[row*n+k] * Fre[k*n+col] - Fre[row*n+k] * T[k*n+col]
                        numIm += T[row*n+k] * Fim[k*n+col] - Fim[row*n+k] * T[k*n+col]
                    }
                    if cAbs(dEig) < 1e-12 {
                        guard let fprime = applyComplexDerivative(fn, (T[row*n+row], 0.0)) else {
                            return false
                        }
                        let deriv = cMul(fprime, (T[row*n+col], 0.0))
                        Fre[row*n+col] = deriv.re; Fim[row*n+col] = deriv.im
                    } else {
                        let val = cDiv((numRe, numIm), dEig)
                        Fre[row*n+col] = val.re; Fim[row*n+col] = val.im
                    }

                } else if rowIs2x2 && !colIs2x2 {
                    // ── 2×2 block_i vs 1×1 block_j ─────────────────────────────
                    // Higham 2008 eq. 4.18 (block Parlett) for this case:
                    //   T_ii · F_ij − F_ij · T_jj = F_ii · T_ij − T_ij · F_jj
                    //                                + Σ_{k intermediate} (F_ik T_kj − T_ik F_kj)
                    // T_jj is scalar (λ_j), so this is:
                    //   (T_ii − λ_j·I) · F_ij = RHS    [2×1 linear system]
                    //
                    // RHS = F_ii @ T_ij − T_ij · f(λ_j) + accumulated block cross-terms
                    // where F_ii is the full 2×2 complex matrix (from the diagonal pass),
                    // NOT just the diagonal entries.  The off-diagonal F[p,q] and F[q,p]
                    // contribute via the matrix product and must be included.

                    let p = rowBlock          // first row of the 2×2 block
                    let q = p + 1             // second row of the 2×2 block
                    let lamJ = T[col*n+col]   // scalar eigenvalue (real)
                    let fjj: C2 = (Fre[col*n+col], Fim[col*n+col])   // f(λ_j)

                    // RHS[p]: (F_ii @ T_ij)[p] − T[p,col] · f(λ_j) + cross-block terms
                    // (F_ii @ T_ij)[p] = F[p,p]*T[p,col] + F[p,q]*T[q,col]
                    let fpp: C2 = (Fre[p*n+p], Fim[p*n+p])
                    let fpq: C2 = (Fre[p*n+q], Fim[p*n+q])
                    let fqp: C2 = (Fre[q*n+p], Fim[q*n+p])
                    let fqq: C2 = (Fre[q*n+q], Fim[q*n+q])
                    let tpc = T[p*n+col];  let tqc = T[q*n+col]
                    // [F_ii @ T_ij][p] − T[p,col]*fjj:
                    var rhs0Re = fpp.re*tpc + fpq.re*tqc - tpc*fjj.re
                    var rhs0Im = fpp.im*tpc + fpq.im*tqc - tpc*fjj.im
                    // [F_ii @ T_ij][q] − T[q,col]*fjj:
                    var rhs1Re = fqp.re*tpc + fqq.re*tqc - tqc*fjj.re
                    var rhs1Im = fqp.im*tpc + fqq.im*tqc - tqc*fjj.im
                    // Cross-block terms: Σ_{k ∉ {p,q}, k < col} (F[p,k]*T[k,col] − T[p,k]*F[k,col])
                    // and equivalently for row q.
                    for k in 0..<n where k != p && k != q && k < col {
                        rhs0Re += Fre[p*n+k]*T[k*n+col] - T[p*n+k]*Fre[k*n+col]
                        rhs0Im += Fim[p*n+k]*T[k*n+col] - T[p*n+k]*Fim[k*n+col]
                        rhs1Re += Fre[q*n+k]*T[k*n+col] - T[q*n+k]*Fre[k*n+col]
                        rhs1Im += Fim[q*n+k]*T[k*n+col] - T[q*n+k]*Fim[k*n+col]
                    }

                    // Coefficient matrix: T_ii − λ_j · I  (real entries since T is real)
                    let A00: C2 = (T[p*n+p] - lamJ, 0.0)
                    let A01: C2 = (T[p*n+q], 0.0)
                    let A10: C2 = (T[q*n+p], 0.0)
                    let A11: C2 = (T[q*n+q] - lamJ, 0.0)
                    let det = cSub(cMul(A00, A11), cMul(A01, A10))
                    guard cAbs(det) > 1e-12 else {
                        // Degenerate: 2×2 block eigenvalue coincides with λ_j.
                        let lam2x2 = complexEigenvalueOfBlock(T: T, n: n, p: p)
                        guard let fp0 = applyComplexDerivative(fn, lam2x2) else { return false }
                        Fre[p*n+col] = (cMul(fp0, (tpc, 0.0))).re
                        Fim[p*n+col] = (cMul(fp0, (tpc, 0.0))).im
                        Fre[q*n+col] = (cMul(fp0, (tqc, 0.0))).re
                        Fim[q*n+col] = (cMul(fp0, (tqc, 0.0))).im
                        continue
                    }
                    // Cramer's rule for 2×2: [A][x] = [rhs]
                    let rhs0: C2 = (rhs0Re, rhs0Im); let rhs1: C2 = (rhs1Re, rhs1Im)
                    let x0 = cDiv(cSub(cMul(A11, rhs0), cMul(A01, rhs1)), det)
                    let x1 = cDiv(cSub(cMul(A00, rhs1), cMul(A10, rhs0)), det)
                    Fre[p*n+col] = x0.re; Fim[p*n+col] = x0.im
                    Fre[q*n+col] = x1.re; Fim[q*n+col] = x1.im

                } else if !rowIs2x2 && colIs2x2 {
                    // ── 1×1 block_i vs 2×2 block_j ─────────────────────────────
                    // Higham 2008 eq. 4.18 for block_i scalar (λ_i) vs block_j 2×2:
                    //   λ_i · F_ij − F_ij · T_jj = F_ii · T_ij − T_ij · F_jj   (*)
                    // Re-arrange: −F_ij · (T_jj − λ_i·I) = RHS
                    //   → (T_jj − λ_i·I)^T · F_ij^T = −RHS^T    [2×1 system]
                    // Note the NEGATION on the right-hand side: this differs from
                    // the 2×1 case above by the direction of the Sylvester (F on
                    // the right of T_jj instead of the left of T_ii).
                    // F_ij is a 1×2 row; unknowns are F[row,p] and F[row,q].
                    //
                    // RHS = F_ii · T_ij − T_ij · F_jj  (scalar F_ii since 1×1 block):
                    //   RHS[c ∈ {p,q}] = F[row,row]*T[row,c]
                    //                  − (T[row,p]*F[p,c] + T[row,q]*F[q,c])
                    //                  + Σ_{k intermediate} (F[row,k]*T[k,c] − T[row,k]*F[k,c])

                    let p = colBlock;  let q = p + 1
                    let lamI = T[row*n+row]   // scalar eigenvalue (real)
                    let fii: C2 = (Fre[row*n+row], Fim[row*n+row])

                    // Compute RHS (before negation):
                    // RHS[0] ↔ column p:
                    var rhs0Re = fii.re*T[row*n+p] - T[row*n+p]*Fre[p*n+p] - T[row*n+q]*Fre[q*n+p]
                    var rhs0Im = fii.im*T[row*n+p] - T[row*n+p]*Fim[p*n+p] - T[row*n+q]*Fim[q*n+p]
                    // RHS[1] ↔ column q:
                    var rhs1Re = fii.re*T[row*n+q] - T[row*n+p]*Fre[p*n+q] - T[row*n+q]*Fre[q*n+q]
                    var rhs1Im = fii.im*T[row*n+q] - T[row*n+p]*Fim[p*n+q] - T[row*n+q]*Fim[q*n+q]
                    // Cross-block accumulated terms (k strictly between row and p):
                    for k in (row+1)..<p {
                        rhs0Re += Fre[row*n+k]*T[k*n+p] - T[row*n+k]*Fre[k*n+p]
                        rhs0Im += Fim[row*n+k]*T[k*n+p] - T[row*n+k]*Fim[k*n+p]
                        rhs1Re += Fre[row*n+k]*T[k*n+q] - T[row*n+k]*Fre[k*n+q]
                        rhs1Im += Fim[row*n+k]*T[k*n+q] - T[row*n+k]*Fim[k*n+q]
                    }

                    // Coefficient matrix: (T_jj − λ_i·I)^T = T_jj^T − λ_i·I
                    // T_jj^T = [[T[p,p], T[q,p]], [T[p,q], T[q,q]]]  (rows/cols swapped)
                    let B00: C2 = (T[p*n+p] - lamI, 0.0);  let B01: C2 = (T[q*n+p], 0.0)
                    let B10: C2 = (T[p*n+q], 0.0);          let B11: C2 = (T[q*n+q] - lamI, 0.0)
                    let det2 = cSub(cMul(B00, B11), cMul(B01, B10))
                    guard cAbs(det2) > 1e-12 else {
                        let lam2x2 = complexEigenvalueOfBlock(T: T, n: n, p: p)
                        guard let fp0 = applyComplexDerivative(fn, lam2x2) else { return false }
                        Fre[row*n+p] = (cMul(fp0, (T[row*n+p], 0.0))).re
                        Fim[row*n+p] = (cMul(fp0, (T[row*n+p], 0.0))).im
                        Fre[row*n+q] = (cMul(fp0, (T[row*n+q], 0.0))).re
                        Fim[row*n+q] = (cMul(fp0, (T[row*n+q], 0.0))).im
                        continue
                    }
                    // Solve (T_jj^T − λ_i·I)·[x0;x1] = −[rhs0;rhs1] (note negation).
                    let nrhs0: C2 = (-rhs0Re, -rhs0Im);  let nrhs1: C2 = (-rhs1Re, -rhs1Im)
                    let x0 = cDiv(cSub(cMul(B11, nrhs0), cMul(B01, nrhs1)), det2)
                    let x1 = cDiv(cSub(cMul(B00, nrhs1), cMul(B10, nrhs0)), det2)
                    Fre[row*n+p] = x0.re; Fim[row*n+p] = x0.im
                    Fre[row*n+q] = x1.re; Fim[row*n+q] = x1.im

                } else {
                    // ── 2×2 block_i vs 2×2 block_j ─────────────────────────────
                    // Higham 2008 eq. 4.18: T_ii · F_ij − F_ij · T_jj = RHS  [2×2 Sylvester]
                    // Vectorise: (I⊗T_ii − T_jj^T⊗I) · vec(F_ij) = vec(RHS)
                    // This is a 4×4 complex linear system solved by Gaussian elimination.
                    let p = rowBlock; let q = p + 1
                    let r = colBlock; let s = r + 1

                    // RHS[a,b] = (F_ii @ T_ij − T_ij @ F_jj)[a,b]
                    //          + Σ_{k: intermediate block, between i and j}
                    //            (F[p+a,k]*T[k,r+b] − T[p+a,k]*F[k,r+b])
                    var rhs: [[C2]] = Array(repeating: Array(repeating: (0.0,0.0) as C2, count: 2), count: 2)
                    for a in 0..<2 {
                        for b in 0..<2 {
                            let ri = p + a;  let ci = r + b
                            // (F_ii @ T_ij)[a,b]:
                            var rhsRe = Fre[ri*n+p]*T[p*n+ci] + Fre[ri*n+q]*T[q*n+ci]
                            var rhsIm = Fim[ri*n+p]*T[p*n+ci] + Fim[ri*n+q]*T[q*n+ci]
                            // −(T_ij @ F_jj)[a,b]:
                            rhsRe -= T[ri*n+r]*Fre[r*n+ci] + T[ri*n+s]*Fre[s*n+ci]
                            rhsIm -= T[ri*n+r]*Fim[r*n+ci] + T[ri*n+s]*Fim[s*n+ci]
                            // Cross-block accumulated terms:
                            for k in 0..<n where k != p && k != q && k != r && k != s && k > p && k < r {
                                rhsRe += Fre[ri*n+k]*T[k*n+ci] - T[ri*n+k]*Fre[k*n+ci]
                                rhsIm += Fim[ri*n+k]*T[k*n+ci] - T[ri*n+k]*Fim[k*n+ci]
                            }
                            rhs[a][b] = (rhsRe, rhsIm)
                        }
                    }
                    // Kronecker 4×4 system: M · vec(F_ij) = vec(RHS)
                    // vec layout: [F[p,r], F[q,r], F[p,s], F[q,s]]
                    let tii00: C2 = (T[p*n+p],0); let tii01: C2 = (T[p*n+q],0)
                    let tii10: C2 = (T[q*n+p],0); let tii11: C2 = (T[q*n+q],0)
                    let tjj00: C2 = (T[r*n+r],0); let tjj01: C2 = (T[r*n+s],0)
                    let tjj10: C2 = (T[s*n+r],0); let tjj11: C2 = (T[s*n+s],0)
                    typealias Row4 = (C2,C2,C2,C2)
                    let M: [Row4] = [
                        (cSub(tii00, tjj00), tii01,              cSub((0,0), tjj10), (0,0)),
                        (tii10,              cSub(tii11, tjj00), (0,0),              cSub((0,0), tjj10)),
                        (cSub((0,0), tjj01), (0,0),              cSub(tii00, tjj11), tii01),
                        ((0,0),              cSub((0,0), tjj01), tii10,              cSub(tii11, tjj11))
                    ]
                    let RHS4: [C2] = [rhs[0][0], rhs[1][0], rhs[0][1], rhs[1][1]]
                    guard let sol4 = solve4x4Complex(M, RHS4) else { return false }
                    Fre[p*n+r] = sol4[0].re; Fim[p*n+r] = sol4[0].im
                    Fre[q*n+r] = sol4[1].re; Fim[q*n+r] = sol4[1].im
                    Fre[p*n+s] = sol4[2].re; Fim[p*n+s] = sol4[2].im
                    Fre[q*n+s] = sol4[3].re; Fim[q*n+s] = sol4[3].im
                }
            }
        }
        return true
    }

    /// Extract the upper eigenvalue (positive imaginary part) of a 2×2 real Schur block
    /// starting at row `p` in the n×n matrix `T`.
    ///
    /// The block has form [[a,b],[c,a]] with complex-conjugate eigenvalues a ± i·|sqrt(-bc)|.
    private static func complexEigenvalueOfBlock(T: [Double], n: Int, p: Int) -> C2 {
        let a = T[p*n+p]; let b = T[p*n+p+1]
        let c = T[(p+1)*n+p]
        let disc = (a - T[(p+1)*n+p+1]) * (a - T[(p+1)*n+p+1]) + 4.0 * b * c
        let sqrtDiscMag = sqrt(max(-disc, 0.0))  // disc < 0 for complex-conjugate pair
        return (a, sqrtDiscMag / 2.0)
    }

    /// Solve a 4×4 complex linear system via Gaussian elimination with partial pivoting.
    ///
    /// `M` is the system matrix (4 rows, each a 4-tuple of C2); `rhs` is the right-hand
    /// side vector. Returns `nil` when the system is singular (|pivot| < 1e-12).
    private static func solve4x4Complex(
        _ M: [(C2,C2,C2,C2)], _ rhs: [C2]
    ) -> [C2]? {
        // Augmented matrix [M | rhs] as a mutable array of rows.
        var aug: [(C2,C2,C2,C2,C2)] = zip(M, rhs).map { (row, r) in
            (row.0, row.1, row.2, row.3, r)
        }

        func getCol(_ row: (C2,C2,C2,C2,C2), _ c: Int) -> C2 {
            switch c { case 0: return row.0; case 1: return row.1
                       case 2: return row.2; case 3: return row.3; default: return row.4 }
        }
        func setAug(_ row: inout (C2,C2,C2,C2,C2), _ c: Int, _ v: C2) {
            switch c { case 0: row.0 = v; case 1: row.1 = v
                       case 2: row.2 = v; case 3: row.3 = v; default: row.4 = v }
        }

        for col in 0..<4 {
            // Find pivot (max magnitude in column ≥ col).
            var pivotRow = col
            var pivotMag = cAbs(getCol(aug[col], col))
            for r in (col+1)..<4 {
                let m = cAbs(getCol(aug[r], col))
                if m > pivotMag { pivotMag = m; pivotRow = r }
            }
            guard pivotMag > 1e-12 else { return nil }
            if pivotRow != col { aug.swapAt(col, pivotRow) }

            let pivot = getCol(aug[col], col)
            // Eliminate below.
            for r in (col+1)..<4 {
                let factor = cDiv(getCol(aug[r], col), pivot)
                for c in col..<5 {
                    let newVal = cSub(getCol(aug[r], c), cMul(factor, getCol(aug[col], c)))
                    setAug(&aug[r], c, newVal)
                }
            }
        }

        // Back substitution.
        var x = [C2](repeating: (0.0, 0.0), count: 4)
        for r in stride(from: 3, through: 0, by: -1) {
            var val = getCol(aug[r], 4)
            for c in (r+1)..<4 {
                val = cSub(val, cMul(getCol(aug[r], c), x[c]))
            }
            x[r] = cDiv(val, getCol(aug[r], r))
        }
        return x
    }

    /// Compute Q · F · Qᵀ where Q is real-orthogonal and F is complex (separate real/imag).
    private static func multiplyQFQT(
        Q: [Double], Fre: [Double], Fim: [Double], n: Int
    ) -> (re: [Double], im: [Double]) {
        let QFre = matmulInternal(Q, Fre, n)
        let QFim = matmulInternal(Q, Fim, n)
        var resRe = [Double](repeating: 0, count: n * n)
        var resIm = [Double](repeating: 0, count: n * n)
        // (Q·F)·Qᵀ via dgemm with transposed Q (Q is row-major, Qᵀ = Q with CblasTrans).
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(n), Int32(n), Int32(n),
                    1.0, QFre, Int32(n), Q, Int32(n), 0.0, &resRe, Int32(n))
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(n), Int32(n), Int32(n),
                    1.0, QFim, Int32(n), Q, Int32(n), 0.0, &resIm, Int32(n))
        return (resRe, resIm)
    }
}
