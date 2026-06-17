//
//  LinAlg+MatrixFunctions.swift
//  Sources/NumericSwift/LinAlg+MatrixFunctions.swift
//
//  Matrix functions: expm, logm, sqrtm, funm, logmComplex, sqrtmComplex.
//
//  Architecture: this file is part of LinAlg (Sources/NumericSwift/), alongside
//  LinAlg+Decompositions, LinAlg+Solvers, LinAlg+Complex, LinAlg+Internal, etc.
//  It depends on the internal Schur helpers and complex-arithmetic primitives
//  defined in LinAlg+Internal.swift.
//
//  Algorithms:
//    - expm: Higham (2005) scaling-and-squaring with variable-degree Padé.
//    - logm, sqrtm, funm: Schur-Parlett via real Schur form (dgees).
//      Reference: N.J. Higham, "Functions of Matrices", SIAM 2008, Ch. 4.
//    - logmComplex, sqrtmComplex: same Schur-Parlett, always returning
//      ComplexMatrix (for matrices whose function is genuinely complex).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Matrix Functions

    /// Supported functions for ``funm(_:_:)``.
    public enum MatrixFunction: String {
        case sin, cos, exp, log, sqrt, sinh, cosh, tanh, abs
    }

    // MARK: - expm

    /// Matrix exponential via scaling-and-squaring with a variable-degree Padé
    /// approximant.
    ///
    /// Implements Higham, "The Scaling and Squaring Method for the Matrix
    /// Exponential Revisited", SIAM J. Matrix Anal. Appl. 26(4):1179–1193 (2005),
    /// Algorithm 10.20 — the algorithm `scipy.linalg.expm` uses. The Padé degree
    /// (3, 5, 7, 9, or 13) is chosen from `‖A‖₁` against the θ-thresholds of
    /// Table 10.2, which bound the backward error at unit roundoff; for larger
    /// norms degree 13 is combined with `s = ⌈log₂(‖A‖₁ / θ₁₃)⌉` halvings and the
    /// result is recovered by `s` squarings. This delivers full double-precision
    /// accuracy, unlike a fixed low-order Padé.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square;
    ///   ``LinAlgError/invalidParameter(_:)`` when any element is non-finite.
    public static func expm(_ m: Matrix) throws -> Matrix {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        let normA = matrixOneNorm(m.data, n)

        // A non-finite norm (any ±inf or NaN element) would make the scaling
        // exponent below ill-defined — guard it as a recoverable error rather
        // than producing garbage. (Audit CR.)
        guard normA.isFinite else {
            throw LinAlgError.invalidParameter(
                "expm requires finite matrix elements; norm is \(normA)")
        }

        // θ_m thresholds (Higham 2005, Table 10.2), 1-norm.
        let theta3 = 1.495585217958292e-2
        let theta5 = 2.539398330063230e-1
        let theta7 = 9.504178996162932e-1
        let theta9 = 2.097847961257068e0
        let theta13 = 5.371920351148152e0

        var s = 0
        let degree: Int
        if normA <= theta3 { degree = 3 }
        else if normA <= theta5 { degree = 5 }
        else if normA <= theta7 { degree = 7 }
        else if normA <= theta9 { degree = 9 }
        else if normA <= theta13 { degree = 13 }
        else {
            degree = 13
            s = Swift.max(0, Int(ceil(log2(normA / theta13))))
        }

        // Scale by 2⁻ˢ via multiplication (2⁻ˢ stays representable for any finite
        // norm, whereas 2ˢ could overflow to +inf and zero out the matrix).
        var A = m.data
        if s > 0 {
            var invScale = pow(2.0, -Double(s))
            vDSP_vsmulD(m.data, 1, &invScale, &A, 1, vDSP_Length(n * n))
        }

        let A2 = matmulInternal(A, A, n)
        let A4 = matmulInternal(A2, A2, n)
        let A6 = matmulInternal(A2, A4, n)
        let A8 = degree == 9 ? matmulInternal(A4, A4, n) : []

        let (U, V) = padeUV(degree: degree, A: A, A2: A2, A4: A4, A6: A6, A8: A8, n: n)

        // Padé approximant r = (V − U)⁻¹ (V + U).
        var VminusU = [Double](repeating: 0, count: n * n)
        var VplusU = [Double](repeating: 0, count: n * n)
        vDSP_vsubD(U, 1, V, 1, &VminusU, 1, vDSP_Length(n * n))  // V − U
        vDSP_vaddD(V, 1, U, 1, &VplusU, 1, vDSP_Length(n * n))   // V + U

        var R = solveLinearSystemInternal(VminusU, VplusU, n)

        // Undo the scaling: r(A/2ˢ)^(2ˢ) = exp(A).
        for _ in 0..<s {
            R = matmulInternal(R, R, n)
        }

        return Matrix(rows: n, cols: n, data: R)
    }

    // MARK: - expm helpers (Higham 2005)

    /// 1-norm (maximum absolute column sum) of an n×n row-major matrix.
    private static func matrixOneNorm(_ a: [Double], _ n: Int) -> Double {
        var maxColSum = 0.0
        for j in 0..<n {
            var colSum = 0.0
            for i in 0..<n { colSum += Swift.abs(a[i * n + j]) }
            maxColSum = Swift.max(maxColSum, colSum)
        }
        return maxColSum
    }

    /// Linear combination `identityCoeff·I + Σ coeffᵢ·Mᵢ` of n×n row-major matrices.
    private static func linearCombination(
        identityCoeff: Double,
        _ terms: [(Double, [Double])],
        _ n: Int
    ) -> [Double] {
        var result = [Double](repeating: 0, count: n * n)
        for (coeff, matrix) in terms {
            var c = coeff
            vDSP_vsmaD(matrix, 1, &c, result, 1, &result, 1, vDSP_Length(n * n))
        }
        if identityCoeff != 0 {
            for i in 0..<n { result[i * n + i] += identityCoeff }
        }
        return result
    }

    /// Padé numerator/denominator split — `U` (odd powers, pre-multiplied by `A`)
    /// and `V` (even powers) — for the requested degree, using the integer `b`
    /// coefficients of Higham (2005). `A2`/`A4`/`A6` are always required; `A8`
    /// only for degree 9. Degree 13 uses the factored evaluation (eq. 10.20) to
    /// minimise matrix products.
    private static func padeUV(
        degree: Int,
        A: [Double], A2: [Double], A4: [Double], A6: [Double], A8: [Double],
        n: Int
    ) -> (U: [Double], V: [Double]) {
        switch degree {
        case 3:
            let b: [Double] = [120, 60, 12, 1]
            let U = matmulInternal(A, linearCombination(identityCoeff: b[1], [(b[3], A2)], n), n)
            let V = linearCombination(identityCoeff: b[0], [(b[2], A2)], n)
            return (U, V)
        case 5:
            let b: [Double] = [30240, 15120, 3360, 420, 30, 1]
            let U = matmulInternal(
                A, linearCombination(identityCoeff: b[1], [(b[5], A4), (b[3], A2)], n), n)
            let V = linearCombination(identityCoeff: b[0], [(b[4], A4), (b[2], A2)], n)
            return (U, V)
        case 7:
            let b: [Double] = [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1]
            let U = matmulInternal(
                A, linearCombination(identityCoeff: b[1], [(b[7], A6), (b[5], A4), (b[3], A2)], n), n)
            let V = linearCombination(identityCoeff: b[0], [(b[6], A6), (b[4], A4), (b[2], A2)], n)
            return (U, V)
        case 9:
            let b: [Double] = [17643225600, 8821612800, 2075673600, 302702400,
                               30270240, 2162160, 110880, 3960, 90, 1]
            let U = matmulInternal(
                A,
                linearCombination(
                    identityCoeff: b[1], [(b[9], A8), (b[7], A6), (b[5], A4), (b[3], A2)], n),
                n)
            let V = linearCombination(
                identityCoeff: b[0], [(b[8], A8), (b[6], A6), (b[4], A4), (b[2], A2)], n)
            return (U, V)
        default:  // degree 13
            let b: [Double] = [
                64764752532480000, 32382376266240000, 7771770303897600,
                1187353796428800, 129060195264000, 10559470521600, 670442572800,
                33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
            ]
            // U = A · ( A6·(b₁₃·A6 + b₁₁·A4 + b₉·A2) + b₇·A6 + b₅·A4 + b₃·A2 + b₁·I )
            let inner1 = linearCombination(identityCoeff: 0, [(b[13], A6), (b[11], A4), (b[9], A2)], n)
            let w1 = matmulInternal(A6, inner1, n)
            let U = matmulInternal(
                A,
                linearCombination(identityCoeff: b[1], [(1.0, w1), (b[7], A6), (b[5], A4), (b[3], A2)], n),
                n)
            // V = A6·(b₁₂·A6 + b₁₀·A4 + b₈·A2) + b₆·A6 + b₄·A4 + b₂·A2 + b₀·I
            let inner2 = linearCombination(identityCoeff: 0, [(b[12], A6), (b[10], A4), (b[8], A2)], n)
            let w2 = matmulInternal(A6, inner2, n)
            let V = linearCombination(identityCoeff: b[0], [(1.0, w2), (b[6], A6), (b[4], A4), (b[2], A2)], n)
            return (U, V)
        }
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
