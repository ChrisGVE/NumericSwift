//
//  LinAlg+Expm.swift
//  Sources/NumericSwift/LinAlg+Expm.swift
//
//  Matrix exponential: expm via scaling-and-squaring with variable-degree Padé
//  approximant (Higham 2005).
//
//  Architecture: part of LinAlg (Sources/NumericSwift/), split from
//  LinAlg+MatrixFunctions.swift at the expm / Schur-Parlett boundary.
//  Private Padé helpers (matrixOneNorm, linearCombination, padeUV) are colocated
//  here because they serve only expm.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

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
        let theta3  = 1.495585217958292e-2
        let theta5  = 2.539398330063230e-1
        let theta7  = 9.504178996162932e-1
        let theta9  = 2.097847961257068e0
        let theta13 = 5.371920351148152e0

        var s = 0
        let degree: Int
        if      normA <= theta3  { degree = 3 }
        else if normA <= theta5  { degree = 5 }
        else if normA <= theta7  { degree = 7 }
        else if normA <= theta9  { degree = 9 }
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
        var VplusU  = [Double](repeating: 0, count: n * n)
        vDSP_vsubD(U, 1, V, 1, &VminusU, 1, vDSP_Length(n * n))  // V − U
        vDSP_vaddD(V, 1, U, 1, &VplusU,  1, vDSP_Length(n * n))  // V + U

        var R = solveLinearSystemInternal(VminusU, VplusU, n)

        // Undo the scaling: r(A/2ˢ)^(2ˢ) = exp(A).
        for _ in 0..<s {
            R = matmulInternal(R, R, n)
        }

        return Matrix(rows: n, cols: n, data: R)
    }

    // MARK: - Padé helpers (private to expm)

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
}
