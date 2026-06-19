//
//  LinAlg+Properties.swift
//  NumericSwift
//
//  Matrix properties: trace, determinant, inverse, rank, condition number,
//  pseudoinverse, and norms.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Matrix Properties

    /// Compute the trace of a square matrix.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func trace(_ m: Matrix) throws -> Double {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        var sum = 0.0
        for i in 0..<m.rows {
            sum += m.data[i * m.cols + i]
        }
        return sum
    }

    /// Compute the determinant of a square matrix.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func det(_ m: Matrix) throws -> Double {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = m.data
        var n1 = __CLPK_integer(n)
        var n2 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&n1, &n2, &a, &lda, &ipiv, &info)

        if info != 0 { return 0.0 }

        var determinant = 1.0
        var sign = 1
        for i in 0..<n {
            determinant *= a[i * n + i]
            if ipiv[i] != Int32(i + 1) {
                sign *= -1
            }
        }

        return determinant * Double(sign)
    }

    /// Compute the matrix inverse.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    ///   Returns `nil` (does not throw) when `m` is square but singular.
    public static func inv(_ m: Matrix) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = m.data
        var n1 = __CLPK_integer(n)
        var n2 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&n1, &n2, &a, &lda, &ipiv, &info)

        if info != 0 { return nil }

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var ngetri = __CLPK_integer(n)
        var ldagetri = __CLPK_integer(n)
        dgetri_(&ngetri, &a, &ldagetri, &ipiv, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        var ngetri2 = __CLPK_integer(n)
        var ldagetri2 = __CLPK_integer(n)
        dgetri_(&ngetri2, &a, &ldagetri2, &ipiv, &work, &lwork, &info)

        if info != 0 { return nil }

        return Matrix(rows: n, cols: n, data: a)
    }

    /// Compute the matrix rank via SVD.
    ///
    /// Returns `0` when the SVD fails to converge or the matrix has no singular
    /// values (defensive — the largest singular value `s[0]` is read for the
    /// tolerance, so an empty `s` must be guarded).
    public static func rank(_ m: Matrix) -> Int {
        guard let (s, _, _) = svd(m), let sMax = s.first else { return 0 }
        let tol = max(Double(m.rows), Double(m.cols)) * sMax * 2.220446049250313e-16
        var r = 0
        for sv in s {
            if sv > tol { r += 1 }
        }
        return r
    }

    /// Compute the condition number of a matrix.
    ///
    /// Returns `NaN` when the SVD fails to converge (the condition number is
    /// undefined), and `+inf` for a singular matrix (`σ_min` below tolerance).
    public static func cond(_ m: Matrix) -> Double {
        guard let (s, _, _) = svd(m) else { return Double.nan }
        let sMax = s.first ?? 0
        let sMin = s.last ?? 0
        let tol = max(Double(m.rows), Double(m.cols)) * sMax * 2.220446049250313e-16
        if sMin <= tol { return Double.infinity }
        return sMax / sMin
    }

    /// Compute the Moore-Penrose pseudoinverse.
    ///
    /// - Returns: The pseudoinverse (shape `cols × rows`), or `nil` when the
    ///   underlying SVD (`dgesvd`) fails to converge. Previously this returned the
    ///   input matrix on failure, which is both the wrong value and — for
    ///   rectangular input — the wrong shape.
    public static func pinv(_ m: Matrix, rcond: Double = 1e-15) -> Matrix? {
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

        if info != 0 { return nil }

        let tol = rcond * s[0]

        // Compute V * Σ^(-1) * U^T
        var result = [Double](repeating: 0, count: m.cols * m.rows)

        for i in 0..<m.cols {
            for j in 0..<m.rows {
                var sum = 0.0
                for k in 0..<minDim {
                    if s[k] > tol {
                        let v_ik = vt[i * m.cols + k]
                        let u_jk = u[k * m.rows + j]
                        sum += v_ik * (1.0 / s[k]) * u_jk
                    }
                }
                result[i * m.rows + j] = sum
            }
        }

        return Matrix(rows: m.cols, cols: m.rows, data: result)
    }

    // MARK: - Norms

    /// Compute a vector or matrix norm, following SciPy's `numpy.linalg.norm`
    /// contract for the order `p`.
    ///
    /// For a column vector (`cols == 1`): `p = 1` (sum of magnitudes), `p = 2`
    /// (Euclidean), `p = .infinity` (max magnitude), otherwise the general
    /// `p`-norm `(Σ|xᵢ|^p)^(1/p)`.
    ///
    /// For a matrix: `p = 1` (max absolute column sum), `p = 2` (**spectral
    /// norm** — the largest singular value, matching SciPy `ord=2`),
    /// `p = .infinity` (max absolute row sum), any other `p` (Frobenius norm).
    ///
    /// - Note: As of the SciPy-parity fix, the default `p = 2` returns the
    ///   spectral norm for matrices (not Frobenius). Use ``frobeniusNorm(_:)``
    ///   or any `p ∉ {1, 2, ∞}` for the Frobenius norm.
    public static func norm(_ m: Matrix, _ p: Double = 2) -> Double {
        if m.cols == 1 {
            return vectorNorm(m, p)
        } else {
            return matrixNorm(m, p)
        }
    }

    /// Frobenius norm: `sqrt(Σ |aᵢⱼ|²)` over all entries. Equivalent to the
    /// Euclidean norm of the flattened matrix and to SciPy `ord='fro'`.
    public static func frobeniusNorm(_ m: Matrix) -> Double {
        var result = 0.0
        vDSP_dotprD(m.data, 1, m.data, 1, &result, vDSP_Length(m.size))
        return sqrt(result)
    }

    // MARK: - Norm helpers

    private static func vectorNorm(_ m: Matrix, _ p: Double) -> Double {
        if p == 1 {
            var result = 0.0
            vDSP_svemgD(m.data, 1, &result, vDSP_Length(m.rows))
            return result
        } else if p == 2 {
            var result = 0.0
            vDSP_dotprD(m.data, 1, m.data, 1, &result, vDSP_Length(m.rows))
            return sqrt(result)
        } else if p == Double.infinity {
            var result = 0.0
            vDSP_maxmgvD(m.data, 1, &result, vDSP_Length(m.rows))
            return result
        } else {
            var sum = 0.0
            for val in m.data {
                sum += pow(abs(val), p)
            }
            return pow(sum, 1.0 / p)
        }
    }

    private static func matrixNorm(_ m: Matrix, _ p: Double) -> Double {
        if p == 1 {
            var maxSum = 0.0
            for j in 0..<m.cols {
                var colSum = 0.0
                for i in 0..<m.rows {
                    colSum += abs(m.data[i * m.cols + j])
                }
                maxSum = max(maxSum, colSum)
            }
            return maxSum
        } else if p == 2 {
            // Spectral norm = largest singular value (SciPy ord=2). A non-convergent
            // SVD yields NaN (the norm is undefined when the factorization fails).
            guard let s = svd(m)?.s else { return Double.nan }
            return s.max() ?? 0
        } else if p == Double.infinity {
            var maxSum = 0.0
            for i in 0..<m.rows {
                var rowSum = 0.0
                for j in 0..<m.cols {
                    rowSum += abs(m.data[i * m.cols + j])
                }
                maxSum = max(maxSum, rowSum)
            }
            return maxSum
        } else {
            // Frobenius norm
            var result = 0.0
            vDSP_dotprD(m.data, 1, m.data, 1, &result, vDSP_Length(m.size))
            return sqrt(result)
        }
    }
}
