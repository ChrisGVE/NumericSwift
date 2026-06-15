//
//  LinAlg+Solvers.swift
//  NumericSwift
//
//  Linear system solvers: solve, lstsq, solveTriangular, choSolve, luSolve.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Linear System Solvers

    /// Solve linear system Ax = b.
    ///
    /// - Parameters:
    ///   - A: Square coefficient matrix
    ///   - b: Right-hand side (vector or matrix)
    /// - Returns: Solution x, or nil if singular
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `A` is not square, or
    ///   ``LinAlgError/dimensionMismatch(_:)`` when `A` and `b` row counts differ.
    ///   Returns `nil` (does not throw) when `A` is square but singular.
    public static func solve(_ A: Matrix, _ b: Matrix) throws -> Matrix? {
        guard A.rows == A.cols else { throw LinAlgError.notSquare(rows: A.rows, cols: A.cols) }
        guard A.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("A.rows (\(A.rows)) must equal b.rows (\(b.rows))")
        }

        var a = toColumnMajor(A)
        var bCol = toColumnMajor(b)

        var n1 = __CLPK_integer(A.rows)
        var nrhs = __CLPK_integer(b.cols)
        var lda = __CLPK_integer(A.rows)
        var ipiv = [__CLPK_integer](repeating: 0, count: A.rows)
        var ldb = __CLPK_integer(A.rows)
        var info: __CLPK_integer = 0

        dgesv_(&n1, &nrhs, &a, &lda, &ipiv, &bCol, &ldb, &info)

        if info != 0 { return nil }

        return fromColumnMajor(bCol, rows: b.rows, cols: b.cols)
    }

    /// Solve least squares problem min ||Ax - b||.
    /// - Throws: ``LinAlgError/dimensionMismatch(_:)`` when `A` and `b` row counts differ.
    ///   Returns `nil` (does not throw) when the factorization fails.
    public static func lstsq(_ A: Matrix, _ b: Matrix) throws -> Matrix? {
        guard A.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("A.rows (\(A.rows)) must equal b.rows (\(b.rows))")
        }

        let maxDim = max(A.rows, A.cols)

        var a = toColumnMajor(A)
        var bCol = buildLstsqRHS(b: b, maxDim: maxDim)

        var m1 = __CLPK_integer(A.rows)
        var n1 = __CLPK_integer(A.cols)
        var nrhs1 = __CLPK_integer(b.cols)
        var lda1 = __CLPK_integer(A.rows)
        var ldb1 = __CLPK_integer(maxDim)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var trans = Int8(UInt8(ascii: "N"))

        dgels_(&trans, &m1, &n1, &nrhs1, &a, &lda1, &bCol, &ldb1, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(A)
        bCol = buildLstsqRHS(b: b, maxDim: maxDim)

        var m2 = __CLPK_integer(A.rows)
        var n2 = __CLPK_integer(A.cols)
        var nrhs2 = __CLPK_integer(b.cols)
        var lda2 = __CLPK_integer(A.rows)
        var ldb2 = __CLPK_integer(maxDim)
        dgels_(&trans, &m2, &n2, &nrhs2, &a, &lda2, &bCol, &ldb2, &work, &lwork, &info)

        if info != 0 { return nil }

        var result = [Double](repeating: 0, count: A.cols * b.cols)
        for i in 0..<A.cols {
            for j in 0..<b.cols {
                result[i * b.cols + j] = bCol[j * maxDim + i]
            }
        }

        return Matrix(rows: A.cols, cols: b.cols, data: result)
    }

    /// Solve triangular system: L*x = b (lower) or U*x = b (upper).
    ///
    /// - Parameters:
    ///   - A: Triangular coefficient matrix
    ///   - b: Right-hand side
    ///   - lower: If true, A is lower triangular; if false, upper triangular
    ///   - trans: If true, solve A^T * x = b instead
    /// - Returns: Solution x, or nil if singular
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `A` is not square, or
    ///   ``LinAlgError/dimensionMismatch(_:)`` when `A` and `b` row counts differ.
    public static func solveTriangular(_ A: Matrix, _ b: Matrix,
                                       lower: Bool = true, trans: Bool = false) throws -> Matrix? {
        guard A.rows == A.cols else { throw LinAlgError.notSquare(rows: A.rows, cols: A.cols) }
        guard A.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("A.rows (\(A.rows)) must equal b.rows (\(b.rows))")
        }

        let n = A.rows

        var a = toColumnMajor(A)
        var bData = toColumnMajor(b)

        var uplo = Int8(UInt8(ascii: lower ? "L" : "U"))
        var transChar = Int8(UInt8(ascii: trans ? "T" : "N"))
        var diag = Int8(UInt8(ascii: "N"))
        var n1 = __CLPK_integer(n)
        var nrhs = __CLPK_integer(b.cols)
        var lda = __CLPK_integer(n)
        var ldb = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        dtrtrs_(&uplo, &transChar, &diag, &n1, &nrhs, &a, &lda, &bData, &ldb, &info)

        if info != 0 { return nil }

        return fromColumnMajor(bData, rows: b.rows, cols: b.cols)
    }

    /// Solve using Cholesky factorization: L*L^T*x = b.
    ///
    /// - Parameters:
    ///   - L: Lower triangular Cholesky factor
    ///   - b: Right-hand side
    /// - Returns: Solution x, or nil if computation fails
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `L` is not square, or
    ///   ``LinAlgError/dimensionMismatch(_:)`` when `L` and `b` row counts differ.
    ///   Returns `nil` (does not throw) when the solve fails.
    public static func choSolve(_ L: Matrix, _ b: Matrix) throws -> Matrix? {
        guard L.rows == L.cols else { throw LinAlgError.notSquare(rows: L.rows, cols: L.cols) }
        guard L.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("L.rows (\(L.rows)) must equal b.rows (\(b.rows))")
        }

        let n = L.rows

        var a = toColumnMajor(L)
        var bData = toColumnMajor(b)

        var uplo = Int8(UInt8(ascii: "L"))
        var n1 = __CLPK_integer(n)
        var nrhs = __CLPK_integer(b.cols)
        var lda = __CLPK_integer(n)
        var ldb = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        dpotrs_(&uplo, &n1, &nrhs, &a, &lda, &bData, &ldb, &info)

        if info != 0 { return nil }

        return fromColumnMajor(bData, rows: b.rows, cols: b.cols)
    }

    /// Solve using LU factorization: P*L*U*x = b.
    ///
    /// - Parameters:
    ///   - L: Lower triangular factor from lu()
    ///   - U: Upper triangular factor from lu()
    ///   - P: Permutation matrix from lu()
    ///   - b: Right-hand side
    /// - Returns: Solution x
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `L`, `U`, or `P` is not square, or
    ///   ``LinAlgError/dimensionMismatch(_:)`` when the factors and `b` leading dimensions differ.
    public static func luSolve(_ L: Matrix, _ U: Matrix, _ P: Matrix, _ b: Matrix) throws -> Matrix {
        guard L.rows == L.cols && U.rows == U.cols && P.rows == P.cols else {
            throw LinAlgError.notSquare(rows: L.rows, cols: L.cols)
        }
        guard L.rows == U.rows && L.rows == P.rows && L.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("L, U, P, and b must share the same leading dimension")
        }

        let n = L.rows

        let pb = applyPermutation(P: P, b: b, n: n)
        let y = forwardSubstitution(L: L, pb: pb, n: n, nrhs: b.cols)
        let x = backSubstitution(U: U, y: y, n: n, nrhs: b.cols)

        return Matrix(rows: b.rows, cols: b.cols, data: x)
    }

    // MARK: - luSolve helpers

    private static func applyPermutation(P: Matrix, b: Matrix, n: Int) -> [Double] {
        var pb = [Double](repeating: 0, count: b.rows * b.cols)
        for j in 0..<b.cols {
            for i in 0..<n {
                var sum = 0.0
                for k in 0..<n {
                    sum += P.data[i * n + k] * b.data[k * b.cols + j]
                }
                pb[i * b.cols + j] = sum
            }
        }
        return pb
    }

    private static func forwardSubstitution(L: Matrix, pb: [Double], n: Int, nrhs: Int) -> [Double] {
        var y = pb
        for j in 0..<nrhs {
            for i in 0..<n {
                var sum = y[i * nrhs + j]
                for k in 0..<i {
                    sum -= L.data[i * n + k] * y[k * nrhs + j]
                }
                y[i * nrhs + j] = sum / L.data[i * n + i]
            }
        }
        return y
    }

    private static func backSubstitution(U: Matrix, y: [Double], n: Int, nrhs: Int) -> [Double] {
        var x = y
        for j in 0..<nrhs {
            for i in stride(from: n - 1, through: 0, by: -1) {
                var sum = x[i * nrhs + j]
                for k in (i + 1)..<n {
                    sum -= U.data[i * n + k] * x[k * nrhs + j]
                }
                x[i * nrhs + j] = sum / U.data[i * n + i]
            }
        }
        return x
    }

    /// Build the extended column-major RHS buffer for dgels (rows padded to maxDim).
    private static func buildLstsqRHS(b: Matrix, maxDim: Int) -> [Double] {
        var bCol = [Double](repeating: 0, count: maxDim * b.cols)
        let bColMajor = toColumnMajor(b)
        for i in 0..<b.rows {
            for j in 0..<b.cols {
                bCol[j * maxDim + i] = bColMajor[j * b.rows + i]
            }
        }
        return bCol
    }
}
