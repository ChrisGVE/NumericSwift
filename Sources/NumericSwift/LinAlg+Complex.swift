//
//  LinAlg+Complex.swift
//  NumericSwift
//
//  Complex matrix operations: csolve, csvd, ceig, ceigvals, cdet, cinv.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Complex Matrix Operations

    /// Solve complex linear system Az = b.
    ///
    /// - Parameters:
    ///   - A: Square complex coefficient matrix
    ///   - b: Right-hand side complex matrix
    /// - Returns: Solution z, or nil if singular
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `A` is not square, or
    ///   ``LinAlgError/dimensionMismatch(_:)`` when `A` and `b` row counts differ.
    ///   Returns `nil` (does not throw) when `A` is square but singular.
    public static func csolve(_ A: ComplexMatrix, _ b: ComplexMatrix) throws -> ComplexMatrix? {
        guard A.rows == A.cols else { throw LinAlgError.notSquare(rows: A.rows, cols: A.cols) }
        guard A.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("A.rows (\(A.rows)) must equal b.rows (\(b.rows))")
        }

        let n = A.rows

        var a = complexToColumnMajor(A, rows: n, cols: n)
        var bData = complexToColumnMajor(b, rows: b.rows, cols: b.cols)

        var n1 = __CLPK_integer(n)
        var nrhs = __CLPK_integer(b.cols)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var ldb = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        zgesv_(&n1, &nrhs, &a, &lda, &ipiv, &bData, &ldb, &info)

        if info != 0 { return nil }

        return complexFromColumnMajor(bData, rows: b.rows, cols: b.cols)
    }

    /// Complex SVD.
    ///
    /// - Parameter m: Complex matrix
    /// - Returns: (s, U, Vt) where s is real, U and Vt are complex
    public static func csvd(_ m: ComplexMatrix) -> (s: [Double], U: ComplexMatrix, Vt: ComplexMatrix)? {
        let minDim = min(m.rows, m.cols)

        var a = complexToColumnMajor(m, rows: m.rows, cols: m.cols)

        var m1 = __CLPK_integer(m.rows)
        var n1 = __CLPK_integer(m.cols)
        var lda1 = __CLPK_integer(m.rows)
        var s = [Double](repeating: 0, count: minDim)
        var u = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0),
                                        count: m.rows * m.rows)
        var vt = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0),
                                         count: m.cols * m.cols)
        var ldu = __CLPK_integer(m.rows)
        var ldvt = __CLPK_integer(m.cols)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)
        var rwork = [Double](repeating: 0, count: max(1, 5 * minDim * minDim + 7 * minDim))
        var iwork = [__CLPK_integer](repeating: 0, count: 8 * minDim)
        var jobz = Int8(UInt8(ascii: "A"))

        zgesdd_(&jobz, &m1, &n1, &a, &lda1, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &rwork, &iwork, &info)

        lwork = __CLPK_integer(work[0].r)
        work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: Int(lwork))

        a = complexToColumnMajor(m, rows: m.rows, cols: m.cols)

        var m2 = __CLPK_integer(m.rows)
        var n2 = __CLPK_integer(m.cols)
        var lda2 = __CLPK_integer(m.rows)

        zgesdd_(&jobz, &m2, &n2, &a, &lda2, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &rwork, &iwork, &info)

        if info != 0 { return nil }

        let U = complexFromColumnMajor(u, rows: m.rows, cols: m.rows)
        let Vt = complexFromColumnMajor(vt, rows: m.cols, cols: m.cols)

        return (s, U, Vt)
    }

    /// Complex eigendecomposition.
    ///
    /// - Parameter m: Square complex matrix
    /// - Returns: (eigenvalues, eigenvectors) both complex
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func ceig(_ m: ComplexMatrix) throws -> (values: ComplexMatrix, vectors: ComplexMatrix)? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = complexToColumnMajor(m, rows: n, cols: n)

        var w = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n)
        var vl = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)
        var ldvl: __CLPK_integer = 1
        var vr = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n * n)
        var ldvr = __CLPK_integer(n)

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)
        var rwork = [Double](repeating: 0, count: 2 * n)
        var jobvl = Int8(UInt8(ascii: "N"))
        var jobvr = Int8(UInt8(ascii: "V"))

        zgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &w, &vl, &ldvl, &vr, &ldvr,
               &work, &lwork, &rwork, &info)

        lwork = __CLPK_integer(work[0].r)
        work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: Int(lwork))

        a = complexToColumnMajor(m, rows: n, cols: n)

        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)

        zgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &w, &vl, &ldvl, &vr, &ldvr,
               &work, &lwork, &rwork, &info)

        if info != 0 { return nil }

        var eigReal = [Double](repeating: 0, count: n)
        var eigImag = [Double](repeating: 0, count: n)
        for i in 0..<n {
            eigReal[i] = w[i].r
            eigImag[i] = w[i].i
        }

        let vecMatrix = complexFromColumnMajor(vr, rows: n, cols: n)

        return (ComplexMatrix(rows: n, cols: 1, real: eigReal, imag: eigImag), vecMatrix)
    }

    /// Complex eigenvalues only.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func ceigvals(_ m: ComplexMatrix) throws -> ComplexMatrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = complexToColumnMajor(m, rows: n, cols: n)

        var w = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n)
        var vl = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)
        var vr = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)
        var ldvl: __CLPK_integer = 1
        var ldvr: __CLPK_integer = 1

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)
        var rwork = [Double](repeating: 0, count: 2 * n)
        var jobvl = Int8(UInt8(ascii: "N"))
        var jobvr = Int8(UInt8(ascii: "N"))

        zgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &w, &vl, &ldvl, &vr, &ldvr,
               &work, &lwork, &rwork, &info)

        lwork = __CLPK_integer(work[0].r)
        work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: Int(lwork))

        a = complexToColumnMajor(m, rows: n, cols: n)

        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)

        zgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &w, &vl, &ldvl, &vr, &ldvr,
               &work, &lwork, &rwork, &info)

        if info != 0 { return nil }

        var eigReal = [Double](repeating: 0, count: n)
        var eigImag = [Double](repeating: 0, count: n)
        for i in 0..<n {
            eigReal[i] = w[i].r
            eigImag[i] = w[i].i
        }

        return ComplexMatrix(rows: n, cols: 1, real: eigReal, imag: eigImag)
    }

    /// Complex determinant.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func cdet(_ m: ComplexMatrix) throws -> (re: Double, im: Double)? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = complexToColumnMajor(m, rows: n, cols: n)

        var m1 = __CLPK_integer(n)
        var n1 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        zgetrf_(&m1, &n1, &a, &lda, &ipiv, &info)

        if info > 0 { return (0, 0) }
        if info < 0 { return nil }

        var detRe = 1.0
        var detIm = 0.0
        var sign = 1
        for i in 0..<n {
            let diagIdx = i * n + i
            let diagRe = a[diagIdx].r
            let diagIm = a[diagIdx].i
            let newRe = detRe * diagRe - detIm * diagIm
            let newIm = detRe * diagIm + detIm * diagRe
            detRe = newRe
            detIm = newIm
            if Int(ipiv[i]) != i + 1 { sign = -sign }
        }

        return (detRe * Double(sign), detIm * Double(sign))
    }

    /// Complex inverse.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    ///   Returns `nil` (does not throw) when `m` is square but singular.
    public static func cinv(_ m: ComplexMatrix) throws -> ComplexMatrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        var a = complexToColumnMajor(m, rows: n, cols: n)

        var m1 = __CLPK_integer(n)
        var n1 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        zgetrf_(&m1, &n1, &a, &lda, &ipiv, &info)

        if info != 0 { return nil }

        var lwork: __CLPK_integer = -1
        var work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)

        zgetri_(&n1, &a, &lda, &ipiv, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0].r)
        work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: Int(lwork))

        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)
        zgetri_(&n2, &a, &lda2, &ipiv, &work, &lwork, &info)

        if info != 0 { return nil }

        return complexFromColumnMajor(a, rows: n, cols: n)
    }

    // MARK: - Complex layout helpers

    /// Convert a ComplexMatrix (row-major) to a column-major interleaved LAPACK array.
    private static func complexToColumnMajor(
        _ m: ComplexMatrix, rows: Int, cols: Int
    ) -> [__CLPK_doublecomplex] {
        var a = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0),
                                        count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                let srcIdx = i * cols + j
                let dstIdx = j * rows + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }
        return a
    }

    /// Convert a column-major interleaved LAPACK complex array back to a ComplexMatrix.
    private static func complexFromColumnMajor(
        _ a: [__CLPK_doublecomplex], rows: Int, cols: Int
    ) -> ComplexMatrix {
        var real = [Double](repeating: 0, count: rows * cols)
        var imag = [Double](repeating: 0, count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                let srcIdx = j * rows + i
                let dstIdx = i * cols + j
                real[dstIdx] = a[srcIdx].r
                imag[dstIdx] = a[srcIdx].i
            }
        }
        return ComplexMatrix(rows: rows, cols: cols, real: real, imag: imag)
    }
}
