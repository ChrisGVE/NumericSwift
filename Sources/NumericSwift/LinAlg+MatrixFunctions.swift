//
//  LinAlg+MatrixFunctions.swift
//  NumericSwift
//
//  Matrix functions: expm, logm, sqrtm, funm.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Matrix Functions

    /// Supported functions for funm
    public enum MatrixFunction: String {
        case sin, cos, exp, log, sqrt, sinh, cosh, tanh, abs
    }

    /// Matrix exponential using Padé approximation with scaling and squaring.
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    public static func expm(_ m: Matrix) throws -> Matrix {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        let normA = cblas_dnrm2(Int32(n * n), m.data, 1)

        // A non-finite norm (any ±inf or NaN element) makes the scaling loop below
        // never terminate (`inf / scale > 1.0` is always true) — guard it as a
        // recoverable error rather than hanging the process. (Audit CR.)
        guard normA.isFinite else {
            throw LinAlgError.invalidParameter(
                "expm requires finite matrix elements; norm is \(normA)")
        }

        var s = 0
        var scale = 1.0
        while normA / scale > 1.0 {
            scale *= 2.0
            s += 1
        }

        var A = [Double](repeating: 0, count: n * n)
        var scaleVal = scale
        vDSP_vsdivD(m.data, 1, &scaleVal, &A, 1, vDSP_Length(n * n))

        let c = [1.0, 0.5, 0.12, 0.01833333333333333,
                 0.001992063492063492, 0.0001575312500000000,
                 0.00000918114788107536]

        let A2 = matmulInternal(A, A, n)
        let A4 = matmulInternal(A2, A2, n)
        let A6 = matmulInternal(A2, A4, n)

        var V = [Double](repeating: 0, count: n * n)
        for i in 0..<n { V[i * n + i] = c[0] }
        var c2 = c[2]
        vDSP_vsmaD(A2, 1, &c2, V, 1, &V, 1, vDSP_Length(n * n))
        var c4 = c[4]
        vDSP_vsmaD(A4, 1, &c4, V, 1, &V, 1, vDSP_Length(n * n))
        var c6 = c[6]
        vDSP_vsmaD(A6, 1, &c6, V, 1, &V, 1, vDSP_Length(n * n))

        var Uinner = [Double](repeating: 0, count: n * n)
        for i in 0..<n { Uinner[i * n + i] = c[1] }
        var c3 = c[3]
        vDSP_vsmaD(A2, 1, &c3, Uinner, 1, &Uinner, 1, vDSP_Length(n * n))
        var c5 = c[5]
        vDSP_vsmaD(A4, 1, &c5, Uinner, 1, &Uinner, 1, vDSP_Length(n * n))

        let U = matmulInternal(A, Uinner, n)

        var VminusU = [Double](repeating: 0, count: n * n)
        var VplusU = [Double](repeating: 0, count: n * n)
        vDSP_vsubD(U, 1, V, 1, &VminusU, 1, vDSP_Length(n * n))
        vDSP_vaddD(V, 1, U, 1, &VplusU, 1, vDSP_Length(n * n))

        var R = solveLinearSystemInternal(VminusU, VplusU, n)

        for _ in 0..<s {
            R = matmulInternal(R, R, n)
        }

        return Matrix(rows: n, cols: n, data: R)
    }

    /// Matrix logarithm using eigendecomposition.
    ///
    /// log(A) = V * diag(log(λ)) * V^(-1) for diagonalizable matrices.
    /// - Parameter m: Square matrix with positive eigenvalues
    /// - Returns: Matrix logarithm, or nil if eigenvalues are non-positive
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    ///   Returns `nil` (does not throw) when eigenvalues are non-positive.
    public static func logm(_ m: Matrix) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        guard let (eigenvalues, eigenvectors) = computeRealEigendecomposition(m.data, n) else {
            return nil
        }

        for ev in eigenvalues {
            if ev <= 0 { return nil }
        }

        let logEigenvalues = eigenvalues.map { log($0) }
        let result = reconstructFromEigen(eigenvectors, logEigenvalues, n)

        return Matrix(rows: n, cols: n, data: result)
    }

    /// Matrix square root using eigendecomposition.
    ///
    /// sqrt(A) = V * diag(sqrt(λ)) * V^(-1) for diagonalizable matrices.
    /// - Parameter m: Square matrix with non-negative eigenvalues
    /// - Returns: Matrix square root, or nil if eigenvalues are negative
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    ///   Returns `nil` (does not throw) when eigenvalues are negative.
    public static func sqrtm(_ m: Matrix) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        guard let (eigenvalues, eigenvectors) = computeRealEigendecomposition(m.data, n) else {
            return nil
        }

        for ev in eigenvalues {
            if ev < 0 { return nil }
        }

        let sqrtEigenvalues = eigenvalues.map { sqrt($0) }
        let result = reconstructFromEigen(eigenvectors, sqrtEigenvalues, n)

        return Matrix(rows: n, cols: n, data: result)
    }

    /// General matrix function using eigendecomposition.
    ///
    /// f(A) = V * diag(f(λ)) * V^(-1) for diagonalizable matrices.
    /// - Parameters:
    ///   - m: Square matrix
    ///   - function: Function to apply (sin, cos, exp, log, sqrt, sinh, cosh, tanh, abs)
    /// - Returns: f(A), or nil if computation fails
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `m` is not square.
    ///   Returns `nil` (does not throw) when the function is undefined on the eigenvalues.
    public static func funm(_ m: Matrix, _ function: MatrixFunction) throws -> Matrix? {
        guard m.rows == m.cols else { throw LinAlgError.notSquare(rows: m.rows, cols: m.cols) }
        let n = m.rows

        guard let (eigenvalues, eigenvectors) = computeRealEigendecomposition(m.data, n) else {
            return nil
        }

        guard let transformedEigenvalues = applyMatrixFunction(function, to: eigenvalues) else {
            return nil
        }

        let result = reconstructFromEigen(eigenvectors, transformedEigenvalues, n)

        return Matrix(rows: n, cols: n, data: result)
    }

    // MARK: - funm helper

    private static func applyMatrixFunction(
        _ function: MatrixFunction,
        to eigenvalues: [Double]
    ) -> [Double]? {
        switch function {
        case .sin:   return eigenvalues.map { sin($0) }
        case .cos:   return eigenvalues.map { cos($0) }
        case .exp:   return eigenvalues.map { exp($0) }
        case .sinh:  return eigenvalues.map { sinh($0) }
        case .cosh:  return eigenvalues.map { cosh($0) }
        case .tanh:  return eigenvalues.map { tanh($0) }
        case .abs:   return eigenvalues.map { Swift.abs($0) }
        case .log:
            for ev in eigenvalues { if ev <= 0 { return nil } }
            return eigenvalues.map { log($0) }
        case .sqrt:
            for ev in eigenvalues { if ev < 0 { return nil } }
            return eigenvalues.map { sqrt($0) }
        }
    }
}
