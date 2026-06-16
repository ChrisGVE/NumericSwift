//
//  UnifiedEvaluatorMatrix.swift
//  NumericSwift
//
//  Matrix and linear-algebra AST node handlers for the unified evaluator.
//
//  This file contains the handlers that are branched to from
//  `UnifiedEvaluatorCore.eval` for:
//    • `.vector([MathLexExpression])`  — column-vector literal
//    • `.matrix([[MathLexExpression]])` — matrix literal
//    • `.conjugateTranspose(matrix:)` — Hermitian adjoint
//    • `.rank(matrix:)` — matrix rank via SVD
//
//  Motivation for splitting from `UnifiedEvaluator.swift`:
//  The per-file line budget is ~400 lines (coding.md §VIII). Keeping matrix
//  helpers here keeps both files comfortably under budget while colocating
//  the matrix logic.
//
//  ## Parser-scope note (§4.9 / ARCH-01)
//
//  `.vector` and `.matrix` literal AST nodes are only emitted by the opt-in
//  mathlex Rust backend (`NUMERICSWIFT_INCLUDE_MATHLEX=1`). The default-build
//  pure-Swift fallback parser has no bracket tokenizer and never produces
//  these nodes. On the default build, matrix values flow in via the `values:`
//  dictionary and are returned from the `.variable` arm in
//  `UnifiedEvaluatorCore.eval`. The handlers here compile and work on both
//  build configurations; they are simply unreachable dead code on the default
//  build — no conditional compilation is needed.
//
//  ## Soft-cap enforcement (CONS-07)
//
//  Every handler that allocates a matrix pre-checks the result shape via
//  `LinAlg.checkSoftCap(rows:cols:)` before allocating. The check throws
//  `LinAlgError.invalidParameter` (never `MathExprError`) per CONS-07.
//
//  ## Element type promotion
//
//  Within a literal node, all elements are evaluated recursively. If every
//  element evaluates to `.scalar`, the result is a `.matrix`. If any element
//  is `.complex`, all elements are promoted to `.complexMatrix`. Mixed
//  matrix-valued elements within a literal throw `.invalidArguments`.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Matrix literal and linear-algebra node handlers

/// Handlers for matrix/vector AST nodes and related linear-algebra operations.
///
/// All methods are `static` and free of stored state; they are called from
/// `UnifiedEvaluatorCore.eval` and delegate complex-matrix construction /
/// linear-algebra operations to `LinAlg`.
enum UnifiedEvaluatorMatrix {

    // MARK: Vector literal (subtask 18.8)

    /// Evaluate a `.vector([MathLexExpression])` node.
    ///
    /// The vector is represented as a column-vector `LinAlg.Matrix` with
    /// `cols == 1`, matching `LinAlg.Matrix.isVector`. Elements are evaluated
    /// recursively; if any element is complex the result is a `.complexMatrix`
    /// column vector.
    ///
    /// - Parameters:
    ///   - elements: The list of element expressions.
    ///   - values: Variable bindings passed through to the recursive evaluator.
    /// - Returns: A `.matrix` (all real elements) or `.complexMatrix` (any complex
    ///   element) column vector.
    /// - Throws: `MathExprError.invalidArguments` if any element evaluates to
    ///   a matrix rather than a scalar/complex. `LinAlgError.invalidParameter`
    ///   if the resulting shape exceeds the soft cap.
    static func evalVector(
        _ elements: [MathLexExpression],
        values: [String: NumericValue],
        complexMode: Bool = false
    ) throws -> NumericValue {
        let rows = elements.count
        guard rows > 0 else {
            return .matrix(LinAlg.Matrix(rows: 0, cols: 1, data: []))
        }
        try LinAlg.checkSoftCap(rows: rows, cols: 1)
        let evaluated = try elements.map {
            try UnifiedEvaluatorCore.eval($0, values: values, complexMode: complexMode)
        }
        return try buildColumnVector(evaluated, expectedRows: rows)
    }

    // MARK: Matrix literal (subtask 18.8)

    /// Evaluate a `.matrix([[MathLexExpression]])` node.
    ///
    /// Each sub-array is a row; all rows must have the same column count or
    /// `LinAlgError.dimensionMismatch` is thrown. Elements within rows are
    /// evaluated recursively; any complex element promotes the result to
    /// `.complexMatrix`.
    ///
    /// - Parameters:
    ///   - rows: The 2-D array of element expressions (outer = rows, inner = cols).
    ///   - values: Variable bindings passed through to the recursive evaluator.
    /// - Returns: A `.matrix` (all real) or `.complexMatrix` (any complex).
    /// - Throws: `LinAlgError.dimensionMismatch` for ragged rows.
    ///   `MathExprError.invalidArguments` if any element evaluates to a matrix.
    ///   `LinAlgError.invalidParameter` if the shape exceeds the soft cap.
    static func evalMatrix(
        _ rows: [[MathLexExpression]],
        values: [String: NumericValue],
        complexMode: Bool = false
    ) throws -> NumericValue {
        guard !rows.isEmpty else {
            return .matrix(LinAlg.Matrix(rows: 0, cols: 0, data: []))
        }
        let nCols = rows[0].count
        for (i, row) in rows.enumerated() {
            guard row.count == nCols else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "matrix literal: row \(i) has \(row.count) columns, expected \(nCols)")
            }
        }
        let nRows = rows.count
        try LinAlg.checkSoftCap(rows: nRows, cols: nCols)
        let evaluated = try rows.map { row -> [NumericValue] in
            try row.map { try UnifiedEvaluatorCore.eval($0, values: values, complexMode: complexMode) }
        }
        return try buildMatrix2D(evaluated, nRows: nRows, nCols: nCols)
    }

    // MARK: Conjugate transpose (subtask 18.9)

    /// Evaluate a `.conjugateTranspose(matrix:)` node — the Hermitian adjoint.
    ///
    /// For a real matrix this equals the regular transpose.
    /// For a complex matrix, each element is conjugated and the result
    /// is transposed: `(A†)[i,j] = conj(A[j,i])`.
    ///
    /// - Parameter operand: The already-evaluated matrix operand.
    /// - Returns: `.matrix` (real input) or `.complexMatrix` (complex input).
    /// - Throws: `MathExprError.invalidArguments` if `operand` is scalar or complex scalar.
    static func evalConjugateTranspose(_ operand: NumericValue) throws -> NumericValue {
        switch operand {
        case .matrix(let m):
            // Real transpose is its own conjugate transpose.
            return .matrix(m.T)

        case .complexMatrix(let cm):
            // Transpose: new[i,j] = old[j,i]; then conjugate imaginary parts.
            let outRows = cm.cols
            let outCols = cm.rows
            try LinAlg.checkSoftCap(rows: outRows, cols: outCols)
            let size = outRows * outCols
            var outReal = [Double](repeating: 0, count: size)
            var outImag = [Double](repeating: 0, count: size)
            for i in 0..<outRows {
                for j in 0..<outCols {
                    // cm[j,i] → out[i,j]; conjugate flips sign of imag part.
                    let src = j * cm.cols + i
                    let dst = i * outCols + j
                    outReal[dst] = cm.real[src]
                    outImag[dst] = -cm.imag[src]   // conjugation
                }
            }
            return .complexMatrix(LinAlg.ComplexMatrix(
                rows: outRows, cols: outCols, real: outReal, imag: outImag))

        case .scalar, .complex:
            throw MathExprError.invalidArguments(
                "conjugateTranspose requires a matrix operand, got \(operand.typeAndShapeDescription)")
        }
    }

    // MARK: Matrix rank (subtask 18.9)

    /// Evaluate a `.rank(matrix:)` node — the numerical rank via SVD.
    ///
    /// Rank is estimated as the number of singular values above a threshold
    /// `tol = max(m, n) * σ_max * ε` (SciPy `matrix_rank` definition), where
    /// ε = `Double.ulpOfOne`. Returns a `.scalar(Double)` equal to the integer rank.
    ///
    /// - Parameter operand: The already-evaluated matrix operand.
    /// - Returns: `.scalar(Double)` — the numerical rank as a `Double`.
    /// - Throws: `MathExprError.invalidArguments` if `operand` is not a real matrix.
    static func evalRank(_ operand: NumericValue) throws -> NumericValue {
        guard case .matrix(let m) = operand else {
            throw MathExprError.invalidArguments(
                "rank requires a real matrix operand, got \(operand.typeAndShapeDescription)")
        }
        let svdResult = LinAlg.svd(m)
        let sigmas = svdResult.s
        guard !sigmas.isEmpty else { return .scalar(0) }
        let sigMax = sigmas[0]  // SVD returns singular values in descending order
        let maxDim = Double(max(m.rows, m.cols))
        let tol = maxDim * sigMax * Double.ulpOfOne
        let rank = sigmas.filter { $0 > tol }.count
        return .scalar(Double(rank))
    }

    // MARK: Private helpers

    /// Build a `NumericValue` column vector from an array of evaluated elements.
    ///
    /// Promotes to `.complexMatrix` if any element is complex. Throws if any
    /// element is a matrix rather than a scalar/complex.
    private static func buildColumnVector(
        _ evaluated: [NumericValue],
        expectedRows: Int
    ) throws -> NumericValue {
        let hasComplex = evaluated.contains { $0.isComplex || $0.isComplexMatrix }

        if hasComplex {
            var real = [Double]()
            var imag = [Double]()
            real.reserveCapacity(expectedRows)
            imag.reserveCapacity(expectedRows)
            for (idx, val) in evaluated.enumerated() {
                switch val {
                case .scalar(let x):
                    real.append(x); imag.append(0)
                case .complex(let z):
                    real.append(z.re); imag.append(z.im)
                default:
                    throw MathExprError.invalidArguments(
                        "vector literal: element \(idx) evaluated to \(val.typeAndShapeDescription); "
                        + "only scalar or complex elements are allowed")
                }
            }
            return .complexMatrix(LinAlg.ComplexMatrix(
                rows: expectedRows, cols: 1, real: real, imag: imag))
        } else {
            var data = [Double]()
            data.reserveCapacity(expectedRows)
            for (idx, val) in evaluated.enumerated() {
                guard case .scalar(let x) = val else {
                    throw MathExprError.invalidArguments(
                        "vector literal: element \(idx) evaluated to \(val.typeAndShapeDescription); "
                        + "only scalar elements are allowed in a real vector")
                }
                data.append(x)
            }
            return .matrix(LinAlg.Matrix(rows: expectedRows, cols: 1, data: data))
        }
    }

    /// Build a `NumericValue` from a 2-D evaluated element grid.
    ///
    /// Row-major layout. Promotes to `.complexMatrix` if any element is complex.
    private static func buildMatrix2D(
        _ evaluated: [[NumericValue]],
        nRows: Int,
        nCols: Int
    ) throws -> NumericValue {
        let hasComplex = evaluated.lazy.flatMap { $0 }.contains { $0.isComplex || $0.isComplexMatrix }

        if hasComplex {
            let size = nRows * nCols
            var real = [Double](repeating: 0, count: size)
            var imag = [Double](repeating: 0, count: size)
            for (rowIdx, row) in evaluated.enumerated() {
                for (colIdx, val) in row.enumerated() {
                    let dst = rowIdx * nCols + colIdx
                    switch val {
                    case .scalar(let x):
                        real[dst] = x; imag[dst] = 0
                    case .complex(let z):
                        real[dst] = z.re; imag[dst] = z.im
                    default:
                        throw MathExprError.invalidArguments(
                            "matrix literal: element [\(rowIdx),\(colIdx)] evaluated to "
                            + "\(val.typeAndShapeDescription); only scalar or complex allowed")
                    }
                }
            }
            return .complexMatrix(LinAlg.ComplexMatrix(
                rows: nRows, cols: nCols, real: real, imag: imag))
        } else {
            var data = [Double](repeating: 0, count: nRows * nCols)
            for (rowIdx, row) in evaluated.enumerated() {
                for (colIdx, val) in row.enumerated() {
                    guard case .scalar(let x) = val else {
                        throw MathExprError.invalidArguments(
                            "matrix literal: element [\(rowIdx),\(colIdx)] evaluated to "
                            + "\(val.typeAndShapeDescription); only scalar elements allowed in real matrix")
                    }
                    data[rowIdx * nCols + colIdx] = x
                }
            }
            return .matrix(LinAlg.Matrix(rows: nRows, cols: nCols, data: data))
        }
    }
}
