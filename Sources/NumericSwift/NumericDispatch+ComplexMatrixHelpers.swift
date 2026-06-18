//
//  NumericDispatch+ComplexMatrixHelpers.swift
//  NumericSwift
//
//  Shared helper functions for complex-matrix arithmetic in the unified
//  numeric pipeline. These are `internal` so all NumericDispatch extension
//  files that implement complex-matrix operations can reference them.
//
//  Contains:
//    - validateCMSameShape   — Group-A shape check for component-wise ops
//    - validateCMMatmulShape — Group-A inner-dimension check for matmul
//    - realBlock             — extract real part of ComplexMatrix as Matrix
//    - imagBlock             — extract imaginary part of ComplexMatrix as Matrix
//    - divideComplex         — (re,im) / Complex scalar formula
//    - complexMatmul         — real-block decomposition CM*CM with coercion
//
//  ## Design notes
//
//  LinAlg has no native complex-matrix operators. The evaluator implements
//  them using LinAlg's real-matrix primitives. The real-block decomposition
//  for matmul follows the standard identity:
//    Cr = Ar·Br − Ai·Bi,  Ci = Ar·Bi + Ai·Br
//  Each product is a LinAlg.dot call (BLAS/Accelerate-backed, Group-B).
//
//  Soft-cap policy (§4.8 / CONS-07): every operation that allocates a result
//  or intermediate matrix checks the soft cap BEFORE the allocation.
//  `complexMatmul` uses a *peak-aware* admission check (Issue #13 / CR-D7):
//  rather than calling `checkSoftCap` on the result shape alone, it divides
//  the cap by `complexMatmulWorkingSetMultiplier` (= 5) and verifies the
//  result element count does not exceed that scaled limit.  This ensures the
//  four intermediate real products plus the two output arrays cannot exceed
//  the intended memory ceiling in aggregate.
//
//  Group-A errors are pre-validated and throw before any LinAlg call,
//  preventing process-trapping preconditions. Group-B errors propagate via
//  `try` from LinAlg.dot.
//
//  1×1 coercion (§4.3a): complexMatmul calls `coerce1x1Complex` on the
//  result so a 1×1 ComplexMatrix collapses to .complex per §15 truth table.
//
//  Licensed under the Apache License, Version 2.0.
//

import Accelerate

// MARK: - Shared shape-validation and block-extraction helpers

extension NumericDispatch {

    // MARK: Shape validation for component-wise ops

    /// Pre-validate that two ComplexMatrix operands share the same shape.
    ///
    /// Throws `MathExprError.shapeMismatch` (Group-A) before any element
    /// operation, preventing accidental misuse of mismatched arrays.
    static func validateCMSameShape(
        _ opName: String,
        lhs: LinAlg.ComplexMatrix,
        rhs: LinAlg.ComplexMatrix
    ) throws {
        guard lhs.rows == rhs.rows && lhs.cols == rhs.cols else {
            throw MathExprError.shapeMismatch(
                "\(opName): shapes (\(lhs.rows)×\(lhs.cols)) "
                + "and (\(rhs.rows)×\(rhs.cols)) must match")
        }
    }

    /// Pre-validate ComplexMatrix matmul inner-dimension compatibility.
    ///
    /// Throws `MathExprError.shapeMismatch` (Group-A) when lhs.cols ≠ rhs.rows,
    /// which prevents the LinAlg.dot precondition from firing.
    static func validateCMMatmulShape(
        lhs: LinAlg.ComplexMatrix,
        rhs: LinAlg.ComplexMatrix
    ) throws {
        // Vec·vec special case: both cols==1 requires same row count
        if lhs.cols == 1 && rhs.cols == 1 {
            guard lhs.rows == rhs.rows else {
                throw MathExprError.shapeMismatch(
                    "dotProduct/matmul: complex vectors must have the same length "
                    + "(\(lhs.rows) vs \(rhs.rows))")
            }
        } else {
            guard lhs.cols == rhs.rows else {
                throw MathExprError.shapeMismatch(
                    "complexMatrix*complexMatrix: lhs.cols (\(lhs.cols)) "
                    + "must equal rhs.rows (\(rhs.rows))")
            }
        }
    }

    // MARK: Block extraction

    /// Extract the real part of a ComplexMatrix as a LinAlg.Matrix.
    ///
    /// The real array is already row-major and directly wrappable; no copy
    /// is performed beyond what Matrix construction requires.
    static func realBlock(_ cm: LinAlg.ComplexMatrix) -> LinAlg.Matrix {
        LinAlg.Matrix(rows: cm.rows, cols: cm.cols, data: cm.real)
    }

    /// Extract the imaginary part of a ComplexMatrix as a LinAlg.Matrix.
    static func imagBlock(_ cm: LinAlg.ComplexMatrix) -> LinAlg.Matrix {
        LinAlg.Matrix(rows: cm.rows, cols: cm.cols, data: cm.imag)
    }

    // MARK: Complex scalar division helper

    /// Divide a complex number (re, im) by a Complex divisor.
    ///
    /// Formula: (a+bi)/(c+di) = [(ac+bd) + i(bc-ad)] / (c²+d²).
    static func divideComplex(
        re: Double, im: Double, by divisor: Complex
    ) -> (re: Double, im: Double) {
        let denom = divisor.re * divisor.re + divisor.im * divisor.im
        let outRe = (re * divisor.re + im * divisor.im) / denom
        let outIm = (im * divisor.re - re * divisor.im) / denom
        return (outRe, outIm)
    }

    // MARK: Core complex matmul

    /// The working-set multiplier for ``complexMatmul(_:_:)`` peak admission control.
    ///
    /// The real-block decomposition (Cr = Ar·Br − Ai·Bi, Ci = Ar·Bi + Ai·Br)
    /// allocates four intermediate `LinAlg.Matrix` objects, each of size M×N
    /// (same as the result), plus two output `[Double]` arrays (crData, ciData)
    /// of the same size.  All six buffers are live simultaneously when the two
    /// vDSP combine passes run, giving a peak working set of 6× the result size.
    ///
    /// The multiplier is the TRUE peak (6), so the admission test
    /// `resultElements × 6 ≤ cap` guarantees the actual peak working set never
    /// exceeds the soft cap. (A smaller multiplier would admit inputs whose 6×
    /// peak overshoots the cap — e.g. with 5, an input at `cap/5` peaks at
    /// `6·cap/5 = 1.2× cap`. Codex pre-0.3.0 audit fix.)
    ///
    /// Reference: Issue #13 / CR-D7 cumulative-bounding fix.
    static let complexMatmulWorkingSetMultiplier: Int = 6

    /// Implements CM*CM via real-block decomposition: Cr=Ar·Br−Ai·Bi, Ci=Ar·Bi+Ai·Br.
    ///
    /// ## Soft-cap admission (§4.8 / Issue #13)
    ///
    /// Admission control bounds the **peak working set**, not just the result shape.
    /// The four intermediate real products (Ar·Br, Ai·Bi, Ar·Bi, Ai·Br) each have
    /// the result shape (M×N), and all four are live simultaneously alongside the
    /// two output arrays when the final vDSP combine passes run.  The peak allocation
    /// is therefore `resultElements × complexMatmulWorkingSetMultiplier`.
    ///
    /// The check is:
    /// ```
    /// resultElements × 5 ≤ maxEvaluatorMatrixElements
    /// ```
    /// equivalently: `resultElements ≤ cap / 5` (integer division, conservative).
    /// This is computed overflow-safely by verifying the result first passes the
    /// ordinary cap, then scaling the cap down before re-checking.
    ///
    /// - Performs Group-A shape pre-validation (throws ``MathExprError/shapeMismatch(_:)``).
    /// - Performs peak-aware soft-cap check (throws ``LinAlg/LinAlgError/invalidParameter(_:)``).
    /// - Calls `coerce1x1Complex` for the 1×1 vec·vec → .complex collapse (§4.3a).
    ///
    /// Each real product is delegated to ``LinAlg/dot(_:_:)`` (BLAS/Accelerate-backed).
    static func complexMatmul(
        lhs: LinAlg.ComplexMatrix,
        rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // Group-A: validate inner dimensions
        try validateCMMatmulShape(lhs: lhs, rhs: rhs)

        // For vec·vec (both cols==1): LinAlg.dot returns a 1×1 scalar result.
        // For mat·vec or mat·mat: result shape is (lhs.rows × rhs.cols).
        let isVecDot = lhs.cols == 1 && rhs.cols == 1
        let resultRows = isVecDot ? 1 : lhs.rows
        let resultCols = isVecDot ? 1 : rhs.cols

        // §4.8 / Issue #13: peak-aware admission control.
        //
        // The decomposition allocates `complexMatmulWorkingSetMultiplier` (= 6)
        // buffers of shape (resultRows × resultCols) simultaneously.  Bound the
        // peak by requiring resultElements × 6 ≤ maxEvaluatorMatrixElements,
        // i.e. resultElements ≤ cap / 6 (floor division, safe from overflow).
        //
        // Implementation: divide the cap by the multiplier and call checkSoftCap
        // with the scaled-down cap limit.  This avoids overflow in the product
        // `resultElements * 5` while still producing an informative error message.
        let cap = LinAlg.maxEvaluatorMatrixElements
        let scaledCap = cap / complexMatmulWorkingSetMultiplier
        let (resultElements, overflow) = LinAlg.elementCount(rows: resultRows, cols: resultCols)
        guard !overflow && resultElements <= scaledCap else {
            let desc = overflow ? "overflows Int" : "\(resultElements)"
            throw LinAlg.LinAlgError.invalidParameter(
                "complexMatmul peak working set (\(desc) × \(complexMatmulWorkingSetMultiplier)"
                + " = \(overflow ? "overflow" : "\(resultElements * complexMatmulWorkingSetMultiplier)")"
                + " elements) exceeds soft cap \(cap)"
                + " (result shape \(resultRows)×\(resultCols),"
                + " per-matrix limit \(scaledCap) for complex matmul)")
        }

        // Extract real and imaginary blocks
        let ar = realBlock(lhs)
        let ai = imagBlock(lhs)
        let br = realBlock(rhs)
        let bi = imagBlock(rhs)

        // Four real products via LinAlg.dot (BLAS; Group-B propagate)
        let arBr = LinAlg.dot(ar, br)  // Ar·Br
        let aiBi = LinAlg.dot(ai, bi)  // Ai·Bi
        let arBi = LinAlg.dot(ar, bi)  // Ar·Bi
        let aiBr = LinAlg.dot(ai, br)  // Ai·Br

        // Cr = Ar·Br − Ai·Bi,   Ci = Ar·Bi + Ai·Br
        let n = arBr.size
        var crData = [Double](repeating: 0, count: n)
        var ciData = [Double](repeating: 0, count: n)
        vDSP_vsubD(aiBi.data, 1, arBr.data, 1, &crData, 1, vDSP_Length(n))
        vDSP_vaddD(arBi.data, 1, aiBr.data, 1, &ciData, 1, vDSP_Length(n))

        let result = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: resultRows, cols: resultCols, real: crData, imag: ciData))
        return coerce1x1Complex(result)
    }

    /// Multiply two complex matrices, always returning a `ComplexMatrix` (no 1×1
    /// scalar coercion). Used by ``evalComplexMatrixPow(cm:exponent:)`` for
    /// exponentiation-by-squaring, where the running base/accumulator must stay a
    /// matrix. Uses the same real-block decomposition as ``complexMatmul(lhs:rhs:)``
    /// (Cr = Ar·Br − Ai·Bi, Ci = Ar·Bi + Ai·Br); callers supply square operands of
    /// the already cap-checked shape, so admission control is not repeated here.
    static func complexMatrixMultiply(
        _ a: LinAlg.ComplexMatrix, _ b: LinAlg.ComplexMatrix
    ) -> LinAlg.ComplexMatrix {
        let ar = realBlock(a), ai = imagBlock(a)
        let br = realBlock(b), bi = imagBlock(b)
        let arBr = LinAlg.dot(ar, br)
        let aiBi = LinAlg.dot(ai, bi)
        let arBi = LinAlg.dot(ar, bi)
        let aiBr = LinAlg.dot(ai, br)
        let count = arBr.size
        var crData = [Double](repeating: 0, count: count)
        var ciData = [Double](repeating: 0, count: count)
        vDSP_vsubD(aiBi.data, 1, arBr.data, 1, &crData, 1, vDSP_Length(count))
        vDSP_vaddD(arBi.data, 1, aiBr.data, 1, &ciData, 1, vDSP_Length(count))
        return LinAlg.ComplexMatrix(
            rows: arBr.rows, cols: arBr.cols, real: crData, imag: ciData)
    }
}
