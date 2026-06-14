//
//  NumericDispatch+EvalStubs.swift
//  NumericSwift
//
//  Evaluator-implemented arithmetic for the §15 truth-table cells that LinAlg
//  does not natively support (complex-matrix operators). These are the implementations
//  for all EVAL seams in the unified numeric pipeline (§4.8).
//
//  ## Complex-matrix arithmetic — ownership model
//
//  LinAlg has no complex-matrix operators (+, -, *, /, hadamard, neg, transpose).
//  The evaluator implements them here using LinAlg's real-matrix primitives.
//
//  ## Real-block decomposition (matmul)
//
//  For CM * CM: given A = Ar + i·Ai and B = Br + i·Bi,
//    Cr = Ar·Br − Ai·Bi
//    Ci = Ar·Bi + Ai·Br
//  Each real product is a LinAlg.dot call on the extracted real/imag blocks.
//
//  ## Bilinear dot (DOM-06)
//
//  Σ aᵢ·bᵢ with NO conjugation. The conjugate form (vdot / Hermitian inner
//  product) is deferred to v-next §14 and is intentionally NOT implemented here.
//
//  ## Soft-cap policy (§4.8 / CONS-07)
//
//  Every operation that allocates a result or intermediate matrix must call
//  LinAlg.checkSoftCap(rows:cols:) BEFORE the allocation. For matmul this
//  covers all four intermediate real products AND the final result. The check
//  throws LinAlgError.invalidParameter (never MathExprError) per CONS-07.
//
//  ## 1×1 coercion (§4.3a)
//
//  CM*CM matmul and dotProduct(CM,CM) must call coerce1x1Complex(_:) on their
//  result so a 1×1 ComplexMatrix collapses to .complex per the §15 truth table.
//
//  ## Group-A vs Group-B error boundary
//
//  Group-A: shape mismatches and soft-cap violations are pre-validated and throw
//           before any LinAlg call. This prevents process-trapping preconditions.
//  Group-B: errors from LinAlg.dot propagate via `try`. In practice Group-A
//           pre-validation ensures LinAlg.dot never sees incompatible shapes.
//
//  Licensed under the Apache License, Version 2.0.
//

import Accelerate

// MARK: - Private helpers

extension NumericDispatch {

    // MARK: Shape validation for component-wise ops

    /// Pre-validate that two ComplexMatrix operands share the same shape.
    ///
    /// Throws `MathExprError.shapeMismatch` (Group-A) before any element
    /// operation, preventing accidental misuse of mismatched arrays.
    private static func validateCMSameShape(
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
    private static func validateCMMatmulShape(
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
    private static func realBlock(_ cm: LinAlg.ComplexMatrix) -> LinAlg.Matrix {
        LinAlg.Matrix(rows: cm.rows, cols: cm.cols, data: cm.real)
    }

    /// Extract the imaginary part of a ComplexMatrix as a LinAlg.Matrix.
    private static func imagBlock(_ cm: LinAlg.ComplexMatrix) -> LinAlg.Matrix {
        LinAlg.Matrix(rows: cm.rows, cols: cm.cols, data: cm.imag)
    }

    // MARK: Complex scalar division helper

    /// Divide a complex number (re, im) by a Complex divisor.
    ///
    /// Formula: (a+bi)/(c+di) = [(ac+bd) + i(bc-ad)] / (c²+d²).
    private static func divideComplex(
        re: Double, im: Double, by divisor: Complex
    ) -> (re: Double, im: Double) {
        let denom = divisor.re * divisor.re + divisor.im * divisor.im
        let outRe = (re * divisor.re + im * divisor.im) / denom
        let outIm = (im * divisor.re - re * divisor.im) / denom
        return (outRe, outIm)
    }
}

// MARK: - Add/sub EVAL implementations

extension NumericDispatch {

    /// scalar ± matrix: broadcast scalar to every element.
    ///
    /// Uses vDSP_vsaddD for the real broadcast; for sub, negates scalar first
    /// (scalar − M or M − scalar depending on operand order).
    static func evalScalarPlusMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        let isAdd = op == .add
        // Determine which is scalar and which is matrix
        let (scalarVal, matrix, scalarIsLHS): (Double, LinAlg.Matrix, Bool)
        if case .scalar(let s) = lhs, case .matrix(let m) = rhs {
            (scalarVal, matrix, scalarIsLHS) = (s, m, true)
        } else if case .matrix(let m) = lhs, case .scalar(let s) = rhs {
            (scalarVal, matrix, scalarIsLHS) = (s, m, false)
        } else {
            throw MathExprError.invalidArguments(
                "evalScalarPlusMatrix: unexpected kinds \(lhs.kind), \(rhs.kind)")
        }
        try LinAlg.checkSoftCap(rows: matrix.rows, cols: matrix.cols)
        let size = matrix.size
        var result = [Double](repeating: 0, count: size)
        if isAdd {
            // scalar + M or M + scalar: both commutative
            var s = scalarVal
            vDSP_vsaddD(matrix.data, 1, &s, &result, 1, vDSP_Length(size))
        } else {
            // sub: either scalar - M or M - scalar
            var s = scalarIsLHS ? scalarVal : -scalarVal
            // scalar - M: negate M then add scalar  →  vDSP_vsaddD(neg(M), scalar)
            // M - scalar: vDSP_vsaddD(M, -scalar)
            if scalarIsLHS {
                var negOne = -1.0
                var negM = [Double](repeating: 0, count: size)
                vDSP_vsmulD(matrix.data, 1, &negOne, &negM, 1, vDSP_Length(size))
                vDSP_vsaddD(negM, 1, &s, &result, 1, vDSP_Length(size))
            } else {
                vDSP_vsaddD(matrix.data, 1, &s, &result, 1, vDSP_Length(size))
            }
        }
        return .matrix(LinAlg.Matrix(rows: matrix.rows, cols: matrix.cols, data: result))
    }

    /// scalar ± complexMatrix: broadcast real scalar to both real and imag blocks.
    static func evalScalarPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        let isAdd = op == .add
        let (scalarVal, cm, scalarIsLHS): (Double, LinAlg.ComplexMatrix, Bool)
        if case .scalar(let s) = lhs, case .complexMatrix(let c) = rhs {
            (scalarVal, cm, scalarIsLHS) = (s, c, true)
        } else if case .complexMatrix(let c) = lhs, case .scalar(let s) = rhs {
            (scalarVal, cm, scalarIsLHS) = (s, c, false)
        } else {
            throw MathExprError.invalidArguments(
                "evalScalarPlusComplexMatrix: unexpected kinds")
        }
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let size = cm.size
        var outReal = cm.real
        var outImag = cm.imag
        // Scalar adds/subtracts only to the real part; imag is unchanged
        if isAdd {
            var s = scalarVal
            vDSP_vsaddD(cm.real, 1, &s, &outReal, 1, vDSP_Length(size))
        } else {
            var s = scalarIsLHS ? scalarVal : -scalarVal
            if scalarIsLHS {
                var negOne = -1.0
                var negReal = [Double](repeating: 0, count: size)
                var negImag = [Double](repeating: 0, count: size)
                vDSP_vsmulD(cm.real, 1, &negOne, &negReal, 1, vDSP_Length(size))
                vDSP_vsmulD(cm.imag, 1, &negOne, &negImag, 1, vDSP_Length(size))
                vDSP_vsaddD(negReal, 1, &s, &outReal, 1, vDSP_Length(size))
                outImag = negImag
            } else {
                vDSP_vsaddD(cm.real, 1, &s, &outReal, 1, vDSP_Length(size))
                // imag unchanged
            }
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// complex ± matrix: promote M → CM, then element-wise complex add/sub.
    static func evalComplexPlusMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        let isAdd = op == .add
        let (z, matrix, complexIsLHS): (Complex, LinAlg.Matrix, Bool)
        if case .complex(let c) = lhs, case .matrix(let m) = rhs {
            (z, matrix, complexIsLHS) = (c, m, true)
        } else if case .matrix(let m) = lhs, case .complex(let c) = rhs {
            (z, matrix, complexIsLHS) = (c, m, false)
        } else {
            throw MathExprError.invalidArguments(
                "evalComplexPlusMatrix: unexpected kinds")
        }
        let cm = promoteToComplexMatrix(matrix)
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        // Broadcast complex scalar over all elements
        var re = isAdd ? z.re : (complexIsLHS ? z.re : -z.re)
        var im = isAdd ? z.im : (complexIsLHS ? z.im : -z.im)
        if !isAdd && !complexIsLHS {
            // M - c: outReal = M.real - c.re, outImag = M.imag - c.im
            var negRe = -z.re
            var negIm = -z.im
            vDSP_vsaddD(cm.real, 1, &negRe, &outReal, 1, vDSP_Length(size))
            vDSP_vsaddD(cm.imag, 1, &negIm, &outImag, 1, vDSP_Length(size))
        } else if !isAdd && complexIsLHS {
            // c - M: outReal = c.re - M.real, outImag = c.im - M.imag
            var negOne = -1.0
            var negReal = [Double](repeating: 0, count: size)
            var negImag = [Double](repeating: 0, count: size)
            vDSP_vsmulD(cm.real, 1, &negOne, &negReal, 1, vDSP_Length(size))
            vDSP_vsmulD(cm.imag, 1, &negOne, &negImag, 1, vDSP_Length(size))
            vDSP_vsaddD(negReal, 1, &re, &outReal, 1, vDSP_Length(size))
            vDSP_vsaddD(negImag, 1, &im, &outImag, 1, vDSP_Length(size))
        } else {
            // add: c + M (commutative)
            vDSP_vsaddD(cm.real, 1, &re, &outReal, 1, vDSP_Length(size))
            vDSP_vsaddD(cm.imag, 1, &im, &outImag, 1, vDSP_Length(size))
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// complex ± complexMatrix: broadcast complex scalar element-wise.
    static func evalComplexPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        let isAdd = op == .add
        let (z, cm, complexIsLHS): (Complex, LinAlg.ComplexMatrix, Bool)
        if case .complex(let c) = lhs, case .complexMatrix(let m) = rhs {
            (z, cm, complexIsLHS) = (c, m, true)
        } else if case .complexMatrix(let m) = lhs, case .complex(let c) = rhs {
            (z, cm, complexIsLHS) = (c, m, false)
        } else {
            throw MathExprError.invalidArguments(
                "evalComplexPlusComplexMatrix: unexpected kinds")
        }
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        if isAdd {
            var re = z.re, im = z.im
            vDSP_vsaddD(cm.real, 1, &re, &outReal, 1, vDSP_Length(size))
            vDSP_vsaddD(cm.imag, 1, &im, &outImag, 1, vDSP_Length(size))
        } else if complexIsLHS {
            // c - CM: outReal = c.re - cm.real, outImag = c.im - cm.imag
            var negOne = -1.0
            var negReal = [Double](repeating: 0, count: size)
            var negImag = [Double](repeating: 0, count: size)
            vDSP_vsmulD(cm.real, 1, &negOne, &negReal, 1, vDSP_Length(size))
            vDSP_vsmulD(cm.imag, 1, &negOne, &negImag, 1, vDSP_Length(size))
            var re = z.re, im = z.im
            vDSP_vsaddD(negReal, 1, &re, &outReal, 1, vDSP_Length(size))
            vDSP_vsaddD(negImag, 1, &im, &outImag, 1, vDSP_Length(size))
        } else {
            // CM - c: outReal = cm.real - c.re, outImag = cm.imag - c.im
            var negRe = -z.re, negIm = -z.im
            vDSP_vsaddD(cm.real, 1, &negRe, &outReal, 1, vDSP_Length(size))
            vDSP_vsaddD(cm.imag, 1, &negIm, &outImag, 1, vDSP_Length(size))
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// matrix ± complexMatrix: promote real M → CM, then element-wise.
    static func evalMatrixPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        let isAdd = op == .add
        let (matrix, cm, matrixIsLHS): (LinAlg.Matrix, LinAlg.ComplexMatrix, Bool)
        if case .matrix(let m) = lhs, case .complexMatrix(let c) = rhs {
            (matrix, cm, matrixIsLHS) = (m, c, true)
        } else if case .complexMatrix(let c) = lhs, case .matrix(let m) = rhs {
            (matrix, cm, matrixIsLHS) = (m, c, false)
        } else {
            throw MathExprError.invalidArguments(
                "evalMatrixPlusComplexMatrix: unexpected kinds")
        }
        let promoted = promoteToComplexMatrix(matrix)
        // Group-A: validate shapes on the promoted pair
        try validateCMSameShape(isAdd ? "add" : "sub", lhs: promoted, rhs: cm)
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        if isAdd {
            vDSP_vaddD(promoted.real, 1, cm.real, 1, &outReal, 1, vDSP_Length(size))
            vDSP_vaddD(promoted.imag, 1, cm.imag, 1, &outImag, 1, vDSP_Length(size))
        } else if matrixIsLHS {
            // M - CM
            vDSP_vsubD(cm.real, 1, promoted.real, 1, &outReal, 1, vDSP_Length(size))
            vDSP_vsubD(cm.imag, 1, promoted.imag, 1, &outImag, 1, vDSP_Length(size))
        } else {
            // CM - M
            vDSP_vsubD(promoted.real, 1, cm.real, 1, &outReal, 1, vDSP_Length(size))
            vDSP_vsubD(promoted.imag, 1, cm.imag, 1, &outImag, 1, vDSP_Length(size))
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// complexMatrix ± complexMatrix: element-wise on real+imag blocks.
    ///
    /// Uses vDSP_vadd/vsubD on the flat row-major arrays for cache-efficient
    /// iteration. Shape validation is Group-A (throw before any allocation).
    static func evalComplexMatrixPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        guard case .complexMatrix(let l) = lhs,
              case .complexMatrix(let r) = rhs else {
            throw MathExprError.invalidArguments(
                "evalComplexMatrixPlusComplexMatrix: expected two complexMatrix operands")
        }
        let isAdd = op == .add
        // Group-A: validate before allocation
        try validateCMSameShape(isAdd ? "add" : "sub", lhs: l, rhs: r)
        try LinAlg.checkSoftCap(rows: l.rows, cols: l.cols)
        let size = l.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        if isAdd {
            vDSP_vaddD(l.real, 1, r.real, 1, &outReal, 1, vDSP_Length(size))
            vDSP_vaddD(l.imag, 1, r.imag, 1, &outImag, 1, vDSP_Length(size))
        } else {
            // vDSP_vsubD(B, 1, A, 1, C, 1, N) computes C = A - B
            vDSP_vsubD(r.real, 1, l.real, 1, &outReal, 1, vDSP_Length(size))
            vDSP_vsubD(r.imag, 1, l.imag, 1, &outImag, 1, vDSP_Length(size))
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: l.rows, cols: l.cols, real: outReal, imag: outImag))
    }
}

// MARK: - Mul EVAL implementations

extension NumericDispatch {

    /// scalar * complexMatrix: broadcast real scalar over both blocks.
    static func evalScalarMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        let (scalar, cm): (Double, LinAlg.ComplexMatrix)
        if case .scalar(let s) = lhs, case .complexMatrix(let c) = rhs {
            (scalar, cm) = (s, c)
        } else if case .complexMatrix(let c) = lhs, case .scalar(let s) = rhs {
            (scalar, cm) = (s, c)
        } else {
            throw MathExprError.invalidArguments(
                "evalScalarMulComplexMatrix: unexpected kinds")
        }
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        var s = scalar
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        vDSP_vsmulD(cm.real, 1, &s, &outReal, 1, vDSP_Length(size))
        vDSP_vsmulD(cm.imag, 1, &s, &outImag, 1, vDSP_Length(size))
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// complex * matrix: promote M → CM, then element-wise complex multiply.
    ///
    /// (a+bi)*(mr+0i) = (a*mr) + (b*mr)*i — the promoted matrix has zero imag.
    static func evalComplexMulMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        let (z, matrix): (Complex, LinAlg.Matrix)
        if case .complex(let c) = lhs, case .matrix(let m) = rhs {
            (z, matrix) = (c, m)
        } else if case .matrix(let m) = lhs, case .complex(let c) = rhs {
            (z, matrix) = (c, m)
        } else {
            throw MathExprError.invalidArguments(
                "evalComplexMulMatrix: unexpected kinds")
        }
        let cm = promoteToComplexMatrix(matrix)
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        // (a+bi)*(mr+0i) = a*mr + i*(b*mr)
        var re = z.re, im = z.im
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        vDSP_vsmulD(cm.real, 1, &re, &outReal, 1, vDSP_Length(size))
        vDSP_vsmulD(cm.real, 1, &im, &outImag, 1, vDSP_Length(size))
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// complex * complexMatrix: broadcast complex scalar over elements.
    ///
    /// Per element: (a+bi)*(cr+ci*i) = (a*cr - b*ci) + i*(a*ci + b*cr).
    static func evalComplexMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        let (z, cm): (Complex, LinAlg.ComplexMatrix)
        if case .complex(let c) = lhs, case .complexMatrix(let m) = rhs {
            (z, cm) = (c, m)
        } else if case .complexMatrix(let m) = lhs, case .complex(let c) = rhs {
            (z, cm) = (c, m)
        } else {
            throw MathExprError.invalidArguments(
                "evalComplexMulComplexMatrix: unexpected kinds")
        }
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        // (a+bi)(cr+ci*i) = (a*cr - b*ci) + i*(a*ci + b*cr)
        for i in 0..<size {
            outReal[i] = z.re * cm.real[i] - z.im * cm.imag[i]
            outImag[i] = z.re * cm.imag[i] + z.im * cm.real[i]
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// matrix * complexMatrix: promote real M → CM, then complex matmul.
    static func evalMatrixMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        let (matrix, cm, matrixIsLHS): (LinAlg.Matrix, LinAlg.ComplexMatrix, Bool)
        if case .matrix(let m) = lhs, case .complexMatrix(let c) = rhs {
            (matrix, cm, matrixIsLHS) = (m, c, true)
        } else if case .complexMatrix(let c) = lhs, case .matrix(let m) = rhs {
            (matrix, cm, matrixIsLHS) = (m, c, false)
        } else {
            throw MathExprError.invalidArguments(
                "evalMatrixMulComplexMatrix: unexpected kinds")
        }
        let promoted = promoteToComplexMatrix(matrix)
        let (lCM, rCM) = matrixIsLHS ? (promoted, cm) : (cm, promoted)
        // Delegate to the core complex matmul
        return try complexMatmul(lhs: lCM, rhs: rCM)
    }

    /// complexMatrix * complexMatrix: real-block decomposition matmul (§4.8).
    ///
    /// **§4.3a coercion contract:** calls `coerce1x1Complex` on the result so
    /// a 1×1 result (vec·vec) collapses to `.complex` per §15 truth table.
    static func evalComplexMatrixMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        guard case .complexMatrix(let l) = lhs,
              case .complexMatrix(let r) = rhs else {
            throw MathExprError.invalidArguments(
                "evalComplexMatrixMulComplexMatrix: expected two complexMatrix operands")
        }
        return try complexMatmul(lhs: l, rhs: r)
    }

    // MARK: Core complex matmul

    /// Implements CM*CM via real-block decomposition: Cr=Ar·Br−Ai·Bi, Ci=Ar·Bi+Ai·Br.
    ///
    /// - Performs Group-A shape pre-validation.
    /// - Soft-cap covers all four intermediate real products AND the final result.
    /// - Calls `coerce1x1Complex` for the 1×1 vec·vec → .complex collapse (§4.3a).
    ///
    /// Each real product is delegated to LinAlg.dot (BLAS/Accelerate-backed).
    private static func complexMatmul(
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
        // §4.8 soft-cap: guard result + all four intermediate real products
        // All five have the same shape (resultRows × resultCols)
        try LinAlg.checkSoftCap(rows: resultRows, cols: resultCols)

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
}

// MARK: - Div EVAL implementations

extension NumericDispatch {

    /// matrix / complex scalar: element-wise division by complex.
    ///
    /// (mr+0i) / (c+di) = [(mr*c + 0*d) + i*(0*c - mr*d)] / (c²+d²)
    ///                   = (mr*c/(c²+d²)) + i*(-mr*d/(c²+d²))
    static func evalMatrixDivComplex(
        matrix: LinAlg.Matrix, divisor: Complex
    ) throws -> NumericValue {
        let cm = promoteToComplexMatrix(matrix)
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let denom = divisor.re * divisor.re + divisor.im * divisor.im
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        let scaleRe = divisor.re / denom
        let scaleIm = -divisor.im / denom
        var sRe = scaleRe, sIm = scaleIm
        vDSP_vsmulD(cm.real, 1, &sRe, &outReal, 1, vDSP_Length(size))
        vDSP_vsmulD(cm.real, 1, &sIm, &outImag, 1, vDSP_Length(size))
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// complexMatrix / scalar: element-wise division by real scalar.
    static func evalComplexMatrixDivScalar(
        cm: LinAlg.ComplexMatrix, scalar: Double
    ) throws -> NumericValue {
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        var s = scalar
        vDSP_vsdivD(cm.real, 1, &s, &outReal, 1, vDSP_Length(size))
        vDSP_vsdivD(cm.imag, 1, &s, &outImag, 1, vDSP_Length(size))
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// complexMatrix / complex scalar: element-wise complex division.
    ///
    /// (cr+ci*i) / (a+bi) = [(cr*a+ci*b) + i*(ci*a-cr*b)] / (a²+b²)
    static func evalComplexMatrixDivComplex(
        cm: LinAlg.ComplexMatrix, divisor: Complex
    ) throws -> NumericValue {
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let denom = divisor.re * divisor.re + divisor.im * divisor.im
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        for i in 0..<size {
            let (re, im) = divideComplex(re: cm.real[i], im: cm.imag[i], by: divisor)
            _ = denom  // suppress unused warning; denom used inside divideComplex
            outReal[i] = re
            outImag[i] = im
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }
}

// MARK: - Pow EVAL implementations

extension NumericDispatch {

    // MARK: Real matrix power (already implemented)

    /// Raise a square real matrix to an integer power via exponentiation-by-squaring.
    ///
    /// Contracts enforced by the caller (`applyPow`) before this function is invoked:
    ///   - `matrix` is square (`rows == cols`)
    ///   - `exponent` has no fractional part (`exponent == exponent.rounded()`)
    ///
    /// Semantics:
    ///   - `n > 0`: repeated matrix multiplication using exponentiation-by-squaring,
    ///     O(log n) multiplications.
    ///   - `n == 0`: identity matrix of the same size (A⁰ = I by convention).
    ///   - `n < 0`: `inv(A^|n|)`; throws `MathExprError.invalidArguments("inverse of singular
    ///     matrix")` when A is singular.
    ///
    /// - Parameters:
    ///   - matrix:   A square `LinAlg.Matrix`.
    ///   - exponent: Integer-valued `Double` exponent (may be negative).
    /// - Returns: `NumericValue.matrix(_)` containing the result.
    /// - Throws: `MathExprError.invalidArguments` when the matrix is singular and `n < 0`;
    ///           `LinAlgError.notSquare` propagated from `LinAlg.inv` if shapes are wrong
    ///           (defensive — caller already checked squareness).
    static func evalMatrixPow(
        matrix: LinAlg.Matrix, exponent: Double
    ) throws -> NumericValue {
        let n = Int(exponent)       // caller guarantees no fractional part

        // n == 0 → A⁰ = identity regardless of A (even singular)
        if n == 0 {
            return .matrix(LinAlg.eye(matrix.rows))
        }

        // For negative exponents compute A^|n| then invert
        let absN = n < 0 ? -n : n

        // Exponentiation by squaring: O(log |n|) multiplications
        var result = LinAlg.eye(matrix.rows)    // accumulator starts as identity
        var base   = matrix                     // running square

        var remaining = absN
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = LinAlg.dot(result, base)
            }
            base      = LinAlg.dot(base, base)
            remaining >>= 1
        }

        if n < 0 {
            // Negative power: invert the positive-power result
            guard let invResult = try LinAlg.inv(result) else {
                throw MathExprError.invalidArguments(
                    "matrix power A^\(n): the matrix (or A^\(absN)) is singular; "
                    + "negative powers require an invertible matrix")
            }
            return .matrix(invResult)
        }
        return .matrix(result)
    }

    /// complexMatrix^n integer power — deferred to a future task.
    ///
    /// Complex-matrix integer power is deferred; the hard preconditions (square,
    /// integer exponent) are already checked by the caller (`applyPow`).
    static func evalComplexMatrixPow(
        cm: LinAlg.ComplexMatrix, exponent: Double
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix^scalar (Task 13)")
    }
}

// MARK: - Unary EVAL implementations

extension NumericDispatch {

    /// neg(complexMatrix): element-wise negate both real and imaginary arrays.
    static func evalNegComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        var negOne = -1.0
        vDSP_vsmulD(cm.real, 1, &negOne, &outReal, 1, vDSP_Length(size))
        vDSP_vsmulD(cm.imag, 1, &negOne, &outImag, 1, vDSP_Length(size))
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
    }

    /// Plain (non-Hermitian) transpose of complexMatrix.
    ///
    /// Swaps rows and cols without conjugation. Conjugate-transpose (†) is
    /// deferred to v-next §14.
    ///
    /// Row-major index mapping: element at (r, c) in the input becomes (c, r)
    /// in the output. Output shape: (cm.cols × cm.rows).
    static func evalTransposeComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        let outRows = cm.cols
        let outCols = cm.rows
        try LinAlg.checkSoftCap(rows: outRows, cols: outCols)
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        for r in 0..<cm.rows {
            for c in 0..<cm.cols {
                let srcIdx = r * cm.cols + c
                let dstIdx = c * outCols + r
                outReal[dstIdx] = cm.real[srcIdx]
                outImag[dstIdx] = cm.imag[srcIdx]
            }
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: outRows, cols: outCols, real: outReal, imag: outImag))
    }
}

// MARK: - Function EVAL implementations

extension NumericDispatch {

    /// abs(complexMatrix): complex Frobenius norm.
    ///
    /// Formula: sqrt(Σ|z_ij|²) = sqrt(Σ(re²_ij + im²_ij)) per Golub & Van Loan §2.3.2.
    /// Returns a `.scalar` value (the norm is a non-negative real number).
    static func evalAbsComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        let size = cm.size
        var sumSq = 0.0
        var reSq = [Double](repeating: 0, count: size)
        var imSq = [Double](repeating: 0, count: size)
        vDSP_vsqD(cm.real, 1, &reSq, 1, vDSP_Length(size))
        vDSP_vsqD(cm.imag, 1, &imSq, 1, vDSP_Length(size))
        var sumVec = [Double](repeating: 0, count: size)
        vDSP_vaddD(reSq, 1, imSq, 1, &sumVec, 1, vDSP_Length(size))
        vDSP_sveD(sumVec, 1, &sumSq, vDSP_Length(size))
        return .scalar(sumSq.squareRoot())
    }

    /// trace(complexMatrix): sum of complex diagonal elements.
    ///
    /// Returns a `.complex` value. The matrix need not be square — trace sums
    /// min(rows, cols) diagonal elements, consistent with NumPy behaviour.
    /// For the evaluator, the caller (NumericDispatch+UnaryFunctions.swift)
    /// routes this function only for complexMatrix arguments; square-ness
    /// enforcement is not required here.
    static func evalTraceComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        let minDim = min(cm.rows, cm.cols)
        var reSum = 0.0
        var imSum = 0.0
        for i in 0..<minDim {
            let idx = i * cm.cols + i
            reSum += cm.real[idx]
            imSum += cm.imag[idx]
        }
        return .complex(Complex(re: reSum, im: imSum))
    }

    /// dotProduct(CM, CM): bilinear complex dot (DOM-06).
    ///
    /// Computes Σ aᵢ·bᵢ with NO conjugation of either operand:
    ///   re = Σ(ar·br − ai·bi)
    ///   im = Σ(ar·bi + ai·br)
    ///
    /// This is the bilinear form, NOT the Hermitian inner product. The conjugated
    /// form (vdot) is deferred to v-next §14.
    ///
    /// **§4.3a coercion contract:** calls `coerce1x1Complex` so a 1×1 result
    /// collapses to `.complex` per §15 truth table.
    ///
    /// Shape semantics: treats operands as flat arrays (matches the vec·vec
    /// contract). Validates that total element counts match.
    static func evalComplexMatrixDotProduct(
        lhs: LinAlg.ComplexMatrix, rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // Group-A: validate conformant shapes for dot product
        try validateCMMatmulShape(lhs: lhs, rhs: rhs)
        // Soft cap: 1×1 result is always trivially under cap, but check uniformly
        try LinAlg.checkSoftCap(rows: 1, cols: 1)

        // Delegate to the core complex matmul (handles vec·vec → 1×1 → coerce)
        return try complexMatmul(lhs: lhs, rhs: rhs)
    }

    /// hadamard(CM, CM): element-wise complex product.
    ///
    /// Per element: (ar+ai·i)(br+bi·i) = (ar·br − ai·bi) + i(ar·bi + ai·br).
    /// This is true per-element complex multiplication, NOT the matrix multiply
    /// from `evalComplexMatrixMulComplexMatrix`.
    static func evalComplexHadamard(
        lhs: LinAlg.ComplexMatrix, rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // Group-A: validate equal shapes
        try validateCMSameShape("hadamard", lhs: lhs, rhs: rhs)
        try LinAlg.checkSoftCap(rows: lhs.rows, cols: lhs.cols)
        let size = lhs.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        for i in 0..<size {
            // (ar+ai·i)(br+bi·i) = (ar·br − ai·bi) + i(ar·bi + ai·br)
            outReal[i] = lhs.real[i] * rhs.real[i] - lhs.imag[i] * rhs.imag[i]
            outImag[i] = lhs.real[i] * rhs.imag[i] + lhs.imag[i] * rhs.real[i]
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: lhs.rows, cols: lhs.cols, real: outReal, imag: outImag))
    }
}
