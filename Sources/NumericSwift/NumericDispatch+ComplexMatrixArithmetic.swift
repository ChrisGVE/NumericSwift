//
//  NumericDispatch+ComplexMatrixArithmetic.swift
//  NumericSwift
//
//  Complex-matrix add, subtract, and multiply EVAL implementations for the
//  unified numeric pipeline.
//
//  Shared helpers (validateCMSameShape, validateCMMatmulShape, realBlock,
//  imagBlock, divideComplex, complexMatmul) live in
//  NumericDispatch+ComplexMatrixHelpers.swift.
//
//  ## File layout for complex-matrix arithmetic
//
//    NumericDispatch+ComplexMatrixHelpers.swift    — shared helpers + complexMatmul
//    NumericDispatch+ComplexMatrixArithmetic.swift — this file: add/sub/mul
//    NumericDispatch+ComplexMatrixFunctions.swift  — div/unary/function impls
//    NumericDispatch+MatrixPower.swift             — real + complex matrix pow
//
//  Licensed under the Apache License, Version 2.0.
//

import Accelerate

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
            var s = scalarVal
            vDSP_vsaddD(matrix.data, 1, &s, &result, 1, vDSP_Length(size))
        } else {
            // sub: either scalar - M or M - scalar
            var s = scalarIsLHS ? scalarVal : -scalarVal
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
        return try evalComplexPlusMatrixImpl(
            cm: cm, z: z, complexIsLHS: complexIsLHS, isAdd: isAdd,
            size: size, outReal: &outReal, outImag: &outImag)
    }

    private static func evalComplexPlusMatrixImpl(
        cm: LinAlg.ComplexMatrix, z: Complex, complexIsLHS: Bool, isAdd: Bool,
        size: Int, outReal: inout [Double], outImag: inout [Double]
    ) throws -> NumericValue {
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
}
