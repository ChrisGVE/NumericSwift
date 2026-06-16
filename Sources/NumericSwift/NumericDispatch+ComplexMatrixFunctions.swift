//
//  NumericDispatch+ComplexMatrixFunctions.swift
//  NumericSwift
//
//  Complex-matrix division, unary, and named-function EVAL implementations
//  for the unified numeric pipeline.
//
//  Covers:
//    - Division: matrix/complex, complexMatrix/scalar, complexMatrix/complex
//    - Unary:    neg(complexMatrix), transpose(complexMatrix)
//    - Functions: abs(complexMatrix), trace(complexMatrix),
//                 dotProduct(CM, CM), hadamard(CM, CM)
//
//  The shared helpers (validateCMSameShape, validateCMMatmulShape, realBlock,
//  imagBlock, divideComplex, complexMatmul) live in
//  NumericDispatch+ComplexMatrixHelpers.swift.
//
//  Licensed under the Apache License, Version 2.0.
//

import Accelerate

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
        let size = cm.size
        var outReal = [Double](repeating: 0, count: size)
        var outImag = [Double](repeating: 0, count: size)
        for i in 0..<size {
            let (re, im) = divideComplex(re: cm.real[i], im: cm.imag[i], by: divisor)
            outReal[i] = re
            outImag[i] = im
        }
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: cm.rows, cols: cm.cols, real: outReal, imag: outImag))
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
