//
//  LinAlg+Arithmetic.swift
//  NumericSwift
//
//  Matrix creation (factory functions) and basic arithmetic operations.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Matrix Creation

    /// Create a matrix/vector of zeros.
    public static func zeros(_ rows: Int, _ cols: Int = 1) -> Matrix {
        // Validate the element count overflow-safely BEFORE `rows * cols` is formed
        // for the array `count:` (a raw multiply would trap on overflow with an
        // opaque message); this surfaces the hard-cap precondition instead.
        assertWithinHardCap(rows: rows, cols: cols)
        return Matrix(rows: rows, cols: cols, data: [Double](repeating: 0, count: rows * cols))
    }

    /// Create a matrix/vector of ones.
    public static func ones(_ rows: Int, _ cols: Int = 1) -> Matrix {
        assertWithinHardCap(rows: rows, cols: cols)
        return Matrix(rows: rows, cols: cols, data: [Double](repeating: 1, count: rows * cols))
    }

    /// Create an identity matrix.
    public static func eye(_ n: Int) -> Matrix {
        assertWithinHardCap(rows: n, cols: n)
        var data = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            data[i * n + i] = 1.0
        }
        return Matrix(rows: n, cols: n, data: data)
    }

    /// Create a diagonal matrix from a vector.
    public static func diag(_ values: [Double]) -> Matrix {
        let n = values.count
        assertWithinHardCap(rows: n, cols: n)
        var data = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            data[i * n + i] = values[i]
        }
        return Matrix(rows: n, cols: n, data: data)
    }

    /// Extract diagonal from a matrix.
    public static func diag(_ m: Matrix) -> [Double] {
        let minDim = min(m.rows, m.cols)
        var result = [Double](repeating: 0, count: minDim)
        for i in 0..<minDim {
            result[i] = m.data[i * m.cols + i]
        }
        return result
    }

    /// Create a range vector.
    /// - Throws: ``LinAlgError/invalidParameter(_:)`` when `step` is zero.
    public static func arange(_ start: Double, _ stop: Double, _ step: Double = 1.0) throws -> Matrix {
        guard step != 0 else { throw LinAlgError.invalidParameter("step must be non-zero") }
        var data = [Double]()
        var current = start
        if step > 0 {
            while current < stop {
                data.append(current)
                current += step
            }
        } else {
            while current > stop {
                data.append(current)
                current += step
            }
        }
        return Matrix(data)
    }

    /// Create a vector of evenly spaced values.
    /// - Throws: ``LinAlgError/invalidParameter(_:)`` when `count` is less than 2.
    public static func linspace(_ start: Double, _ stop: Double, _ count: Int) throws -> Matrix {
        guard count >= 2 else { throw LinAlgError.invalidParameter("count must be at least 2") }
        var data = [Double]()
        data.reserveCapacity(count)
        let step = (stop - start) / Double(count - 1)
        for i in 0..<count {
            data.append(start + Double(i) * step)
        }
        return Matrix(data)
    }

    // MARK: - Basic Arithmetic

    /// Add two matrices element-wise.
    public static func add(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var result = [Double](repeating: 0, count: lhs.size)
        vDSP_vaddD(lhs.data, 1, rhs.data, 1, &result, 1, vDSP_Length(lhs.size))
        return Matrix(rows: lhs.rows, cols: lhs.cols, data: result)
    }

    /// Subtract two matrices element-wise.
    public static func sub(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var result = [Double](repeating: 0, count: lhs.size)
        vDSP_vsubD(rhs.data, 1, lhs.data, 1, &result, 1, vDSP_Length(lhs.size))
        return Matrix(rows: lhs.rows, cols: lhs.cols, data: result)
    }

    /// Negate a matrix.
    public static func neg(_ m: Matrix) -> Matrix {
        var result = [Double](repeating: 0, count: m.size)
        var neg = -1.0
        vDSP_vsmulD(m.data, 1, &neg, &result, 1, vDSP_Length(m.size))
        return Matrix(rows: m.rows, cols: m.cols, data: result)
    }

    /// Multiply matrix by scalar.
    public static func mul(_ scalar: Double, _ m: Matrix) -> Matrix {
        var result = [Double](repeating: 0, count: m.size)
        var s = scalar
        vDSP_vsmulD(m.data, 1, &s, &result, 1, vDSP_Length(m.size))
        return Matrix(rows: m.rows, cols: m.cols, data: result)
    }

    /// Multiply matrix by scalar.
    public static func mul(_ m: Matrix, _ scalar: Double) -> Matrix {
        mul(scalar, m)
    }

    /// Divide matrix by scalar.
    public static func div(_ m: Matrix, _ scalar: Double) -> Matrix {
        precondition(scalar != 0, "Division by zero")
        var result = [Double](repeating: 0, count: m.size)
        var s = scalar
        vDSP_vsdivD(m.data, 1, &s, &result, 1, vDSP_Length(m.size))
        return Matrix(rows: m.rows, cols: m.cols, data: result)
    }

    /// Element-wise multiplication (Hadamard product).
    public static func hadamard(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var result = [Double](repeating: 0, count: lhs.size)
        vDSP_vmulD(lhs.data, 1, rhs.data, 1, &result, 1, vDSP_Length(lhs.size))
        return Matrix(rows: lhs.rows, cols: lhs.cols, data: result)
    }

    /// Element-wise division.
    public static func elementDiv(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var result = [Double](repeating: 0, count: lhs.size)
        vDSP_vdivD(rhs.data, 1, lhs.data, 1, &result, 1, vDSP_Length(lhs.size))
        return Matrix(rows: lhs.rows, cols: lhs.cols, data: result)
    }

    // MARK: - Matrix Multiplication

    /// Matrix multiplication or dot product.
    public static func dot(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
        // Vector dot product
        if lhs.cols == 1 && rhs.cols == 1 {
            precondition(lhs.rows == rhs.rows, "Vectors must have same length")
            var result: Double = 0
            vDSP_dotprD(lhs.data, 1, rhs.data, 1, &result, vDSP_Length(lhs.rows))
            return Matrix([result])
        }

        // Matrix-vector multiplication
        if rhs.cols == 1 {
            precondition(lhs.cols == rhs.rows, "Incompatible dimensions")
            var result = [Double](repeating: 0, count: lhs.rows)
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        Int32(lhs.rows), Int32(lhs.cols),
                        1.0, lhs.data, Int32(lhs.cols),
                        rhs.data, 1,
                        0.0, &result, 1)
            return Matrix(rows: lhs.rows, cols: 1, data: result)
        }

        // Matrix multiplication
        precondition(lhs.cols == rhs.rows, "Incompatible dimensions")
        var result = [Double](repeating: 0, count: lhs.rows * rhs.cols)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(lhs.rows), Int32(rhs.cols), Int32(lhs.cols),
                    1.0, lhs.data, Int32(lhs.cols),
                    rhs.data, Int32(rhs.cols),
                    0.0, &result, Int32(rhs.cols))
        return Matrix(rows: lhs.rows, cols: rhs.cols, data: result)
    }
}
