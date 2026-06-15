//
//  LinAlg+Matrix.swift
//  NumericSwift
//
//  The `Matrix` struct and its operator extensions.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Matrix Structure

    /// A matrix stored in row-major order.
    ///
    /// Provides linear algebra operations using the Accelerate framework.
    /// For vectors, use a Matrix with cols = 1.
    ///
    /// `Matrix` is `Sendable`: its storage consists solely of `Int` fields and a
    /// `[Double]` array, both of which are value types safe to share across
    /// concurrency boundaries without additional synchronisation.
    public struct Matrix: Equatable, CustomStringConvertible, Sendable {
        /// Number of rows
        public let rows: Int
        /// Number of columns
        public let cols: Int
        /// Data stored in row-major order
        public var data: [Double]

        /// Whether this is a vector (single column)
        public var isVector: Bool { cols == 1 }

        /// Total number of elements
        public var size: Int { rows * cols }

        /// Shape as (rows, cols)
        public var shape: (Int, Int) { (rows, cols) }

        /// Create a matrix from data in row-major order.
        ///
        /// - Parameters:
        ///   - rows: Number of rows
        ///   - cols: Number of columns
        ///   - data: Data in row-major order
        public init(rows: Int, cols: Int, data: [Double]) {
            LinAlg.assertWithinHardCap(rows: rows, cols: cols)
            precondition(data.count == rows * cols, "Data size must equal rows * cols")
            self.rows = rows
            self.cols = cols
            self.data = data
        }

        /// Create a matrix from a 2D array.
        ///
        /// - Parameter array: 2D array where each inner array is a row
        public init(_ array: [[Double]]) {
            precondition(!array.isEmpty, "Array cannot be empty")
            let rows = array.count
            let cols = array[0].count
            precondition(array.allSatisfy { $0.count == cols }, "All rows must have same length")
            LinAlg.assertWithinHardCap(rows: rows, cols: cols)
            self.rows = rows
            self.cols = cols
            self.data = array.flatMap { $0 }
        }

        /// Create a vector from a 1D array.
        ///
        /// - Parameter array: 1D array of values
        public init(_ array: [Double]) {
            LinAlg.assertWithinHardCap(rows: array.count, cols: 1)
            self.rows = array.count
            self.cols = 1
            self.data = array
        }

        /// Access element at (row, col) using 0-based indexing.
        public subscript(row: Int, col: Int) -> Double {
            get {
                precondition(row >= 0 && row < rows && col >= 0 && col < cols)
                return data[row * cols + col]
            }
            set {
                precondition(row >= 0 && row < rows && col >= 0 && col < cols)
                data[row * cols + col] = newValue
            }
        }

        /// Access element at index (for vectors).
        public subscript(index: Int) -> Double {
            get {
                precondition(cols == 1 && index >= 0 && index < rows)
                return data[index]
            }
            set {
                precondition(cols == 1 && index >= 0 && index < rows)
                data[index] = newValue
            }
        }

        public var description: String {
            if cols == 1 {
                return "Vector(\(rows))"
            }
            return "Matrix(\(rows)x\(cols))"
        }

        /// Convert to 2D array.
        public func toArray() -> [[Double]] {
            var result = [[Double]]()
            for i in 0..<rows {
                var row = [Double]()
                for j in 0..<cols {
                    row.append(data[i * cols + j])
                }
                result.append(row)
            }
            return result
        }

        /// Get a single row as a row vector.
        public func row(_ i: Int) -> Matrix {
            precondition(i >= 0 && i < rows)
            let start = i * cols
            return Matrix(rows: 1, cols: cols, data: Array(data[start..<(start + cols)]))
        }

        /// Get a single column as a column vector.
        public func col(_ j: Int) -> Matrix {
            precondition(j >= 0 && j < cols)
            var colData = [Double](repeating: 0, count: rows)
            for i in 0..<rows {
                colData[i] = data[i * cols + j]
            }
            return Matrix(rows: rows, cols: 1, data: colData)
        }

        /// Transpose the matrix.
        public var T: Matrix {
            var result = [Double](repeating: 0, count: rows * cols)
            for i in 0..<rows {
                for j in 0..<cols {
                    result[j * rows + i] = data[i * cols + j]
                }
            }
            return Matrix(rows: cols, cols: rows, data: result)
        }

        public static func == (lhs: Matrix, rhs: Matrix) -> Bool {
            guard lhs.rows == rhs.rows && lhs.cols == rhs.cols else { return false }
            for i in 0..<lhs.data.count {
                if abs(lhs.data[i] - rhs.data[i]) > 1e-10 {
                    return false
                }
            }
            return true
        }

        /// Compute the matrix inverse.
        ///
        /// - Returns: The inverse matrix, or `nil` if the matrix is singular.
        /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when the matrix is not square.
        public func inverse() throws -> Matrix? {
            return try LinAlg.inv(self)
        }

        /// Compute the Moore-Penrose pseudoinverse.
        ///
        /// For non-square or rank-deficient matrices, returns the least-squares solution.
        /// For invertible square matrices, equals `inverse()`.
        ///
        /// - Parameter rcond: Cutoff for small singular values (default: 1e-15).
        /// - Returns: The pseudoinverse matrix.
        public func pinv(rcond: Double = 1e-15) -> Matrix {
            return LinAlg.pinv(self, rcond: rcond)
        }
    }
}

// MARK: - Matrix Operators (for convenience)

extension LinAlg.Matrix {
    /// Add two matrices.
    public static func + (lhs: LinAlg.Matrix, rhs: LinAlg.Matrix) -> LinAlg.Matrix {
        LinAlg.add(lhs, rhs)
    }

    /// Subtract two matrices.
    public static func - (lhs: LinAlg.Matrix, rhs: LinAlg.Matrix) -> LinAlg.Matrix {
        LinAlg.sub(lhs, rhs)
    }

    /// Negate a matrix.
    public static prefix func - (m: LinAlg.Matrix) -> LinAlg.Matrix {
        LinAlg.neg(m)
    }

    /// Multiply matrix by scalar.
    public static func * (lhs: Double, rhs: LinAlg.Matrix) -> LinAlg.Matrix {
        LinAlg.mul(lhs, rhs)
    }

    /// Multiply matrix by scalar.
    public static func * (lhs: LinAlg.Matrix, rhs: Double) -> LinAlg.Matrix {
        LinAlg.mul(lhs, rhs)
    }

    /// Matrix multiplication.
    public static func * (lhs: LinAlg.Matrix, rhs: LinAlg.Matrix) -> LinAlg.Matrix {
        LinAlg.dot(lhs, rhs)
    }

    /// Divide matrix by scalar.
    public static func / (lhs: LinAlg.Matrix, rhs: Double) -> LinAlg.Matrix {
        LinAlg.div(lhs, rhs)
    }
}
