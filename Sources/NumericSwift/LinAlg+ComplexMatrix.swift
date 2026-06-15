//
//  LinAlg+ComplexMatrix.swift
//  NumericSwift
//
//  The `ComplexMatrix` struct.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Complex Matrix Structure

    /// A complex matrix with separate real and imaginary components.
    ///
    /// Stored in row-major order for both real and imaginary parts.
    ///
    /// `ComplexMatrix` is `Sendable`: its storage consists solely of `Int` fields and
    /// two `[Double]` arrays (real and imaginary parts), all of which are value types
    /// safe to share across concurrency boundaries without additional synchronisation.
    public struct ComplexMatrix: Equatable, CustomStringConvertible, Sendable {
        /// Number of rows
        public let rows: Int
        /// Number of columns
        public let cols: Int
        /// Real part data in row-major order
        public var real: [Double]
        /// Imaginary part data in row-major order
        public var imag: [Double]

        /// Whether this is a vector (single column)
        public var isVector: Bool { cols == 1 }

        /// Total number of elements
        public var size: Int { rows * cols }

        /// Shape as (rows, cols)
        public var shape: (Int, Int) { (rows, cols) }

        /// Create a complex matrix from separate real and imaginary data.
        public init(rows: Int, cols: Int, real: [Double], imag: [Double]) {
            LinAlg.assertWithinHardCap(rows: rows, cols: cols)
            precondition(real.count == rows * cols && imag.count == rows * cols,
                         "Data size must equal rows * cols")
            self.rows = rows
            self.cols = cols
            self.real = real
            self.imag = imag
        }

        /// Create a complex matrix from a real matrix (zero imaginary part).
        ///
        /// The source `Matrix` was already hard-cap validated at construction.
        /// The assertion here is defensive — it is a no-op in practice.
        public init(_ matrix: Matrix) {
            LinAlg.assertWithinHardCap(rows: matrix.rows, cols: matrix.cols)
            self.rows = matrix.rows
            self.cols = matrix.cols
            self.real = matrix.data
            self.imag = [Double](repeating: 0, count: matrix.size)
        }

        /// Create a complex matrix from 2D arrays of real and imaginary parts.
        public init(real: [[Double]], imag: [[Double]]) {
            precondition(!real.isEmpty && real.count == imag.count)
            let rows = real.count
            let cols = real[0].count
            precondition(real.allSatisfy { $0.count == cols } &&
                         imag.allSatisfy { $0.count == cols })
            LinAlg.assertWithinHardCap(rows: rows, cols: cols)
            self.rows = rows
            self.cols = cols
            self.real = real.flatMap { $0 }
            self.imag = imag.flatMap { $0 }
        }

        /// Access element at (row, col).
        public subscript(row: Int, col: Int) -> (re: Double, im: Double) {
            get {
                precondition(row >= 0 && row < rows && col >= 0 && col < cols)
                let idx = row * cols + col
                return (real[idx], imag[idx])
            }
            set {
                precondition(row >= 0 && row < rows && col >= 0 && col < cols)
                let idx = row * cols + col
                real[idx] = newValue.re
                imag[idx] = newValue.im
            }
        }

        public var description: String {
            if cols == 1 {
                return "ComplexVector(\(rows))"
            }
            return "ComplexMatrix(\(rows)x\(cols))"
        }

        public static func == (lhs: ComplexMatrix, rhs: ComplexMatrix) -> Bool {
            guard lhs.rows == rhs.rows && lhs.cols == rhs.cols else { return false }
            for i in 0..<lhs.real.count {
                if abs(lhs.real[i] - rhs.real[i]) > 1e-10 ||
                   abs(lhs.imag[i] - rhs.imag[i]) > 1e-10 {
                    return false
                }
            }
            return true
        }
    }
}
