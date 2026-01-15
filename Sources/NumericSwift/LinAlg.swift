//
//  LinAlg.swift
//  NumericSwift
//
//  Linear algebra operations using Accelerate framework for hardware-accelerated computation.
//  Follows NumPy/scipy.linalg patterns with namespaced API.
//
//  Licensed under the MIT License.
//

import Foundation
import Accelerate

// MARK: - LinAlg Namespace

/// Linear algebra operations namespace.
///
/// Provides matrix operations, decompositions, and solvers using the Accelerate framework.
/// All operations use hardware-accelerated LAPACK/BLAS routines.
///
/// Example usage:
/// ```swift
/// let A = LinAlg.Matrix([[1, 2], [3, 4]])
/// let b = LinAlg.Matrix([5, 6])
/// let x = LinAlg.solve(A, b)
/// ```
public enum LinAlg {

    // MARK: - Matrix Structure

    /// A matrix stored in row-major order.
    ///
    /// Provides linear algebra operations using the Accelerate framework.
    /// For vectors, use a Matrix with cols = 1.
    public struct Matrix: Equatable, CustomStringConvertible {
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
            self.rows = rows
            self.cols = cols
            self.data = array.flatMap { $0 }
        }

        /// Create a vector from a 1D array.
        ///
        /// - Parameter array: 1D array of values
        public init(_ array: [Double]) {
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
    }

    // MARK: - Complex Matrix Structure

    /// A complex matrix with separate real and imaginary components.
    ///
    /// Stored in row-major order for both real and imaginary parts.
    public struct ComplexMatrix: Equatable, CustomStringConvertible {
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
            precondition(real.count == rows * cols && imag.count == rows * cols,
                         "Data size must equal rows * cols")
            self.rows = rows
            self.cols = cols
            self.real = real
            self.imag = imag
        }

        /// Create a complex matrix from a real matrix (zero imaginary part).
        public init(_ matrix: Matrix) {
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

    // MARK: - Matrix Creation

    /// Create a matrix/vector of zeros.
    public static func zeros(_ rows: Int, _ cols: Int = 1) -> Matrix {
        Matrix(rows: rows, cols: cols, data: [Double](repeating: 0, count: rows * cols))
    }

    /// Create a matrix/vector of ones.
    public static func ones(_ rows: Int, _ cols: Int = 1) -> Matrix {
        Matrix(rows: rows, cols: cols, data: [Double](repeating: 1, count: rows * cols))
    }

    /// Create an identity matrix.
    public static func eye(_ n: Int) -> Matrix {
        var data = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            data[i * n + i] = 1.0
        }
        return Matrix(rows: n, cols: n, data: data)
    }

    /// Create a diagonal matrix from a vector.
    public static func diag(_ values: [Double]) -> Matrix {
        let n = values.count
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
    public static func arange(_ start: Double, _ stop: Double, _ step: Double = 1.0) -> Matrix {
        precondition(step != 0, "Step cannot be zero")
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
    public static func linspace(_ start: Double, _ stop: Double, _ count: Int) -> Matrix {
        precondition(count >= 2, "Count must be at least 2")
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

    // MARK: - Matrix Properties

    /// Compute the trace of a square matrix.
    public static func trace(_ m: Matrix) -> Double {
        precondition(m.rows == m.cols, "Matrix must be square")
        var sum = 0.0
        for i in 0..<m.rows {
            sum += m.data[i * m.cols + i]
        }
        return sum
    }

    /// Compute the determinant of a square matrix.
    public static func det(_ m: Matrix) -> Double {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        var a = m.data
        var n1 = __CLPK_integer(n)
        var n2 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&n1, &n2, &a, &lda, &ipiv, &info)

        if info != 0 { return 0.0 }

        var determinant = 1.0
        var sign = 1
        for i in 0..<n {
            determinant *= a[i * n + i]
            if ipiv[i] != Int32(i + 1) {
                sign *= -1
            }
        }

        return determinant * Double(sign)
    }

    /// Compute the matrix inverse.
    public static func inv(_ m: Matrix) -> Matrix? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        var a = m.data
        var n1 = __CLPK_integer(n)
        var n2 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&n1, &n2, &a, &lda, &ipiv, &info)

        if info != 0 { return nil }

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var ngetri = __CLPK_integer(n)
        var ldagetri = __CLPK_integer(n)
        dgetri_(&ngetri, &a, &ldagetri, &ipiv, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        var ngetri2 = __CLPK_integer(n)
        var ldagetri2 = __CLPK_integer(n)
        dgetri_(&ngetri2, &a, &ldagetri2, &ipiv, &work, &lwork, &info)

        if info != 0 { return nil }

        return Matrix(rows: n, cols: n, data: a)
    }

    /// Compute the matrix rank via SVD.
    public static func rank(_ m: Matrix) -> Int {
        let (s, _, _) = svd(m)
        let tol = max(Double(m.rows), Double(m.cols)) * s[0] * 2.220446049250313e-16
        var r = 0
        for sv in s {
            if sv > tol { r += 1 }
        }
        return r
    }

    /// Compute the condition number of a matrix.
    public static func cond(_ m: Matrix) -> Double {
        let (s, _, _) = svd(m)
        let sMax = s.first ?? 0
        let sMin = s.last ?? 0
        let tol = max(Double(m.rows), Double(m.cols)) * sMax * 2.220446049250313e-16
        if sMin <= tol { return Double.infinity }
        return sMax / sMin
    }

    /// Compute the Moore-Penrose pseudoinverse.
    public static func pinv(_ m: Matrix, rcond: Double = 1e-15) -> Matrix {
        let minDim = min(m.rows, m.cols)

        var a = toColumnMajor(m)

        var m1 = __CLPK_integer(m.rows)
        var n1 = __CLPK_integer(m.cols)
        var lda1 = __CLPK_integer(m.rows)
        var s = [Double](repeating: 0, count: minDim)
        var u = [Double](repeating: 0, count: m.rows * m.rows)
        var vt = [Double](repeating: 0, count: m.cols * m.cols)
        var ldu = __CLPK_integer(m.rows)
        var ldvt = __CLPK_integer(m.cols)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var jobu = Int8(UInt8(ascii: "A"))
        var jobvt = Int8(UInt8(ascii: "A"))

        dgesvd_(&jobu, &jobvt, &m1, &n1, &a, &lda1, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(m)
        var m2 = __CLPK_integer(m.rows)
        var n2 = __CLPK_integer(m.cols)
        var lda2 = __CLPK_integer(m.rows)
        dgesvd_(&jobu, &jobvt, &m2, &n2, &a, &lda2, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)

        if info != 0 { return m }

        let tol = rcond * s[0]

        // Compute V * Σ^(-1) * U^T
        var result = [Double](repeating: 0, count: m.cols * m.rows)

        for i in 0..<m.cols {
            for j in 0..<m.rows {
                var sum = 0.0
                for k in 0..<minDim {
                    if s[k] > tol {
                        let v_ik = vt[i * m.cols + k]
                        let u_jk = u[k * m.rows + j]
                        sum += v_ik * (1.0 / s[k]) * u_jk
                    }
                }
                result[i * m.rows + j] = sum
            }
        }

        return Matrix(rows: m.cols, cols: m.rows, data: result)
    }

    // MARK: - Norms

    /// Compute vector or matrix norm.
    public static func norm(_ m: Matrix, _ p: Double = 2) -> Double {
        if m.cols == 1 {
            // Vector norm
            if p == 1 {
                var result = 0.0
                vDSP_svemgD(m.data, 1, &result, vDSP_Length(m.rows))
                return result
            } else if p == 2 {
                var result = 0.0
                vDSP_dotprD(m.data, 1, m.data, 1, &result, vDSP_Length(m.rows))
                return sqrt(result)
            } else if p == Double.infinity {
                var result = 0.0
                vDSP_maxmgvD(m.data, 1, &result, vDSP_Length(m.rows))
                return result
            } else {
                var sum = 0.0
                for val in m.data {
                    sum += pow(abs(val), p)
                }
                return pow(sum, 1.0/p)
            }
        } else {
            // Matrix norm
            if p == 1 {
                var maxSum = 0.0
                for j in 0..<m.cols {
                    var colSum = 0.0
                    for i in 0..<m.rows {
                        colSum += abs(m.data[i * m.cols + j])
                    }
                    maxSum = max(maxSum, colSum)
                }
                return maxSum
            } else if p == Double.infinity {
                var maxSum = 0.0
                for i in 0..<m.rows {
                    var rowSum = 0.0
                    for j in 0..<m.cols {
                        rowSum += abs(m.data[i * m.cols + j])
                    }
                    maxSum = max(maxSum, rowSum)
                }
                return maxSum
            } else {
                // Frobenius norm (default)
                var result = 0.0
                vDSP_dotprD(m.data, 1, m.data, 1, &result, vDSP_Length(m.size))
                return sqrt(result)
            }
        }
    }

    // MARK: - Decompositions

    /// LU decomposition with partial pivoting.
    ///
    /// - Parameter m: Square matrix
    /// - Returns: (L, U, P) where P @ L @ U = m
    public static func lu(_ m: Matrix) -> (L: Matrix, U: Matrix, P: Matrix) {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        // Convert to column-major for LAPACK
        var a = toColumnMajor(m)
        var n1 = __CLPK_integer(n)
        var n2 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&n1, &n2, &a, &lda, &ipiv, &info)

        // Extract L and U from column-major result, converting to row-major
        var L = [Double](repeating: 0, count: n * n)
        var U = [Double](repeating: 0, count: n * n)

        for i in 0..<n {
            for j in 0..<n {
                // a is in column-major: element (i,j) is at a[j * n + i]
                let colMajorIdx = j * n + i
                if i > j {
                    // Below diagonal -> L
                    L[i * n + j] = a[colMajorIdx]
                } else if i == j {
                    // Diagonal: L gets 1, U gets the value
                    L[i * n + j] = 1.0
                    U[i * n + j] = a[colMajorIdx]
                } else {
                    // Above diagonal -> U
                    U[i * n + j] = a[colMajorIdx]
                }
            }
        }

        // Create permutation matrix from ipiv
        // ipiv[i] means row i was swapped with row ipiv[i]-1 (1-indexed)
        var P = [Double](repeating: 0, count: n * n)
        var perm = Array(0..<n)
        for i in 0..<n {
            let pivot = Int(ipiv[i]) - 1
            if pivot != i {
                perm.swapAt(i, pivot)
            }
        }
        for i in 0..<n {
            P[i * n + perm[i]] = 1.0
        }

        return (Matrix(rows: n, cols: n, data: L),
                Matrix(rows: n, cols: n, data: U),
                Matrix(rows: n, cols: n, data: P))
    }

    /// QR decomposition.
    ///
    /// - Parameter m: Matrix (m×n)
    /// - Returns: (Q, R) where Q @ R = m
    public static func qr(_ m: Matrix) -> (Q: Matrix, R: Matrix) {
        let minDim = min(m.rows, m.cols)

        var a = toColumnMajor(m)

        var m1 = __CLPK_integer(m.rows)
        var n1 = __CLPK_integer(m.cols)
        var lda1 = __CLPK_integer(m.rows)
        var tau = [Double](repeating: 0, count: minDim)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        dgeqrf_(&m1, &n1, &a, &lda1, &tau, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        var m2 = __CLPK_integer(m.rows)
        var n2 = __CLPK_integer(m.cols)
        var lda2 = __CLPK_integer(m.rows)
        dgeqrf_(&m2, &n2, &a, &lda2, &tau, &work, &lwork, &info)

        // Extract R
        var R = [Double](repeating: 0, count: minDim * m.cols)
        for i in 0..<minDim {
            for j in i..<m.cols {
                R[i * m.cols + j] = a[j * m.rows + i]
            }
        }

        // Generate Q
        var m3 = __CLPK_integer(m.rows)
        var k1 = __CLPK_integer(minDim)
        var k2 = __CLPK_integer(minDim)
        var lda3 = __CLPK_integer(m.rows)
        lwork = -1
        dorgqr_(&m3, &k1, &k2, &a, &lda3, &tau, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))
        var m4 = __CLPK_integer(m.rows)
        var k3 = __CLPK_integer(minDim)
        var k4 = __CLPK_integer(minDim)
        var lda4 = __CLPK_integer(m.rows)
        dorgqr_(&m4, &k3, &k4, &a, &lda4, &tau, &work, &lwork, &info)

        // Convert Q to row-major
        var Q = [Double](repeating: 0, count: m.rows * minDim)
        for i in 0..<m.rows {
            for j in 0..<minDim {
                Q[i * minDim + j] = a[j * m.rows + i]
            }
        }

        return (Matrix(rows: m.rows, cols: minDim, data: Q),
                Matrix(rows: minDim, cols: m.cols, data: R))
    }

    /// Singular Value Decomposition.
    ///
    /// - Parameter m: Matrix (m×n)
    /// - Returns: (s, U, Vt) where U @ diag(s) @ Vt = m
    public static func svd(_ m: Matrix) -> (s: [Double], U: Matrix, Vt: Matrix) {
        let minDim = min(m.rows, m.cols)

        var a = toColumnMajor(m)

        var m1 = __CLPK_integer(m.rows)
        var n1 = __CLPK_integer(m.cols)
        var lda1 = __CLPK_integer(m.rows)
        var s = [Double](repeating: 0, count: minDim)
        var u = [Double](repeating: 0, count: m.rows * m.rows)
        var vt = [Double](repeating: 0, count: m.cols * m.cols)
        var ldu = __CLPK_integer(m.rows)
        var ldvt = __CLPK_integer(m.cols)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var jobu = Int8(UInt8(ascii: "A"))
        var jobvt = Int8(UInt8(ascii: "A"))

        dgesvd_(&jobu, &jobvt, &m1, &n1, &a, &lda1, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(m)
        var m2 = __CLPK_integer(m.rows)
        var n2 = __CLPK_integer(m.cols)
        var lda2 = __CLPK_integer(m.rows)
        dgesvd_(&jobu, &jobvt, &m2, &n2, &a, &lda2, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)

        // Convert U to row-major
        var U = [Double](repeating: 0, count: m.rows * m.rows)
        for i in 0..<m.rows {
            for j in 0..<m.rows {
                U[i * m.rows + j] = u[j * m.rows + i]
            }
        }

        // Convert Vt to row-major
        var Vt = [Double](repeating: 0, count: m.cols * m.cols)
        for i in 0..<m.cols {
            for j in 0..<m.cols {
                Vt[i * m.cols + j] = vt[j * m.cols + i]
            }
        }

        return (s, Matrix(rows: m.rows, cols: m.rows, data: U),
                Matrix(rows: m.cols, cols: m.cols, data: Vt))
    }

    /// Eigenvalue decomposition.
    ///
    /// - Parameter m: Square matrix
    /// - Returns: (eigenvalues, imagParts, eigenvectors)
    public static func eig(_ m: Matrix) -> (values: [Double], imagParts: [Double], vectors: Matrix) {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        var a = toColumnMajor(m)

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var wr = [Double](repeating: 0, count: n)
        var wi = [Double](repeating: 0, count: n)
        var vl = [Double](repeating: 0, count: 1)
        var vr = [Double](repeating: 0, count: n * n)
        var ldvl: __CLPK_integer = 1
        var ldvr = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var jobvl = Int8(UInt8(ascii: "N"))
        var jobvr = Int8(UInt8(ascii: "V"))

        dgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(m)
        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)
        dgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        // Convert eigenvectors to row-major
        var vecs = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                vecs[i * n + j] = vr[j * n + i]
            }
        }

        return (wr, wi, Matrix(rows: n, cols: n, data: vecs))
    }

    /// Eigenvalues only (more efficient than full decomposition).
    public static func eigvals(_ m: Matrix) -> (real: [Double], imag: [Double]) {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        var a = toColumnMajor(m)

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var wr = [Double](repeating: 0, count: n)
        var wi = [Double](repeating: 0, count: n)
        var vl = [Double](repeating: 0, count: 1)
        var vr = [Double](repeating: 0, count: 1)
        var ldvl: __CLPK_integer = 1
        var ldvr: __CLPK_integer = 1
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var jobvl = Int8(UInt8(ascii: "N"))
        var jobvr = Int8(UInt8(ascii: "N"))

        dgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(m)
        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)
        dgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        return (wr, wi)
    }

    /// Cholesky decomposition (for positive definite matrices).
    ///
    /// - Parameter m: Symmetric positive definite matrix
    /// - Returns: Lower triangular L where L @ L^T = m, or nil if not positive definite
    public static func cholesky(_ m: Matrix) -> Matrix? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        var a = toColumnMajor(m)

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var info: __CLPK_integer = 0
        var uplo = Int8(UInt8(ascii: "L"))

        dpotrf_(&uplo, &n1, &a, &lda1, &info)

        if info != 0 { return nil }

        // Convert to row-major and zero upper triangle
        var L = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0...i {
                L[i * n + j] = a[j * n + i]
            }
        }

        return Matrix(rows: n, cols: n, data: L)
    }

    // MARK: - Linear System Solvers

    /// Solve linear system Ax = b.
    ///
    /// - Parameters:
    ///   - A: Square coefficient matrix
    ///   - b: Right-hand side (vector or matrix)
    /// - Returns: Solution x, or nil if singular
    public static func solve(_ A: Matrix, _ b: Matrix) -> Matrix? {
        precondition(A.rows == A.cols, "A must be square")
        precondition(A.rows == b.rows, "Dimensions must match")

        var a = toColumnMajor(A)
        var bCol = toColumnMajor(b)

        var n1 = __CLPK_integer(A.rows)
        var nrhs = __CLPK_integer(b.cols)
        var lda = __CLPK_integer(A.rows)
        var ipiv = [__CLPK_integer](repeating: 0, count: A.rows)
        var ldb = __CLPK_integer(A.rows)
        var info: __CLPK_integer = 0

        dgesv_(&n1, &nrhs, &a, &lda, &ipiv, &bCol, &ldb, &info)

        if info != 0 { return nil }

        return fromColumnMajor(bCol, rows: b.rows, cols: b.cols)
    }

    /// Solve least squares problem min ||Ax - b||.
    public static func lstsq(_ A: Matrix, _ b: Matrix) -> Matrix? {
        precondition(A.rows == b.rows, "Dimensions must match")

        let maxDim = max(A.rows, A.cols)

        var a = toColumnMajor(A)
        var bCol = [Double](repeating: 0, count: maxDim * b.cols)
        let bColMajor = toColumnMajor(b)
        for i in 0..<b.rows {
            for j in 0..<b.cols {
                bCol[j * maxDim + i] = bColMajor[j * b.rows + i]
            }
        }

        var m1 = __CLPK_integer(A.rows)
        var n1 = __CLPK_integer(A.cols)
        var nrhs1 = __CLPK_integer(b.cols)
        var lda1 = __CLPK_integer(A.rows)
        var ldb1 = __CLPK_integer(maxDim)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var trans = Int8(UInt8(ascii: "N"))

        dgels_(&trans, &m1, &n1, &nrhs1, &a, &lda1, &bCol, &ldb1, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        a = toColumnMajor(A)
        bCol = [Double](repeating: 0, count: maxDim * b.cols)
        let bColMajor2 = toColumnMajor(b)
        for i in 0..<b.rows {
            for j in 0..<b.cols {
                bCol[j * maxDim + i] = bColMajor2[j * b.rows + i]
            }
        }

        var m2 = __CLPK_integer(A.rows)
        var n2 = __CLPK_integer(A.cols)
        var nrhs2 = __CLPK_integer(b.cols)
        var lda2 = __CLPK_integer(A.rows)
        var ldb2 = __CLPK_integer(maxDim)
        dgels_(&trans, &m2, &n2, &nrhs2, &a, &lda2, &bCol, &ldb2, &work, &lwork, &info)

        if info != 0 { return nil }

        var result = [Double](repeating: 0, count: A.cols * b.cols)
        for i in 0..<A.cols {
            for j in 0..<b.cols {
                result[i * b.cols + j] = bCol[j * maxDim + i]
            }
        }

        return Matrix(rows: A.cols, cols: b.cols, data: result)
    }

    /// Solve triangular system: L*x = b (lower) or U*x = b (upper).
    ///
    /// - Parameters:
    ///   - A: Triangular coefficient matrix
    ///   - b: Right-hand side
    ///   - lower: If true, A is lower triangular; if false, upper triangular
    ///   - trans: If true, solve A^T * x = b instead
    /// - Returns: Solution x, or nil if singular
    public static func solveTriangular(_ A: Matrix, _ b: Matrix,
                                       lower: Bool = true, trans: Bool = false) -> Matrix? {
        precondition(A.rows == A.cols, "A must be square")
        precondition(A.rows == b.rows, "Dimensions must match")

        let n = A.rows

        // Convert to column-major for LAPACK
        var a = toColumnMajor(A)
        var bData = toColumnMajor(b)

        var uplo = Int8(UInt8(ascii: lower ? "L" : "U"))
        var transChar = Int8(UInt8(ascii: trans ? "T" : "N"))
        var diag = Int8(UInt8(ascii: "N"))
        var n1 = __CLPK_integer(n)
        var nrhs = __CLPK_integer(b.cols)
        var lda = __CLPK_integer(n)
        var ldb = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        dtrtrs_(&uplo, &transChar, &diag, &n1, &nrhs, &a, &lda, &bData, &ldb, &info)

        if info != 0 { return nil }

        return fromColumnMajor(bData, rows: b.rows, cols: b.cols)
    }

    /// Solve using Cholesky factorization: L*L^T*x = b.
    ///
    /// - Parameters:
    ///   - L: Lower triangular Cholesky factor
    ///   - b: Right-hand side
    /// - Returns: Solution x, or nil if computation fails
    public static func choSolve(_ L: Matrix, _ b: Matrix) -> Matrix? {
        precondition(L.rows == L.cols, "L must be square")
        precondition(L.rows == b.rows, "Dimensions must match")

        let n = L.rows

        var a = toColumnMajor(L)
        var bData = toColumnMajor(b)

        var uplo = Int8(UInt8(ascii: "L"))
        var n1 = __CLPK_integer(n)
        var nrhs = __CLPK_integer(b.cols)
        var lda = __CLPK_integer(n)
        var ldb = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        dpotrs_(&uplo, &n1, &nrhs, &a, &lda, &bData, &ldb, &info)

        if info != 0 { return nil }

        return fromColumnMajor(bData, rows: b.rows, cols: b.cols)
    }

    /// Solve using LU factorization: P*L*U*x = b.
    ///
    /// - Parameters:
    ///   - L: Lower triangular factor from lu()
    ///   - U: Upper triangular factor from lu()
    ///   - P: Permutation matrix from lu()
    ///   - b: Right-hand side
    /// - Returns: Solution x
    public static func luSolve(_ L: Matrix, _ U: Matrix, _ P: Matrix, _ b: Matrix) -> Matrix {
        precondition(L.rows == L.cols && U.rows == U.cols && P.rows == P.cols)
        precondition(L.rows == U.rows && L.rows == P.rows)
        precondition(L.rows == b.rows)

        let n = L.rows

        // Apply P to b: P * b
        var pb = [Double](repeating: 0, count: b.rows * b.cols)
        for j in 0..<b.cols {
            for i in 0..<n {
                var sum = 0.0
                for k in 0..<n {
                    sum += P.data[i * n + k] * b.data[k * b.cols + j]
                }
                pb[i * b.cols + j] = sum
            }
        }

        // Solve L * y = P * b (forward substitution)
        var y = pb
        for j in 0..<b.cols {
            for i in 0..<n {
                var sum = y[i * b.cols + j]
                for k in 0..<i {
                    sum -= L.data[i * n + k] * y[k * b.cols + j]
                }
                y[i * b.cols + j] = sum / L.data[i * n + i]
            }
        }

        // Solve U * x = y (back substitution)
        var x = y
        for j in 0..<b.cols {
            for i in stride(from: n - 1, through: 0, by: -1) {
                var sum = x[i * b.cols + j]
                for k in (i + 1)..<n {
                    sum -= U.data[i * n + k] * x[k * b.cols + j]
                }
                x[i * b.cols + j] = sum / U.data[i * n + i]
            }
        }

        return Matrix(rows: b.rows, cols: b.cols, data: x)
    }

    // MARK: - Matrix Functions

    /// Matrix exponential using Padé approximation with scaling and squaring.
    public static func expm(_ m: Matrix) -> Matrix {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        let normA = cblas_dnrm2(Int32(n * n), m.data, 1)

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
    public static func logm(_ m: Matrix) -> Matrix? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        guard let (eigenvalues, eigenvectors) = computeRealEigendecomposition(m.data, n) else {
            return nil
        }

        // Check for non-positive eigenvalues
        for ev in eigenvalues {
            if ev <= 0 { return nil }
        }

        // Apply log to eigenvalues
        let logEigenvalues = eigenvalues.map { log($0) }

        // Reconstruct: logA = V * diag(log(λ)) * V^(-1)
        let result = reconstructFromEigen(eigenvectors, logEigenvalues, n)

        return Matrix(rows: n, cols: n, data: result)
    }

    /// Matrix square root using eigendecomposition.
    ///
    /// sqrt(A) = V * diag(sqrt(λ)) * V^(-1) for diagonalizable matrices.
    /// - Parameter m: Square matrix with non-negative eigenvalues
    /// - Returns: Matrix square root, or nil if eigenvalues are negative
    public static func sqrtm(_ m: Matrix) -> Matrix? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        guard let (eigenvalues, eigenvectors) = computeRealEigendecomposition(m.data, n) else {
            return nil
        }

        // Check for negative eigenvalues
        for ev in eigenvalues {
            if ev < 0 { return nil }
        }

        // Apply sqrt to eigenvalues
        let sqrtEigenvalues = eigenvalues.map { sqrt($0) }

        // Reconstruct: sqrtA = V * diag(sqrt(λ)) * V^(-1)
        let result = reconstructFromEigen(eigenvectors, sqrtEigenvalues, n)

        return Matrix(rows: n, cols: n, data: result)
    }

    /// Supported functions for funm
    public enum MatrixFunction: String {
        case sin, cos, exp, log, sqrt, sinh, cosh, tanh, abs
    }

    /// General matrix function using eigendecomposition.
    ///
    /// f(A) = V * diag(f(λ)) * V^(-1) for diagonalizable matrices.
    /// - Parameters:
    ///   - m: Square matrix
    ///   - function: Function to apply (sin, cos, exp, log, sqrt, sinh, cosh, tanh, abs)
    /// - Returns: f(A), or nil if computation fails
    public static func funm(_ m: Matrix, _ function: MatrixFunction) -> Matrix? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        guard let (eigenvalues, eigenvectors) = computeRealEigendecomposition(m.data, n) else {
            return nil
        }

        // Apply the function to eigenvalues
        let transformedEigenvalues: [Double]
        switch function {
        case .sin:
            transformedEigenvalues = eigenvalues.map { sin($0) }
        case .cos:
            transformedEigenvalues = eigenvalues.map { cos($0) }
        case .exp:
            transformedEigenvalues = eigenvalues.map { exp($0) }
        case .log:
            for ev in eigenvalues {
                if ev <= 0 { return nil }
            }
            transformedEigenvalues = eigenvalues.map { log($0) }
        case .sqrt:
            for ev in eigenvalues {
                if ev < 0 { return nil }
            }
            transformedEigenvalues = eigenvalues.map { sqrt($0) }
        case .sinh:
            transformedEigenvalues = eigenvalues.map { sinh($0) }
        case .cosh:
            transformedEigenvalues = eigenvalues.map { cosh($0) }
        case .tanh:
            transformedEigenvalues = eigenvalues.map { tanh($0) }
        case .abs:
            transformedEigenvalues = eigenvalues.map { Swift.abs($0) }
        }

        // Reconstruct: f(A) = V * diag(f(λ)) * V^(-1)
        let result = reconstructFromEigen(eigenvectors, transformedEigenvalues, n)

        return Matrix(rows: n, cols: n, data: result)
    }

    // MARK: - Complex Matrix Operations

    /// Solve complex linear system Az = b.
    ///
    /// - Parameters:
    ///   - A: Square complex coefficient matrix
    ///   - b: Right-hand side complex matrix
    /// - Returns: Solution z, or nil if singular
    public static func csolve(_ A: ComplexMatrix, _ b: ComplexMatrix) -> ComplexMatrix? {
        precondition(A.rows == A.cols, "A must be square")
        precondition(A.rows == b.rows, "Dimensions must match")

        let n = A.rows

        // Convert to column-major interleaved complex for LAPACK
        var a = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = i * n + j
                let dstIdx = j * n + i  // Column-major
                a[dstIdx] = __CLPK_doublecomplex(r: A.real[srcIdx], i: A.imag[srcIdx])
            }
        }

        var bData = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: b.rows * b.cols)
        for i in 0..<b.rows {
            for j in 0..<b.cols {
                let srcIdx = i * b.cols + j
                let dstIdx = j * b.rows + i  // Column-major
                bData[dstIdx] = __CLPK_doublecomplex(r: b.real[srcIdx], i: b.imag[srcIdx])
            }
        }

        var n1 = __CLPK_integer(n)
        var nrhs = __CLPK_integer(b.cols)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var ldb = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        zgesv_(&n1, &nrhs, &a, &lda, &ipiv, &bData, &ldb, &info)

        if info != 0 { return nil }

        // Convert back to row-major
        var resultReal = [Double](repeating: 0, count: b.rows * b.cols)
        var resultImag = [Double](repeating: 0, count: b.rows * b.cols)
        for i in 0..<b.rows {
            for j in 0..<b.cols {
                let srcIdx = j * b.rows + i
                let dstIdx = i * b.cols + j
                resultReal[dstIdx] = bData[srcIdx].r
                resultImag[dstIdx] = bData[srcIdx].i
            }
        }

        return ComplexMatrix(rows: b.rows, cols: b.cols, real: resultReal, imag: resultImag)
    }

    /// Complex SVD.
    ///
    /// - Parameter m: Complex matrix
    /// - Returns: (s, U, Vt) where s is real, U and Vt are complex
    public static func csvd(_ m: ComplexMatrix) -> (s: [Double], U: ComplexMatrix, Vt: ComplexMatrix)? {
        let minDim = min(m.rows, m.cols)

        // Convert to column-major interleaved
        var a = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: m.rows * m.cols)
        for i in 0..<m.rows {
            for j in 0..<m.cols {
                let srcIdx = i * m.cols + j
                let dstIdx = j * m.rows + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }

        var m1 = __CLPK_integer(m.rows)
        var n1 = __CLPK_integer(m.cols)
        var lda1 = __CLPK_integer(m.rows)
        var s = [Double](repeating: 0, count: minDim)
        var u = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: m.rows * m.rows)
        var vt = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: m.cols * m.cols)
        var ldu = __CLPK_integer(m.rows)
        var ldvt = __CLPK_integer(m.cols)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)
        var rwork = [Double](repeating: 0, count: max(1, 5 * minDim * minDim + 7 * minDim))
        var iwork = [__CLPK_integer](repeating: 0, count: 8 * minDim)
        var jobz = Int8(UInt8(ascii: "A"))

        zgesdd_(&jobz, &m1, &n1, &a, &lda1, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &rwork, &iwork, &info)

        lwork = __CLPK_integer(work[0].r)
        work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: Int(lwork))

        // Reset a
        for i in 0..<m.rows {
            for j in 0..<m.cols {
                let srcIdx = i * m.cols + j
                let dstIdx = j * m.rows + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }

        var m2 = __CLPK_integer(m.rows)
        var n2 = __CLPK_integer(m.cols)
        var lda2 = __CLPK_integer(m.rows)

        zgesdd_(&jobz, &m2, &n2, &a, &lda2, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &rwork, &iwork, &info)

        if info != 0 { return nil }

        // Convert U to row-major
        var Ureal = [Double](repeating: 0, count: m.rows * m.rows)
        var Uimag = [Double](repeating: 0, count: m.rows * m.rows)
        for i in 0..<m.rows {
            for j in 0..<m.rows {
                let srcIdx = j * m.rows + i
                let dstIdx = i * m.rows + j
                Ureal[dstIdx] = u[srcIdx].r
                Uimag[dstIdx] = u[srcIdx].i
            }
        }

        // Convert Vt to row-major
        var Vtreal = [Double](repeating: 0, count: m.cols * m.cols)
        var Vtimag = [Double](repeating: 0, count: m.cols * m.cols)
        for i in 0..<m.cols {
            for j in 0..<m.cols {
                let srcIdx = j * m.cols + i
                let dstIdx = i * m.cols + j
                Vtreal[dstIdx] = vt[srcIdx].r
                Vtimag[dstIdx] = vt[srcIdx].i
            }
        }

        return (s,
                ComplexMatrix(rows: m.rows, cols: m.rows, real: Ureal, imag: Uimag),
                ComplexMatrix(rows: m.cols, cols: m.cols, real: Vtreal, imag: Vtimag))
    }

    /// Complex eigendecomposition.
    ///
    /// - Parameter m: Square complex matrix
    /// - Returns: (eigenvalues, eigenvectors) both complex
    public static func ceig(_ m: ComplexMatrix) -> (values: ComplexMatrix, vectors: ComplexMatrix)? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        // Convert to column-major
        var a = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = i * n + j
                let dstIdx = j * n + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }

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

        zgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &w, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &rwork, &info)

        lwork = __CLPK_integer(work[0].r)
        work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: Int(lwork))

        // Reset a
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = i * n + j
                let dstIdx = j * n + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }

        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)

        zgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &w, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &rwork, &info)

        if info != 0 { return nil }

        // Extract eigenvalues
        var eigReal = [Double](repeating: 0, count: n)
        var eigImag = [Double](repeating: 0, count: n)
        for i in 0..<n {
            eigReal[i] = w[i].r
            eigImag[i] = w[i].i
        }

        // Convert eigenvectors to row-major
        var vecReal = [Double](repeating: 0, count: n * n)
        var vecImag = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = j * n + i
                let dstIdx = i * n + j
                vecReal[dstIdx] = vr[srcIdx].r
                vecImag[dstIdx] = vr[srcIdx].i
            }
        }

        return (ComplexMatrix(rows: n, cols: 1, real: eigReal, imag: eigImag),
                ComplexMatrix(rows: n, cols: n, real: vecReal, imag: vecImag))
    }

    /// Complex eigenvalues only.
    public static func ceigvals(_ m: ComplexMatrix) -> ComplexMatrix? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        var a = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = i * n + j
                let dstIdx = j * n + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }

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

        zgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &w, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &rwork, &info)

        lwork = __CLPK_integer(work[0].r)
        work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: Int(lwork))

        // Reset a
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = i * n + j
                let dstIdx = j * n + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }

        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)

        zgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &w, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &rwork, &info)

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
    public static func cdet(_ m: ComplexMatrix) -> (re: Double, im: Double)? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        var a = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = i * n + j
                let dstIdx = j * n + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }

        var m1 = __CLPK_integer(n)
        var n1 = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        zgetrf_(&m1, &n1, &a, &lda, &ipiv, &info)

        if info > 0 {
            return (0, 0)  // Singular
        }
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

            if Int(ipiv[i]) != i + 1 {
                sign = -sign
            }
        }

        return (detRe * Double(sign), detIm * Double(sign))
    }

    /// Complex inverse.
    public static func cinv(_ m: ComplexMatrix) -> ComplexMatrix? {
        precondition(m.rows == m.cols, "Matrix must be square")
        let n = m.rows

        var a = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = i * n + j
                let dstIdx = j * n + i
                a[dstIdx] = __CLPK_doublecomplex(r: m.real[srcIdx], i: m.imag[srcIdx])
            }
        }

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

        var resultReal = [Double](repeating: 0, count: n * n)
        var resultImag = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let srcIdx = j * n + i
                let dstIdx = i * n + j
                resultReal[dstIdx] = a[srcIdx].r
                resultImag[dstIdx] = a[srcIdx].i
            }
        }

        return ComplexMatrix(rows: n, cols: n, real: resultReal, imag: resultImag)
    }

    // MARK: - Private Helpers

    /// Convert row-major to column-major.
    private static func toColumnMajor(_ m: Matrix) -> [Double] {
        var result = [Double](repeating: 0, count: m.size)
        for i in 0..<m.rows {
            for j in 0..<m.cols {
                result[j * m.rows + i] = m.data[i * m.cols + j]
            }
        }
        return result
    }

    /// Convert column-major to row-major Matrix.
    private static func fromColumnMajor(_ data: [Double], rows: Int, cols: Int) -> Matrix {
        var result = [Double](repeating: 0, count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                result[i * cols + j] = data[j * rows + i]
            }
        }
        return Matrix(rows: rows, cols: cols, data: result)
    }

    /// Internal matrix multiplication for n×n matrices.
    private static func matmulInternal(_ A: [Double], _ B: [Double], _ n: Int) -> [Double] {
        var C = [Double](repeating: 0, count: n * n)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(n), Int32(n), Int32(n),
                    1.0, A, Int32(n), B, Int32(n),
                    0.0, &C, Int32(n))
        return C
    }

    /// Internal linear system solver for n×n matrices.
    private static func solveLinearSystemInternal(_ A: [Double], _ B: [Double], _ n: Int) -> [Double] {
        var a = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                a[j * n + i] = A[i * n + j]
            }
        }

        var b = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                b[j * n + i] = B[i * n + j]
            }
        }

        var n1 = __CLPK_integer(n)
        var nrhs = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var ldb = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        dgesv_(&n1, &nrhs, &a, &lda, &ipiv, &b, &ldb, &info)

        var result = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                result[i * n + j] = b[j * n + i]
            }
        }

        return result
    }

    /// Compute eigendecomposition returning real eigenvalues only.
    private static func computeRealEigendecomposition(_ data: [Double], _ n: Int) -> (eigenvalues: [Double], eigenvectors: [Double])? {
        // Convert to column-major for LAPACK
        var a = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                a[j * n + i] = data[i * n + j]
            }
        }

        var n1 = __CLPK_integer(n)
        var lda1 = __CLPK_integer(n)
        var wr = [Double](repeating: 0, count: n)
        var wi = [Double](repeating: 0, count: n)
        var vl = [Double](repeating: 0, count: 1)
        var vr = [Double](repeating: 0, count: n * n)
        var ldvl: __CLPK_integer = 1
        var ldvr = __CLPK_integer(n)
        var info: __CLPK_integer = 0

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var jobvl = Int8(UInt8(ascii: "N"))
        var jobvr = Int8(UInt8(ascii: "V"))

        dgeev_(&jobvl, &jobvr, &n1, &a, &lda1, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        // Reset a
        for i in 0..<n {
            for j in 0..<n {
                a[j * n + i] = data[i * n + j]
            }
        }

        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)
        dgeev_(&jobvl, &jobvr, &n2, &a, &lda2, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)

        if info != 0 { return nil }

        // Check for complex eigenvalues
        for im in wi {
            if abs(im) > 1e-10 { return nil }
        }

        // Convert eigenvectors to row-major
        var vecs = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                vecs[i * n + j] = vr[j * n + i]
            }
        }

        return (wr, vecs)
    }

    /// Reconstruct matrix from eigendecomposition: A = V * diag(λ) * V^(-1).
    private static func reconstructFromEigen(_ V: [Double], _ eigenvalues: [Double], _ n: Int) -> [Double] {
        // Compute V^(-1)
        let Vinv = invertMatrixInternal(V, n)

        // Compute V * diag(λ)
        var VD = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                VD[i * n + j] = V[i * n + j] * eigenvalues[j]
            }
        }

        // Compute (V * diag(λ)) * V^(-1)
        return matmulInternal(VD, Vinv, n)
    }

    /// Invert matrix using LU decomposition.
    private static func invertMatrixInternal(_ M: [Double], _ n: Int) -> [Double] {
        var a = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                a[j * n + i] = M[i * n + j]
            }
        }

        var n1 = __CLPK_integer(n)
        var n1b = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&n1, &n1b, &a, &lda, &ipiv, &info)

        if info != 0 { return M }

        var lwork: __CLPK_integer = -1
        var work = [Double](repeating: 0, count: 1)
        var n2 = __CLPK_integer(n)
        var lda2 = __CLPK_integer(n)

        dgetri_(&n2, &a, &lda2, &ipiv, &work, &lwork, &info)

        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0, count: Int(lwork))

        var n3 = __CLPK_integer(n)
        var lda3 = __CLPK_integer(n)
        dgetri_(&n3, &a, &lda3, &ipiv, &work, &lwork, &info)

        var result = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                result[i * n + j] = a[j * n + i]
            }
        }

        return result
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
