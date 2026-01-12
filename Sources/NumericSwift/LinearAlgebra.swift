//
//  LinearAlgebra.swift
//  NumericSwift
//
//  Matrix and vector operations using Accelerate framework for hardware-accelerated computation.
//  Follows NumPy/scipy.linalg patterns.
//
//  Licensed under the MIT License.
//

import Foundation
import Accelerate

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

// MARK: - Matrix Creation

/// Create a matrix/vector of zeros.
///
/// - Parameters:
///   - rows: Number of rows
///   - cols: Number of columns (default 1 for vector)
/// - Returns: Matrix filled with zeros
public func zeros(_ rows: Int, _ cols: Int = 1) -> Matrix {
    Matrix(rows: rows, cols: cols, data: [Double](repeating: 0, count: rows * cols))
}

/// Create a matrix/vector of ones.
///
/// - Parameters:
///   - rows: Number of rows
///   - cols: Number of columns (default 1 for vector)
/// - Returns: Matrix filled with ones
public func ones(_ rows: Int, _ cols: Int = 1) -> Matrix {
    Matrix(rows: rows, cols: cols, data: [Double](repeating: 1, count: rows * cols))
}

/// Create an identity matrix.
///
/// - Parameter n: Size of the square matrix
/// - Returns: n×n identity matrix
public func eye(_ n: Int) -> Matrix {
    var data = [Double](repeating: 0, count: n * n)
    for i in 0..<n {
        data[i * n + i] = 1.0
    }
    return Matrix(rows: n, cols: n, data: data)
}

/// Create a diagonal matrix from a vector.
///
/// - Parameter values: Diagonal values
/// - Returns: Square matrix with values on the diagonal
public func diag(_ values: [Double]) -> Matrix {
    let n = values.count
    var data = [Double](repeating: 0, count: n * n)
    for i in 0..<n {
        data[i * n + i] = values[i]
    }
    return Matrix(rows: n, cols: n, data: data)
}

/// Create a range vector.
///
/// - Parameters:
///   - start: Starting value
///   - stop: Stop value (exclusive)
///   - step: Step size (default 1)
/// - Returns: Vector from start to stop (exclusive) with given step
public func arange(_ start: Double, _ stop: Double, _ step: Double = 1.0) -> Matrix {
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
///
/// - Parameters:
///   - start: Starting value
///   - stop: Ending value (inclusive)
///   - count: Number of values
/// - Returns: Vector with count evenly spaced values from start to stop
public func linspace(_ start: Double, _ stop: Double, _ count: Int) -> Matrix {
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
public func + (lhs: Matrix, rhs: Matrix) -> Matrix {
    precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
    var result = [Double](repeating: 0, count: lhs.size)
    vDSP_vaddD(lhs.data, 1, rhs.data, 1, &result, 1, vDSP_Length(lhs.size))
    return Matrix(rows: lhs.rows, cols: lhs.cols, data: result)
}

/// Subtract two matrices element-wise.
public func - (lhs: Matrix, rhs: Matrix) -> Matrix {
    precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
    var result = [Double](repeating: 0, count: lhs.size)
    vDSP_vsubD(rhs.data, 1, lhs.data, 1, &result, 1, vDSP_Length(lhs.size))
    return Matrix(rows: lhs.rows, cols: lhs.cols, data: result)
}

/// Negate a matrix.
public prefix func - (m: Matrix) -> Matrix {
    var result = [Double](repeating: 0, count: m.size)
    var neg = -1.0
    vDSP_vsmulD(m.data, 1, &neg, &result, 1, vDSP_Length(m.size))
    return Matrix(rows: m.rows, cols: m.cols, data: result)
}

/// Multiply matrix by scalar.
public func * (lhs: Double, rhs: Matrix) -> Matrix {
    var result = [Double](repeating: 0, count: rhs.size)
    var scalar = lhs
    vDSP_vsmulD(rhs.data, 1, &scalar, &result, 1, vDSP_Length(rhs.size))
    return Matrix(rows: rhs.rows, cols: rhs.cols, data: result)
}

/// Multiply matrix by scalar.
public func * (lhs: Matrix, rhs: Double) -> Matrix {
    rhs * lhs
}

/// Divide matrix by scalar.
public func / (lhs: Matrix, rhs: Double) -> Matrix {
    precondition(rhs != 0, "Division by zero")
    var result = [Double](repeating: 0, count: lhs.size)
    var scalar = rhs
    vDSP_vsdivD(lhs.data, 1, &scalar, &result, 1, vDSP_Length(lhs.size))
    return Matrix(rows: lhs.rows, cols: lhs.cols, data: result)
}

/// Element-wise multiplication (Hadamard product).
public func hadamard(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
    precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
    var result = [Double](repeating: 0, count: lhs.size)
    vDSP_vmulD(lhs.data, 1, rhs.data, 1, &result, 1, vDSP_Length(lhs.size))
    return Matrix(rows: lhs.rows, cols: lhs.cols, data: result)
}

// MARK: - Matrix Multiplication

/// Matrix multiplication or dot product.
///
/// - Parameters:
///   - lhs: Left matrix (m×k)
///   - rhs: Right matrix (k×n)
/// - Returns: Product matrix (m×n)
public func dot(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
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

/// Matrix multiplication operator.
public func * (lhs: Matrix, rhs: Matrix) -> Matrix {
    dot(lhs, rhs)
}

// MARK: - Matrix Properties

/// Compute the trace of a square matrix.
public func trace(_ m: Matrix) -> Double {
    precondition(m.rows == m.cols, "Matrix must be square")
    var sum = 0.0
    for i in 0..<m.rows {
        sum += m.data[i * m.cols + i]
    }
    return sum
}

/// Compute the determinant of a square matrix.
public func det(_ m: Matrix) -> Double {
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
public func inv(_ m: Matrix) -> Matrix? {
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
public func rank(_ m: Matrix) -> Int {
    let (s, _, _) = svd(m)
    let tol = max(Double(m.rows), Double(m.cols)) * s[0] * 2.220446049250313e-16
    var r = 0
    for sv in s {
        if sv > tol { r += 1 }
    }
    return r
}

/// Compute the condition number of a matrix.
public func cond(_ m: Matrix) -> Double {
    let (s, _, _) = svd(m)
    let sMax = s.first ?? 0
    let sMin = s.last ?? 0
    let tol = max(Double(m.rows), Double(m.cols)) * sMax * 2.220446049250313e-16
    if sMin <= tol { return Double.infinity }
    return sMax / sMin
}

/// Compute the Moore-Penrose pseudoinverse.
public func pinv(_ m: Matrix, rcond: Double = 1e-15) -> Matrix {
    let minDim = min(m.rows, m.cols)

    // Convert to column-major
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
///
/// - Parameters:
///   - m: Matrix or vector
///   - p: Norm type (default 2)
///     - 1: L1 norm (sum of absolute values) / column sum norm
///     - 2: L2 norm (Euclidean) / Frobenius norm
///     - .infinity: max norm / row sum norm
/// - Returns: Norm value
public func norm(_ m: Matrix, _ p: Double = 2) -> Double {
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
public func lu(_ m: Matrix) -> (L: Matrix, U: Matrix, P: Matrix) {
    precondition(m.rows == m.cols, "Matrix must be square")
    let n = m.rows

    var a = m.data
    var n1 = __CLPK_integer(n)
    var n2 = __CLPK_integer(n)
    var lda = __CLPK_integer(n)
    var ipiv = [__CLPK_integer](repeating: 0, count: n)
    var info: __CLPK_integer = 0

    dgetrf_(&n1, &n2, &a, &lda, &ipiv, &info)

    // Extract L and U
    var L = [Double](repeating: 0, count: n * n)
    var U = [Double](repeating: 0, count: n * n)

    for i in 0..<n {
        for j in 0..<n {
            if i > j {
                L[i * n + j] = a[i * n + j]
            } else if i == j {
                L[i * n + j] = 1.0
                U[i * n + j] = a[i * n + j]
            } else {
                U[i * n + j] = a[i * n + j]
            }
        }
    }

    // Create permutation matrix
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
public func qr(_ m: Matrix) -> (Q: Matrix, R: Matrix) {
    let minDim = min(m.rows, m.cols)

    // Convert to column-major
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
public func svd(_ m: Matrix) -> (s: [Double], U: Matrix, Vt: Matrix) {
    let minDim = min(m.rows, m.cols)

    // Convert to column-major
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
/// - Returns: (eigenvalues, eigenvectors) - eigenvalues may have imaginary parts
public func eig(_ m: Matrix) -> (values: [Double], imagParts: [Double], vectors: Matrix) {
    precondition(m.rows == m.cols, "Matrix must be square")
    let n = m.rows

    // Convert to column-major
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
public func eigvals(_ m: Matrix) -> (real: [Double], imag: [Double]) {
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
public func cholesky(_ m: Matrix) -> Matrix? {
    precondition(m.rows == m.cols, "Matrix must be square")
    let n = m.rows

    // Convert to column-major
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
public func solve(_ A: Matrix, _ b: Matrix) -> Matrix? {
    precondition(A.rows == A.cols, "A must be square")
    precondition(A.rows == b.rows, "Dimensions must match")

    // Convert to column-major
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
///
/// - Parameters:
///   - A: Coefficient matrix (m×n)
///   - b: Right-hand side (m×1 or m×k)
/// - Returns: Solution x that minimizes ||Ax - b||
public func lstsq(_ A: Matrix, _ b: Matrix) -> Matrix? {
    precondition(A.rows == b.rows, "Dimensions must match")

    let maxDim = max(A.rows, A.cols)

    // Convert to column-major
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

    // Extract solution
    var result = [Double](repeating: 0, count: A.cols * b.cols)
    for i in 0..<A.cols {
        for j in 0..<b.cols {
            result[i * b.cols + j] = bCol[j * maxDim + i]
        }
    }

    return Matrix(rows: A.cols, cols: b.cols, data: result)
}

// MARK: - Matrix Functions

/// Matrix exponential using Padé approximation with scaling and squaring.
public func expm(_ m: Matrix) -> Matrix {
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

// MARK: - Helper Functions

/// Convert row-major to column-major.
private func toColumnMajor(_ m: Matrix) -> [Double] {
    var result = [Double](repeating: 0, count: m.size)
    for i in 0..<m.rows {
        for j in 0..<m.cols {
            result[j * m.rows + i] = m.data[i * m.cols + j]
        }
    }
    return result
}

/// Convert column-major to row-major Matrix.
private func fromColumnMajor(_ data: [Double], rows: Int, cols: Int) -> Matrix {
    var result = [Double](repeating: 0, count: rows * cols)
    for i in 0..<rows {
        for j in 0..<cols {
            result[i * cols + j] = data[j * rows + i]
        }
    }
    return Matrix(rows: rows, cols: cols, data: result)
}

/// Internal matrix multiplication for n×n matrices.
private func matmulInternal(_ A: [Double], _ B: [Double], _ n: Int) -> [Double] {
    var C = [Double](repeating: 0, count: n * n)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(n), Int32(n), Int32(n),
                1.0, A, Int32(n), B, Int32(n),
                0.0, &C, Int32(n))
    return C
}

/// Internal linear system solver for n×n matrices.
private func solveLinearSystemInternal(_ A: [Double], _ B: [Double], _ n: Int) -> [Double] {
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
