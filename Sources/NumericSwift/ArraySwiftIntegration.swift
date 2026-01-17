//
//  ArraySwiftIntegration.swift
//  NumericSwift
//
//  Integration with ArraySwift for enhanced N-dimensional array support.
//  This file is only compiled when NUMERICSWIFT_ARRAYSWIFT is defined.
//
//  Licensed under the MIT License.
//

#if NUMERICSWIFT_ARRAYSWIFT

import Foundation
import ArraySwift

// MARK: - LinAlg.Matrix ↔ NDArray Conversion

extension LinAlg.Matrix {

    /// Create a Matrix from an NDArray.
    ///
    /// - Parameter ndarray: A 1D or 2D NDArray (real values only)
    /// - Returns: A Matrix, or nil if the NDArray is not 1D/2D or is complex
    ///
    /// Example:
    /// ```swift
    /// let arr = NDArray([[1.0, 2.0], [3.0, 4.0]])
    /// let matrix = LinAlg.Matrix(ndarray: arr)
    /// ```
    public init?(ndarray: NDArray) {
        guard !ndarray.isComplex else { return nil }

        switch ndarray.ndim {
        case 1:
            // 1D array becomes a column vector
            self.init(rows: ndarray.size, cols: 1, data: ndarray.real)
        case 2:
            // 2D array maps directly
            self.init(rows: ndarray.shape[0], cols: ndarray.shape[1], data: ndarray.real)
        default:
            return nil
        }
    }

    /// Convert this Matrix to an NDArray.
    ///
    /// - Returns: An NDArray with the same shape and data
    ///
    /// Example:
    /// ```swift
    /// let matrix = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
    /// let arr = matrix.toNDArray()
    /// ```
    public func toNDArray() -> NDArray {
        if isVector {
            return NDArray(shape: [rows], data: data)
        }
        return NDArray(shape: [rows, cols], data: data)
    }
}

// MARK: - NDArray Extensions for LinAlg

extension NDArray {

    /// Convert this NDArray to a LinAlg.Matrix.
    ///
    /// - Returns: A Matrix, or nil if the NDArray is not 1D/2D or is complex
    ///
    /// Example:
    /// ```swift
    /// let arr = NDArray.arange(start: 0, stop: 6).reshape([2, 3])
    /// if let matrix = arr.toMatrix() {
    ///     let result = LinAlg.solve(matrix, b)
    /// }
    /// ```
    public func toMatrix() -> LinAlg.Matrix? {
        LinAlg.Matrix(ndarray: self)
    }

    /// Solve a linear system Ax = b where this array is A.
    ///
    /// - Parameter b: Right-hand side vector or matrix
    /// - Returns: Solution x, or nil if the system cannot be solved
    ///
    /// Requires this array to be 2D and square.
    public func solve(_ b: NDArray) -> NDArray? {
        guard let A = self.toMatrix(),
              let bMat = b.toMatrix(),
              let x = LinAlg.solve(A, bMat) else {
            return nil
        }
        return x.toNDArray()
    }

    /// Compute the inverse of this matrix.
    ///
    /// - Returns: The inverse matrix, or nil if singular
    ///
    /// Requires this array to be 2D and square.
    public func inv() -> NDArray? {
        guard let matrix = self.toMatrix(),
              let invMat = LinAlg.inv(matrix) else {
            return nil
        }
        return invMat.toNDArray()
    }

    /// Compute the determinant of this matrix.
    ///
    /// - Returns: The determinant, or nil if not a square matrix
    ///
    /// Requires this array to be 2D and square.
    public func det() -> Double? {
        guard let matrix = self.toMatrix() else { return nil }
        return LinAlg.det(matrix)
    }

    /// Compute the matrix rank.
    ///
    /// - Returns: The rank, or nil if not a 2D array
    public func rank() -> Int? {
        guard let matrix = self.toMatrix() else { return nil }
        return LinAlg.rank(matrix)
    }

    /// Compute the trace (sum of diagonal elements).
    ///
    /// - Returns: The trace, or nil if not a 2D array
    public func trace() -> Double? {
        guard let matrix = self.toMatrix() else { return nil }
        return LinAlg.trace(matrix)
    }

    /// Compute the Frobenius norm.
    ///
    /// - Returns: The Frobenius norm
    public func norm() -> Double {
        if let matrix = self.toMatrix() {
            return LinAlg.norm(matrix)
        }
        // Fallback for non-2D arrays
        var sumSq = 0.0
        for v in real {
            sumSq += v * v
        }
        if let imagPart = imag {
            for v in imagPart {
                sumSq += v * v
            }
        }
        return Darwin.sqrt(sumSq)
    }

    /// Perform LU decomposition.
    ///
    /// - Returns: Tuple of (P, L, U) matrices, or nil if not a square 2D array
    public func lu() -> (P: NDArray, L: NDArray, U: NDArray)? {
        guard let matrix = self.toMatrix(), matrix.rows == matrix.cols else {
            return nil
        }
        let result = LinAlg.lu(matrix)
        return (result.P.toNDArray(), result.L.toNDArray(), result.U.toNDArray())
    }

    /// Perform QR decomposition.
    ///
    /// - Returns: Tuple of (Q, R) matrices, or nil if not a 2D array
    public func qr() -> (Q: NDArray, R: NDArray)? {
        guard let matrix = self.toMatrix() else {
            return nil
        }
        let result = LinAlg.qr(matrix)
        return (result.Q.toNDArray(), result.R.toNDArray())
    }

    /// Perform Singular Value Decomposition.
    ///
    /// - Returns: Tuple of (U, s, Vt) where A = U * diag(s) * Vt, or nil if not a 2D array
    public func svd() -> (U: NDArray, s: NDArray, Vt: NDArray)? {
        guard let matrix = self.toMatrix() else {
            return nil
        }
        let result = LinAlg.svd(matrix)
        let sArray = NDArray(result.s)
        return (result.U.toNDArray(), sArray, result.Vt.toNDArray())
    }

    /// Compute eigenvalues and eigenvectors.
    ///
    /// - Returns: Tuple of (real eigenvalues, imaginary parts, eigenvectors), or nil if not square
    ///
    /// Note: For general matrices, eigenvalues may be complex. Use `imagParts` to check.
    public func eig() -> (values: NDArray, imagParts: NDArray, vectors: NDArray)? {
        guard let matrix = self.toMatrix(), matrix.rows == matrix.cols else {
            return nil
        }
        let result = LinAlg.eig(matrix)
        return (NDArray(result.values), NDArray(result.imagParts), result.vectors.toNDArray())
    }

    /// Perform Cholesky decomposition.
    ///
    /// - Returns: Lower triangular matrix L where A = L * L^T, or nil if not positive definite
    public func cholesky() -> NDArray? {
        guard let matrix = self.toMatrix(),
              let L = LinAlg.cholesky(matrix) else {
            return nil
        }
        return L.toNDArray()
    }
}

// MARK: - Statistics Functions for NDArray

/// Sum of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: Sum of all elements
public func sum(_ array: NDArray) -> Double {
    array.sum()
}

/// Mean of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: Arithmetic mean
public func mean(_ array: NDArray) -> Double {
    array.mean()
}

/// Median of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: Median value
public func median(_ array: NDArray) -> Double {
    array.median()
}

/// Variance of NDArray elements.
///
/// - Parameters:
///   - array: The NDArray
///   - ddof: Delta degrees of freedom (0 for population, 1 for sample)
/// - Returns: Variance
public func variance(_ array: NDArray, ddof: Int = 0) -> Double {
    array.variance(ddof: ddof)
}

/// Standard deviation of NDArray elements.
///
/// - Parameters:
///   - array: The NDArray
///   - ddof: Delta degrees of freedom (0 for population, 1 for sample)
/// - Returns: Standard deviation
public func stddev(_ array: NDArray, ddof: Int = 0) -> Double {
    array.std(ddof: ddof)
}

/// Percentile of NDArray elements.
///
/// - Parameters:
///   - array: The NDArray
///   - p: Percentile (0-100)
/// - Returns: The p-th percentile value
public func percentile(_ array: NDArray, _ p: Double) -> Double {
    array.percentile(p)
}

/// Minimum value in NDArray.
///
/// - Parameter array: The NDArray
/// - Returns: Minimum value
public func amin(_ array: NDArray) -> Double {
    array.min()
}

/// Maximum value in NDArray.
///
/// - Parameter array: The NDArray
/// - Returns: Maximum value
public func amax(_ array: NDArray) -> Double {
    array.max()
}

/// Range (max - min) of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: Range
public func ptp(_ array: NDArray) -> Double {
    array.ptp()
}

/// Cumulative sum of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray of cumulative sums
public func cumsum(_ array: NDArray) -> NDArray {
    array.cumsum()
}

/// Cumulative product of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray of cumulative products
public func cumprod(_ array: NDArray) -> NDArray {
    array.cumprod()
}

/// First-order discrete difference of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray of differences
public func diff(_ array: NDArray) -> NDArray {
    array.diff()
}

// MARK: - Utility Functions for NDArray

/// Clip NDArray values to a range.
///
/// - Parameters:
///   - array: The NDArray
///   - min: Lower bound
///   - max: Upper bound
/// - Returns: NDArray with clipped values
public func clip(_ array: NDArray, min: Double, max: Double) -> NDArray {
    array.clip(min: min, max: max)
}

/// Round NDArray elements to the nearest integer.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray with rounded values
public func roundArray(_ array: NDArray) -> NDArray {
    array.round()
}

/// Floor NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray with floored values
public func floorArray(_ array: NDArray) -> NDArray {
    array.floor()
}

/// Ceiling NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray with ceiling values
public func ceilArray(_ array: NDArray) -> NDArray {
    array.ceil()
}

/// Absolute value of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray with absolute values
public func absArray(_ array: NDArray) -> NDArray {
    array.abs()
}

/// Square root of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray with square roots
public func sqrtArray(_ array: NDArray) -> NDArray {
    array.sqrt()
}

/// Natural logarithm of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray with natural logarithms
public func logArray(_ array: NDArray) -> NDArray {
    array.log()
}

/// Exponential of NDArray elements.
///
/// - Parameter array: The NDArray
/// - Returns: NDArray with exponentials
public func expArray(_ array: NDArray) -> NDArray {
    array.exp()
}

/// Sine of NDArray elements.
///
/// - Parameter array: The NDArray (angles in radians)
/// - Returns: NDArray with sine values
public func sinArray(_ array: NDArray) -> NDArray {
    array.sin()
}

/// Cosine of NDArray elements.
///
/// - Parameter array: The NDArray (angles in radians)
/// - Returns: NDArray with cosine values
public func cosArray(_ array: NDArray) -> NDArray {
    array.cos()
}

/// Tangent of NDArray elements.
///
/// - Parameter array: The NDArray (angles in radians)
/// - Returns: NDArray with tangent values
public func tanArray(_ array: NDArray) -> NDArray {
    array.tan()
}

#endif // NUMERICSWIFT_ARRAYSWIFT
