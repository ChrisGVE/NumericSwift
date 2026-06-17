//
//  Sparse.swift
//  NumericSwift
//
//  Sparse matrix support for NumericSwift.  API semantics follow scipy.sparse
//  (https://docs.scipy.org/doc/scipy/reference/sparse.html).
//
//  Formats supported
//  -----------------
//  - COO (coordinate) — ``COOMatrix``
//    Three parallel arrays: `rows`, `cols`, `values`.  Duplicate (row, col)
//    entries are SUMMED on construction, matching scipy.sparse.coo_matrix
//    semantics.  COO is the natural input format for assembly from triplets.
//
//  - CSR (compressed sparse row) — ``CSRMatrix``
//    Three arrays: `indptr` (row pointers, length rows+1), `indices` (column
//    indices), `data` (nonzero values).  indptr[i] is the index into `indices`
//    and `data` where row i begins; indptr[i+1]-indptr[i] is the number of
//    nonzeros in row i.  Indices within each row are kept sorted ascending.
//    CSR is the preferred format for arithmetic (SpMV, SpMM, spsolve).
//
//  Operations
//  ----------
//  - ``COOMatrix.toCSR()``       — COO → CSR conversion
//  - ``COOMatrix.toDense()``     — COO → dense LinAlg.Matrix
//  - ``CSRMatrix.toDense()``     — CSR → dense LinAlg.Matrix
//  - ``CSRMatrix.spmv(_:)``      — sparse matrix–vector product (y = A·x)
//  - ``CSRMatrix.spmm(_:)``      — sparse matrix–dense matrix product (C = A·B)
//  - ``CSRMatrix.add(_:)``       — element-wise addition (same shape required)
//  - ``CSRMatrix.transpose()``   — matrix transpose
//  - ``Sparse.spsolve(_:_:)``    — solve A·x = b; uses Conjugate Gradient for
//    symmetric positive-definite A, falling back to dense LU (via
//    ``LinAlg.solve``) for general/non-SPD matrices.
//
//  Deferred (out of scope for this MVP)
//  -------------------------------------
//  - CSC (compressed sparse column) format
//  - Sparse eigensolvers
//  - Sparse Cholesky / LU / QR factorizations
//  - Fancy indexing (row/column slicing)
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

// MARK: - Sparse Namespace

/// Sparse matrix operations namespace.
///
/// Provides COO and CSR sparse matrix formats, conversion utilities, and
/// sparse linear algebra operations following scipy.sparse semantics.
///
/// ## Quick example
/// ```swift
/// // Build a 3×3 sparse matrix from triplets
/// let A = try Sparse.COOMatrix(rows: 3, cols: 3, triplets: [
///     (row: 0, col: 0, value: 4.0),
///     (row: 0, col: 1, value: 1.0),
///     (row: 1, col: 0, value: 1.0),
///     (row: 1, col: 1, value: 3.0),
/// ])
/// let csr = A.toCSR()
/// let b = [1.0, 2.0, 3.0]
/// let x = try Sparse.spsolve(csr, b)
/// ```
///
/// ## Deferred scope
/// The following items are explicitly out of scope for this MVP release and
/// will be addressed in future issues: CSC format, sparse eigensolvers, sparse
/// factorizations (Cholesky/LU/QR), and fancy indexing.
public enum Sparse {

    // MARK: - Errors

    /// Recoverable error conditions raised by sparse matrix operations.
    public enum SparseError: Error, Equatable, Sendable {
        /// An index in the triplet list is out of bounds for the declared shape.
        case indexOutOfBounds(row: Int, col: Int, rows: Int, cols: Int)
        /// Two matrices have incompatible shapes for the requested operation.
        case shapeMismatch(String)
        /// A square matrix is required but a non-square one was supplied.
        case notSquare(rows: Int, cols: Int)
        /// The system A·x = b is singular (no unique solution).
        case singularMatrix
        /// The supplied vector length does not match the matrix column or row count.
        case dimensionMismatch(String)
    }

    // MARK: - COO Matrix

    /// Sparse matrix in coordinate (COO) format.
    ///
    /// Stores nonzeros as three parallel arrays: row indices, column indices,
    /// and values.  Duplicate (row, col) entries supplied at construction time
    /// are **summed**, matching `scipy.sparse.coo_matrix` semantics.
    ///
    /// COO is convenient for assembling a matrix from triplets but is not
    /// directly efficient for arithmetic.  Convert to ``CSRMatrix`` via
    /// ``toCSR()`` before performing SpMV or other operations.
    public struct COOMatrix: Sendable {
        /// Number of rows.
        public let rows: Int
        /// Number of columns.
        public let cols: Int
        /// Row index of each nonzero (length nnz, after duplicate summation).
        public let rowIndices: [Int]
        /// Column index of each nonzero (length nnz, after duplicate summation).
        public let colIndices: [Int]
        /// Value of each nonzero (length nnz, after duplicate summation).
        public let values: [Double]

        /// Number of stored nonzeros (after duplicate summation).
        public var nnz: Int { values.count }

        /// Construct a COO matrix from (row, col, value) triplets.
        ///
        /// Duplicate (row, col) entries are **summed** (scipy.sparse COO
        /// semantics).  Entries whose summed value is zero are still retained
        /// (scipy behaviour); callers that need structural cancellation should
        /// convert to CSR and re-check.
        ///
        /// - Parameters:
        ///   - rows: Number of rows.
        ///   - cols: Number of columns.
        ///   - triplets: Array of `(row, col, value)` tuples.  Row and column
        ///     indices must be in `[0, rows)` and `[0, cols)` respectively.
        /// - Throws: ``SparseError/indexOutOfBounds(row:col:rows:cols:)`` when
        ///   any triplet index is out of bounds.
        public init(rows: Int, cols: Int,
                    triplets: [(row: Int, col: Int, value: Double)]) throws {
            self.rows = rows
            self.cols = cols

            // Validate indices
            for t in triplets {
                guard t.row >= 0 && t.row < rows && t.col >= 0 && t.col < cols else {
                    throw SparseError.indexOutOfBounds(
                        row: t.row, col: t.col, rows: rows, cols: cols)
                }
            }

            // Deduplicate + sum using a dictionary keyed by (row, col).
            // Iteration order of a Dictionary is non-deterministic; we will
            // re-sort by (row, col) for a stable canonical ordering matching
            // scipy's row-major output order.
            var accumulator: [Int64: Double] = [:]
            // Encode (row, col) into a single Int64 key (safe: rows, cols < 2^31)
            for t in triplets {
                let key = Int64(t.row) * Int64(cols) + Int64(t.col)
                accumulator[key, default: 0.0] += t.value
            }

            // Sort by (row, col) for canonical order
            let sorted = accumulator.sorted { $0.key < $1.key }
            var rowIdx = [Int]()
            var colIdx = [Int]()
            var vals   = [Double]()
            rowIdx.reserveCapacity(sorted.count)
            colIdx.reserveCapacity(sorted.count)
            vals.reserveCapacity(sorted.count)
            for (key, v) in sorted {
                rowIdx.append(Int(key / Int64(cols == 0 ? 1 : cols)))
                colIdx.append(Int(key % Int64(cols == 0 ? 1 : cols)))
                vals.append(v)
            }
            self.rowIndices = rowIdx
            self.colIndices = colIdx
            self.values     = vals
        }

        /// Construct a COO matrix from a dense ``LinAlg.Matrix``.
        ///
        /// Exact zeros in the dense matrix are excluded from the sparse
        /// representation (structural sparsity).
        ///
        /// - Parameter matrix: The source dense matrix.
        public init(_ matrix: LinAlg.Matrix) {
            var rowIdx = [Int]()
            var colIdx = [Int]()
            var vals   = [Double]()
            for i in 0..<matrix.rows {
                for j in 0..<matrix.cols {
                    let v = matrix[i, j]
                    if v != 0.0 {
                        rowIdx.append(i)
                        colIdx.append(j)
                        vals.append(v)
                    }
                }
            }
            self.rows       = matrix.rows
            self.cols       = matrix.cols
            self.rowIndices = rowIdx
            self.colIndices = colIdx
            self.values     = vals
        }

        /// Convert this COO matrix to CSR (compressed sparse row) format.
        ///
        /// Column indices within each row are sorted in ascending order,
        /// matching scipy's CSR convention.
        ///
        /// - Returns: An equivalent ``CSRMatrix``.
        public func toCSR() -> CSRMatrix {
            // Build the CSR structure from COO arrays.
            // 1. Compute nnz per row (indptr will be the prefix-sum).
            var rowCount = [Int](repeating: 0, count: rows)
            for r in rowIndices { rowCount[r] += 1 }

            // 2. indptr: exclusive prefix sum of rowCount.
            var indptr = [Int](repeating: 0, count: rows + 1)
            for i in 0..<rows { indptr[i + 1] = indptr[i] + rowCount[i] }

            // 3. Fill indices and data; track current insertion position per row.
            var indices = [Int](repeating: 0, count: nnz)
            var data    = [Double](repeating: 0, count: nnz)
            var pos     = Array(indptr.dropLast())  // mutable insertion cursor per row

            for k in 0..<nnz {
                let r   = rowIndices[k]
                let dst = pos[r]
                indices[dst] = colIndices[k]
                data[dst]    = values[k]
                pos[r] += 1
            }

            // 4. Sort indices within each row (COO may not be sorted).
            for i in 0..<rows {
                let start = indptr[i]
                let end   = indptr[i + 1]
                if end - start <= 1 { continue }
                // Gather (col, val) pairs, sort by col, scatter back.
                var pairs = [(col: Int, val: Double)]()
                pairs.reserveCapacity(end - start)
                for k in start..<end { pairs.append((indices[k], data[k])) }
                pairs.sort { $0.col < $1.col }
                for (k, p) in pairs.enumerated() {
                    indices[start + k] = p.col
                    data[start + k]    = p.val
                }
            }

            return CSRMatrix(rows: rows, cols: cols,
                             indptr: indptr, indices: indices, data: data)
        }

        /// Convert to a dense ``LinAlg.Matrix``.
        ///
        /// - Returns: A dense matrix equivalent to this sparse matrix.
        public func toDense() -> LinAlg.Matrix {
            var flat = [Double](repeating: 0, count: rows * cols)
            for k in 0..<nnz {
                flat[rowIndices[k] * cols + colIndices[k]] += values[k]
            }
            return LinAlg.Matrix(rows: rows, cols: cols, data: flat)
        }
    }

    // MARK: - CSR Matrix

    /// Sparse matrix in compressed sparse row (CSR) format.
    ///
    /// ## Layout
    ///
    /// Three arrays describe the nonzero structure:
    ///
    /// - `indptr`: Row pointer array of length `rows + 1`.  Row `i` occupies
    ///   `data[indptr[i] ..< indptr[i+1]]`.  `indptr[0]` is always 0.
    /// - `indices`: Column indices of all nonzeros, in row-major order.
    ///   Within each row, indices are sorted in ascending order.
    /// - `data`: Nonzero values, parallel to `indices`.
    ///
    /// This is identical to the CSR layout used by scipy.sparse and LAPACK
    /// sparse extensions.
    ///
    /// ## Construction
    ///
    /// Prefer ``COOMatrix/toCSR()`` or ``init(_:)`` (from dense).  Direct
    /// construction via ``init(rows:cols:indptr:indices:data:)`` is available
    /// for interoperability with external CSR data.
    ///
    /// ## Deferred scope
    /// CSC format, sparse eigensolvers, sparse factorizations, and fancy
    /// indexing are out of scope for this MVP.  See the Sparse module
    /// documentation for the full deferred-scope list.
    public struct CSRMatrix: Sendable {
        /// Number of rows.
        public let rows: Int
        /// Number of columns.
        public let cols: Int
        /// Row pointer array (length `rows + 1`); `indptr[i]` is the index of
        /// the first nonzero in row `i`.
        public let indptr: [Int]
        /// Column indices of nonzeros, parallel to ``data``.
        public let indices: [Int]
        /// Nonzero values, parallel to ``indices``.
        public let data: [Double]

        /// Number of stored nonzeros.
        public var nnz: Int { data.count }

        /// Construct a CSR matrix directly from its three CSR arrays.
        ///
        /// The caller is responsible for supplying a structurally valid CSR
        /// layout: `indptr.count == rows + 1`, `indptr[0] == 0`,
        /// `indptr[rows] == nnz`, `indices.count == data.count == nnz`.
        ///
        /// - Parameters:
        ///   - rows: Number of rows.
        ///   - cols: Number of columns.
        ///   - indptr: Row pointer array (length `rows + 1`).
        ///   - indices: Column indices of nonzeros.
        ///   - data: Nonzero values.
        public init(rows: Int, cols: Int,
                    indptr: [Int], indices: [Int], data: [Double]) {
            self.rows    = rows
            self.cols    = cols
            self.indptr  = indptr
            self.indices = indices
            self.data    = data
        }

        /// Construct a CSR matrix from a dense ``LinAlg.Matrix``.
        ///
        /// Exact zeros are excluded (structural sparsity).
        ///
        /// - Parameter matrix: The source dense matrix.
        public init(_ matrix: LinAlg.Matrix) {
            let coo = COOMatrix(matrix)
            let csr = coo.toCSR()
            self.rows    = csr.rows
            self.cols    = csr.cols
            self.indptr  = csr.indptr
            self.indices = csr.indices
            self.data    = csr.data
        }

        /// Convert to a dense ``LinAlg.Matrix``.
        ///
        /// - Returns: A dense matrix equivalent to this sparse matrix.
        public func toDense() -> LinAlg.Matrix {
            var flat = [Double](repeating: 0.0, count: rows * cols)
            for i in 0..<rows {
                for k in indptr[i]..<indptr[i + 1] {
                    flat[i * cols + indices[k]] = data[k]
                }
            }
            return LinAlg.Matrix(rows: rows, cols: cols, data: flat)
        }

        // MARK: SpMV

        /// Compute the sparse matrix–vector product `y = A · x`.
        ///
        /// This operation is the workhorse of iterative sparse solvers and
        /// graph algorithms.  Complexity is O(nnz).
        ///
        /// scipy reference: `csr_matrix @ x`
        ///
        /// - Parameter x: Dense input vector; length must equal ``cols``.
        /// - Returns: Dense output vector of length ``rows``.
        /// - Throws: ``SparseError/dimensionMismatch(_:)`` when
        ///   `x.count ≠ cols`.
        public func spmv(_ x: [Double]) throws -> [Double] {
            guard x.count == cols else {
                throw SparseError.dimensionMismatch(
                    "SpMV: vector length \(x.count) ≠ matrix cols \(cols)")
            }
            var y = [Double](repeating: 0.0, count: rows)
            for i in 0..<rows {
                var acc = 0.0
                for k in indptr[i]..<indptr[i + 1] {
                    acc += data[k] * x[indices[k]]
                }
                y[i] = acc
            }
            return y
        }

        // MARK: SpMM

        /// Compute the sparse matrix–dense matrix product `C = A · B`.
        ///
        /// `A` is this `rows × cols` sparse matrix; `B` is a dense matrix
        /// with `cols` rows.  The result `C` is a dense `rows × B.cols` matrix.
        ///
        /// scipy reference: `csr_matrix @ dense_matrix`
        ///
        /// - Parameter B: Dense right-hand-side matrix.  `B.rows` must equal
        ///   ``cols``.
        /// - Returns: Dense result matrix of shape `rows × B.cols`.
        /// - Throws: ``SparseError/dimensionMismatch(_:)`` when `B.rows ≠ cols`.
        public func spmm(_ B: LinAlg.Matrix) throws -> LinAlg.Matrix {
            guard B.rows == cols else {
                throw SparseError.dimensionMismatch(
                    "SpMM: B.rows \(B.rows) ≠ A.cols \(cols)")
            }
            var Cdata = [Double](repeating: 0.0, count: rows * B.cols)
            for i in 0..<rows {
                for k in indptr[i]..<indptr[i + 1] {
                    let aval = data[k]
                    let acol = indices[k]
                    // Row `acol` of B contributes `aval * B[acol, j]` to C[i, j]
                    for j in 0..<B.cols {
                        Cdata[i * B.cols + j] += aval * B[acol, j]
                    }
                }
            }
            return LinAlg.Matrix(rows: rows, cols: B.cols, data: Cdata)
        }

        // MARK: Element-wise Add

        /// Element-wise addition of two sparse matrices with the same shape.
        ///
        /// scipy reference: `A + B` for two csr_matrices with the same shape.
        ///
        /// - Parameter other: Matrix to add; must have the same ``rows`` and
        ///   ``cols`` as the receiver.
        /// - Returns: A new ``CSRMatrix`` equal to `self + other`.
        /// - Throws: ``SparseError/shapeMismatch(_:)`` when shapes differ.
        public func add(_ other: CSRMatrix) throws -> CSRMatrix {
            guard rows == other.rows && cols == other.cols else {
                throw SparseError.shapeMismatch(
                    "add: shapes (\(rows)×\(cols)) and (\(other.rows)×\(other.cols)) differ")
            }
            // Direct sorted-merge of two CSR matrices (both have sorted column indices
            // within each row).  We walk each row simultaneously, merging the two
            // sorted index streams and summing values at matching column positions.
            var newIndptr  = [Int](repeating: 0, count: rows + 1)
            var newIndices = [Int]()
            var newData    = [Double]()
            newIndices.reserveCapacity(nnz + other.nnz)
            newData.reserveCapacity(nnz + other.nnz)

            for i in 0..<rows {
                let aStart = indptr[i];       let aEnd = indptr[i + 1]
                let bStart = other.indptr[i]; let bEnd = other.indptr[i + 1]
                var ai = aStart
                var bi = bStart
                // Merge two sorted index runs
                while ai < aEnd || bi < bEnd {
                    let aCol = ai < aEnd ? indices[ai] : Int.max
                    let bCol = bi < bEnd ? other.indices[bi] : Int.max
                    if aCol < bCol {
                        newIndices.append(aCol)
                        newData.append(data[ai])
                        ai += 1
                    } else if bCol < aCol {
                        newIndices.append(bCol)
                        newData.append(other.data[bi])
                        bi += 1
                    } else {
                        // Same column: sum
                        newIndices.append(aCol)
                        newData.append(data[ai] + other.data[bi])
                        ai += 1
                        bi += 1
                    }
                }
                newIndptr[i + 1] = newIndices.count
            }

            return CSRMatrix(rows: rows, cols: cols,
                             indptr: newIndptr, indices: newIndices, data: newData)
        }

        // MARK: Transpose

        /// Return the transpose of this matrix.
        ///
        /// The result has shape `cols × rows`.  The transposed matrix is
        /// returned in valid CSR form (sorted column indices within each row).
        ///
        /// scipy reference: `csr_matrix.T`
        ///
        /// - Returns: A new ``CSRMatrix`` equal to `self.T`.
        public func transpose() -> CSRMatrix {
            // Transpose of an R×C CSR matrix is a C×R CSR matrix.
            // Equivalent to converting to COO, swapping row/col, then rebuilding CSR.
            let tRows = cols
            let tCols = rows

            var rowCount = [Int](repeating: 0, count: tRows)
            for c in indices { rowCount[c] += 1 }

            var tIndptr = [Int](repeating: 0, count: tRows + 1)
            for i in 0..<tRows { tIndptr[i + 1] = tIndptr[i] + rowCount[i] }

            var tIndices = [Int](repeating: 0, count: nnz)
            var tData    = [Double](repeating: 0, count: nnz)
            var pos      = Array(tIndptr.dropLast())  // mutable cursor

            for i in 0..<rows {
                for k in indptr[i]..<indptr[i + 1] {
                    let c   = indices[k]
                    let dst = pos[c]
                    tIndices[dst] = i
                    tData[dst]    = data[k]
                    pos[c] += 1
                }
            }

            // The column indices within each new row may be unsorted because we
            // inserted them in the original matrix's row order.  Sort them.
            for i in 0..<tRows {
                let start = tIndptr[i]
                let end   = tIndptr[i + 1]
                if end - start <= 1 { continue }
                var pairs = [(col: Int, val: Double)]()
                pairs.reserveCapacity(end - start)
                for k in start..<end { pairs.append((tIndices[k], tData[k])) }
                pairs.sort { $0.col < $1.col }
                for (k, p) in pairs.enumerated() {
                    tIndices[start + k] = p.col
                    tData[start + k]    = p.val
                }
            }

            return CSRMatrix(rows: tRows, cols: tCols,
                             indptr: tIndptr, indices: tIndices, data: tData)
        }
    }

    // MARK: - spsolve

    /// Solve the sparse linear system A · x = b.
    ///
    /// ## Solver routing
    ///
    /// The solver selects the algorithm based on the detected structure of `A`:
    ///
    /// - **Conjugate Gradient (CG)** — used when `A` is detected as *symmetric
    ///   positive definite* (SPD).  Detection tests symmetry (|A[i,j] - A[j,i]|
    ///   ≤ 1e-10 for all stored entries) and positive definiteness (via a LAPACK
    ///   Cholesky factorization — a non-nil result certifies PD exactly).  CG
    ///   converges in at
    ///   most `n` iterations for exact arithmetic; the implementation stops at
    ///   `maxIter = 10 * n` with relative residual tolerance `rtol = 1e-10`.
    ///   scipy reference: `scipy.sparse.linalg.cg`.
    ///
    /// - **Dense LU fallback** — used for all other matrices.  The sparse
    ///   matrix is converted to dense and solved via the existing
    ///   ``LinAlg.solve(_:_:)`` routine (LAPACK dgesv).  This is exact but
    ///   O(n³); it is intended for small-to-medium systems where sparsity
    ///   benefits are secondary to correctness.
    ///   scipy reference: `scipy.sparse.linalg.spsolve` (with SuperLU).
    ///
    /// ## Deferred
    ///
    /// Sparse direct solvers (sparse Cholesky, sparse LU via UMFPACK/SuperLU)
    /// are out of scope for this MVP.  The dense fallback has O(n³) cost and
    /// is not suitable for large systems.
    ///
    /// - Parameters:
    ///   - A: Square sparse coefficient matrix in CSR format.
    ///   - b: Right-hand-side vector; length must equal `A.rows`.
    /// - Returns: Solution vector `x` of length `A.rows`.
    /// - Throws:
    ///   - ``SparseError/notSquare(rows:cols:)`` when `A` is not square.
    ///   - ``SparseError/dimensionMismatch(_:)`` when `b.count ≠ A.rows`.
    ///   - ``SparseError/singularMatrix`` when no unique solution exists.
    public static func spsolve(_ A: CSRMatrix, _ b: [Double]) throws -> [Double] {
        guard A.rows == A.cols else {
            throw SparseError.notSquare(rows: A.rows, cols: A.cols)
        }
        guard b.count == A.rows else {
            throw SparseError.dimensionMismatch(
                "spsolve: b.count \(b.count) ≠ A.rows \(A.rows)")
        }

        // Attempt CG for SPD matrices; fall through to dense LU otherwise.
        if isSPD(A) {
            if let x = conjugateGradient(A, b) {
                return x
            }
            // CG failed (can happen for ill-conditioned SPD matrices) → fallback
        }

        // Dense LU fallback via LinAlg.solve (LAPACK dgesv)
        return try denseLUSolve(A, b)
    }

    // MARK: - Private: SPD detection

    /// Test whether `A` is symmetric positive definite.
    ///
    /// The test is exact (not heuristic) via:
    /// 1. Symmetry check: |A[i,j] - A[j,i]| ≤ 1e-10 for all stored nonzeros.
    /// 2. Positive definiteness: convert to dense and attempt LAPACK Cholesky
    ///    (``LinAlg.cholesky``).  A non-nil return confirms SPD.
    ///
    /// This matches the scipy approach where `cg` is applied to SPD systems;
    /// Cholesky failure is the definitive negative criterion.
    private static func isSPD(_ A: CSRMatrix) -> Bool {
        let n = A.rows
        guard n > 0 else { return false }

        // 1. Symmetry check: for each stored A[i,j] find A[j,i]
        for i in 0..<n {
            for k in A.indptr[i]..<A.indptr[i + 1] {
                let j   = A.indices[k]
                let aij = A.data[k]
                if j == i { continue }  // diagonal is trivially symmetric
                // Find A[j, i] in row j
                var aji: Double? = nil
                for m in A.indptr[j]..<A.indptr[j + 1] {
                    if A.indices[m] == i { aji = A.data[m]; break }
                }
                guard let v = aji, abs(aij - v) <= 1e-10 else { return false }
            }
        }

        // 2. Positive definiteness via Cholesky (LAPACK dpotrf).
        //    LinAlg.cholesky returns nil for non-PD matrices; a non-nil result
        //    certifies positive definiteness.
        let dense = A.toDense()
        guard let _ = try? LinAlg.cholesky(dense) else { return false }
        return true
    }

    // MARK: - Private: Conjugate Gradient

    /// Unpreconditioned Conjugate Gradient for SPD systems.
    ///
    /// Implements the standard CG algorithm (Hestenes & Stiefel 1952):
    ///
    /// ```
    /// r₀ = b - A·x₀   (x₀ = 0)
    /// p₀ = r₀
    /// for k = 0, 1, 2, …:
    ///     αₖ  = (rₖ·rₖ) / (pₖ·A·pₖ)
    ///     xₖ₊₁ = xₖ + αₖ·pₖ
    ///     rₖ₊₁ = rₖ - αₖ·A·pₖ
    ///     if ‖rₖ₊₁‖ < rtol·‖b‖: converged
    ///     βₖ  = (rₖ₊₁·rₖ₊₁) / (rₖ·rₖ)
    ///     pₖ₊₁ = rₖ₊₁ + βₖ·pₖ
    /// ```
    ///
    /// scipy reference: `scipy.sparse.linalg.cg` with `rtol=1e-10`.
    ///
    /// - Returns: Solution vector, or `nil` if CG did not converge within
    ///   `maxIter = 10 * n` iterations.
    private static func conjugateGradient(_ A: CSRMatrix, _ b: [Double]) -> [Double]? {
        let n = b.count
        let rtol = 1e-10
        let maxIter = 10 * n

        var x = [Double](repeating: 0.0, count: n)
        var r = b                               // r₀ = b - A·x₀ = b (x₀ = 0)
        var p = r                               // p₀ = r₀
        var rsold = dot(r, r)                   // rᵀ·r

        let bnorm = sqrt(dot(b, b))
        let atol  = rtol * (bnorm > 0 ? bnorm : 1.0)

        for _ in 0..<maxIter {
            // A is square (n×n) and p has length n = A.cols, so spmv cannot throw.
            // The guard in spsolve ensures A.rows == A.cols == b.count == n.
            guard let Ap = try? A.spmv(p) else { return nil }
            let pAp = dot(p, Ap)
            guard abs(pAp) > 0 else { return nil }  // breakdown
            let alpha = rsold / pAp

            // xₖ₊₁ = xₖ + α·p
            for i in 0..<n { x[i] += alpha * p[i] }
            // rₖ₊₁ = rₖ - α·A·p
            for i in 0..<n { r[i] -= alpha * Ap[i] }

            let rsnew = dot(r, r)
            if sqrt(rsnew) < atol { return x }  // converged

            let beta = rsnew / rsold
            // pₖ₊₁ = rₖ₊₁ + β·pₖ
            for i in 0..<n { p[i] = r[i] + beta * p[i] }
            rsold = rsnew
        }

        return nil   // did not converge
    }

    // MARK: - Private: Dense LU Fallback

    /// Solve via dense LU (LAPACK dgesv) by converting the CSR matrix to dense.
    ///
    /// Throws ``SparseError/singularMatrix`` when ``LinAlg/solve(_:_:)``
    /// returns `nil` (singular input).
    private static func denseLUSolve(_ A: CSRMatrix, _ b: [Double]) throws -> [Double] {
        let densA = A.toDense()
        let densB = LinAlg.Matrix(b)   // column vector
        guard let x = try LinAlg.solve(densA, densB) else {
            throw SparseError.singularMatrix
        }
        return x.data
    }

    // MARK: - Private: BLAS-free dot product

    /// Compute the dot product of two vectors (length must match).
    private static func dot(_ a: [Double], _ b: [Double]) -> Double {
        // Use vDSP for performance when vectors are non-trivial.
        var result = 0.0
        vDSP_dotprD(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }
}
