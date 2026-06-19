//
//  LinAlg+Solvers.swift
//  Sources/NumericSwift/
//
//  Linear system solvers: solve, lstsq, solveTriangular, choSolve, luSolve.
//
//  This file is one of several `enum LinAlg` extension files (alongside
//  `LinAlg+Expm.swift` and others in the same directory). Each extension file
//  groups a coherent subset of LinAlg's public API.
//
//  The `*Diagnosed` overloads (solveDiagnosed, lstsqDiagnosed,
//  solveTriangularDiagnosed, choSolveDiagnosed) return `Diagnosed<Matrix?>`
//  defined in `NumericDiagnostic.swift`.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

extension LinAlg {

    // MARK: - Linear System Solvers

    /// Solve linear system Ax = b.
    ///
    /// - Parameters:
    ///   - A: Square coefficient matrix
    ///   - b: Right-hand side (vector or matrix)
    /// - Returns: Solution x, or nil if singular
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `A` is not square, or
    ///   ``LinAlgError/dimensionMismatch(_:)`` when `A` and `b` row counts differ.
    ///   Returns `nil` (does not throw) when `A` is square but singular.
    public static func solve(_ A: Matrix, _ b: Matrix) throws -> Matrix? {
        guard A.rows == A.cols else { throw LinAlgError.notSquare(rows: A.rows, cols: A.cols) }
        guard A.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("A.rows (\(A.rows)) must equal b.rows (\(b.rows))")
        }

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
    /// - Throws: ``LinAlgError/dimensionMismatch(_:)`` when `A` and `b` row counts differ.
    ///   Returns `nil` (does not throw) when the factorization fails.
    public static func lstsq(_ A: Matrix, _ b: Matrix) throws -> Matrix? {
        guard A.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("A.rows (\(A.rows)) must equal b.rows (\(b.rows))")
        }

        let maxDim = max(A.rows, A.cols)

        var a = toColumnMajor(A)
        var bCol = buildLstsqRHS(b: b, maxDim: maxDim)

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

        // LAPACK workspace-query idiom (LAPACK Users' Guide §2.4):
        // The first dgels_ call above (with lwork = -1) is a dry run — it does not
        // solve anything; instead it writes the optimal workspace size into work[0].
        // The workspace is now allocated and the matrices are re-supplied because
        // dgels_ overwrites both A and b in place during the factorisation.
        a = toColumnMajor(A)
        bCol = buildLstsqRHS(b: b, maxDim: maxDim)

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
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `A` is not square, or
    ///   ``LinAlgError/dimensionMismatch(_:)`` when `A` and `b` row counts differ.
    public static func solveTriangular(_ A: Matrix, _ b: Matrix,
                                       lower: Bool = true, trans: Bool = false) throws -> Matrix? {
        guard A.rows == A.cols else { throw LinAlgError.notSquare(rows: A.rows, cols: A.cols) }
        guard A.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("A.rows (\(A.rows)) must equal b.rows (\(b.rows))")
        }

        let n = A.rows

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
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `L` is not square, or
    ///   ``LinAlgError/dimensionMismatch(_:)`` when `L` and `b` row counts differ.
    ///   Returns `nil` (does not throw) when the solve fails.
    public static func choSolve(_ L: Matrix, _ b: Matrix) throws -> Matrix? {
        guard L.rows == L.cols else { throw LinAlgError.notSquare(rows: L.rows, cols: L.cols) }
        guard L.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("L.rows (\(L.rows)) must equal b.rows (\(b.rows))")
        }

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
    /// - Throws: ``LinAlgError/notSquare(rows:cols:)`` when `L`, `U`, or `P` is not square;
    ///   ``LinAlgError/dimensionMismatch(_:)`` when the factors and `b` leading dimensions differ;
    ///   ``LinAlgError/invalidParameter(_:)`` when `U` has a zero pivot (the factored matrix is
    ///   singular — back-substitution would otherwise divide by zero and return ±inf/NaN).
    public static func luSolve(_ L: Matrix, _ U: Matrix, _ P: Matrix, _ b: Matrix) throws -> Matrix {
        guard L.rows == L.cols && U.rows == U.cols && P.rows == P.cols else {
            throw LinAlgError.notSquare(rows: L.rows, cols: L.cols)
        }
        guard L.rows == U.rows && L.rows == P.rows && L.rows == b.rows else {
            throw LinAlgError.dimensionMismatch("L, U, P, and b must share the same leading dimension")
        }

        let n = L.rows

        // A zero U pivot means the factored matrix is singular; back-substitution
        // (`sum / U[i,i]`) would divide by zero and silently yield ±inf/NaN.
        // `dgetrf` sets an exact-zero pivot for exact singularity, so this is a
        // clean, non-arbitrary check (near-singular but nonzero pivots remain the
        // caller's conditioning concern, surfaced via `cond`/the diagnosed solvers).
        for i in 0..<n where U.data[i * n + i] == 0 {
            throw LinAlgError.invalidParameter(
                "luSolve: U factor is singular (zero pivot at index \(i)); "
                + "the system has no unique solution")
        }

        let pb = applyPermutation(P: P, b: b, n: n)
        let y = forwardSubstitution(L: L, pb: pb, n: n, nrhs: b.cols)
        let x = backSubstitution(U: U, y: y, n: n, nrhs: b.cols)

        return Matrix(rows: b.rows, cols: b.cols, data: x)
    }

    // MARK: - luSolve helpers

    private static func applyPermutation(P: Matrix, b: Matrix, n: Int) -> [Double] {
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
        return pb
    }

    private static func forwardSubstitution(L: Matrix, pb: [Double], n: Int, nrhs: Int) -> [Double] {
        var y = pb
        for j in 0..<nrhs {
            for i in 0..<n {
                var sum = y[i * nrhs + j]
                for k in 0..<i {
                    sum -= L.data[i * n + k] * y[k * nrhs + j]
                }
                y[i * nrhs + j] = sum / L.data[i * n + i]
            }
        }
        return y
    }

    private static func backSubstitution(U: Matrix, y: [Double], n: Int, nrhs: Int) -> [Double] {
        var x = y
        for j in 0..<nrhs {
            for i in stride(from: n - 1, through: 0, by: -1) {
                var sum = x[i * nrhs + j]
                for k in (i + 1)..<n {
                    sum -= U.data[i * n + k] * x[k * nrhs + j]
                }
                x[i * nrhs + j] = sum / U.data[i * n + i]
            }
        }
        return x
    }

    /// Build the extended column-major RHS buffer for dgels (rows padded to maxDim).
    private static func buildLstsqRHS(b: Matrix, maxDim: Int) -> [Double] {
        var bCol = [Double](repeating: 0, count: maxDim * b.cols)
        let bColMajor = toColumnMajor(b)
        for i in 0..<b.rows {
            for j in 0..<b.cols {
                bCol[j * maxDim + i] = bColMajor[j * b.rows + i]
            }
        }
        return bCol
    }

    // MARK: - Self-aware (diagnostic-bearing) solver overloads

    /// The condition-number ceiling beyond which a linear-system solve is
    /// declared *outside its accuracy envelope*.
    ///
    /// A square system `Ax = b` loses roughly `log10(cond(A))` significant digits
    /// to the conditioning of `A`. At `cond(A) ≈ 1e12` only ~4 of a `Double`'s ~16
    /// significant digits survive, so a result computed past this threshold may be
    /// dominated by round-off and is no longer trustworthy at full precision. This
    /// is the boundary the diagnostic-bearing solvers (``solveDiagnosed(_:_:)`` and
    /// friends) use to emit ``NumericDiagnostic/outsideEnvelope(method:reason:)``.
    ///
    /// Equivalently, a *reciprocal* condition estimate `rcond = 1 / cond(A)` below
    /// `1e-12` trips the same diagnostic.
    public static let solveConditionEnvelope: Double = 1e12

    /// Build the ill-conditioning diagnostic for a system whose condition number
    /// exceeds ``solveConditionEnvelope``, or `nil` when the system is inside the
    /// envelope.
    ///
    /// `cond` here is the SVD-based 2-norm condition number from ``cond(_:)``; an
    /// exactly-singular `A` yields `+inf` and trips the diagnostic.
    private static func conditioningDiagnostic(method: String, conditionNumber: Double) -> NumericDiagnostic? {
        guard conditionNumber > solveConditionEnvelope else { return nil }
        let rcond = conditionNumber.isFinite ? 1.0 / conditionNumber : 0.0
        return .outsideEnvelope(
            method: method,
            reason: "matrix is ill-conditioned (cond ≈ \(conditionNumber), rcond ≈ \(rcond)) "
                + "— exceeds the cond ≤ \(solveConditionEnvelope) accuracy envelope; "
                + "the solution may be dominated by round-off and unreliable"
        )
    }

    /// Solve `Ax = b` (LU), returning the solution paired with any limitation
    /// diagnostic — the self-aware companion to ``solve(_:_:)``.
    ///
    /// The bare ``solve(_:_:)`` is unchanged and remains the zero-overhead path;
    /// this overload additionally estimates `cond(A)` (SVD-based) and emits a
    /// ``NumericDiagnostic/outsideEnvelope(method:reason:)`` when the system is
    /// ill-conditioned beyond ``solveConditionEnvelope`` (e.g. a high-order Hilbert
    /// matrix). The numeric value is identical to ``solve(_:_:)``; only the
    /// diagnostic is added.
    ///
    /// - Parameters:
    ///   - A: Square coefficient matrix.
    ///   - b: Right-hand side (vector or matrix).
    /// - Returns: A ``Diagnosed`` wrapping the solution (`nil` when singular) and
    ///   any ill-conditioning diagnostic.
    /// - Throws: The same errors as ``solve(_:_:)``.
    public static func solveDiagnosed(_ A: Matrix, _ b: Matrix) throws -> Diagnosed<Matrix?> {
        let x = try solve(A, b)
        let diag = conditioningDiagnostic(method: "LinAlg.solve", conditionNumber: cond(A))
        return Diagnosed(x, diagnostics: diag.map { [$0] } ?? [])
    }

    /// Solve the least-squares problem `min ||Ax - b||`, returning the solution
    /// paired with any limitation diagnostic — the self-aware companion to
    /// ``lstsq(_:_:)``.
    ///
    /// Emits a ``NumericDiagnostic/outsideEnvelope(method:reason:)`` when `A` is
    /// rank-deficient or ill-conditioned beyond ``solveConditionEnvelope``. The
    /// bare ``lstsq(_:_:)`` is unchanged.
    ///
    /// - Parameters:
    ///   - A: Coefficient matrix (any shape).
    ///   - b: Right-hand side.
    /// - Returns: A ``Diagnosed`` wrapping the least-squares solution (`nil` on
    ///   factorization failure) and any ill-conditioning diagnostic.
    /// - Throws: The same errors as ``lstsq(_:_:)``.
    public static func lstsqDiagnosed(_ A: Matrix, _ b: Matrix) throws -> Diagnosed<Matrix?> {
        let x = try lstsq(A, b)
        let diag = conditioningDiagnostic(method: "LinAlg.lstsq", conditionNumber: cond(A))
        return Diagnosed(x, diagnostics: diag.map { [$0] } ?? [])
    }

    /// Solve a triangular system `Ax = b`, returning the solution paired with any
    /// limitation diagnostic — the self-aware companion to ``solveTriangular(_:_:lower:trans:)``.
    ///
    /// For a triangular matrix the condition number is governed by the diagonal;
    /// this overload uses the SVD-based ``cond(_:)`` and emits a
    /// ``NumericDiagnostic/outsideEnvelope(method:reason:)`` when it exceeds
    /// ``solveConditionEnvelope`` (e.g. a near-zero diagonal entry). The bare
    /// ``solveTriangular(_:_:lower:trans:)`` is unchanged.
    ///
    /// - Returns: A ``Diagnosed`` wrapping the solution (`nil` when singular) and
    ///   any ill-conditioning diagnostic.
    /// - Throws: The same errors as ``solveTriangular(_:_:lower:trans:)``.
    public static func solveTriangularDiagnosed(_ A: Matrix, _ b: Matrix,
                                                lower: Bool = true, trans: Bool = false) throws -> Diagnosed<Matrix?> {
        let x = try solveTriangular(A, b, lower: lower, trans: trans)
        let diag = conditioningDiagnostic(method: "LinAlg.solveTriangular", conditionNumber: cond(A))
        return Diagnosed(x, diagnostics: diag.map { [$0] } ?? [])
    }

    /// Solve `(L Lᵀ) x = b` from a Cholesky factor, returning the solution paired
    /// with any limitation diagnostic — the self-aware companion to ``choSolve(_:_:)``.
    ///
    /// `choSolve` is only valid for a **symmetric positive-definite** system. This
    /// overload reconstructs `A = L Lᵀ` and checks two preconditions:
    ///
    /// 1. **non-SPD input** — `A` is not (numerically) symmetric positive-definite
    ///    (its symmetric part fails Cholesky), the primary out-of-envelope case;
    /// 2. **ill-conditioning** — `cond(A)` exceeds ``solveConditionEnvelope``.
    ///
    /// Either condition emits a ``NumericDiagnostic/outsideEnvelope(method:reason:)``.
    /// The bare ``choSolve(_:_:)`` is unchanged.
    ///
    /// - Parameters:
    ///   - L: Lower-triangular Cholesky factor.
    ///   - b: Right-hand side.
    /// - Returns: A ``Diagnosed`` wrapping the solution (`nil` when the solve
    ///   fails) and any limitation diagnostic.
    /// - Throws: The same errors as ``choSolve(_:_:)``.
    public static func choSolveDiagnosed(_ L: Matrix, _ b: Matrix) throws -> Diagnosed<Matrix?> {
        let x = try choSolve(L, b)
        // Reconstruct the implied system matrix A = L·Lᵀ and judge its SPD-ness +
        // conditioning. A non-SPD A means choSolve was applied outside its envelope.
        let A = dot(L, L.T)
        var diagnostics: [NumericDiagnostic] = []
        if !isSymmetricPositiveDefinite(A) {
            diagnostics.append(.outsideEnvelope(
                method: "LinAlg.choSolve",
                reason: "the implied system matrix L·Lᵀ is not symmetric positive-definite "
                    + "— choSolve is valid only for SPD systems; the solution is unreliable"
            ))
        }
        if let diag = conditioningDiagnostic(method: "LinAlg.choSolve", conditionNumber: cond(A)) {
            diagnostics.append(diag)
        }
        return Diagnosed(x, diagnostics: diagnostics)
    }

    /// Whether `A` is (numerically) symmetric positive-definite: symmetric to a
    /// relative tolerance AND admitting a Cholesky factorization.
    private static func isSymmetricPositiveDefinite(_ A: Matrix) -> Bool {
        guard A.rows == A.cols else { return false }
        let n = A.rows
        // Symmetry within a scaled tolerance (LAPACK-style: relative to the
        // largest magnitude entry, never a bare absolute epsilon).
        let scale = A.data.reduce(0.0) { Swift.max($0, abs($1)) }
        // Relative tolerance: ~7 orders of magnitude above Double machine epsilon
        // (~2.2e-16), giving slack for the floating-point assembly round-off of
        // L·Lᵀ while still catching genuinely asymmetric input. The relative form
        // (scaled by the largest element) avoids false negatives for large-magnitude
        // matrices where an absolute threshold like 1e-10 would be far too tight.
        // Note: Sparse.isSPD deliberately uses an absolute 1e-10 because it operates
        // on triplet-entry data where only O(1) entries are tested.
        let tol = (scale == 0 ? 1.0 : scale) * 1e-9
        for i in 0..<n {
            for j in (i + 1)..<n where abs(A[i, j] - A[j, i]) > tol {
                return false
            }
        }
        // Positive-definiteness: Cholesky succeeds iff SPD.
        return (try? cholesky(A)) ?? nil != nil
    }
}
