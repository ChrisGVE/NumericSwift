//
//  NumericValue+Accessors.swift
//  Sources/NumericSwift
//
//  Accessors and shape-introspection extensions for NumericValue.
//
//  This file adds:
//    • NumericValue.Kind — a flat enum for dispatch-truth-table tagging (§15)
//    • kind            — the canonical kind of this value
//    • isScalar / isComplex / isMatrix / isComplexMatrix — Boolean flags
//    • isMatrixLike / isComplexLike — convenience group predicates
//    • asScalar / asComplex / asMatrix / asComplexMatrix — optional extractors
//    • asScalarThrowing / asComplexThrowing / asMatrixThrowing /
//      asComplexMatrixThrowing — throwing extractors following the project
//      LinAlgError-style convention
//    • rows / cols / shape / elementCount — shape introspection; nil for scalars
//    • is1x1 — §4.3a coercion gate: true only for 1×1 matrix or complexMatrix
//    • typeAndShapeDescription — diagnostic string combining kind + shape
//
//  Design notes:
//
//    Optional vs. throwing — The plain `asX` accessors return Optional because
//    the caller is asking "is this a scalar?" — an absent value is the natural
//    answer to a type query, not an error. The throwing `asXThrowing` variants
//    are provided for call sites that want to propagate a typed error rather than
//    unwrap an Optional themselves (e.g. pipeline stages that cannot accept a
//    mismatch). Both sets co-exist in the stdlib (Array.first vs. Array[index]).
//
//    rows/cols for scalars — Scalars are conceptually 1×1, but returning 1
//    for rows/cols would break the distinction between a 1×1 matrix and a plain
//    scalar, which is load-bearing for the §4.3a coercion logic. Therefore
//    rows/cols return nil for .scalar and .complex, and callers that need a
//    uniform shape should use is1x1 or the throwing extractors.
//
//  Licensed under the Apache License, Version 2.0.
//

// MARK: - NumericValue.Kind

extension NumericValue {

    /// A flat enumeration naming the four numeric kinds.
    ///
    /// `Kind` is the key into the dispatch truth table (§15): every pipeline
    /// stage switches on `value.kind` rather than on the enum cases themselves,
    /// keeping dispatch logic decoupled from payload extraction.
    ///
    /// ```swift
    /// switch value.kind {
    /// case .scalar:        handleScalar(value.asScalar!)
    /// case .complex:       handleComplex(value.asComplex!)
    /// case .matrix:        handleMatrix(value.asMatrix!)
    /// case .complexMatrix: handleComplexMatrix(value.asComplexMatrix!)
    /// }
    /// ```
    public enum Kind: Equatable, Hashable, Sendable, CaseIterable, CustomStringConvertible {
        /// A real scalar (`Double`).
        case scalar
        /// A complex scalar (`Complex`).
        case complex
        /// A real matrix (`LinAlg.Matrix`).
        case matrix
        /// A complex matrix (`LinAlg.ComplexMatrix`).
        case complexMatrix

        /// A short lowercase label matching the `NumericValue` case name.
        public var description: String {
            switch self {
            case .scalar:        return "scalar"
            case .complex:       return "complex"
            case .matrix:        return "matrix"
            case .complexMatrix: return "complexMatrix"
            }
        }
    }
}

// MARK: - NumericValue.AccessorError

extension NumericValue {

    /// Errors thrown by the `asXThrowing` family of accessors.
    public enum AccessorError: Error, Equatable, Sendable, CustomStringConvertible {
        /// The value's kind does not match the requested kind.
        ///
        /// - Parameters:
        ///   - expected: The kind the caller requested.
        ///   - actual: The kind the value actually has.
        case kindMismatch(expected: Kind, actual: Kind)

        public var description: String {
            switch self {
            case let .kindMismatch(expected, actual):
                return "NumericValue kind mismatch: expected \(expected), got \(actual)"
            }
        }
    }
}

// MARK: - Kind accessor

extension NumericValue {

    /// The dispatch kind of this value.
    ///
    /// Use `kind` as the key into the §15 truth table. The four cases are
    /// mutually exclusive and exhaustive; every `NumericValue` has exactly one.
    public var kind: Kind {
        switch self {
        case .scalar:        return .scalar
        case .complex:       return .complex
        case .matrix:        return .matrix
        case .complexMatrix: return .complexMatrix
        }
    }
}

// MARK: - Boolean kind flags

extension NumericValue {

    /// `true` when the value is a real scalar.
    public var isScalar: Bool { kind == .scalar }

    /// `true` when the value is a complex scalar.
    public var isComplex: Bool { kind == .complex }

    /// `true` when the value is a real matrix.
    public var isMatrix: Bool { kind == .matrix }

    /// `true` when the value is a complex matrix.
    public var isComplexMatrix: Bool { kind == .complexMatrix }

    // MARK: Group predicates

    /// `true` when the value is a real matrix or a real scalar.
    ///
    /// Useful in pipeline stages that promote scalars to 1×1 matrices and then
    /// operate uniformly on the matrix representation.
    public var isMatrixLike: Bool { isScalar || isMatrix }

    /// `true` when the value is a complex scalar or a complex matrix.
    ///
    /// Mirrors `isMatrixLike` for the complex domain.
    public var isComplexLike: Bool { isComplex || isComplexMatrix }
}

// MARK: - Optional extractors

extension NumericValue {

    /// Returns the `Double` payload when `self` is `.scalar`, otherwise `nil`.
    public var asScalar: Double? {
        guard case .scalar(let x) = self else { return nil }
        return x
    }

    /// Returns the `Complex` payload when `self` is `.complex`, otherwise `nil`.
    public var asComplex: Complex? {
        guard case .complex(let z) = self else { return nil }
        return z
    }

    /// Returns the `LinAlg.Matrix` payload when `self` is `.matrix`, otherwise `nil`.
    public var asMatrix: LinAlg.Matrix? {
        guard case .matrix(let m) = self else { return nil }
        return m
    }

    /// Returns the `LinAlg.ComplexMatrix` payload when `self` is `.complexMatrix`,
    /// otherwise `nil`.
    public var asComplexMatrix: LinAlg.ComplexMatrix? {
        guard case .complexMatrix(let cm) = self else { return nil }
        return cm
    }
}

// MARK: - Throwing extractors

extension NumericValue {

    /// Returns the `Double` payload or throws `AccessorError.kindMismatch`.
    ///
    /// Use at call sites that cannot handle a mismatch and want to propagate a
    /// typed error rather than unwrap an Optional.
    ///
    /// - Throws: `AccessorError.kindMismatch(expected: .scalar, actual:)` when
    ///   `self` is not `.scalar`.
    public func asScalarThrowing() throws -> Double {
        guard case .scalar(let x) = self else {
            throw AccessorError.kindMismatch(expected: .scalar, actual: kind)
        }
        return x
    }

    /// Returns the `Complex` payload or throws `AccessorError.kindMismatch`.
    ///
    /// - Throws: `AccessorError.kindMismatch(expected: .complex, actual:)` when
    ///   `self` is not `.complex`.
    public func asComplexThrowing() throws -> Complex {
        guard case .complex(let z) = self else {
            throw AccessorError.kindMismatch(expected: .complex, actual: kind)
        }
        return z
    }

    /// Returns the `LinAlg.Matrix` payload or throws `AccessorError.kindMismatch`.
    ///
    /// - Throws: `AccessorError.kindMismatch(expected: .matrix, actual:)` when
    ///   `self` is not `.matrix`.
    public func asMatrixThrowing() throws -> LinAlg.Matrix {
        guard case .matrix(let m) = self else {
            throw AccessorError.kindMismatch(expected: .matrix, actual: kind)
        }
        return m
    }

    /// Returns the `LinAlg.ComplexMatrix` payload or throws `AccessorError.kindMismatch`.
    ///
    /// - Throws: `AccessorError.kindMismatch(expected: .complexMatrix, actual:)` when
    ///   `self` is not `.complexMatrix`.
    public func asComplexMatrixThrowing() throws -> LinAlg.ComplexMatrix {
        guard case .complexMatrix(let cm) = self else {
            throw AccessorError.kindMismatch(expected: .complexMatrix, actual: kind)
        }
        return cm
    }
}

// MARK: - Shape introspection

extension NumericValue {

    /// Number of rows for matrix kinds; `nil` for scalar and complex.
    ///
    /// Scalars are conceptually dimensionless (not 1×1). Use `is1x1` to test
    /// for a 1×1 matrix or complex-matrix; use `shape` for a uniform tuple.
    public var rows: Int? {
        switch self {
        case .scalar, .complex:
            return nil
        case .matrix(let m):
            return m.rows
        case .complexMatrix(let cm):
            return cm.rows
        }
    }

    /// Number of columns for matrix kinds; `nil` for scalar and complex.
    public var cols: Int? {
        switch self {
        case .scalar, .complex:
            return nil
        case .matrix(let m):
            return m.cols
        case .complexMatrix(let cm):
            return cm.cols
        }
    }

    /// Shape as `(rows, cols)` for matrix kinds; `nil` for scalar and complex.
    public var shape: (rows: Int, cols: Int)? {
        guard let r = rows, let c = cols else { return nil }
        return (r, c)
    }

    /// Total number of stored elements.
    ///
    /// - For `.scalar` and `.complex`, returns `1`.
    /// - For `.matrix` and `.complexMatrix`, returns `rows * cols`.
    public var elementCount: Int {
        switch self {
        case .scalar, .complex:
            return 1
        case .matrix(let m):
            return m.size
        case .complexMatrix(let cm):
            return cm.size
        }
    }

    // MARK: §4.3a coercion gate

    /// `true` only when `self` is a 1×1 `.matrix` or a 1×1 `.complexMatrix`.
    ///
    /// This is the gate for the §4.3a coercion rule: a 1×1 matrix result may be
    /// collapsed to a scalar by the pipeline's output normaliser (Task 14).
    /// Plain `.scalar` and `.complex` values are **not** 1×1 by this definition
    /// — they are already scalars and do not need coercion.
    public var is1x1: Bool {
        switch self {
        case .matrix(let m):
            return m.rows == 1 && m.cols == 1
        case .complexMatrix(let cm):
            return cm.rows == 1 && cm.cols == 1
        case .scalar, .complex:
            return false
        }
    }
}

// MARK: - Diagnostic description

extension NumericValue {

    /// A concise string combining the kind label and shape for use in
    /// log messages, assertion failures, and error descriptions.
    ///
    /// Examples:
    /// ```
    /// NumericValue.scalar(3.14)               → "scalar"
    /// NumericValue.complex(Complex(1, 2))     → "complex"
    /// NumericValue.matrix(2×3 matrix)         → "matrix(2x3)"
    /// NumericValue.complexMatrix(1×4 matrix)  → "complexMatrix(1x4)"
    /// ```
    public var typeAndShapeDescription: String {
        switch self {
        case .scalar:
            return "scalar"
        case .complex:
            return "complex"
        case .matrix(let m):
            return "matrix(\(m.rows)x\(m.cols))"
        case .complexMatrix(let cm):
            return "complexMatrix(\(cm.rows)x\(cm.cols))"
        }
    }
}
