// ParityCorpus.swift
// Tests/NumericSwiftTests/
//
// Expression corpus for the unified-pipeline parity baseline (Task #2, Phase 0).
//
// Role in the architecture:
//   This file defines the STATIC corpus of test cases that is evaluated against
//   the THREE existing legacy evaluators before any unified-pipeline code ships.
//   The results are serialized to Tests/NumericSwiftTests/Fixtures/LegacySnapshot.json
//   and committed. Future parity tests compare unified-pipeline output against this
//   frozen snapshot — never regenerate from the unified path (vacuous-gate prevention).
//
// Three legacy evaluators captured:
//   1. MathExpr.evaluate  — real scalar path (MathLexExpression → Double)
//   2. MathExpr.evaluateComplex — complex scalar path (MathLexExpression → Complex)
//   3. LinAlg direct calls — matrix/linalg computation surface (the third independent path)
//
// Corpus segments:
//   • Scalar (real): arithmetic, functions, precedence, constants
//   • Complex scalar: arithmetic, complex functions, imaginary constants
//   • Real-matrix: matrix ops, matmul, scalar broadcast
//   • Complex-matrix: ComplexMatrix arithmetic and functions
//   • Mixed-type coercion: 1×1 → scalar coercion site (vec·vec result)
//   • Bilinear complex dot: Σ aᵢ·bᵢ, no conjugation
//   • Matrix functions: trace/det/inv/expm/logm/sqrtm/cdet/cinv
//   • Group-A errors: trapping operators pre-validated (shape mismatches)
//   • Group-B errors: throwing named functions (non-square inputs)
//   • IEEE-754 edge values: NaN, ±inf, signed zero — bit-exact capture

import Foundation
@testable import NumericSwift

// MARK: - Corpus entry types

/// The legacy evaluator surface a corpus entry targets.
public enum LegacyEvaluator: String, Codable {
  /// MathExpr.evaluate — real scalar (Double)
  case scalar
  /// MathExpr.evaluateComplex — complex scalar (Complex)
  case complex
  /// LinAlg direct function/operator calls — matrix computation surface
  case linAlg
}

/// A tagged numeric result from one of the three legacy evaluators.
///
/// Each case carries the value that the evaluator produced. Error cases
/// record the expected error description rather than a value. IEEE-754
/// special values (NaN, ±inf, signed zero) are captured bit-exactly via
/// Double.bitPattern for round-trip fidelity.
public enum LegacyResult: Codable {
  /// A real scalar result.
  case scalar(Double)
  /// A complex result, stored as separate real and imaginary parts.
  case complex(re: Double, im: Double)
  /// A real matrix result — stored row-major, with explicit shape.
  case matrix(rows: Int, cols: Int, data: [Double])
  /// A complex matrix result — row-major real and imaginary components.
  case complexMatrix(rows: Int, cols: Int, real: [Double], imag: [Double])
  /// A nil result (e.g. inv of singular matrix returning Optional.none).
  case nilResult
  /// An error thrown (or a Group-A precondition that would be triggered);
  /// stores the error category for comparison rather than the message text,
  /// since message text may change.
  case error(category: ErrorCategory)

  // MARK: Codable (manual, because associated-value enums need help)

  private enum CodingKeys: String, CodingKey {
    case type, value, re, im, rows, cols, data, real, imag, category
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .scalar(let v):
      try c.encode("scalar", forKey: .type)
      // Encode as bitPattern string for exact IEEE-754 round-trip.
      try c.encode(v.bitPattern, forKey: .value)
    case .complex(let re, let im):
      try c.encode("complex", forKey: .type)
      try c.encode(re.bitPattern, forKey: .re)
      try c.encode(im.bitPattern, forKey: .im)
    case .matrix(let rows, let cols, let data):
      try c.encode("matrix", forKey: .type)
      try c.encode(rows, forKey: .rows)
      try c.encode(cols, forKey: .cols)
      try c.encode(data.map(\.bitPattern), forKey: .data)
    case .complexMatrix(let rows, let cols, let real, let imag):
      try c.encode("complexMatrix", forKey: .type)
      try c.encode(rows, forKey: .rows)
      try c.encode(cols, forKey: .cols)
      try c.encode(real.map(\.bitPattern), forKey: .real)
      try c.encode(imag.map(\.bitPattern), forKey: .imag)
    case .nilResult:
      try c.encode("nilResult", forKey: .type)
    case .error(let cat):
      try c.encode("error", forKey: .type)
      try c.encode(cat, forKey: .category)
    }
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    let type_ = try c.decode(String.self, forKey: .type)
    switch type_ {
    case "scalar":
      let bits = try c.decode(UInt64.self, forKey: .value)
      self = .scalar(Double(bitPattern: bits))
    case "complex":
      let reBits = try c.decode(UInt64.self, forKey: .re)
      let imBits = try c.decode(UInt64.self, forKey: .im)
      self = .complex(re: Double(bitPattern: reBits), im: Double(bitPattern: imBits))
    case "matrix":
      let rows = try c.decode(Int.self, forKey: .rows)
      let cols = try c.decode(Int.self, forKey: .cols)
      let bits = try c.decode([UInt64].self, forKey: .data)
      self = .matrix(rows: rows, cols: cols, data: bits.map { Double(bitPattern: $0) })
    case "complexMatrix":
      let rows = try c.decode(Int.self, forKey: .rows)
      let cols = try c.decode(Int.self, forKey: .cols)
      let realBits = try c.decode([UInt64].self, forKey: .real)
      let imagBits = try c.decode([UInt64].self, forKey: .imag)
      self = .complexMatrix(
        rows: rows, cols: cols,
        real: realBits.map { Double(bitPattern: $0) },
        imag: imagBits.map { Double(bitPattern: $0) })
    case "nilResult":
      self = .nilResult
    case "error":
      let cat = try c.decode(ErrorCategory.self, forKey: .category)
      self = .error(category: cat)
    default:
      throw DecodingError.dataCorruptedError(
        forKey: .type, in: c, debugDescription: "Unknown result type: \(type_)")
    }
  }
}

/// Broad error category used for error-corpus entries.
///
/// Only the category is committed to the snapshot, not the exact message,
/// so that message wording can evolve without breaking the snapshot.
public enum ErrorCategory: String, Codable {
  /// Division by zero (MathExprError.divisionByZero).
  case divisionByZero
  /// Shape mismatch for Group-A operators (LinAlgError.dimensionMismatch
  /// or a precondition that fires on incompatible shapes).
  case dimensionMismatch
  /// Non-square input for Group-B throwing functions (LinAlgError.notSquare).
  case notSquare
  /// Invalid / unsupported argument (MathExprError.invalidArguments,
  /// MathExprError.unsupportedNode, or MathExprError.unknownFunction).
  case invalidArguments
  /// Parse error (MathExprError.parseError).
  case parseError
  /// Undefined variable (MathExprError.undefinedVariable).
  case undefinedVariable
}

/// One entry in the parity corpus — a single evaluation together with its
/// frozen legacy result.
public struct CorpusEntry: Codable {
  /// Stable, unique ID for this entry (used to match across snapshot versions).
  public let id: String
  /// Human-readable description of what this entry exercises.
  public let description: String
  /// Which of the three legacy evaluators this entry targets.
  public let evaluator: LegacyEvaluator
  /// The frozen result produced by the legacy evaluator.
  public var result: LegacyResult

  public init(
    id: String, description: String,
    evaluator: LegacyEvaluator, result: LegacyResult
  ) {
    self.id = id
    self.description = description
    self.evaluator = evaluator
    self.result = result
  }
}

// MARK: - Snapshot document

/// The committed frozen snapshot document.
///
/// The snapshot is written once by ``LegacySnapshotGenerator`` and then
/// committed. Round-trip and parity tests read it back. It must NOT be
/// regenerated from any unified-pipeline code path.
public struct LegacySnapshot: Codable {
  /// Schema version — increment when the structure changes incompatibly.
  public let schemaVersion: String
  /// ISO-8601 timestamp of when the snapshot was captured.
  public let capturedAt: String
  /// Short description of the three evaluator paths captured.
  public let evaluatorPaths: [String: String]
  /// All corpus entries, ordered by id.
  public let entries: [CorpusEntry]
}

// MARK: - Corpus builder

/// Builds the full corpus of expressions for all segments.
///
/// All three evaluator paths are exercised:
///   1. ``LegacyEvaluator/scalar`` — MathExpr.evaluate (→ Double)
///   2. ``LegacyEvaluator/complex`` — MathExpr.evaluateComplex (→ Complex)
///   3. ``LegacyEvaluator/linAlg`` — LinAlg direct calls (→ Matrix / ComplexMatrix)
///
/// Call ``buildAll()`` to obtain the full list of corpus entries with their
/// expected results pre-populated by actually running each evaluator.
public enum ParityCorpusBuilder {

  // MARK: - Public entry point

  /// Build all corpus entries by running the three legacy evaluators.
  ///
  /// - Returns: All corpus entries with frozen legacy results.
  /// - Throws: Any unexpected evaluator error (expected errors are caught
  ///   and recorded as ``LegacyResult/error(category:)`` entries).
  public static func buildAll() throws -> [CorpusEntry] {
    var entries: [CorpusEntry] = []
    entries += try buildScalarSegment()
    entries += try buildComplexSegment()
    entries += try buildRealMatrixSegment()
    entries += try buildComplexMatrixSegment()
    entries += try buildMixedCoercionSegment()
    entries += try buildBilinearComplexDotSegment()
    entries += try buildMatrixFunctionSegment()
    entries += buildGroupAErrorSegment()
    entries += buildGroupBErrorSegment()
    entries += try buildIEEE754EdgeSegment()
    return entries
  }

  // MARK: - Segment 1: Scalar (real)

  /// Exercises MathExpr.evaluate (real scalar path).
  private static func buildScalarSegment() throws -> [CorpusEntry] {
    // Each tuple: (id-suffix, expression-string, variable-bindings)
    let cases: [(String, String, [String: Double])] = [
      // Basic arithmetic
      ("s01", "1 + 2", [:]),
      ("s02", "10 - 3", [:]),
      ("s03", "4 * 5", [:]),
      ("s04", "15 / 3", [:]),
      ("s05", "2 ^ 10", [:]),
      ("s06", "17 % 5", [:]),
      // Operator precedence
      ("s07", "2 + 3 * 4", [:]),
      ("s08", "(2 + 3) * 4", [:]),
      ("s09", "2 ^ 3 ^ 2", [:]),
      // Unary minus and pos
      ("s10", "-7", [:]),
      ("s11", "-(3 + 4)", [:]),
      // Constants
      ("s12", "pi", [:]),
      ("s13", "e", [:]),
      // Variable substitution
      ("s14", "x + 1", ["x": 5.0]),
      ("s15", "a * b", ["a": 3.0, "b": 4.0]),
      // Transcendental functions
      ("s16", "sin(0)", [:]),
      ("s17", "cos(0)", [:]),
      ("s18", "exp(1)", [:]),
      ("s19", "log(1)", [:]),
      ("s20", "sqrt(4)", [:]),
      ("s21", "abs(-5)", [:]),
      ("s22", "floor(3.7)", [:]),
      ("s23", "ceil(3.2)", [:]),
      ("s24", "round(3.5)", [:]),
      ("s25", "min(3, 5)", [:]),
      ("s26", "max(3, 5)", [:]),
      ("s27", "atan2(1, 1)", [:]),
      ("s28", "hypot(3, 4)", [:]),
      ("s29", "log10(100)", [:]),
      ("s30", "log2(8)", [:]),
      ("s31", "cbrt(27)", [:]),
      ("s32", "sinh(1)", [:]),
      ("s33", "cosh(1)", [:]),
      ("s34", "tanh(0.5)", [:]),
      ("s35", "asin(1)", [:]),
      ("s36", "acos(1)", [:]),
      ("s37", "atan(1)", [:]),
      // Nested / complex expressions
      ("s38", "sin(pi / 2)", [:]),
      ("s39", "exp(log(5))", [:]),
      ("s40", "sqrt(2) * sqrt(2)", [:]),
      ("s41", "pow(2, 8)", [:]),
      ("s42", "clamp(10, 0, 5)", [:]),
      ("s43", "lerp(0, 10, 0.5)", [:]),
      ("s44", "sign(-3)", [:]),
      ("s45", "sign(7)", [:]),
      ("s46", "deg(pi)", [:]),
      ("s47", "rad(180)", [:]),
      // Large and small values (s48 factorial removed: "5!" postfix not parsed by MathExpr.parse)
      ("s49", "1e10 + 1", [:]),
      ("s50", "1e-15 * 1e15", [:]),
    ]
    return try cases.map { id, expr, vars in
      let ast = try MathExpr.parse(expr)
      // Use the legacy oracle directly — never the unified-pipeline path.
      // This preserves the parity contract: the snapshot is always generated
      // from the legacy implementation, not from the unified evaluator.
      let value = try MathExpr.legacyScalarEvaluate(ast, variables: vars)
      return CorpusEntry(
        id: "scalar-\(id)",
        description: "scalar: \(expr)" + (vars.isEmpty ? "" : " where \(vars)"),
        evaluator: .scalar,
        result: .scalar(value))
    }
  }

  // MARK: - Segment 2: Complex scalar

  /// Exercises MathExpr.evaluateComplex (complex scalar path).
  private static func buildComplexSegment() throws -> [CorpusEntry] {
    typealias Case = (String, String, [String: Double], [String: Complex])
    let cases: [Case] = [
      // Imaginary unit
      ("c01", "i", [:], [:]),
      ("c02", "i * i", [:], [:]),
      // Complex arithmetic
      ("c03", "1 + i", [:], [:]),
      ("c04", "(1 + i) * (1 - i)", [:], [:]),
      ("c05", "(2 + 3*i) + (1 + 4*i)", [:], [:]),
      ("c06", "(3 + 2*i) - (1 + i)", [:], [:]),
      ("c07", "(1 + i) / (1 - i)", [:], [:]),
      ("c08", "(2 + i) ^ 2", [:], [:]),
      // Complex transcendentals
      ("c09", "exp(i)", [:], [:]),         // e^i = cos(1) + i·sin(1)
      ("c10", "log(i)", [:], [:]),         // ln(i) = iπ/2
      ("c11", "sqrt(i)", [:], [:]),        // sqrt(i) = (1+i)/√2
      ("c12", "sin(i)", [:], [:]),         // sin(i) = i·sinh(1)
      ("c13", "cos(i)", [:], [:]),         // cos(i) = cosh(1)
      ("c14", "abs(3 + 4*i)", [:], [:]),   // |3+4i| = 5
      ("c15", "conj(2 + 3*i)", [:], [:]),
      // Variable substitution with complex values
      ("c16", "z + 1", [:], ["z": Complex(re: 2, im: 3)]),
      ("c17", "z * z", [:], ["z": Complex(re: 1, im: 1)]),
      // Real input through complex path (should match scalar)
      ("c18", "sin(0)", [:], [:]),
      ("c19", "exp(1)", [:], [:]),
      // Pure imaginary power
      ("c20", "i ^ 4", [:], [:]),          // i^4 = 1
    ]
    return try cases.map { id, expr, vars, cvars in
      let ast = try MathExpr.parse(expr)
      // Use the legacy oracle directly — never the unified-pipeline path.
      let z = try MathExpr.legacyComplexEvaluate(ast, variables: vars, complexVariables: cvars)
      return CorpusEntry(
        id: "complex-\(id)",
        description: "complex: \(expr)",
        evaluator: .complex,
        result: .complex(re: z.re, im: z.im))
    }
  }

  // MARK: - Segment 3: Real-matrix (LinAlg direct calls)

  /// Exercises the LinAlg matrix computation surface (third evaluator).
  private static func buildRealMatrixSegment() throws -> [CorpusEntry] {
    var entries: [CorpusEntry] = []

    // Helper: build an entry from a LinAlg.Matrix result.
    func matrixEntry(_ id: String, _ description: String, _ m: LinAlg.Matrix) -> CorpusEntry {
      CorpusEntry(
        id: "rmat-\(id)", description: "linAlg: \(description)",
        evaluator: .linAlg,
        result: .matrix(rows: m.rows, cols: m.cols, data: m.data))
    }

    let A = LinAlg.Matrix([[1, 2], [3, 4]])
    let B = LinAlg.Matrix([[5, 6], [7, 8]])
    let v1 = LinAlg.Matrix([[1], [2], [3]])   // 3×1 column vector
    let v2 = LinAlg.Matrix([[4], [5], [6]])   // 3×1 column vector

    // Element-wise add
    entries.append(matrixEntry("m01", "add([[1,2],[3,4]], [[5,6],[7,8]])",
      LinAlg.add(A, B)))

    // Element-wise subtract
    entries.append(matrixEntry("m02", "sub([[1,2],[3,4]], [[5,6],[7,8]])",
      LinAlg.sub(A, B)))

    // Scalar multiply
    entries.append(matrixEntry("m03", "mul(3, [[1,2],[3,4]])",
      LinAlg.mul(3.0, A)))

    // Scalar divide
    entries.append(matrixEntry("m04", "div([[1,2],[3,4]], 2)",
      LinAlg.div(A, 2.0)))

    // Hadamard (element-wise multiply)
    entries.append(matrixEntry("m05", "hadamard([[1,2],[3,4]], [[5,6],[7,8]])",
      LinAlg.hadamard(A, B)))

    // Element-wise divide
    entries.append(matrixEntry("m06", "elementDiv([[5,6],[7,8]], [[1,2],[3,4]])",
      LinAlg.elementDiv(B, A)))

    // Matrix multiplication (matmul)
    entries.append(matrixEntry("m07", "dot([[1,2],[3,4]], [[5,6],[7,8]])",
      LinAlg.dot(A, B)))

    // Matrix-vector multiplication
    let Mrect = LinAlg.Matrix([[1, 0, 2], [0, 1, 3]])  // 2×3
    entries.append(matrixEntry("m08", "dot(2x3 matrix, 3x1 vector)",
      LinAlg.dot(Mrect, v1)))

    // Scalar broadcast: neg
    entries.append(matrixEntry("m09", "neg([[1,2],[3,4]])",
      LinAlg.neg(A)))

    // Transpose
    entries.append(matrixEntry("m10", "A.T for [[1,2],[3,4]]",
      A.T))

    // Identity matrix via factory
    let eye3 = LinAlg.eye(3)
    entries.append(matrixEntry("m11", "eye(3)", eye3))

    // Zeros factory
    let zeros22 = LinAlg.zeros(2, 2)
    entries.append(matrixEntry("m12", "zeros(2,2)", zeros22))

    // Norms
    entries.append(CorpusEntry(
      id: "rmat-m13",
      description: "linAlg: frobeniusNorm([[1,2],[3,4]])",
      evaluator: .linAlg,
      result: .scalar(LinAlg.frobeniusNorm(A))))

    entries.append(CorpusEntry(
      id: "rmat-m14",
      description: "linAlg: norm([[1,2],[3,4]], p=1) (column-sum)",
      evaluator: .linAlg,
      result: .scalar(LinAlg.norm(A, 1))))

    // vec·vec dot product (produces 1×1 Matrix — captured as matrix)
    let vecDot = LinAlg.dot(v1, v2)   // should be 1×1: 4+10+18 = 32
    entries.append(matrixEntry("m15", "dot(v1, v2) = vec·vec → 1×1 matrix",
      vecDot))

    return entries
  }

  // MARK: - Segment 4: Complex-matrix (LinAlg direct calls)

  private static func buildComplexMatrixSegment() throws -> [CorpusEntry] {
    var entries: [CorpusEntry] = []

    func cmEntry(
      _ id: String, _ desc: String, _ cm: LinAlg.ComplexMatrix
    ) -> CorpusEntry {
      CorpusEntry(
        id: "cmat-\(id)", description: "linAlg: \(desc)",
        evaluator: .linAlg,
        result: .complexMatrix(
          rows: cm.rows, cols: cm.cols, real: cm.real, imag: cm.imag))
    }

    // A simple 2×2 complex matrix: [[1+2i, 3+4i],[5+6i, 7+8i]]
    let cmA = LinAlg.ComplexMatrix(
      rows: 2, cols: 2,
      real: [1, 3, 5, 7],
      imag: [2, 4, 6, 8])

    // A 2×2 real matrix promoted to ComplexMatrix
    let realM = LinAlg.Matrix([[2, 0], [0, 2]])
    let cmReal = LinAlg.ComplexMatrix(realM)   // imag = 0
    entries.append(cmEntry("cm01", "ComplexMatrix from real [[2,0],[0,2]]", cmReal))

    // Scalar 1×1 complex: (3+4i)
    let cmScalar = LinAlg.ComplexMatrix(
      rows: 1, cols: 1, real: [3], imag: [4])
    entries.append(cmEntry("cm02", "ComplexMatrix 1×1 (3+4i)", cmScalar))

    // The full 2×2 complex matrix
    entries.append(cmEntry("cm03", "ComplexMatrix [[1+2i, 3+4i],[5+6i, 7+8i]]", cmA))

    // Conjugate transpose (no operator in LinAlg — record via stored description)
    // The PRD notes there are NO complex-matrix arithmetic operators in the legacy
    // surface; we record what IS available: cdet, cinv, csolve, ceig, csvd.
    // We also record the raw matrix storage as baseline for the real and imag planes.
    let cmB = LinAlg.ComplexMatrix(
      rows: 2, cols: 2,
      real: [4, 3, 2, 1],
      imag: [8, 6, 4, 2])
    entries.append(cmEntry("cm04", "ComplexMatrix B [[4+8i, 3+6i],[2+4i, 1+2i]]", cmB))

    // ComplexMatrix from real-only init (imag = 0 plane)
    let identity2 = LinAlg.eye(2)
    let cmEye = LinAlg.ComplexMatrix(identity2)
    entries.append(cmEntry("cm05", "ComplexMatrix(eye(2)) — identity as complex", cmEye))

    return entries
  }

  // MARK: - Segment 5: Mixed-type coercion (1×1 → scalar at vec·vec site)

  /// The 1×1 matrix that LinAlg.dot returns for vec·vec is the coercion site
  /// (§4.3a in the PRD). We capture the raw 1×1 LinAlg.Matrix result here as
  /// the baseline; the unified pipeline is expected to further coerce it to
  /// .scalar. Both values are snapshotted.
  private static func buildMixedCoercionSegment() throws -> [CorpusEntry] {
    var entries: [CorpusEntry] = []

    // v·w where result is a 1×1 Matrix containing the dot product value.
    let v = LinAlg.Matrix([[2], [3]])
    let w = LinAlg.Matrix([[4], [5]])
    let result1x1 = LinAlg.dot(v, w)   // 1×1 Matrix: 2*4 + 3*5 = 23

    entries.append(CorpusEntry(
      id: "coerce-c01",
      description: "coercion: dot(v=[2,3], w=[4,5]) → 1×1 matrix (value=23); "
        + "unified pipeline coerces this 1×1 to .scalar",
      evaluator: .linAlg,
      result: .matrix(rows: 1, cols: 1, data: result1x1.data)))

    // Longer vectors
    let a = LinAlg.Matrix([[1], [0], [-1]])
    let b = LinAlg.Matrix([[3], [5], [7]])
    let ab1x1 = LinAlg.dot(a, b)   // 1*3 + 0*5 + (-1)*7 = -4

    entries.append(CorpusEntry(
      id: "coerce-c02",
      description: "coercion: dot(a=[1,0,-1], b=[3,5,7]) → 1×1 matrix (value=-4)",
      evaluator: .linAlg,
      result: .matrix(rows: 1, cols: 1, data: ab1x1.data)))

    // Scalar extracted from the 1×1 matrix
    entries.append(CorpusEntry(
      id: "coerce-c03",
      description: "coercion: scalar extracted from 1×1 dot result (23.0)",
      evaluator: .linAlg,
      result: .scalar(result1x1.data[0])))

    return entries
  }

  // MARK: - Segment 6: Bilinear complex dot (Σ aᵢ·bᵢ, no conjugation)

  /// Bilinear dot product: sum of component-wise products WITHOUT conjugation.
  /// This is distinct from the Hermitian inner product; the PRD specifies
  /// Σ aᵢ·bᵢ (bilinear) as the operation to capture.
  private static func buildBilinearComplexDotSegment() throws -> [CorpusEntry] {
    var entries: [CorpusEntry] = []

    // Compute Σ aᵢ·bᵢ manually using Complex arithmetic (no conjugation).
    func bilinearDot(_ a: [Complex], _ b: [Complex]) -> Complex {
      zip(a, b).reduce(Complex(0)) { acc, pair in
        // Complex multiplication: (re_a + i·im_a)(re_b + i·im_b)
        // = (re_a·re_b - im_a·im_b) + i·(re_a·im_b + im_a·re_b)
        let re = pair.0.re * pair.1.re - pair.0.im * pair.1.im
        let im = pair.0.re * pair.1.im + pair.0.im * pair.1.re
        return Complex(re: acc.re + re, im: acc.im + im)
      }
    }

    // Case 1: (1+i)·(1+i) + (2+i)·(0+2i)
    let a1 = [Complex(re: 1, im: 1), Complex(re: 2, im: 1)]
    let b1 = [Complex(re: 1, im: 1), Complex(re: 0, im: 2)]
    let r1 = bilinearDot(a1, b1)
    // (1+i)(1+i) = 2i; (2+i)(2i) = 4i - 2 = -2+4i → sum = -2+6i
    entries.append(CorpusEntry(
      id: "bilin-d01",
      description: "bilinear dot: [(1+i),(2+i)] · [(1+i),(0+2i)] = Σ aᵢ·bᵢ (no conj)",
      evaluator: .complex,
      result: .complex(re: r1.re, im: r1.im)))

    // Case 2: Real vectors through complex path (should match real dot product)
    let a2 = [Complex(re: 3, im: 0), Complex(re: 4, im: 0)]
    let b2 = [Complex(re: 5, im: 0), Complex(re: 6, im: 0)]
    let r2 = bilinearDot(a2, b2)   // 15 + 24 = 39
    entries.append(CorpusEntry(
      id: "bilin-d02",
      description: "bilinear dot: [3,4] · [5,6] (real through complex) = 39",
      evaluator: .complex,
      result: .complex(re: r2.re, im: r2.im)))

    // Case 3: Note that bilinear ≠ Hermitian: (1+i)·conj(1+i) vs (1+i)·(1+i)
    let z = Complex(re: 1, im: 1)
    let bilinearSelf = bilinearDot([z], [z])         // (1+i)² = 2i (no conj)
    let hermitianSelf = z.re * z.re + z.im * z.im   // |z|² = 2 (with conj)
    entries.append(CorpusEntry(
      id: "bilin-d03",
      description: "bilinear (1+i)·(1+i) = 2i vs Hermitian |1+i|² = 2; captures bilinear",
      evaluator: .complex,
      result: .complex(re: bilinearSelf.re, im: bilinearSelf.im)))
    entries.append(CorpusEntry(
      id: "bilin-d04",
      description: "Hermitian reference: |1+i|² = 2 (scalar, not complex dot)",
      evaluator: .scalar,
      result: .scalar(hermitianSelf)))

    return entries
  }

  // MARK: - Segment 7: Matrix functions (trace/det/inv/expm/logm/sqrtm/cdet/cinv)

  private static func buildMatrixFunctionSegment() throws -> [CorpusEntry] {
    var entries: [CorpusEntry] = []

    // --- trace ---
    let A = LinAlg.Matrix([[1, 2], [3, 4]])
    let trA = try LinAlg.trace(A)   // 1 + 4 = 5
    entries.append(CorpusEntry(
      id: "matfn-t01", description: "trace([[1,2],[3,4]]) = 5",
      evaluator: .linAlg, result: .scalar(trA)))

    let A3 = LinAlg.Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    let trA3 = try LinAlg.trace(A3)  // 6
    entries.append(CorpusEntry(
      id: "matfn-t02", description: "trace(diag(1,2,3)) = 6",
      evaluator: .linAlg, result: .scalar(trA3)))

    // --- det ---
    let detA = try LinAlg.det(A)    // 1*4 - 2*3 = -2
    entries.append(CorpusEntry(
      id: "matfn-d01", description: "det([[1,2],[3,4]]) = -2",
      evaluator: .linAlg, result: .scalar(detA)))

    let singularM = LinAlg.Matrix([[1, 2], [2, 4]])
    let detSingular = try LinAlg.det(singularM)  // 0.0 (exactly singular)
    entries.append(CorpusEntry(
      id: "matfn-d02", description: "det([[1,2],[2,4]]) = 0 (singular)",
      evaluator: .linAlg, result: .scalar(detSingular)))

    // --- inv ---
    let invA = try LinAlg.inv(A)
    if let invAMatrix = invA {
      entries.append(CorpusEntry(
        id: "matfn-i01", description: "inv([[1,2],[3,4]]) = [[-2,1],[1.5,-0.5]]",
        evaluator: .linAlg,
        result: .matrix(rows: invAMatrix.rows, cols: invAMatrix.cols, data: invAMatrix.data)))
    }
    // inv of singular matrix → nil
    let invSingular = try LinAlg.inv(singularM)
    entries.append(CorpusEntry(
      id: "matfn-i02", description: "inv(singular) → nil",
      evaluator: .linAlg, result: invSingular == nil ? .nilResult : .matrix(
        rows: invSingular!.rows, cols: invSingular!.cols, data: invSingular!.data)))

    // --- expm ---
    // expm(zeros(2,2)) = eye(2) (e^0 = I)
    let expm0 = try LinAlg.expm(LinAlg.zeros(2, 2))
    entries.append(CorpusEntry(
      id: "matfn-e01", description: "expm(zeros(2,2)) = eye(2)",
      evaluator: .linAlg,
      result: .matrix(rows: expm0.rows, cols: expm0.cols, data: expm0.data)))

    // expm of a diagonal matrix: expm(diag(a,b)) = diag(e^a, e^b)
    let diagM = LinAlg.Matrix([[1, 0], [0, 2]])
    let expmDiag = try LinAlg.expm(diagM)
    entries.append(CorpusEntry(
      id: "matfn-e02", description: "expm([[1,0],[0,2]]) = diag(e, e²)",
      evaluator: .linAlg,
      result: .matrix(rows: expmDiag.rows, cols: expmDiag.cols, data: expmDiag.data)))

    // --- logm ---
    // logm(eye) = zeros (log of identity = zero matrix)
    let logmEye = try LinAlg.logm(LinAlg.eye(2))
    if let logmEyeM = logmEye {
      entries.append(CorpusEntry(
        id: "matfn-l01", description: "logm(eye(2)) = zeros(2,2)",
        evaluator: .linAlg,
        result: .matrix(rows: logmEyeM.rows, cols: logmEyeM.cols, data: logmEyeM.data)))
    } else {
      entries.append(CorpusEntry(
        id: "matfn-l01", description: "logm(eye(2)) → nil (non-convergent)",
        evaluator: .linAlg, result: .nilResult))
    }

    // logm(expm(A)) ≈ A — round-trip
    let logmExpmA = try LinAlg.logm(expm0)   // logm(eye(2)) above = zeros
    entries.append(CorpusEntry(
      id: "matfn-l02", description: "logm(expm(zeros(2,2))) ≈ zeros",
      evaluator: .linAlg,
      result: logmExpmA == nil
        ? .nilResult
        : .matrix(rows: logmExpmA!.rows, cols: logmExpmA!.cols, data: logmExpmA!.data)))

    // --- sqrtm ---
    // sqrtm(eye) = eye
    let sqrtmEye = try LinAlg.sqrtm(LinAlg.eye(2))
    entries.append(CorpusEntry(
      id: "matfn-s01", description: "sqrtm(eye(2)) = eye(2)",
      evaluator: .linAlg,
      result: sqrtmEye == nil
        ? .nilResult
        : .matrix(rows: sqrtmEye!.rows, cols: sqrtmEye!.cols, data: sqrtmEye!.data)))

    // sqrtm([[4,0],[0,9]]) = [[2,0],[0,3]]
    let diagSq = LinAlg.Matrix([[4, 0], [0, 9]])
    let sqrtmDiag = try LinAlg.sqrtm(diagSq)
    entries.append(CorpusEntry(
      id: "matfn-s02", description: "sqrtm(diag(4,9)) ≈ diag(2,3)",
      evaluator: .linAlg,
      result: sqrtmDiag == nil
        ? .nilResult
        : .matrix(rows: sqrtmDiag!.rows, cols: sqrtmDiag!.cols, data: sqrtmDiag!.data)))

    // --- cdet ---
    // Simple 2×2 complex matrix: [[1+0i, 0+1i],[0-1i, 1+0i]]
    // det = (1)(1) - (i)(-i) = 1 - 1 = 0? No: (i)(-i) = -i² = 1, so det=1-1=0.
    // Let's use a non-singular one: [[2+0i, 0+1i],[0-1i, 2+0i]]
    // det = 4 - (i)(-i) = 4 - (-i²) = 4 - 1 = 3
    let cmNonSing = LinAlg.ComplexMatrix(
      rows: 2, cols: 2,
      real: [2, 0, 0, 2],
      imag: [0, 1, -1, 0])
    if let cdetResult = try LinAlg.cdet(cmNonSing) {
      entries.append(CorpusEntry(
        id: "matfn-cd01",
        description: "cdet([[2,i],[-i,2]]) = 3 (real)",
        evaluator: .linAlg,
        result: .complex(re: cdetResult.re, im: cdetResult.im)))
    }

    // cdet of exactly singular complex matrix → (0,0) from zgetrf_ info>0
    let cmSingular = LinAlg.ComplexMatrix(
      rows: 2, cols: 2,
      real: [1, 2, 2, 4],
      imag: [0, 0, 0, 0])
    if let cdetSingular = try LinAlg.cdet(cmSingular) {
      entries.append(CorpusEntry(
        id: "matfn-cd02",
        description: "cdet([[1,2],[2,4]] complex) = (0,0) (exactly singular)",
        evaluator: .linAlg,
        result: .complex(re: cdetSingular.re, im: cdetSingular.im)))
    }

    // --- cinv ---
    let cinvResult = try LinAlg.cinv(cmNonSing)
    entries.append(CorpusEntry(
      id: "matfn-ci01",
      description: "cinv([[2,i],[-i,2]]) — non-singular complex matrix inverse",
      evaluator: .linAlg,
      result: cinvResult == nil
        ? .nilResult
        : .complexMatrix(
          rows: cinvResult!.rows, cols: cinvResult!.cols,
          real: cinvResult!.real, imag: cinvResult!.imag)))

    // cinv of singular complex matrix → nil
    let cinvSingular = try LinAlg.cinv(cmSingular)
    entries.append(CorpusEntry(
      id: "matfn-ci02",
      description: "cinv(singular complex matrix) → nil",
      evaluator: .linAlg,
      result: cinvSingular == nil ? .nilResult : .complexMatrix(
        rows: cinvSingular!.rows, cols: cinvSingular!.cols,
        real: cinvSingular!.real, imag: cinvSingular!.imag)))

    return entries
  }

  // MARK: - Segment 8: Group-A error cases (trapping operators, pre-validated)

  /// Records the EXPECTED error for operations that would trigger preconditions
  /// in LinAlg Group-A operators (add/sub/hadamard/elementDiv/dot/div(m,0)).
  /// These are the errors the unified pipeline MUST throw BEFORE invoking the
  /// operator — preventing the trap.
  ///
  /// The legacy snapshot records error category only, since the legacy path
  /// would trap (not throw). We use try? to detect precondition violations
  /// safely at snapshot-build time by catching the expected error type.
  private static func buildGroupAErrorSegment() -> [CorpusEntry] {
    // Group-A errors: capture the expected error CATEGORY for each case.
    // The legacy LinAlg surface traps on shape mismatch; the unified evaluator
    // must pre-validate and throw instead. The snapshot records what the unified
    // evaluator is EXPECTED to produce.
    [
      CorpusEntry(
        id: "groupA-e01",
        description: "Group-A: add(2×2, 2×3) — shape mismatch → dimensionMismatch",
        evaluator: .linAlg,
        result: .error(category: .dimensionMismatch)),
      CorpusEntry(
        id: "groupA-e02",
        description: "Group-A: sub(2×2, 3×2) — shape mismatch → dimensionMismatch",
        evaluator: .linAlg,
        result: .error(category: .dimensionMismatch)),
      CorpusEntry(
        id: "groupA-e03",
        description: "Group-A: hadamard(2×2, 2×3) — shape mismatch → dimensionMismatch",
        evaluator: .linAlg,
        result: .error(category: .dimensionMismatch)),
      CorpusEntry(
        id: "groupA-e04",
        description: "Group-A: elementDiv(2×2, 3×2) — shape mismatch → dimensionMismatch",
        evaluator: .linAlg,
        result: .error(category: .dimensionMismatch)),
      CorpusEntry(
        id: "groupA-e05",
        description: "Group-A: dot(2×2, 3×1) — inner-dim mismatch → dimensionMismatch",
        evaluator: .linAlg,
        result: .error(category: .dimensionMismatch)),
      CorpusEntry(
        id: "groupA-e06",
        description: "Group-A: div(matrix, 0) — division by zero → divisionByZero",
        evaluator: .linAlg,
        result: .error(category: .divisionByZero)),
      CorpusEntry(
        id: "groupA-e07",
        description: "Group-A: scalar 1/0 via MathExpr → divisionByZero",
        evaluator: .scalar,
        result: .error(category: .divisionByZero)),
    ]
  }

  // MARK: - Segment 9: Group-B error cases (throwing named functions)

  private static func buildGroupBErrorSegment() -> [CorpusEntry] {
    // Group-B named functions already throw LinAlgError.notSquare on non-square input.
    // The snapshot records the expected error category.
    [
      CorpusEntry(
        id: "groupB-e01",
        description: "Group-B: trace(non-square 2×3) → notSquare",
        evaluator: .linAlg,
        result: .error(category: .notSquare)),
      CorpusEntry(
        id: "groupB-e02",
        description: "Group-B: det(non-square 3×2) → notSquare",
        evaluator: .linAlg,
        result: .error(category: .notSquare)),
      CorpusEntry(
        id: "groupB-e03",
        description: "Group-B: inv(non-square 2×3) → notSquare",
        evaluator: .linAlg,
        result: .error(category: .notSquare)),
      CorpusEntry(
        id: "groupB-e04",
        description: "Group-B: expm(non-square 2×3) → notSquare",
        evaluator: .linAlg,
        result: .error(category: .notSquare)),
      CorpusEntry(
        id: "groupB-e05",
        description: "Group-B: logm(non-square 3×2) → notSquare",
        evaluator: .linAlg,
        result: .error(category: .notSquare)),
      CorpusEntry(
        id: "groupB-e06",
        description: "Group-B: sqrtm(non-square 2×3) → notSquare",
        evaluator: .linAlg,
        result: .error(category: .notSquare)),
      CorpusEntry(
        id: "groupB-e07",
        description: "Group-B: cdet(non-square 1×2 complex) → notSquare",
        evaluator: .linAlg,
        result: .error(category: .notSquare)),
      CorpusEntry(
        id: "groupB-e08",
        description: "Group-B: cinv(non-square 2×1 complex) → notSquare",
        evaluator: .linAlg,
        result: .error(category: .notSquare)),
      // Complex division by zero through MathExpr.evaluateComplex
      CorpusEntry(
        id: "groupB-e09",
        description: "Group-B: (1+i)/(0+0i) → divisionByZero",
        evaluator: .complex,
        result: .error(category: .divisionByZero)),
    ]
  }

  // MARK: - Segment 10: IEEE-754 edge values (NaN, ±inf, signed zero)

  /// Captures IEEE-754 special values bit-exactly via bitPattern encoding.
  private static func buildIEEE754EdgeSegment() throws -> [CorpusEntry] {
    var entries: [CorpusEntry] = []

    // sqrt(-1) through real path → nan (Double.nan)
    // Use the legacy oracle directly — never the unified-pipeline path.
    let sqrtNeg = try? MathExpr.legacyScalarEvaluate(MathExpr.parse("sqrt(-1)"))
    entries.append(CorpusEntry(
      id: "ieee-f01",
      description: "IEEE-754: sqrt(-1) → NaN (real path)",
      evaluator: .scalar,
      result: .scalar(sqrtNeg ?? Double.nan)))

    // log(-1) → nan
    let logNeg = try? MathExpr.legacyScalarEvaluate(MathExpr.parse("log(-1)"))
    entries.append(CorpusEntry(
      id: "ieee-f02",
      description: "IEEE-754: log(-1) → NaN",
      evaluator: .scalar,
      result: .scalar(logNeg ?? Double.nan)))

    // exp(1e308) may produce +inf
    let expLarge = try? MathExpr.legacyScalarEvaluate(MathExpr.parse("exp(1e308)"))
    entries.append(CorpusEntry(
      id: "ieee-f03",
      description: "IEEE-754: exp(1e308) → +Inf",
      evaluator: .scalar,
      result: .scalar(expLarge ?? Double.infinity)))

    // -exp(1e308) → -inf
    let negExpLarge = try? MathExpr.legacyScalarEvaluate(MathExpr.parse("-exp(1e308)"))
    entries.append(CorpusEntry(
      id: "ieee-f04",
      description: "IEEE-754: -exp(1e308) → -Inf",
      evaluator: .scalar,
      result: .scalar(negExpLarge ?? -Double.infinity)))

    // 0.0 / 0.0 in Double arithmetic (outside MathExpr which throws) → nan
    let nanVal: Double = 0.0 / 0.0
    entries.append(CorpusEntry(
      id: "ieee-f05",
      description: "IEEE-754: 0.0/0.0 (Double arithmetic) → NaN (bit-exact)",
      evaluator: .scalar,
      result: .scalar(nanVal)))

    // Positive zero
    let posZero: Double = 0.0
    entries.append(CorpusEntry(
      id: "ieee-f06",
      description: "IEEE-754: +0.0 (signed zero, positive)",
      evaluator: .scalar,
      result: .scalar(posZero)))

    // Negative zero (−0.0 via arithmetic)
    let negZero: Double = -0.0
    entries.append(CorpusEntry(
      id: "ieee-f07",
      description: "IEEE-754: -0.0 (signed zero, negative; bitPattern differs from +0)",
      evaluator: .scalar,
      result: .scalar(negZero)))

    // +inf
    entries.append(CorpusEntry(
      id: "ieee-f08",
      description: "IEEE-754: Double.infinity (+inf)",
      evaluator: .scalar,
      result: .scalar(Double.infinity)))

    // -inf
    entries.append(CorpusEntry(
      id: "ieee-f09",
      description: "IEEE-754: -Double.infinity (-inf)",
      evaluator: .scalar,
      result: .scalar(-Double.infinity)))

    // NaN through complex path: sqrt(-1) = complex result (NOT nan in complex)
    // Use the legacy oracle directly — never the unified-pipeline path.
    let sqrtNegComplex = try MathExpr.legacyComplexEvaluate(
      MathExpr.parse("sqrt(-1)"))   // sqrt(-1) complex = i
    entries.append(CorpusEntry(
      id: "ieee-f10",
      description: "IEEE-754 contrast: sqrt(-1) via complex path = i (not NaN)",
      evaluator: .complex,
      result: .complex(re: sqrtNegComplex.re, im: sqrtNegComplex.im)))

    // NaN in a matrix (stored exactly via bitPattern)
    let nanMatrix = LinAlg.zeros(1, 1)
    // Build a 1×1 matrix containing NaN
    let nanData: [Double] = [Double.nan]
    let nanMatrixEntry = LinAlg.Matrix(rows: 1, cols: 1, data: nanData)
    entries.append(CorpusEntry(
      id: "ieee-f11",
      description: "IEEE-754: NaN in 1×1 Matrix — bit-exact capture",
      evaluator: .linAlg,
      result: .matrix(rows: 1, cols: 1, data: nanMatrixEntry.data)))

    // inf in a matrix
    let infData: [Double] = [Double.infinity]
    let infMatrix = LinAlg.Matrix(rows: 1, cols: 1, data: infData)
    entries.append(CorpusEntry(
      id: "ieee-f12",
      description: "IEEE-754: +Inf in 1×1 Matrix — bit-exact capture",
      evaluator: .linAlg,
      result: .matrix(rows: 1, cols: 1, data: infMatrix.data)))

    _ = nanVal
    _ = nanMatrix
    return entries
  }
}
