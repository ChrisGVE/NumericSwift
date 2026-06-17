//
//  LinSolve.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: LinAlg linear-system solvers.
//
//  Strategies compared (WORKBENCH.md §4): `solve` (LU / dgesv), `lstsq` (dgels),
//  `choSolve` (Cholesky / dpotrs), `solveTriangular` (dtrtrs).
//
//  ## Comparison scalar
//
//  Each strategy yields a solution vector x; the workbench compares a single
//  scalar. We use **x[0]** — the first solution component — a deterministic
//  functional of the solution, matching the oracle (`linsolve.py` returns x[0]).
//
//  ## Input reconstruction
//
//  Fixtures carry the matrices as flat row-major `Double` arrays via the
//  ``InputValue/array(_:)`` mechanism:
//    • solve / solveTriangular : `A` (flat, n×n), `n`, `b` (length n); triangular
//      also carries `lower` (Bool).
//    • lstsq                   : `A` (flat, rows×cols), `rows`, `cols`, `b`.
//    • choSolve                : `L` (flat Cholesky factor, n×n), `n`, `b`. The
//      implied system matrix is A = L·Lᵀ.
//
//  ## Self-awareness
//
//  Out-of-envelope cases are tagged `inEnvelope: false`: ill-conditioned /
//  near-singular systems (cond(A) > 1e12 — e.g. high-order Hilbert matrices), and
//  for `choSolve`, non-SPD input. For the gate to pass the *library* must emit a
//  ``NumericDiagnostic/outsideEnvelope`` for those — the closures only forward
//  what the diagnostic-bearing solver overloads
//  (``LinAlg/solveDiagnosed(_:_:)`` etc.) produced; they never fabricate one.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The LinAlg linear-system-solver domain suite.
    public static let linsolveSuite = DomainSuite(
        name: "linsolve",
        registerStrategies: registerLinSolveStrategies,
        makeEnvelopeRegistry: makeLinSolveEnvelopeRegistry
    )
}

// MARK: - Input reconstruction

/// Read a flat `Double` array from an ``InputValue/array(_:)`` input.
@Sendable
private func doubleArray(_ value: InputValue?) -> [Double]? {
    guard let elements = value?.arrayValue else { return nil }
    let doubles = elements.compactMap(\.doubleValue)
    return doubles.count == elements.count ? doubles : nil
}

/// Reconstruct a row-major `LinAlg.Matrix` from a flat array and explicit shape.
@Sendable
private func matrix(_ flat: [Double], rows: Int, cols: Int) -> LinAlg.Matrix? {
    guard rows > 0, cols > 0, flat.count == rows * cols else { return nil }
    return LinAlg.Matrix(rows: rows, cols: cols, data: flat)
}

/// Reconstruct the RHS column vector `b` (an n×1 matrix).
@Sendable
private func rhs(_ value: InputValue?) -> LinAlg.Matrix? {
    guard let b = doubleArray(value), !b.isEmpty else { return nil }
    return LinAlg.Matrix(b)
}

/// The comparison scalar: the first component of a solution column vector.
@Sendable
private func firstComponent(_ x: LinAlg.Matrix?) -> Double? {
    guard let x = x, x.size > 0 else { return nil }
    return x.data[0]
}

// MARK: - Strategy registrations

/// Populate `registry` with the LinAlg linear-system-solver strategies.
@Sendable
public func registerLinSolveStrategies(into registry: inout StrategyRegistry) {

    // solve — LU (dgesv) on a square system; diagnostic on ill-conditioned A.
    registry.register(id: "solve") { inputs in
        guard let n = inputs["n"]?.intValue,
              let flat = doubleArray(inputs["A"]),
              let A = matrix(flat, rows: n, cols: n),
              let b = rhs(inputs["b"]) else { return nil }
        guard let d = try? LinAlg.solveDiagnosed(A, b) else { return nil }
        guard let value = firstComponent(d.value) else { return nil }
        return StrategyResult(value: value, diagnostics: d.diagnostics)
    }

    // lstsq — least squares (dgels) on an (rows×cols) system; diagnostic on
    // rank-deficient / ill-conditioned A.
    registry.register(id: "lstsq") { inputs in
        guard let rows = inputs["rows"]?.intValue,
              let cols = inputs["cols"]?.intValue,
              let flat = doubleArray(inputs["A"]),
              let A = matrix(flat, rows: rows, cols: cols),
              let b = rhs(inputs["b"]) else { return nil }
        guard let d = try? LinAlg.lstsqDiagnosed(A, b) else { return nil }
        guard let value = firstComponent(d.value) else { return nil }
        return StrategyResult(value: value, diagnostics: d.diagnostics)
    }

    // choSolve — Cholesky solve (dpotrs) given the factor L; diagnostic on
    // non-SPD or ill-conditioned implied system A = L·Lᵀ.
    registry.register(id: "choSolve") { inputs in
        guard let n = inputs["n"]?.intValue,
              let flat = doubleArray(inputs["L"]),
              let L = matrix(flat, rows: n, cols: n),
              let b = rhs(inputs["b"]) else { return nil }
        guard let d = try? LinAlg.choSolveDiagnosed(L, b) else { return nil }
        guard let value = firstComponent(d.value) else { return nil }
        return StrategyResult(value: value, diagnostics: d.diagnostics)
    }

    // solveTriangular — triangular solve (dtrtrs); diagnostic on ill-conditioned A.
    registry.register(id: "solveTriangular") { inputs in
        guard let n = inputs["n"]?.intValue,
              let flat = doubleArray(inputs["A"]),
              let A = matrix(flat, rows: n, cols: n),
              let b = rhs(inputs["b"]) else { return nil }
        let lower: Bool
        if case let .bool(value)? = inputs["lower"] { lower = value } else { lower = true }
        guard let d = try? LinAlg.solveTriangularDiagnosed(A, b, lower: lower) else { return nil }
        guard let value = firstComponent(d.value) else { return nil }
        return StrategyResult(value: value, diagnostics: d.diagnostics)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the LinSolve domain.
///
/// All four solvers route to LAPACK and achieve near machine precision on
/// well-conditioned systems; the looser `hard`/`edge` bounds absorb the
/// round-off that grows with the (still in-envelope) condition number of the
/// random/Hilbert test matrices. Edge bounds apply only to the in-envelope edge
/// guards — out-of-envelope edge cases are not accuracy-scored (WORKBENCH.md §5).
@Sendable
public func makeLinSolveEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let tol: Double = tier == .trivial ? 1e-10 : tier == .hard ? 1e-7 : 1e-6
        for strategy in ["solve", "lstsq", "choSolve", "solveTriangular"] {
            reg.register(EnvelopeEntry(
                strategy: strategy, tier: tier, maxAbsError: tol,
                description: "LAPACK \(strategy) — \(tier.rawValue) cases (cond ≤ 1e12 envelope)"))
        }
    }
    return reg
}
