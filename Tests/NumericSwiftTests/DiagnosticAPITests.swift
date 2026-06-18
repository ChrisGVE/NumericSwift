//
//  DiagnosticAPITests.swift
//  Tests/NumericSwiftTests/
//
//  Dedicated unit tests for the wave-1 self-awareness / diagnostics public API
//  introduced in the hardening sweep (0.3.0):
//
//    • `NumericDiagnostic` enum + `Diagnosed<Value>` wrapper (NumericDiagnostic.swift)
//    • `brentq` bracketing root finder (Optimization.swift)
//    • the `*Diagnosed` overloads across Optimization / SpecialFunctions /
//      Distributions / Interpolation / Series / Spatial / LinAlg+Solvers
//    • the `diagnostics` fields carried on result structs.
//
//  These exercise the diagnostic-EMISSION contract directly, independent of the
//  workbench fixture corpus (which validates the same contract end-to-end via
//  `WorkbenchGateTests`). Reference roots/values are analytic or come from the
//  bare (non-diagnosed) overload, which is the documented invariant: a
//  `*Diagnosed` overload returns the SAME best-effort value as its bare twin and
//  only adds the diagnostic.
//
import XCTest

@testable import NumericSwift

final class DiagnosticAPITests: XCTestCase {

    // MARK: - NumericDiagnostic enum

    func testNumericDiagnostic_isOutsideEnvelope() {
        XCTAssertTrue(NumericDiagnostic.outsideEnvelope(method: "m", reason: "r").isOutsideEnvelope)
        XCTAssertFalse(NumericDiagnostic.precisionDegraded(method: "m", approxDigits: 5).isOutsideEnvelope)
        XCTAssertFalse(NumericDiagnostic.nonConvergence(method: "m", reason: "r").isOutsideEnvelope)
    }

    func testNumericDiagnostic_description() {
        XCTAssertEqual(
            NumericDiagnostic.outsideEnvelope(method: "tDist.ppf", reason: "tail").description,
            "[outsideEnvelope] tDist.ppf: tail")
        XCTAssertEqual(
            NumericDiagnostic.precisionDegraded(method: "erfinv", approxDigits: 8).description,
            "[precisionDegraded] erfinv: ~8 significant digit(s)")
        XCTAssertEqual(
            NumericDiagnostic.nonConvergence(method: "bisect", reason: "maxiter").description,
            "[nonConvergence] bisect: maxiter")
    }

    func testNumericDiagnostic_equatable() {
        XCTAssertEqual(
            NumericDiagnostic.outsideEnvelope(method: "m", reason: "r"),
            NumericDiagnostic.outsideEnvelope(method: "m", reason: "r"))
        XCTAssertNotEqual(
            NumericDiagnostic.outsideEnvelope(method: "m", reason: "r"),
            NumericDiagnostic.outsideEnvelope(method: "m", reason: "other"))
        XCTAssertNotEqual(
            NumericDiagnostic.precisionDegraded(method: "m", approxDigits: 5),
            NumericDiagnostic.precisionDegraded(method: "m", approxDigits: 6))
    }

    // MARK: - Diagnosed<Value>

    func testDiagnosed_defaultEmptyIsReliable() {
        let d = Diagnosed(42.0)
        XCTAssertEqual(d.value, 42.0)
        XCTAssertTrue(d.diagnostics.isEmpty)
        XCTAssertTrue(d.isReliable)
    }

    func testDiagnosed_outsideEnvelopeMakesUnreliable() {
        let d = Diagnosed(1.0, diagnostics: [.outsideEnvelope(method: "m", reason: "r")])
        XCTAssertFalse(d.isReliable)
    }

    func testDiagnosed_nonOutsideDiagnosticsStayReliable() {
        // precisionDegraded / nonConvergence are warnings, NOT reliability failures.
        let d = Diagnosed(
            1.0,
            diagnostics: [
                .precisionDegraded(method: "m", approxDigits: 5),
                .nonConvergence(method: "m", reason: "r"),
            ])
        XCTAssertTrue(d.isReliable)
        XCTAssertEqual(d.diagnostics.count, 2)
    }

    func testDiagnosed_mapCarriesDiagnosticsThrough() {
        let degrees = Diagnosed(180.0, diagnostics: [.precisionDegraded(method: "m", approxDigits: 5)])
        let radians = degrees.map { $0 * .pi / 180 }
        XCTAssertEqual(radians.value, .pi, accuracy: 1e-15)
        XCTAssertEqual(radians.diagnostics, degrees.diagnostics)
    }

    func testDiagnosed_mapChangesType() {
        let d = Diagnosed(3.0).map { "value=\($0)" }
        XCTAssertEqual(d.value, "value=3.0")
        XCTAssertTrue(d.isReliable)
    }

    // MARK: - brentq (Brent's method root finder)

    func testBrentq_findsRootOfCos() {
        // cos(x) = 0 has a root at π/2 in [0, 2].
        let r = brentq({ Darwin.cos($0) }, a: 0, b: 2)
        XCTAssertTrue(r.converged)
        XCTAssertEqual(r.root, .pi / 2, accuracy: 1e-7)  // brentq honours xtol (1e-8)
        XCTAssertTrue(r.diagnostics.isEmpty)
    }

    func testBrentq_findsSqrt2() {
        // x² - 2 = 0 → √2 in [0, 2].
        let r = brentq({ $0 * $0 - 2 }, a: 0, b: 2)
        XCTAssertTrue(r.converged)
        XCTAssertEqual(r.root, 2.0.squareRoot(), accuracy: 1e-7)  // brentq honours xtol (1e-8)
        XCTAssertTrue(r.diagnostics.isEmpty)
    }

    func testBrentq_exactEndpointRoot() {
        // f(a) == 0 → return a immediately, no diagnostics, zero iterations.
        let r = brentq({ $0 }, a: 0, b: 1)
        XCTAssertTrue(r.converged)
        XCTAssertEqual(r.root, 0)
        XCTAssertEqual(r.iterations, 0)
        XCTAssertTrue(r.diagnostics.isEmpty)
    }

    func testBrentq_invalidBracketEmitsDiagnostic() {
        // f(a)·f(b) > 0 — no sign change. SciPy raises ValueError; we flag + NaN.
        let r = brentq({ $0 * $0 + 1 }, a: 1, b: 2)
        XCTAssertFalse(r.converged)
        XCTAssertTrue(r.root.isNaN)
        XCTAssertEqual(r.diagnostics.count, 1)
        XCTAssertTrue(r.diagnostics.first?.isOutsideEnvelope ?? false)
    }

    func testBrentq_maxiterExhaustionEmitsDiagnostic() {
        // A starved iteration budget cannot converge → outsideEnvelope diagnostic.
        let r = brentq({ Darwin.cos($0) }, a: 0, b: 2, maxiter: 1)
        XCTAssertFalse(r.converged)
        XCTAssertEqual(r.flag, "maxiter reached")
        XCTAssertTrue(r.diagnostics.contains { $0.isOutsideEnvelope })
    }

    // MARK: - erfinvDiagnosed

    func testErfinvDiagnosed_inEnvelopeIsClean() {
        let d = erfinvDiagnosed(0.5)
        XCTAssertEqual(d.value, erfinv(0.5))  // same best-effort value as bare
        XCTAssertTrue(d.diagnostics.isEmpty)
        XCTAssertTrue(d.isReliable)
    }

    func testErfinvDiagnosed_extremeTailIsFlagged() {
        let x = 1.0 - 1e-13  // inside (−1,1), beyond the 1e-11 tail boundary
        let d = erfinvDiagnosed(x)
        XCTAssertEqual(d.value, erfinv(x))
        XCTAssertFalse(d.isReliable)
        XCTAssertTrue(d.diagnostics.first?.isOutsideEnvelope ?? false)
    }

    func testErfinvDiagnosed_outOfDomainNotFlagged() {
        // ±1 / out-of-domain already carry their own ±inf/NaN signal — not flagged.
        XCTAssertTrue(erfinvDiagnosed(1.0).diagnostics.isEmpty)
        XCTAssertTrue(erfinvDiagnosed(2.0).diagnostics.isEmpty)
    }

    // MARK: - TDistribution.ppfDiagnosed

    func testTDistPpfDiagnosed_centralIsClean() {
        let t = TDistribution(df: 10)
        let d = t.ppfDiagnosed(0.5)
        XCTAssertEqual(d.value, t.ppf(0.5), accuracy: 1e-12)
        XCTAssertTrue(d.diagnostics.isEmpty)
    }

    func testTDistPpfDiagnosed_extremeTailIsFlagged() {
        let t = TDistribution(df: 10)
        let d = t.ppfDiagnosed(1e-5)  // |p| beyond 0.9999
        XCTAssertEqual(d.value, t.ppf(1e-5))
        XCTAssertFalse(d.isReliable)
        XCTAssertTrue(d.diagnostics.first?.isOutsideEnvelope ?? false)
    }

    // MARK: - LinAlg solveDiagnosed / lstsqDiagnosed

    /// `n × n` Hilbert matrix: `H[i][j] = 1/(i+j+1)`. Famously ill-conditioned —
    /// `cond` grows ~e^(3.5n), so a 12×12 Hilbert is ~10¹⁶, far past the 1e12 envelope.
    private func hilbert(_ n: Int) -> LinAlg.Matrix {
        LinAlg.Matrix((0..<n).map { i in (0..<n).map { j in 1.0 / Double(i + j + 1) } })
    }

    func testSolveDiagnosed_wellConditionedIsClean() throws {
        let A = LinAlg.Matrix([[2.0, 0.0], [0.0, 3.0]])
        let b = LinAlg.Matrix([1.0, 1.0])
        let d = try LinAlg.solveDiagnosed(A, b)
        XCTAssertNotNil(d.value)
        XCTAssertTrue(d.diagnostics.isEmpty)
        XCTAssertTrue(d.isReliable)
    }

    func testSolveDiagnosed_illConditionedIsFlagged() throws {
        let A = hilbert(12)
        let b = LinAlg.Matrix((0..<12).map { _ in 1.0 })
        let d = try LinAlg.solveDiagnosed(A, b)
        XCTAssertFalse(d.isReliable)
        XCTAssertTrue(d.diagnostics.first?.isOutsideEnvelope ?? false)
    }

    func testLstsqDiagnosed_illConditionedIsFlagged() throws {
        let A = hilbert(12)
        let b = LinAlg.Matrix((0..<12).map { _ in 1.0 })
        let d = try LinAlg.lstsqDiagnosed(A, b)
        XCTAssertFalse(d.isReliable)
    }

    // MARK: - Interpolation *Diagnosed overloads

    func testEvalCubicSplineDiagnosed_inRangeClean_outOfRangeFlagged() {
        let x = [0.0, 1.0, 2.0, 3.0]
        let y = [0.0, 1.0, 4.0, 9.0]
        let coeffs = computeSplineCoeffs(x: x, y: y)

        let inside = evalCubicSplineDiagnosed(x: x, coeffs: coeffs, xNew: 1.5)
        XCTAssertTrue(inside.diagnostics.isEmpty)
        XCTAssertEqual(inside.value, evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.5, extrapolate: true))

        let outside = evalCubicSplineDiagnosed(x: x, coeffs: coeffs, xNew: 5.0)
        XCTAssertFalse(outside.isReliable)
        XCTAssertTrue(outside.diagnostics.first?.isOutsideEnvelope ?? false)
    }

    func testEvalPchipDiagnosed_outOfRangeFlagged() {
        let x = [0.0, 1.0, 2.0]
        let y = [0.0, 1.0, 8.0]
        let d = computePchipDerivatives(x: x, y: y)
        XCTAssertTrue(evalPchipDiagnosed(x: x, y: y, d: d, xNew: 1.0).diagnostics.isEmpty)
        XCTAssertFalse(evalPchipDiagnosed(x: x, y: y, d: d, xNew: -1.0).isReliable)
    }

    func testEvalAkimaDiagnosed_outOfRangeFlagged() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0]
        let y = [0.0, 1.0, 0.0, 1.0, 0.0]
        let coeffs = computeAkimaCoeffs(x: x, y: y)
        XCTAssertTrue(evalAkimaDiagnosed(x: x, coeffs: coeffs, xNew: 2.0).diagnostics.isEmpty)
        XCTAssertFalse(evalAkimaDiagnosed(x: x, coeffs: coeffs, xNew: 10.0).isReliable)
    }

    func testEvalBarycentricDiagnosed_outOfRangeFlagged() {
        let x = [0.0, 1.0, 2.0]
        let y = [1.0, 2.0, 5.0]
        let w = computeBarycentricWeights(x: x)
        XCTAssertTrue(evalBarycentricDiagnosed(x: x, y: y, w: w, xNew: 1.0).diagnostics.isEmpty)
        XCTAssertFalse(evalBarycentricDiagnosed(x: x, y: y, w: w, xNew: 3.0).isReliable)
    }

    // MARK: - Series taylor*Diagnosed

    func testTaylorEvalDiagnosed_unboundedGeneratorIsClean() {
        // sin has a closed-form generator → no support limit → never flagged.
        let d = Series.taylorEvalDiagnosed("sin", at: 0.5, terms: 30)
        XCTAssertNotNil(d)
        XCTAssertTrue(d?.diagnostics.isEmpty ?? false)
    }

    func testTaylorCoefficientsDiagnosed_tanBeyondSupportIsFlagged() {
        // tan is the only bounded generator (support limit = 12 terms).
        let within = Series.taylorCoefficientsDiagnosed(for: "tan", terms: 12)
        XCTAssertTrue(within?.diagnostics.isEmpty ?? false)

        let beyond = Series.taylorCoefficientsDiagnosed(for: "tan", terms: 20)
        XCTAssertNotNil(beyond)
        XCTAssertFalse(beyond?.isReliable ?? true)
        XCTAssertTrue(beyond?.diagnostics.first?.isOutsideEnvelope ?? false)
    }

    func testTaylorEvalDiagnosed_unknownFunctionIsNil() {
        XCTAssertNil(Series.taylorEvalDiagnosed("not_a_function", at: 1.0, terms: 5))
        XCTAssertNil(Series.taylorCoefficientsDiagnosed(for: "not_a_function", terms: 5))
    }

    // MARK: - Spatial kNN *Diagnosed (degenerate-query + crash-safety)

    func testQueryDiagnosed_wellPosedIsClean() {
        let tree = KDTree([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        let d = tree.queryDiagnosed([0.1, 0.1], k: 2)
        XCTAssertTrue(d.diagnostics.isEmpty)
        XCTAssertEqual(d.value.indices.count, 2)
    }

    func testQueryDiagnosed_kGreaterThanNIsFlaggedNotCrash() {
        // Regression guard: the bare query force-unwraps an empty best-list when
        // k > n. queryDiagnosed must detect the degeneracy and NOT crash.
        let tree = KDTree([[0.0, 0.0], [1.0, 1.0]])
        let d = tree.queryDiagnosed([0.0, 0.0], k: 5)
        XCTAssertFalse(d.isReliable)
        XCTAssertTrue(d.diagnostics.first?.isOutsideEnvelope ?? false)
        XCTAssertLessThanOrEqual(d.value.indices.count, 2)  // only the neighbours that exist
    }

    func testQueryDiagnosed_nonPositiveKIsFlagged() {
        let tree = KDTree([[0.0, 0.0], [1.0, 1.0]])
        let d = tree.queryDiagnosed([0.0, 0.0], k: 0)
        XCTAssertFalse(d.isReliable)
        XCTAssertTrue(d.value.indices.isEmpty)
    }

    func testBruteForceKNNDiagnosed_matchesAndFlagsDegenerate() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        let clean = Spatial.bruteForceKNNDiagnosed(points: points, query: [0.1, 0.1], k: 2)
        XCTAssertTrue(clean.diagnostics.isEmpty)
        XCTAssertEqual(clean.value.indices.first, 0)  // nearest is the origin point

        let degenerate = Spatial.bruteForceKNNDiagnosed(points: points, query: [0.0, 0.0], k: 10)
        XCTAssertFalse(degenerate.isReliable)
    }

    func testBruteForceKNNDiagnosed_emptySetIsFlagged() {
        let d = Spatial.bruteForceKNNDiagnosed(points: [], query: [0.0], k: 1)
        XCTAssertFalse(d.isReliable)
        XCTAssertTrue(d.value.indices.isEmpty)
    }

    // MARK: - diagnostics field on result structs (default-empty contract)

    func testQuadResult_cleanIntegralHasEmptyDiagnostics() {
        // ∫₀¹ x² dx = 1/3, well inside quad's envelope → no diagnostics.
        let r = quad({ $0 * $0 }, 0, 1)
        XCTAssertEqual(r.value, 1.0 / 3.0, accuracy: 1e-10)
        XCTAssertTrue(r.diagnostics.isEmpty)
    }

    func testRootScalarResult_diagnosticsDefaultsEmpty() {
        // The bisect bracketing finder on a clean sign-change → empty diagnostics.
        let r = brentq({ $0 - 0.5 }, a: 0, b: 1)
        XCTAssertTrue(r.converged)
        XCTAssertTrue(r.diagnostics.isEmpty)
    }
}
