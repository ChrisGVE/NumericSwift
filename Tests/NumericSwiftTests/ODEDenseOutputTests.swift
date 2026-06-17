//
//  ODEDenseOutputTests.swift
//  NumericSwiftTests
//
//  Tests for dense-output (higher-order continuous interpolant) in solveIVP
//  and contiguous-state odeint.  Each test has an analytic oracle — no SciPy
//  dependency required.
//
//  Analytic oracles used:
//    y' = y,  y(0)=1          →  y(t) = e^t
//    y' = -y, y(0)=1          →  y(t) = e^(-t)
//    y'' + y = 0, y(0)=1,y'(0)=0 →  y(t) = cos(t)  (harmonic oscillator)
//
//  The dense-output tests place evaluation points *between* solver steps to
//  expose errors that only manifest when the interpolant — not just the solver
//  quadrature node — is evaluated.  Linear interpolation cannot satisfy the
//  tighter tolerances used here; higher-order continuous-extension interpolants
//  can and must.
//

import XCTest
@testable import NumericSwift

// swiftlint:disable type_body_length

final class ODEDenseOutputTests: XCTestCase {

  // MARK: - Helpers

  /// Maximum relative error across a result array compared to the analytic oracle.
  private func maxRelErr(
    _ result: [[Double]],
    component: Int = 0,
    oracle: (Double) -> Double,
    times: [Double]
  ) -> Double {
    zip(times, result).map { (t, y) -> Double in
      let exact = oracle(t)
      return abs(exact) > 1e-12 ? abs(y[component] - exact) / abs(exact) : abs(y[component] - exact)
    }.max() ?? 0
  }

  // MARK: - RK45 dense output: y' = y (exponential growth)

  /// Uses a fine tEval grid (50 pts) over [0,2] so that many evaluation
  /// points fall between solver steps.  Linear interpolation error on e^t with
  /// step ~0.4 is O(h²)~0.08; the RK45 quartic dense-output interpolant gives
  /// O(h^5) error far below 1e-4.
  func testRK45DenseOutput_ExponentialGrowth() {
    let tEval = stride(from: 0.0, through: 2.0, by: 0.04).map { $0 }  // 51 pts
    let result = solveIVP(
      { y, _ in [y[0]] },
      tSpan: (0, 2),
      y0: [1.0],
      method: .rk45,
      tEval: tEval,
      rtol: 1e-4,
      atol: 1e-7
    )

    XCTAssertTrue(result.success, "Solver failed: \(result.message)")
    XCTAssertEqual(result.t.count, tEval.count, "Output count mismatch")

    let err = maxRelErr(result.y, oracle: { Darwin.exp($0) }, times: tEval)
    // Linear interpolation would give ~5e-3 relative error; quartic dense
    // output must achieve < 5e-5.
    XCTAssertLessThan(err, 5e-5, "RK45 dense output relative error \(err) too large (linear interp would be ~5e-3)")
  }

  /// Harmonic oscillator (cos) with a fine tEval grid.
  func testRK45DenseOutput_HarmonicOscillator() {
    let tEval = stride(from: 0.0, through: Double.pi, by: Double.pi / 100).map { $0 }
    let result = solveIVP(
      { y, _ in [y[1], -y[0]] },       // y'' + y = 0
      tSpan: (0, Double.pi),
      y0: [1.0, 0.0],                   // y(0)=cos(0)=1, y'(0)=-sin(0)=0
      method: .rk45,
      tEval: tEval,
      rtol: 1e-5,
      atol: 1e-8
    )

    XCTAssertTrue(result.success, "Solver failed: \(result.message)")
    XCTAssertEqual(result.t.count, tEval.count, "Output count mismatch")

    let err = maxRelErr(result.y, component: 0, oracle: { Darwin.cos($0) }, times: tEval)
    // Require < 1e-4 relative error (linear interp would be ~1e-2).
    XCTAssertLessThan(err, 1e-4, "RK45 dense output (oscillator) relative error \(err) too large")
  }

  // MARK: - RK23 dense output: y' = -y (exponential decay)

  /// Cubic Hermite dense-output for RK23: fine grid over [0,1].
  /// RK23 is lower order (3rd), but its cubic Hermite dense output gives
  /// O(h^4) error between steps — far better than linear O(h^2).
  /// We use tight tolerances so that step sizes are small enough to let
  /// the dense-output accuracy advantage over linear interpolation show.
  func testRK23DenseOutput_ExponentialDecay() {
    let tEval = stride(from: 0.0, through: 1.0, by: 0.02).map { $0 }  // 51 pts
    let result = solveIVP(
      { y, _ in [-y[0]] },
      tSpan: (0, 1),
      y0: [1.0],
      method: .rk23,
      tEval: tEval,
      rtol: 1e-6,
      atol: 1e-9
    )

    XCTAssertTrue(result.success, "Solver failed: \(result.message)")
    XCTAssertEqual(result.t.count, tEval.count, "Output count mismatch")

    let err = maxRelErr(result.y, oracle: { Darwin.exp(-$0) }, times: tEval)
    // With rtol=1e-6, solver step h ≈ 0.05-0.1.  Linear interp error ~ h²/8 ≈ 3e-4;
    // cubic Hermite gives O(h^4) ≈ 1e-7.  Require < 1e-4 to prove dense output is used.
    XCTAssertLessThan(err, 1e-4, "RK23 dense output relative error \(err) too large (linear interp would be ~3e-4)")
  }

  // MARK: - Continuity at solver step boundaries

  /// Dense-output at solver step boundaries must match the solver's own
  /// quadrature output — not a separately-seeded run.  We achieve this by
  /// running once without tEval to get the step-node values, then running
  /// again with those same times as tEval and comparing both outputs against
  /// the analytic solution.  The two sets must agree with each other to 1e-9.
  func testRK45DenseOutput_ContinuousAtStepBoundary() {
    // First run: capture internal solver step times and values.
    let baseResult = solveIVP(
      { y, _ in [y[0]] },
      tSpan: (0, 1),
      y0: [1.0],
      method: .rk45,
      rtol: 1e-6,
      atol: 1e-9
    )
    let stepTimes = baseResult.t

    // Second run: use dense-output at the same time points.
    let denseResult = solveIVP(
      { y, _ in [y[0]] },
      tSpan: (0, 1),
      y0: [1.0],
      method: .rk45,
      tEval: stepTimes,
      rtol: 1e-6,
      atol: 1e-9
    )

    XCTAssertEqual(denseResult.t.count, stepTimes.count, "Boundary count mismatch")

    // Both outputs must be accurate against the analytic e^t and match each other.
    for (i, t) in stepTimes.enumerated() {
      let exact = Darwin.exp(t)

      // Each run must be close to the analytic solution.
      XCTAssertEqual(baseResult.y[i][0], exact, accuracy: 1e-5,
        "Base result at t=\(t) outside analytic tolerance")
      XCTAssertEqual(denseResult.y[i][0], exact, accuracy: 1e-5,
        "Dense result at t=\(t) outside analytic tolerance")

      // The two runs must agree with each other (both use the same tolerances
      // so their step-endpoint values should be numerically identical for the
      // same starting conditions and tolerances).
      XCTAssertEqual(
        denseResult.y[i][0], baseResult.y[i][0],
        accuracy: 1e-9,
        "Dense output at step boundary t=\(t) disagrees with base solver output by more than 1e-9"
      )
    }
  }

  // MARK: - odeint state reuse: continuity across intervals

  /// odeint must give the same result on [0,2] whether we ask for 5 or 20
  /// evenly spaced output points.  If odeint restarts the solver per interval,
  /// the accumulated initial-step-estimation overhead introduces tiny but
  /// detectable differences vs a single continuous integration.
  ///
  /// More importantly: odeint must match the analytic solution to within the
  /// solver tolerance — which validates that state is correctly threaded across
  /// interval boundaries.
  func testOdeint_StateReuse_ExponentialDecay() {
    let tFine = stride(from: 0.0, through: 2.0, by: 0.1).map { $0 }   // 21 pts
    let result = odeint(
      { y, _ in [-y[0]] },
      y0: [1.0],
      t: tFine,
      rtol: 1.49e-8,
      atol: 1.49e-8
    )

    XCTAssertEqual(result.count, tFine.count, "odeint output count mismatch")

    // Each output must match e^(-t) to 1e-6 relative error.
    for (i, tVal) in tFine.enumerated() {
      let exact = Darwin.exp(-tVal)
      let got = result[i][0]
      let relErr = abs(exact) > 1e-12 ? abs(got - exact) / abs(exact) : abs(got - exact)
      XCTAssertLessThan(
        relErr, 1e-6,
        "odeint[t=\(tVal)]: got \(got), exact \(exact), relErr \(relErr)"
      )
    }
  }

  /// odeint result must be monotonically decreasing for y'=-y (it always is
  /// analytically) — this catches pathological state-threading bugs where a
  /// restart injects a jump discontinuity in the solution array.
  func testOdeint_StateReuse_MonotonicDecay() {
    let t = stride(from: 0.0, through: 3.0, by: 0.05).map { $0 }
    let result = odeint(
      { y, _ in [-y[0]] },
      y0: [1.0],
      t: t,
      rtol: 1e-8,
      atol: 1e-10
    )

    for i in 1..<result.count {
      XCTAssertLessThan(
        result[i][0], result[i - 1][0],
        "odeint not monotonically decreasing at index \(i): \(result[i-1][0]) → \(result[i][0])"
      )
    }
  }

  /// odeint on the harmonic oscillator must match the analytic solution to
  /// within tight tolerance across many time points.
  func testOdeint_StateReuse_HarmonicOscillator() {
    let t = stride(from: 0.0, through: 2 * Double.pi, by: Double.pi / 20).map { $0 }  // 41 pts
    let result = odeint(
      { y, _ in [y[1], -y[0]] },    // y'' + y = 0
      y0: [1.0, 0.0],               // y(0)=1, y'(0)=0  →  y(t)=cos(t)
      t: t,
      rtol: 1e-8,
      atol: 1e-10
    )

    XCTAssertEqual(result.count, t.count, "odeint output count mismatch")

    for (i, tVal) in t.enumerated() {
      let exact = Darwin.cos(tVal)
      let got = result[i][0]
      XCTAssertEqual(
        got, exact,
        accuracy: 1e-5,
        "odeint harmonic oscillator at t=\(tVal): got \(got), exact \(exact)"
      )
    }
  }

  // MARK: - tEval ordering: all output points must respect tEval order

  /// When tEval is provided, result.t must equal tEval exactly (not sorted,
  /// not filtered — just the requested points in requested order).
  func testRK45_tEvalOrderPreserved() {
    let tEval = [0.0, 0.5, 1.0, 1.5, 2.0]
    let result = solveIVP(
      { y, _ in [y[0]] },
      tSpan: (0, 2),
      y0: [1.0],
      method: .rk45,
      tEval: tEval,
      rtol: 1e-6,
      atol: 1e-9
    )

    XCTAssertEqual(result.t, tEval, "result.t does not equal tEval")
    for (i, tVal) in tEval.enumerated() {
      XCTAssertEqual(result.y[i][0], Darwin.exp(tVal), accuracy: 1e-4,
        "y at t=\(tVal) outside tolerance")
    }
  }

  // MARK: - RK45 dense output: interpolation accuracy degrades gracefully near endpoints

  /// At t=t0 the dense output must return y0 exactly (s=0 edge).
  func testRK45DenseOutput_AtT0_ReturnsY0() {
    let result = solveIVP(
      { y, _ in [y[0]] },
      tSpan: (0, 1),
      y0: [2.718281828],  // e ≈ e^1 boundary condition
      method: .rk45,
      tEval: [0.0],
      rtol: 1e-8,
      atol: 1e-10
    )

    XCTAssertEqual(result.t.count, 1)
    XCTAssertEqual(result.y[0][0], 2.718281828, accuracy: 1e-10,
      "Dense output at t=t0 must return y0 exactly")
  }

  /// At t=tf the dense output must return the final solution within solver tolerance.
  func testRK45DenseOutput_AtTf_ReturnsFinalY() {
    let result = solveIVP(
      { y, _ in [y[0]] },
      tSpan: (0, 1),
      y0: [1.0],
      method: .rk45,
      tEval: [1.0],
      rtol: 1e-8,
      atol: 1e-10
    )

    XCTAssertEqual(result.t.count, 1)
    XCTAssertEqual(result.y[0][0], Darwin.exp(1.0), accuracy: 1e-6,
      "Dense output at t=tf must return final y")
  }

  // MARK: - RK45 dense output: multi-component system

  /// Two-component system: y[0]'= y[1], y[1]'= -y[0].
  /// Both components must be interpolated correctly.
  func testRK45DenseOutput_MultiComponent() {
    let tEval = stride(from: 0.0, through: Double.pi, by: Double.pi / 50).map { $0 }
    let result = solveIVP(
      { y, _ in [y[1], -y[0]] },
      tSpan: (0, Double.pi),
      y0: [1.0, 0.0],
      method: .rk45,
      tEval: tEval,
      rtol: 1e-6,
      atol: 1e-9
    )

    XCTAssertEqual(result.t.count, tEval.count)

    for (i, tVal) in tEval.enumerated() {
      // y[0] = cos(t), y[1] = -sin(t)
      XCTAssertEqual(result.y[i][0], Darwin.cos(tVal), accuracy: 1e-4,
        "y[0] at t=\(tVal)")
      XCTAssertEqual(result.y[i][1], -Darwin.sin(tVal), accuracy: 1e-4,
        "y[1] at t=\(tVal)")
    }
  }
}
