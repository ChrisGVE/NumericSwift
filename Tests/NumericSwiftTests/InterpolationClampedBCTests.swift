// InterpolationClampedBCTests.swift
// Tests/NumericSwiftTests/
//
// Tests for the clamped cubic-spline boundary condition with user-supplied
// endpoint first-derivative values (issue #9 / M19).
//
// Oracle: scipy.interpolate.CubicSpline(bc_type=((1, d0), (1, d1)))
// Python venv: /tmp/.nsoracle/bin/python
//
// All expected values are frozen literals obtained from SciPy 1.x and must
// NOT be regenerated from this library's evaluator.

import XCTest

@testable import NumericSwift

final class InterpolationClampedBCTests: XCTestCase {

  // MARK: - Default-clamped regression (f'=0 at both ends)
  //
  // Existing callers write `.clamped`; the static-var shim must continue to
  // deliver a spline whose endpoint slopes are exactly zero.  These values
  // serve as the backwards-compatibility regression guard.

  func testDefaultClampedEquivalentToZeroSlopes() {
    // Oracle: scipy CubicSpline([0,1,2,3], [0,1,0,1], bc_type=((1,0),(1,0)))
    let x = [0.0, 1.0, 2.0, 3.0]
    let y = [0.0, 1.0, 0.0, 1.0]

    let coeffs = computeSplineCoeffs(x: x, y: y, bc: .clamped)

    // Interior values (SciPy frozen literals)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.5), 0.5, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.0), 1.0, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.5), 0.5, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 2.0), 0.0, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 2.5), 0.5, accuracy: 1e-12)

    // Endpoint first-derivatives must be zero (the defining property)
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 0.0), 0.0, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 3.0), 0.0, accuracy: 1e-12)
  }

  // MARK: - Explicit zero slopes (same as default; must be identical)

  func testExplicitZeroSlopesMatchDefault() {
    let x = [0.0, 1.0, 2.0, 3.0]
    let y = [0.0, 1.0, 0.0, 1.0]

    let defaultCoeffs = computeSplineCoeffs(x: x, y: y, bc: .clamped)
    let explicitCoeffs = computeSplineCoeffs(x: x, y: y, bc: .clamped(dStart: 0.0, dEnd: 0.0))

    // Both APIs must produce identical coefficient arrays
    XCTAssertEqual(defaultCoeffs.count, explicitCoeffs.count)
    for i in 0..<defaultCoeffs.count {
      XCTAssertEqual(defaultCoeffs[i].a, explicitCoeffs[i].a, accuracy: 1e-15,
                     "a[\(i)] mismatch")
      XCTAssertEqual(defaultCoeffs[i].b, explicitCoeffs[i].b, accuracy: 1e-15,
                     "b[\(i)] mismatch")
      XCTAssertEqual(defaultCoeffs[i].c, explicitCoeffs[i].c, accuracy: 1e-15,
                     "c[\(i)] mismatch")
      XCTAssertEqual(defaultCoeffs[i].d, explicitCoeffs[i].d, accuracy: 1e-15,
                     "d[\(i)] mismatch")
    }
  }

  // MARK: - Nonzero endpoint slopes (d0=2.0, d1=-1.0)
  //
  // Oracle: scipy CubicSpline([0,1,2,3], [0,1,0,1], bc_type=((1,2.0),(1,-1.0)))

  func testNonzeroEndpointSlopes() {
    let x = [0.0, 1.0, 2.0, 3.0]
    let y = [0.0, 1.0, 0.0, 1.0]
    let coeffs = computeSplineCoeffs(x: x, y: y, bc: .clamped(dStart: 2.0, dEnd: -1.0))

    // Interior values (SciPy frozen literals)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.5), 0.825,  accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.0), 1.0,    accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.5), 0.375,  accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 2.0), 0.0,    accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 2.5), 0.675,  accuracy: 1e-12)

    // Endpoint first-derivatives must match the prescribed values
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 0.0),  2.0, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 3.0), -1.0, accuracy: 1e-12)
  }

  // MARK: - Steep ends on non-uniform grid (d0=5.0, d1=-3.0)
  //
  // Oracle: scipy CubicSpline([0,1,3,6], [0,2,1,4], bc_type=((1,5.0),(1,-3.0)))

  func testSteepEndsNonUniformGrid() {
    let x = [0.0, 1.0, 3.0, 6.0]
    let y = [0.0, 2.0, 1.0, 4.0]
    let coeffs = computeSplineCoeffs(x: x, y: y, bc: .clamped(dStart: 5.0, dEnd: -3.0))

    // Interior values (SciPy frozen literals)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.0), 0.0,               accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.5), 1.63048245614035,  accuracy: 1e-10)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.0), 2.0,               accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 2.0), 1.29824561403509,  accuracy: 1e-10)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 3.0), 1.0,               accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 4.5), 3.91118421052632,  accuracy: 1e-10)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 6.0), 4.0,               accuracy: 1e-12)

    // Endpoint derivatives
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 0.0),  5.0, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 6.0), -3.0, accuracy: 1e-12)
  }

  // MARK: - Near-flat ends (d0=0.1, d1=0.1)
  //
  // Oracle: scipy CubicSpline([0,1,2,3], [1,1.5,1.2,1.8], bc_type=((1,0.1),(1,0.1)))

  func testNearFlatEnds() {
    let x = [0.0, 1.0, 2.0, 3.0]
    let y = [1.0, 1.5, 1.2, 1.8]
    let coeffs = computeSplineCoeffs(x: x, y: y, bc: .clamped(dStart: 0.1, dEnd: 0.1))

    // Interior values (SciPy frozen literals)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.5), 1.2525, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.5), 1.3375, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 2.5), 1.51,   accuracy: 1e-12)

    // Endpoint derivatives
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 0.0), 0.1, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 3.0), 0.1, accuracy: 1e-12)
  }

  // MARK: - Two-point edge case (d0=1.0, d1=2.0)
  //
  // Oracle: scipy CubicSpline([0,1], [0,1], bc_type=((1,1.0),(1,2.0)))
  // With n=2 the "clamped" BC is fully determined by a single cubic segment
  // whose Hermite conditions impose both endpoint derivatives explicitly.

  func testTwoPointClampedWithDerivatives() {
    let x = [0.0, 1.0]
    let y = [0.0, 1.0]
    let coeffs = computeSplineCoeffs(x: x, y: y, bc: .clamped(dStart: 1.0, dEnd: 2.0))

    XCTAssertEqual(coeffs.count, 1)

    // Interior values (SciPy frozen literals)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.00), 0.0,       accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.25), 0.203125,  accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.50), 0.375,     accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0.75), 0.609375,  accuracy: 1e-12)
    XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.00), 1.0,       accuracy: 1e-12)

    // Endpoint derivatives
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 0.0), 1.0, accuracy: 1e-12)
    XCTAssertEqual(evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 1.0), 2.0, accuracy: 1e-12)
  }

  // MARK: - Spline interpolates through all data points regardless of BC

  func testClampedSplinePassesThroughDataPoints() {
    let x = [0.0, 1.0, 2.0, 3.0]
    let y = [0.0, 1.0, 0.0, 1.0]
    let coeffs = computeSplineCoeffs(x: x, y: y, bc: .clamped(dStart: 2.0, dEnd: -1.0))

    for (xi, yi) in zip(x, y) {
      XCTAssertEqual(
        evalCubicSpline(x: x, coeffs: coeffs, xNew: xi), yi, accuracy: 1e-12,
        "Spline must interpolate data point at x=\(xi)")
    }
  }

  // MARK: - RawRepresentable round-trip (restores 0.2.x public surface)
  //
  // Original raw strings (recovered from git history, commit e0fe3aa):
  //   .natural   → "natural"
  //   .clamped   → "clamped"
  //   .notAKnot  → "not-a-knot"
  //
  // These are FROZEN: changing them is a breaking change to the published API.

  func testNaturalRawValue() {
    XCTAssertEqual(SplineBoundaryCondition.natural.rawValue, "natural")
  }

  func testNaturalRoundTrip() {
    let bc = SplineBoundaryCondition(rawValue: "natural")
    XCTAssertNotNil(bc, "init?(rawValue:\"natural\") must succeed")
    if case .natural = bc! {
      // correct
    } else {
      XCTFail("Expected .natural, got \(bc!)")
    }
  }

  func testNotAKnotRawValue() {
    XCTAssertEqual(SplineBoundaryCondition.notAKnot.rawValue, "not-a-knot")
  }

  func testNotAKnotRoundTrip() {
    let bc = SplineBoundaryCondition(rawValue: "not-a-knot")
    XCTAssertNotNil(bc, "init?(rawValue:\"not-a-knot\") must succeed")
    if case .notAKnot = bc! {
      // correct
    } else {
      XCTFail("Expected .notAKnot, got \(bc!)")
    }
  }

  func testClampedRawValue() {
    // The static-var `.clamped` shim (zero derivatives) must map to "clamped".
    XCTAssertEqual(SplineBoundaryCondition.clamped.rawValue, "clamped")
    // The associated-value form must also map to "clamped" (rawValue is
    // derivative-independent: it names the BC kind, not the derivative values).
    XCTAssertEqual(SplineBoundaryCondition.clamped(dStart: 1.0, dEnd: 2.0).rawValue, "clamped")
  }

  func testClampedRoundTrip() {
    // init?(rawValue: "clamped") must return .clamped(dStart: 0, dEnd: 0),
    // preserving the historic zero-slope default.
    let bc = SplineBoundaryCondition(rawValue: "clamped")
    XCTAssertNotNil(bc, "init?(rawValue:\"clamped\") must succeed")
    if case .clamped(let ds, let de) = bc! {
      XCTAssertEqual(ds, 0.0, "init?(rawValue:\"clamped\").dStart must be 0 (historic default)")
      XCTAssertEqual(de, 0.0, "init?(rawValue:\"clamped\").dEnd  must be 0 (historic default)")
    } else {
      XCTFail("Expected .clamped(dStart:dEnd:), got \(bc!)")
    }
  }

  func testUnknownRawValueReturnsNil() {
    XCTAssertNil(SplineBoundaryCondition(rawValue: "bogus"))
    XCTAssertNil(SplineBoundaryCondition(rawValue: ""))
    XCTAssertNil(SplineBoundaryCondition(rawValue: "Clamped"))  // case-sensitive
  }
}
