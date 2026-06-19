//
//  InterpolationNDTests.swift
//  Tests/NumericSwiftTests/
//
//  XCTest coverage for InterpolationND.interpn / InterpolationND.swift.
//  Oracle values were generated with SciPy's interpn function (scipy 1.x):
//
//      from scipy.interpolate import interpn
//      interpn((x, y), values, [point], method='linear', bounds_error=True)
//
//  Run the oracle script at /tmp/oracle.py against the project's
//  /tmp/.nsoracle venv to regenerate if SciPy ever changes behaviour.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class InterpolationNDTests: XCTestCase {

  // MARK: - Shared grid fixtures

  // 2-D non-uniform grid: x=[0,1,3], y=[0,2,5], f(x,y)=x+2y
  // Values (row-major, x-axis outer): [[0,4,10],[1,5,11],[3,7,13]]
  let x2 = [0.0, 1.0, 3.0]
  let y2 = [0.0, 2.0, 5.0]
  let v2: [Double] = [
    0.0, 4.0, 10.0,   // x=0: 0+2*0, 0+2*2, 0+2*5
    1.0, 5.0, 11.0,   // x=1: 1+2*0, 1+2*2, 1+2*5
    3.0, 7.0, 13.0,   // x=3: 3+2*0, 3+2*2, 3+2*5
  ]

  // 3-D unit cube grid: f(x,y,z)=x+y+z
  // Values (row-major, x outer): [[[0,1],[1,2]],[[1,2],[2,3]]]
  let x3 = [0.0, 1.0]
  let y3 = [0.0, 1.0]
  let z3 = [0.0, 1.0]
  let v3: [Double] = [
    0.0, 1.0, 1.0, 2.0,   // x=0: (0,0,0)=0, (0,0,1)=1, (0,1,0)=1, (0,1,1)=2
    1.0, 2.0, 2.0, 3.0,   // x=1: (1,0,0)=1, (1,0,1)=2, (1,1,0)=2, (1,1,1)=3
  ]

  // 1-D grid: x=[0,1,2], f(x)=[0,1,4]  (non-uniform spacing in y)
  let x1 = [0.0, 1.0, 2.0]
  let v1 = [0.0, 1.0, 4.0]

  // MARK: - 2-D linear interpolation

  /// Querying exactly on grid nodes must recover the stored value with no error.
  func test2DLinearOnNode() throws {
    let onNodeCases: [(point: [Double], expected: Double)] = [
      ([0.0, 0.0], 0.0),
      ([1.0, 2.0], 5.0),
      ([1.0, 5.0], 11.0),
      ([3.0, 5.0], 13.0),
    ]
    for (point, expected) in onNodeCases {
      let result = try InterpolationND.interpn(
        points: [x2, y2], values: v2, xi: [point],
        method: .linear, boundsHandling: .error)
      XCTAssertEqual(
        result[0], expected, accuracy: 1e-12,
        "Linear on-node query at \(point) should return \(expected)")
    }
  }

  /// Interior query points (oracle: scipy interpn linear).
  func test2DLinearInterior() throws {
    // Oracle: (0.5, 1.0) -> 2.5,  (2.0, 3.5) -> 9.0
    let cases: [(point: [Double], expected: Double)] = [
      ([0.5, 1.0], 2.5),
      ([2.0, 3.5], 9.0),
    ]
    for (point, expected) in cases {
      let result = try InterpolationND.interpn(
        points: [x2, y2], values: v2, xi: [point],
        method: .linear, boundsHandling: .error)
      XCTAssertEqual(
        result[0], expected, accuracy: 1e-10,
        "2D linear at \(point) should be \(expected)")
    }
  }

  /// Multiple query points returned in the correct order.
  func test2DLinearBatchOrder() throws {
    let pts: [[Double]] = [[0.0, 0.0], [1.0, 2.0], [0.5, 1.0], [2.0, 3.5], [1.0, 5.0], [3.0, 5.0]]
    let expected = [0.0, 5.0, 2.5, 9.0, 11.0, 13.0]
    let results = try InterpolationND.interpn(
      points: [x2, y2], values: v2, xi: pts,
      method: .linear, boundsHandling: .error)
    XCTAssertEqual(results.count, expected.count)
    for (i, (got, want)) in zip(results, expected).enumerated() {
      XCTAssertEqual(got, want, accuracy: 1e-10, "Batch result[\(i)] mismatch")
    }
  }

  // MARK: - 2-D nearest interpolation

  /// Oracle: scipy interpn nearest for the 2-D fixture.
  func test2DNearest() throws {
    // (0.0,0.0)->0, (1.0,2.0)->5, (0.5,1.0)->0, (2.0,3.5)->5, (1.0,5.0)->11, (3.0,5.0)->13
    let cases: [(point: [Double], expected: Double)] = [
      ([0.0, 0.0], 0.0),
      ([1.0, 2.0], 5.0),
      ([0.5, 1.0], 0.0),
      ([2.0, 3.5], 5.0),
      ([1.0, 5.0], 11.0),
      ([3.0, 5.0], 13.0),
    ]
    for (point, expected) in cases {
      let result = try InterpolationND.interpn(
        points: [x2, y2], values: v2, xi: [point],
        method: .nearest, boundsHandling: .error)
      XCTAssertEqual(
        result[0], expected, accuracy: 1e-12,
        "2D nearest at \(point) should be \(expected)")
    }
  }

  // MARK: - 3-D linear interpolation

  /// Oracle: scipy interpn linear on the unit cube.
  func test3DLinear() throws {
    // (0.25,0.5,0.75)->1.5, (1.0,1.0,1.0)->3.0, (0.0,0.0,0.0)->0.0, (0.5,0.5,0.5)->1.5
    let cases: [(point: [Double], expected: Double)] = [
      ([0.25, 0.5, 0.75], 1.5),
      ([1.0, 1.0, 1.0], 3.0),
      ([0.0, 0.0, 0.0], 0.0),
      ([0.5, 0.5, 0.5], 1.5),
    ]
    for (point, expected) in cases {
      let result = try InterpolationND.interpn(
        points: [x3, y3, z3], values: v3, xi: [point],
        method: .linear, boundsHandling: .error)
      XCTAssertEqual(
        result[0], expected, accuracy: 1e-10,
        "3D linear at \(point) should be \(expected)")
    }
  }

  // MARK: - 3-D nearest interpolation

  /// Oracle: scipy interpn nearest on the unit cube (scipy 1.x, `method='nearest'`).
  ///
  /// ## Tie-break contract
  ///
  /// SciPy `interpn` with `method='nearest'` uses `numpy.round` internally, which
  /// applies *round-half-to-even* (banker's rounding) for scalars.  On a two-node
  /// axis [0.0, 1.0] the midpoint 0.5 is equidistant from both nodes; numpy rounds
  /// 0.5 → 0 (even), so the **left / lower** node wins.  Our implementation matches
  /// this: `distToLeft <= distToRight → chosenIndex = lo`.
  ///
  /// The tie-break DIRECTION is pinned by:
  ///   - `[0.5, 0.5, 0.5]` → maps to index (0,0,0) → value 0.0  (left wins on all 3 axes)
  ///   - `[0.51, 0.51, 0.51]` → just past midpoint → right wins on all 3 axes → index
  ///     (1,1,1) → value 3.0  (confirms direction is correct)
  func test3DNearest() throws {
    // Oracle: (0.25,0.5,0.75)->1.0, (1.0,1.0,1.0)->3.0, (0.0,0.0,0.0)->0.0
    // Tie-break: (0.5,0.5,0.5)->0.0  (left/lower node wins — SciPy round-half-to-even)
    let cases: [(point: [Double], expected: Double)] = [
      ([0.25, 0.5, 0.75], 1.0),
      ([1.0, 1.0, 1.0], 3.0),
      ([0.0, 0.0, 0.0], 0.0),
      // Exact midpoint on all three axes: left node wins (tie-break = left/lower).
      ([0.5, 0.5, 0.5], 0.0),
    ]
    for (point, expected) in cases {
      let result = try InterpolationND.interpn(
        points: [x3, y3, z3], values: v3, xi: [point],
        method: .nearest, boundsHandling: .error)
      XCTAssertEqual(
        result[0], expected, accuracy: 1e-12,
        "3D nearest at \(point) should be \(expected)")
    }

    // Opposite-corner case: a query just past the midpoint (0.51) must snap to the
    // RIGHT / higher node, confirming the tie-break boundary is at 0.5 (not > 0.5).
    // Oracle: scipy interpn nearest at [0.51, 0.51, 0.51] → index (1,1,1) → 3.0
    let pastMidpoint = try InterpolationND.interpn(
      points: [x3, y3, z3], values: v3, xi: [[0.51, 0.51, 0.51]],
      method: .nearest, boundsHandling: .error)
    XCTAssertEqual(
      pastMidpoint[0], 3.0, accuracy: 1e-12,
      "3D nearest at [0.51,0.51,0.51] should snap to right node → 3.0")
  }

  // MARK: - Out-of-bounds behaviour

  /// .error mode must throw when a query point is outside the grid.
  func testOutOfBoundsThrows() {
    // x=4.0 exceeds x-axis max of 3.0
    XCTAssertThrowsError(
      try InterpolationND.interpn(
        points: [x2, y2], values: v2, xi: [[4.0, 1.0]],
        method: .linear, boundsHandling: .error)
    ) { error in
      guard let ndError = error as? InterpolationND.InterpError else {
        XCTFail("Expected InterpolationND.InterpError, got \(error)")
        return
      }
      if case .outOfBounds(let axis, _, _, _) = ndError {
        XCTAssertEqual(axis, 0, "Out-of-bounds should be on axis 0")
      } else {
        XCTFail("Expected .outOfBounds, got \(ndError)")
      }
    }

    // y=-0.5 is below y-axis min of 0.0
    XCTAssertThrowsError(
      try InterpolationND.interpn(
        points: [x2, y2], values: v2, xi: [[1.0, -0.5]],
        method: .linear, boundsHandling: .error)
    ) { error in
      guard case InterpolationND.InterpError.outOfBounds(let axis, _, _, _) = error else {
        XCTFail("Expected .outOfBounds on axis 1, got \(error)")
        return
      }
      XCTAssertEqual(axis, 1, "Out-of-bounds should be on axis 1")
    }
  }

  /// .fillValue(.nan) must return NaN for out-of-bounds points and correct values for in-bounds.
  func testOutOfBoundsFillNaN() throws {
    // Oracle: (4.0, 1.0)->nan, (-0.5, 1.0)->nan
    let oob: [[Double]] = [[4.0, 1.0], [-0.5, 1.0]]
    for point in oob {
      let result = try InterpolationND.interpn(
        points: [x2, y2], values: v2, xi: [point],
        method: .linear, boundsHandling: .fillValue(.nan))
      XCTAssertTrue(result[0].isNaN, "Expected NaN for OOB point \(point), got \(result[0])")
    }

    // In-bounds point should still interpolate correctly
    let inBounds = try InterpolationND.interpn(
      points: [x2, y2], values: v2, xi: [[1.0, 2.0]],
      method: .linear, boundsHandling: .fillValue(.nan))
    XCTAssertEqual(inBounds[0], 5.0, accuracy: 1e-10)
  }

  /// .fillValue(-999.0) must return the constant for out-of-bounds points.
  func testOutOfBoundsFillConstant() throws {
    // Oracle: (4.0, 1.0) -> -999.0
    let result = try InterpolationND.interpn(
      points: [x2, y2], values: v2, xi: [[4.0, 1.0]],
      method: .linear, boundsHandling: .fillValue(-999.0))
    XCTAssertEqual(result[0], -999.0, accuracy: 1e-12)
  }

  // MARK: - 1-D consistency (degenerate case)

  /// Oracle: scipy 1D interpn linear — [0.5]->0.5, [1.5]->2.5
  func test1DLinearConsistency() throws {
    let cases: [(point: [Double], expected: Double)] = [
      ([0.5], 0.5),
      ([1.5], 2.5),
    ]
    for (point, expected) in cases {
      let result = try InterpolationND.interpn(
        points: [x1], values: v1, xi: [point],
        method: .linear, boundsHandling: .error)
      XCTAssertEqual(
        result[0], expected, accuracy: 1e-10,
        "1D linear at \(point) should be \(expected)")
    }
  }

  /// Oracle: scipy 1D interpn nearest — [0.5]->0.0, [1.5]->1.0
  func test1DNearest() throws {
    // [0.5] is equidistant between x[0]=0 and x[1]=1 → left wins → f[0]=0.0
    // [1.5] is equidistant between x[1]=1 and x[2]=2 → left wins → f[1]=1.0
    let cases: [(point: [Double], expected: Double)] = [
      ([0.5], 0.0),
      ([1.5], 1.0),
    ]
    for (point, expected) in cases {
      let result = try InterpolationND.interpn(
        points: [x1], values: v1, xi: [point],
        method: .nearest, boundsHandling: .error)
      XCTAssertEqual(
        result[0], expected, accuracy: 1e-12,
        "1D nearest at \(point) should be \(expected)")
    }
  }

  // MARK: - Non-uniform axis spacing

  /// Verifies interpolation is correct on a deliberately non-uniform grid.
  /// f(x,y) = x+2y on x=[0,1,3], y=[0,2,5] — axes are non-uniform.
  func testNonUniformAxisSpacing() throws {
    // Interior point mid-way in the non-uniform x-interval [1,3]:
    // x=2 is at t=(2-1)/(3-1)=0.5 along [1,3].
    // y=3.5 is at t=(3.5-2)/(5-2)=0.5 along [2,5].
    // Bilinear result: oracle says 9.0.
    let result = try InterpolationND.interpn(
      points: [x2, y2], values: v2, xi: [[2.0, 3.5]],
      method: .linear, boundsHandling: .error)
    XCTAssertEqual(result[0], 9.0, accuracy: 1e-10)

    // On-node exact at non-first index
    let onNode = try InterpolationND.interpn(
      points: [x2, y2], values: v2, xi: [[3.0, 2.0]],
      method: .linear, boundsHandling: .error)
    XCTAssertEqual(onNode[0], 7.0, accuracy: 1e-12)
  }

  // MARK: - Corner weight sum

  /// For any interior query, the sum of all 2^N corner weights must equal 1.
  /// We verify this indirectly: interpolating a constant field must return that constant.
  func testLinearWeightSumEqualsOne() throws {
    // Constant field: every value = 42.0 on the 2D grid
    let constValues = [Double](repeating: 42.0, count: x2.count * y2.count)
    let pts: [[Double]] = [[0.5, 1.0], [2.0, 3.5], [1.5, 4.0]]
    let results = try InterpolationND.interpn(
      points: [x2, y2], values: constValues, xi: pts,
      method: .linear, boundsHandling: .error)
    for (i, r) in results.enumerated() {
      XCTAssertEqual(r, 42.0, accuracy: 1e-12, "Weight sum test failed at query \(i)")
    }
  }

  // MARK: - Upper boundary (on the maximum grid value)

  /// Querying exactly at the upper boundary of each axis must not throw.
  func testUpperBoundaryIsInBounds() throws {
    // x=3.0 is the exact maximum x-coordinate
    let result = try InterpolationND.interpn(
      points: [x2, y2], values: v2, xi: [[3.0, 5.0]],
      method: .linear, boundsHandling: .error)
    XCTAssertEqual(result[0], 13.0, accuracy: 1e-12)
  }

  // MARK: - Validation errors

  /// Mismatched query point dimension must throw dimensionMismatch.
  func testDimensionMismatchThrows() {
    // 2D grid but 3-element query point
    XCTAssertThrowsError(
      try InterpolationND.interpn(
        points: [x2, y2], values: v2, xi: [[1.0, 2.0, 3.0]],
        method: .linear, boundsHandling: .error)
    ) { error in
      guard case InterpolationND.InterpError.dimensionMismatch = error else {
        XCTFail("Expected .dimensionMismatch, got \(error)")
        return
      }
    }
  }

  /// A grid axis with only one point must throw invalidGrid.
  func testSinglePointAxisThrows() {
    XCTAssertThrowsError(
      try InterpolationND.interpn(
        points: [[0.0], [0.0, 1.0]], values: [0.0, 1.0], xi: [[0.0, 0.5]],
        method: .linear, boundsHandling: .error)
    ) { error in
      guard case InterpolationND.InterpError.invalidGrid = error else {
        XCTFail("Expected .invalidGrid, got \(error)")
        return
      }
    }
  }

  /// A values array whose size doesn't match the grid shape must throw invalidGrid.
  func testValuesSizeMismatchThrows() {
    // Grid shape 3×3 = 9 elements, but only 8 values supplied
    XCTAssertThrowsError(
      try InterpolationND.interpn(
        points: [x2, y2], values: [Double](repeating: 0, count: 8), xi: [[1.0, 2.0]],
        method: .linear, boundsHandling: .error)
    ) { error in
      guard case InterpolationND.InterpError.invalidGrid = error else {
        XCTFail("Expected .invalidGrid, got \(error)")
        return
      }
    }
  }
}
