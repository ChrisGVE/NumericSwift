//
//  InterpolationND.swift
//  Sources/NumericSwift/
//
//  N-dimensional interpolation on a regular (but not necessarily uniform) grid.
//  API mirrors scipy.interpolate.interpn / RegularGridInterpolator.
//
//  Supported methods: linear (multilinear) and nearest.
//  Out-of-bounds behaviour: throw or fill with a constant value.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Public types

/// Interpolation method for N-dimensional grid interpolation.
public enum InterpolationNDMethod: Sendable {
  /// Multilinear interpolation: weighted sum over the 2^N enclosing corners.
  case linear
  /// Nearest-neighbour: returns the value at the closest grid node.
  case nearest
}

/// Out-of-bounds policy for N-dimensional grid interpolation.
public enum InterpolationNDBounds: Sendable {
  /// Throw `InterpolationNDError.outOfBounds` when any query coordinate lies
  /// outside the axis range.
  case error
  /// Return `fill` for any query point where at least one coordinate is
  /// outside its axis range.  Use `.nan` to reproduce SciPy's default.
  case fillValue(Double)
}

/// Errors produced by `interpn`.
public enum InterpolationNDError: Error, Equatable, Sendable {
  /// A query coordinate is outside the axis bounds.
  ///
  /// - Parameters:
  ///   - axis: Zero-based axis index.
  ///   - value: The out-of-range coordinate.
  ///   - min: Lower bound of the axis.
  ///   - max: Upper bound of the axis.
  case outOfBounds(axis: Int, value: Double, min: Double, max: Double)

  /// The grid specification is malformed (axis too short, wrong values count,
  /// or axis not strictly increasing).
  case invalidGrid(reason: String)

  /// A query point has a different number of components than the grid has axes.
  ///
  /// - Parameters:
  ///   - expected: Number of axes.
  ///   - got: Number of components in the query point.
  case dimensionMismatch(expected: Int, got: Int)
}

// MARK: - Public API

/// Interpolate on a regular N-dimensional grid.
///
/// This function evaluates an N-dimensional interpolant at a set of query
/// points.  The grid is defined by one coordinate array per axis; the data
/// values are given as a flat, row-major (C-order) array whose shape matches
/// the grid.
///
/// The API mirrors SciPy's `scipy.interpolate.interpn`:
///
/// ```python
/// from scipy.interpolate import interpn
/// v = interpn((x, y), values, [(0.5, 1.0)], method='linear')
/// ```
///
/// Equivalent Swift:
///
/// ```swift
/// let v = try interpn(
///     points: [x, y], values: values, xi: [[0.5, 1.0]],
///     method: .linear, boundsHandling: .error)
/// ```
///
/// ## Algorithm
///
/// **Linear:** multilinear interpolation over the 2^N enclosing corners.
/// For each axis *d*, binary search finds the bracketing interval, and a
/// fractional weight `t_d` is computed from the distance within that interval.
/// The result is the weighted sum
///
///     Σ_c  (Π_d weight_d(c)) · value(c)
///
/// over all 2^N corner combinations, where `weight_d(c) = 1−t_d` if the
/// corner selects the left node on axis *d*, and `t_d` otherwise.
///
/// **Nearest:** each axis independently rounds to the closest grid node
/// (ties go to the left/lower node, matching SciPy's behaviour).
///
/// - Parameters:
///   - points: Coordinate arrays for each axis, each strictly increasing
///     and containing at least two elements.
///   - values: Flat, row-major array of data values.  Its count must equal
///     the product of the axis lengths.
///   - xi: Query points.  Each element is an N-component coordinate vector.
///   - method: Interpolation method (`.linear` or `.nearest`).
///   - boundsHandling: What to do when a query point is outside the grid
///     (`.error` throws; `.fillValue(v)` returns `v`).
/// - Returns: Interpolated values, one per query point, in the same order.
/// - Throws: `InterpolationNDError.invalidGrid` when `points` or `values`
///   are malformed; `InterpolationNDError.dimensionMismatch` when a query
///   point has the wrong number of components; `InterpolationNDError.outOfBounds`
///   when a query point is out of range and `boundsHandling` is `.error`.
public func interpn(
  points: [[Double]],
  values: [Double],
  xi: [[Double]],
  method: InterpolationNDMethod = .linear,
  boundsHandling: InterpolationNDBounds = .error
) throws -> [Double] {
  let grid = try RegularGrid(points: points, values: values)
  return try xi.map { queryPoint in
    try grid.evaluate(at: queryPoint, method: method, boundsHandling: boundsHandling)
  }
}

// MARK: - Internal grid representation

/// Pre-validated grid descriptor used internally by `interpn`.
///
/// Holds the per-axis coordinate arrays, the strides needed for flat
/// (row-major) index arithmetic, and a reference to the values array.
private struct RegularGrid {

  /// Coordinate arrays, one per axis (each strictly increasing, ≥ 2 elements).
  let axes: [[Double]]

  /// Flat, row-major data values.
  let values: [Double]

  /// Row-major strides: `strides[d]` is the number of elements to skip in
  /// `values` when the index on axis `d` increases by one.
  let strides: [Int]

  /// Number of axes (dimensions).
  var ndim: Int { axes.count }

  // MARK: Initialiser — validate everything upfront

  init(points: [[Double]], values: [Double]) throws {
    guard !points.isEmpty else {
      throw InterpolationNDError.invalidGrid(reason: "points must contain at least one axis")
    }

    // Validate each axis
    for (d, axis) in points.enumerated() {
      guard axis.count >= 2 else {
        throw InterpolationNDError.invalidGrid(
          reason: "axis \(d) must have at least 2 elements, got \(axis.count)")
      }
      for i in 1..<axis.count {
        guard axis[i] > axis[i - 1] else {
          throw InterpolationNDError.invalidGrid(
            reason: "axis \(d) must be strictly increasing: \(axis[i-1]) >= \(axis[i]) at index \(i)")
        }
      }
    }

    // Validate values count against grid shape
    let expectedCount = points.reduce(1) { $0 * $1.count }
    guard values.count == expectedCount else {
      throw InterpolationNDError.invalidGrid(
        reason: "values count \(values.count) does not match grid shape \(points.map(\.count)) = \(expectedCount)")
    }

    // Compute row-major strides (last axis has stride 1)
    var strides = [Int](repeating: 1, count: points.count)
    for d in stride(from: points.count - 2, through: 0, by: -1) {
      strides[d] = strides[d + 1] * points[d + 1].count
    }

    self.axes = points
    self.values = values
    self.strides = strides
  }

  // MARK: Evaluation

  /// Evaluate the interpolant at one query point.
  func evaluate(
    at point: [Double],
    method: InterpolationNDMethod,
    boundsHandling: InterpolationNDBounds
  ) throws -> Double {
    guard point.count == ndim else {
      throw InterpolationNDError.dimensionMismatch(expected: ndim, got: point.count)
    }

    // Bounds check — performed before any interpolation work.
    for d in 0..<ndim {
      let lo = axes[d].first!
      let hi = axes[d].last!
      if point[d] < lo || point[d] > hi {
        switch boundsHandling {
        case .error:
          throw InterpolationNDError.outOfBounds(axis: d, value: point[d], min: lo, max: hi)
        case .fillValue(let fill):
          return fill
        }
      }
    }

    switch method {
    case .linear:
      return multilinear(at: point)
    case .nearest:
      return nearestNeighbour(at: point)
    }
  }

  // MARK: - Multilinear interpolation

  /// Multilinear interpolation at an in-bounds query point.
  ///
  /// For each axis *d* the algorithm finds the bracketing interval [i_d, i_d+1]
  /// via binary search and computes the fractional position within that interval.
  /// It then iterates over all 2^N corners using bit-manipulation to encode
  /// which boundary (left=0, right=1) is selected on each axis, accumulating
  /// the weighted sum of corner values.
  ///
  /// Reference: Wikipedia, "Multilinear interpolation",
  /// https://en.wikipedia.org/wiki/Multilinear_interpolation
  private func multilinear(at point: [Double]) -> Double {
    // Step 1: find left-edge index and fractional weight for each axis.
    var leftIndices = [Int](repeating: 0, count: ndim)
    var weights = [Double](repeating: 0.0, count: ndim)

    for d in 0..<ndim {
      let lo = findIntervalND(axes[d], point[d])
      leftIndices[d] = lo
      let span = axes[d][lo + 1] - axes[d][lo]
      // span > 0 is guaranteed by the strictly-increasing validation.
      weights[d] = (point[d] - axes[d][lo]) / span
    }

    // Step 2: accumulate weighted sum over 2^N corners.
    // Corner c is encoded as an N-bit integer: bit d = 0 → left node, 1 → right.
    let cornerCount = 1 << ndim
    var result = 0.0

    for corner in 0..<cornerCount {
      var flatIndex = 0
      var cornerWeight = 1.0

      for d in 0..<ndim {
        let bit = (corner >> d) & 1  // 0 = left, 1 = right on axis d
        flatIndex += (leftIndices[d] + bit) * strides[d]
        cornerWeight *= bit == 0 ? (1.0 - weights[d]) : weights[d]
      }

      result += cornerWeight * values[flatIndex]
    }

    return result
  }

  // MARK: - Nearest-neighbour interpolation

  /// Nearest-neighbour interpolation at an in-bounds query point.
  ///
  /// For each axis, the coordinate is snapped to the closer of the two
  /// bracketing grid nodes.  Ties (equidistant case) go to the left/lower
  /// node, matching SciPy's behaviour.
  private func nearestNeighbour(at point: [Double]) -> Double {
    var flatIndex = 0

    for d in 0..<ndim {
      let lo = findIntervalND(axes[d], point[d])
      let hi = min(lo + 1, axes[d].count - 1)

      let distToLeft = point[d] - axes[d][lo]
      let distToRight = axes[d][hi] - point[d]

      // Tie-break: left node wins (matches SciPy's `nearest` rule).
      let chosenIndex = distToLeft <= distToRight ? lo : hi
      flatIndex += chosenIndex * strides[d]
    }

    return values[flatIndex]
  }
}

// MARK: - Internal binary search

/// Find the largest index `i` in `sortedAxis` such that `sortedAxis[i] <= x`,
/// clamped to `sortedAxis.count - 2` so that `i + 1` is always a valid index.
///
/// Pre-condition: `sortedAxis` has ≥ 2 elements and `x` is in
/// `[sortedAxis.first!, sortedAxis.last!]` (validated by the caller).
private func findIntervalND(_ sortedAxis: [Double], _ x: Double) -> Int {
  let n = sortedAxis.count
  let lastValidIndex = n - 2

  // Clamp the upper boundary so that the returned index i satisfies i+1 < n.
  if x >= sortedAxis[n - 1] { return lastValidIndex }

  // Binary search: find the largest lo with sortedAxis[lo] <= x.
  var lo = 0
  var hi = n - 1
  while hi - lo > 1 {
    let mid = (lo + hi) / 2
    if sortedAxis[mid] > x {
      hi = mid
    } else {
      lo = mid
    }
  }
  // lo is now the left-bracket index; clamp defensively.
  return min(lo, lastValidIndex)
}
