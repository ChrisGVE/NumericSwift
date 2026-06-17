# Spatial Algorithms

KDTree, Voronoi diagrams, Delaunay triangulation, and distance metrics.

## Overview

The `Spatial` module provides data structures and algorithms for spatial data
analysis, following `scipy.spatial` patterns.

Distance metrics and pairwise computation live under the `Spatial` namespace.
The computational geometry primitives (`KDTree`, result types) are top-level
types. The old top-level free functions (`voronoi`, `delaunay`, `convexHull`)
are still available as `@available(*, deprecated)` shims so existing code
continues to compile with a deprecation warning. New code should use the
namespaced forms.

## Migration from Top-Level Functions

```swift
// Before (deprecated)
voronoi(points)
delaunay(points)
convexHull(points)

// After
Spatial.voronoi(points)
Spatial.delaunay(points)
Spatial.convexHull(points)
```

## KDTree

Efficient spatial indexing for nearest-neighbour queries:

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
let tree = KDTree(points)

// Find k nearest neighbours
let (indices, distances) = tree.query([0.5, 0.5], k: 2)

// Find all points within radius
let (nearIndices, nearDists) = tree.queryRadius([0.5, 0.5], radius: 0.8)
```

## Distance Metrics

```swift
let a = [0.0, 0.0]
let b = [3.0, 4.0]

// Common metrics (all under Spatial namespace)
let euclidean   = Spatial.euclideanDistance(a, b)    // 5.0
let manhattan   = Spatial.manhattanDistance(a, b)    // 7.0
let chebyshev   = Spatial.chebyshevDistance(a, b)    // 4.0
let minkowski   = Spatial.minkowskiDistance(a, b, p: 3)
let cosine      = Spatial.cosineDistance(a, b)
let correlation = Spatial.correlationDistance(a, b)
```

### Distance Matrices

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

// Pairwise distances between all points (condensed form)
let condensed = pdist(points)

// Convert to square matrix
let square = squareformToMatrix(condensed)

// Distance matrix between two sets of points
let X = [[0.0, 0.0], [1.0, 1.0]]
let Y = [[0.5, 0.5], [2.0, 2.0]]
let dist = cdist(X, Y, metric: .euclidean)
```

## Voronoi Diagrams

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
let result = Spatial.voronoi(points)

print(result.vertices)       // Voronoi vertices
print(result.regions)        // region indices for each point
print(result.ridgeVertices)  // vertices forming each ridge
print(result.ridgePoints)    // pairs of input points sharing a ridge
```

## Delaunay Triangulation

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.5]]
let result = Spatial.delaunay(points)

print(result.simplices)    // triangle vertex indices
print(result.neighbors)    // neighbouring triangles
print(result.hullIndices)  // convex hull indices
```

## Convex Hull

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.3]]
let result = Spatial.convexHull(points)

print(result.vertices)  // hull vertex coordinates
print(result.indices)   // indices of hull vertices
print(result.area)      // area of convex hull
```

## Topics

### Namespace

- ``Spatial``

### KDTree

- ``KDTree``
- ``KDTreeNode``

### Distance Functions

- ``Spatial/euclideanDistance(_:_:)``
- ``Spatial/squaredEuclideanDistance(_:_:)``
- ``Spatial/manhattanDistance(_:_:)``
- ``Spatial/chebyshevDistance(_:_:)``
- ``Spatial/minkowskiDistance(_:_:p:)``
- ``Spatial/cosineDistance(_:_:)``
- ``Spatial/correlationDistance(_:_:)``
- ``DistanceMetric``
- ``distanceFunction(for:)``

### Distance Matrices

- ``cdist(_:_:metric:)``
- ``pdist(_:metric:)``
- ``squareform(_:)``
- ``squareformToMatrix(_:)``

### Voronoi

- ``Spatial/voronoi(_:)``
- ``VoronoiResult``

### Delaunay

- ``Spatial/delaunay(_:)``
- ``DelaunayResult``

### Convex Hull

- ``Spatial/convexHull(_:)``
- ``ConvexHullResult``
