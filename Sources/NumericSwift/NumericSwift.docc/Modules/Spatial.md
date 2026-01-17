# Spatial Algorithms

KDTree, Voronoi diagrams, Delaunay triangulation, and distance metrics.

## Overview

The Spatial module provides data structures and algorithms for spatial data analysis, following scipy.spatial patterns.

## KDTree

Efficient spatial indexing for nearest neighbor queries:

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
let tree = KDTree(points)

// Find k nearest neighbors
let (distances, indices) = tree.query([0.5, 0.5], k: 2)
```

## Distance Metrics

```swift
let a = [0.0, 0.0]
let b = [3.0, 4.0]

// Common metrics
let euclidean = euclideanDistance(a, b)    // 5.0
let manhattan = manhattanDistance(a, b)    // 7.0
let chebyshev = chebyshevDistance(a, b)    // 4.0
let minkowski = minkowskiDistance(a, b, p: 3)
let cosine = cosineDistance(a, b)
let correlation = correlationDistance(a, b)
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
let result = voronoi(points)

print(result.vertices)      // Voronoi vertices
print(result.regions)       // Region indices for each point
print(result.ridgeVertices) // Vertices forming each ridge
print(result.ridgePoints)   // Pairs of input points sharing a ridge
```

## Delaunay Triangulation

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.5]]
let result = delaunay(points)

print(result.simplices)    // Triangle vertex indices
print(result.neighbors)    // Neighboring triangles
print(result.hullIndices)  // Convex hull indices
```

## Convex Hull

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.3]]
let result = convexHull(points)

print(result.vertices)   // Hull vertex coordinates
print(result.indices)    // Indices of hull vertices
print(result.area)       // Area of convex hull
```

## Topics

### KDTree

- ``KDTree``
- ``KDTreeNode``

### Distance Functions

- ``euclideanDistance(_:_:)``
- ``manhattanDistance(_:_:)``
- ``cityblockDistance(_:_:)``
- ``chebyshevDistance(_:_:)``
- ``minkowskiDistance(_:_:p:)``
- ``cosineDistance(_:_:)``
- ``correlationDistance(_:_:)``
- ``DistanceMetric``
- ``distanceFunction(for:)``

### Distance Matrices

- ``cdist(_:_:metric:)``
- ``pdist(_:metric:)``
- ``squareform(_:)``
- ``squareformToMatrix(_:)``

### Voronoi

- ``voronoi(_:)``
- ``VoronoiResult``

### Delaunay

- ``delaunay(_:)``
- ``DelaunayResult``

### Convex Hull

- ``convexHull(_:)``
- ``ConvexHullResult``
