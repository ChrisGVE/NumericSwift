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

// Find all points within radius
let indices = tree.queryBallPoint([0.5, 0.5], r: 1.0)
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
```

### Distance Matrices

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

// Pairwise distance matrix
let D = pdist(points)  // Condensed form
let squareD = squareform(D)  // Square matrix form
```

## Voronoi Diagrams

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
let voronoi = Voronoi(points)

print(voronoi.vertices)    // Voronoi vertices
print(voronoi.regions)     // Region indices for each point
print(voronoi.ridgePoints) // Pairs of input points sharing a ridge
```

## Delaunay Triangulation

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.5]]
let tri = Delaunay(points)

print(tri.simplices)    // Triangle vertex indices
print(tri.neighbors)    // Neighboring triangles
print(tri.convexHull)   // Convex hull indices
```

## Convex Hull

```swift
let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.3]]
let hull = ConvexHull(points)

print(hull.vertices)  // Indices of hull vertices
print(hull.area)      // Area of convex hull
```

## Topics

### KDTree

- ``KDTree``
- ``KDTree/query(_:k:)``
- ``KDTree/queryBallPoint(_:r:)``

### Distance Functions

- ``euclideanDistance(_:_:)``
- ``manhattanDistance(_:_:)``
- ``chebyshevDistance(_:_:)``
- ``minkowskiDistance(_:_:p:)``
- ``pdist(_:metric:)``
- ``squareform(_:)``

### Voronoi

- ``Voronoi``

### Delaunay

- ``Delaunay``

### Convex Hull

- ``ConvexHull``
