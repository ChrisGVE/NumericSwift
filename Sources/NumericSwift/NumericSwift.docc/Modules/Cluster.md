# Clustering

Clustering algorithms for data analysis.

## Overview

The Cluster module provides implementations of popular clustering algorithms including K-means, DBSCAN, and hierarchical clustering.

## K-means Clustering

```swift
let data = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]

let result = kmeans(data, k: 2)

print(result.labels)      // Cluster assignment for each point
print(result.centroids)   // Final centroid positions
print(result.inertia)     // Sum of squared distances to centroids
```

### K-means Options

```swift
let result = kmeans(
    data,
    k: 3,
    maxIterations: 300,
    tolerance: 1e-4,
    nInit: 10,              // Number of random initializations
    initMethod: "kmeans++"  // Initialization method
)
```

## DBSCAN

Density-based clustering that finds arbitrarily shaped clusters:

```swift
let result = dbscan(
    data,
    eps: 0.5,        // Maximum distance between neighbors
    minSamples: 5    // Minimum points to form a cluster
)

print(result.labels)        // -1 indicates noise points
print(result.coreIndices)   // Indices of core points
```

## Hierarchical Clustering

```swift
// Compute linkage matrix and cut tree at specific number of clusters
let result = hierarchicalClustering(data, linkage: .ward, nClusters: 3)
print(result.linkageMatrix)  // Linkage matrix for dendrogram
print(result.labels)         // Cluster labels

// Or cut at a specific distance threshold
let result2 = hierarchicalClustering(data, linkage: .ward, distanceThreshold: 5.0)
```

### Linkage Methods

- `.single` - Minimum distance between clusters
- `.complete` - Maximum distance between clusters
- `.average` - Average distance (UPGMA)
- `.ward` - Ward's minimum variance method

## Distance Metrics

Used by clustering algorithms:

```swift
let dist = euclideanDistance(point1, point2)
let dist = manhattanDistance(point1, point2)
let dist = chebyshevDistance(point1, point2)
```

## Topics

### K-means

- ``kmeans(_:k:maxIterations:tolerance:nInit:initMethod:)``
- ``KMeansResult``

### DBSCAN

- ``dbscan(_:eps:minSamples:)``
- ``DBSCANResult``

### Hierarchical Clustering

- ``hierarchicalClustering(_:linkage:nClusters:distanceThreshold:)``
- ``HierarchicalResult``
- ``LinkageMethod``

### Cluster Evaluation

- ``silhouetteScore(_:labels:)``
- ``elbowMethod(_:maxK:)``
- ``ElbowResult``

### Distance Functions

- ``euclideanDistance(_:_:)``
- ``squaredEuclideanDistance(_:_:)``
- ``manhattanDistance(_:_:)``
- ``chebyshevDistance(_:_:)``
- ``pairwiseDistances(_:)``
