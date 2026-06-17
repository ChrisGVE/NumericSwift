# Clustering

Clustering algorithms for data analysis.

## Overview

The `Cluster` module provides implementations of popular clustering algorithms
including K-means, DBSCAN, and hierarchical clustering.

All functions live under the `Cluster` namespace. The old top-level free
functions (e.g. `kmeans`, `dbscan`) are still available as
`@available(*, deprecated)` shims so existing code continues to compile with a
deprecation warning. New code should use the namespaced forms.

## Migration from Top-Level Functions

```swift
// Before (deprecated)
kmeans(data, k: 2)
dbscan(data, eps: 0.5, minSamples: 5)

// After
Cluster.kmeans(data, k: 2)
Cluster.dbscan(data, eps: 0.5, minSamples: 5)
```

## K-means Clustering

```swift
let data = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]

let result = Cluster.kmeans(data, k: 2)

print(result.labels)      // cluster assignment for each point
print(result.centroids)   // final centroid positions
print(result.inertia)     // sum of squared distances to centroids
```

### K-means Options

```swift
let result = Cluster.kmeans(
    data,
    k: 3,
    maxIterations: 300,
    tolerance: 1e-4,
    nInit: 10,              // number of random initializations
    initMethod: "kmeans++"  // initialization method
)
```

## DBSCAN

Density-based clustering that finds arbitrarily shaped clusters:

```swift
let result = Cluster.dbscan(
    data,
    eps: 0.5,        // maximum distance between neighbors
    minSamples: 5    // minimum points to form a cluster
)

print(result.labels)        // -1 indicates noise points
print(result.coreIndices)   // indices of core points
```

## Hierarchical Clustering

```swift
// Cut tree at a specific number of clusters
let result = Cluster.hierarchicalClustering(data, linkage: .ward, nClusters: 3)
print(result.linkageMatrix)  // linkage matrix for dendrogram
print(result.labels)         // cluster labels

// Or cut at a distance threshold
let result2 = Cluster.hierarchicalClustering(data, linkage: .ward, distanceThreshold: 5.0)
```

### Linkage Methods

- `.single`   — minimum distance between clusters
- `.complete` — maximum distance between clusters
- `.average`  — average distance (UPGMA)
- `.ward`     — Ward's minimum variance method

## Cluster Evaluation

```swift
// Silhouette score: −1 (worst) to +1 (best)
let score = Cluster.silhouetteScore(data, labels: result.labels)

// Elbow method to select k
let elbow = Cluster.elbowMethod(data, maxK: 10)
print(elbow.inertias)   // inertia per k
```

## Distance Functions

The clustering algorithms use Euclidean distance internally. For richer
distance metrics (cosine, Minkowski, correlation) and pairwise distance
matrices (`cdist`, `pdist`), see <doc:Spatial>.

## Topics

### Namespace

- ``Cluster``

### K-means

- ``Cluster/kmeans(_:k:maxIterations:tolerance:nInit:initMethod:)``
- ``Cluster/KMeansResult``

### DBSCAN

- ``Cluster/dbscan(_:eps:minSamples:)``
- ``Cluster/DBSCANResult``

### Hierarchical Clustering

- ``Cluster/hierarchicalClustering(_:linkage:nClusters:distanceThreshold:)``
- ``Cluster/HierarchicalResult``
- ``Cluster/LinkageMethod``

### Cluster Evaluation

- ``Cluster/silhouetteScore(_:labels:)``
- ``Cluster/elbowMethod(_:maxK:)``
- ``Cluster/ElbowResult``

### Distance Functions

For richer distance metrics and pairwise matrix computation, see
``Spatial`` and <doc:Spatial>.
