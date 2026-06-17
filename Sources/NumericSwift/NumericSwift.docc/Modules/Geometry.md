# Geometry

2D/3D geometry with SIMD vectors and coordinate transforms.

## Overview

The `Geometry` module provides vector and matrix types backed by SIMD,
coordinate system transformations, and geometric algorithms.

Coordinate conversion and utility functions live under the `Geometry` namespace.
The SIMD type aliases (`Vec2`, `Vec3`, `Vec4`, `Quat`, `Mat4`) and geometric
result types remain at module level for brevity. The old top-level free
functions for coordinate conversion (e.g. `deg2rad`, `polarToCart`,
`cartToSpherical`) are still available as `@available(*, deprecated)` shims so
existing code continues to compile with a deprecation warning. New code should
use the namespaced forms.

## Migration from Top-Level Functions

```swift
// Before (deprecated)
deg2rad(180)
polarToCart(r: 1, theta: .pi / 4)
cartToSpherical(x: 1, y: 0, z: 0)

// After
Geometry.deg2rad(180)
Geometry.polarToCart(r: 1, theta: .pi / 4)
Geometry.cartToSpherical(x: 1, y: 0, z: 0)
```

## Vector Types

```swift
// 2D, 3D, and 4D vectors (SIMD-backed)
let v2 = Vec2(1, 2)
let v3 = Vec3(1, 2, 3)
let v4 = Vec4(1, 2, 3, 4)

// Vector operations
let sum = v3 + Vec3(1, 1, 1)
let scaled = v3 * 2.0
let magnitude = simd_length(v3)
let normalized = simd_normalize(v3)
```

## Matrix Types

```swift
// 4x4 matrices for transformations
let m4 = Mat4(diagonal: Vec4(1, 1, 1, 1))  // Identity

// Quaternions for rotations
let q = Quat(angle: .pi/4, axis: Vec3(0, 1, 0))
```

## Coordinate Conversions

### 2D Polar/Cartesian

```swift
// Cartesian to polar
let (r, theta) = Geometry.cartToPolar(x: 1, y: 1)  // (√2, π/4)

// Polar to Cartesian
let (x, y) = Geometry.polarToCart(r: 1, theta: .pi/4)  // (0.707, 0.707)
```

### 3D Spherical/Cartesian

The library uses the **physics/ISO convention**: θ is the polar angle (from
the z-axis) and φ is the azimuthal angle (in the xy-plane). This matches
SciPy's `sph2cart` / `cart2sph` semantics.

```swift
// Cartesian to spherical (r, θ, φ) — physics convention
let (r, theta, phi) = Geometry.cartToSpherical(x: 1, y: 0, z: 0)

// Spherical to Cartesian
let (x, y, z) = Geometry.sphericalToCart(r: 1, theta: .pi/2, phi: 0)
```

### Angle Conversions

```swift
let radians = Geometry.deg2rad(180)  // π
let degrees = Geometry.rad2deg(.pi)  // 180
```

## Distance Calculations

```swift
// 2D distance
let d2 = Geometry.distance2D(Vec2(0, 0), Vec2(3, 4))  // 5

// 3D distance
let d3 = Geometry.distance3D(Vec3(0, 0, 0), Vec3(1, 2, 2))  // 3

// Angle between vectors
let angle2d = Geometry.angleBetween2D(Vec2(1, 0), Vec2(0, 1))  // π/2
let angle3d = Geometry.angleBetween3D(Vec3(1, 0, 0), Vec3(0, 1, 0))  // π/2
```

## Geometric Algorithms

### Circle and Ellipse Fitting

```swift
// Fit circle through 3 points
if let circle = circleFrom3Points(p1, p2, p3) {
    print(circle.center, circle.radius)
}

// Fit circle to point cloud (algebraic)
if let fit = circleFitAlgebraic(points) {
    print(fit.center, fit.radius, fit.rmse)
}

// Fit circle to point cloud (Taubin method - more robust)
if let fit = circleFitTaubin(points) {
    print(fit.center, fit.radius, fit.rmse)
}

// Fit ellipse using Fitzgibbon's direct method
if let ellipse = ellipseFitDirect(points: points) {
    print(ellipse.center, ellipse.semiAxes, ellipse.angle)
}
```

### Planes and Spheres

```swift
// Plane from 3 points
if let plane = planeFrom3Points(p1, p2, p3) {
    print(plane.normal, plane.d)
}

// Point-to-plane distance
let dist = pointPlaneDistance(point, plane)

// Sphere from 4 points
if let sphere = sphereFrom4Points(p1, p2, p3, p4) {
    print(sphere.center, sphere.radius)
}

// Fit sphere to point cloud
if let fit = sphereFitAlgebraic(points) {
    print(fit.center, fit.radius, fit.rmse)
}
```

### B-Splines

```swift
// Evaluate B-spline curve
let point = bsplineEvaluate(controlPoints: controlPts, degree: 3, t: 0.5)

// B-spline derivative
let tangent = bsplineDerivative(controlPoints: controlPts, degree: 3, t: 0.5)

// Fit B-spline to data
if let fit = bsplineFit(points: dataPoints, degree: 3, numControlPoints: 10) {
    print(fit.controlPoints, fit.rmse)
}
```

## Topics

### Namespace

- ``Geometry``

### Vector Types

- ``Vec2``
- ``Vec3``
- ``Vec4``
- ``Quat``

### Matrix Types

- ``Mat4``

### Coordinate Conversions

- ``Geometry/polarToCart(r:theta:)``
- ``Geometry/cartToPolar(x:y:)``
- ``Geometry/sphericalToCart(r:theta:phi:)``
- ``Geometry/cartToSpherical(x:y:z:)``
- ``Geometry/deg2rad(_:)``
- ``Geometry/rad2deg(_:)``

### Distance and Angle Functions

- ``Geometry/distance2D(_:_:)``
- ``Geometry/distance3D(_:_:)``
- ``Geometry/angleBetween2D(_:_:)``
- ``Geometry/angleBetween3D(_:_:)``

### Circle Fitting

- ``Geometry/circleFrom3Points(_:_:_:)``
- ``Geometry/circleFitAlgebraic(_:)``
- ``Geometry/circleFitTaubin(_:)``
- ``CircleResult``
- ``CircleFitResult``

### Ellipse Fitting

- ``Geometry/ellipseFitDirect(points:)``
- ``EllipseFitResult``

### Plane Operations

- ``Geometry/planeFrom3Points(_:_:_:)``
- ``Geometry/pointPlaneDistance(_:_:)``
- ``Geometry/linePlaneIntersection(linePoint:lineDir:plane:)``
- ``Geometry/planePlaneIntersection(_:_:)``
- ``PlaneResult``

### Sphere Operations

- ``Geometry/sphereFrom4Points(_:_:_:_:)``
- ``Geometry/sphereFitAlgebraic(_:)``
- ``SphereResult``
- ``SphereFitResult``

### B-Splines

- ``Geometry/bsplineEvaluate(controlPoints:degree:t:knots:)``
- ``Geometry/bsplineEvaluate3D(controlPoints:degree:t:knots:)``
- ``Geometry/bsplineDerivative(controlPoints:degree:t:knots:)``
- ``Geometry/bsplineFit(points:degree:numControlPoints:parameterization:)``
- ``Geometry/bsplineFit3D(points:degree:numControlPoints:parameterization:)``
- ``BSplineFitResult``
- ``BSplineFitResult3D``
- ``BSplineParameterization``

### Polygon Operations

- ``Geometry/convexHull2D(_:)``
- ``Geometry/pointInPolygon(_:_:)``
- ``Geometry/triangleArea2D(_:_:_:)``
- ``Geometry/triangleArea3D(_:_:_:)``
- ``Geometry/centroid2D(_:)``
- ``Geometry/centroid3D(_:)``
