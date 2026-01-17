# Geometry

2D/3D geometry with SIMD vectors and coordinate transforms.

## Overview

The Geometry module provides vector and matrix types backed by SIMD, coordinate system transformations, and geometric algorithms.

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
let (r, theta) = cartToPolar(x: 1, y: 1)  // (sqrt(2), pi/4)

// Polar to Cartesian
let (x, y) = polarToCart(r: 1, theta: .pi/4)  // (0.707, 0.707)
```

### 3D Spherical/Cartesian

```swift
// Cartesian to spherical (r, theta, phi)
let (r, theta, phi) = cartToSpherical(x: 1, y: 0, z: 0)

// Spherical to Cartesian
let (x, y, z) = sphericalToCart(r: 1, theta: 0, phi: .pi/2)
```

### Angle Conversions

```swift
let radians = deg2rad(180)  // pi
let degrees = rad2deg(.pi)  // 180
```

## Distance Calculations

```swift
// 2D distance
let d2 = distance2D(Vec2(0, 0), Vec2(3, 4))  // 5

// 3D distance
let d3 = distance3D(Vec3(0, 0, 0), Vec3(1, 2, 2))  // 3

// Angle between vectors
let angle2d = angleBetween2D(Vec2(1, 0), Vec2(0, 1))  // pi/2
let angle3d = angleBetween3D(Vec3(1, 0, 0), Vec3(0, 1, 0))  // pi/2
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

### Vector Types

- ``Vec2``
- ``Vec3``
- ``Vec4``
- ``Quat``

### Matrix Types

- ``Mat4``

### Coordinate Conversions

- ``polarToCart(r:theta:)``
- ``cartToPolar(x:y:)``
- ``sphericalToCart(r:theta:phi:)``
- ``cartToSpherical(x:y:z:)``
- ``deg2rad(_:)``
- ``rad2deg(_:)``

### Distance and Angle Functions

- ``distance2D(_:_:)``
- ``distance3D(_:_:)``
- ``angleBetween2D(_:_:)``
- ``angleBetween3D(_:_:)``

### Circle Fitting

- ``circleFrom3Points(_:_:_:)``
- ``circleFitAlgebraic(_:)``
- ``circleFitTaubin(_:)``
- ``CircleResult``
- ``CircleFitResult``

### Ellipse Fitting

- ``ellipseFitDirect(points:)``
- ``EllipseFitResult``

### Plane Operations

- ``planeFrom3Points(_:_:_:)``
- ``pointPlaneDistance(_:_:)``
- ``linePlaneIntersection(linePoint:lineDir:plane:)``
- ``planePlaneIntersection(_:_:)``
- ``PlaneResult``

### Sphere Operations

- ``sphereFrom4Points(_:_:_:_:)``
- ``sphereFitAlgebraic(_:)``
- ``SphereResult``
- ``SphereFitResult``

### B-Splines

- ``bsplineEvaluate(controlPoints:degree:t:knots:)``
- ``bsplineEvaluate3D(controlPoints:degree:t:knots:)``
- ``bsplineDerivative(controlPoints:degree:t:knots:)``
- ``bsplineFit(points:degree:numControlPoints:parameterization:)``
- ``bsplineFit3D(points:degree:numControlPoints:parameterization:)``
- ``BSplineFitResult``
- ``BSplineFitResult3D``
- ``BSplineParameterization``

### Polygon Operations

- ``convexHull2D(_:)``
- ``pointInPolygon(_:_:)``
- ``triangleArea2D(_:_:_:)``
- ``triangleArea3D(_:_:_:)``
- ``centroid2D(_:)``
- ``centroid3D(_:)``
