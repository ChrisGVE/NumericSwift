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
let dot = v3.dot(Vec3(1, 0, 0))
let cross = v3.cross(Vec3(0, 1, 0))
let magnitude = v3.length
let normalized = v3.normalized
```

## Matrix Types

```swift
// 3x3 and 4x4 matrices
let m3 = Mat3.identity
let m4 = Mat4.identity

// Transformation matrices
let rotation = Mat4.rotation(angle: .pi/4, axis: Vec3(0, 1, 0))
let translation = Mat4.translation(Vec3(1, 2, 3))
let scale = Mat4.scale(Vec3(2, 2, 2))

// Matrix operations
let combined = translation * rotation * scale
let transformed = combined * Vec4(1, 0, 0, 1)
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
let d2 = distance2D((0, 0), (3, 4))  // 5

// 3D distance
let d3 = distance3D((0, 0, 0), (1, 2, 2))  // 3
```

## Geometric Algorithms

### Ellipse Fitting

```swift
let points: [(Double, Double)] = [...]
let result = ellipseFitDirect(points)

print(result.center)      // (cx, cy)
print(result.semiAxes)    // (a, b)
print(result.angle)       // Rotation angle
```

### Plane Fitting

```swift
let points: [(Double, Double, Double)] = [...]
let plane = fitPlane(points)

print(plane.normal)   // Normal vector
print(plane.d)        // Distance from origin
```

## Topics

### Vector Types

- ``Vec2``
- ``Vec3``
- ``Vec4``

### Matrix Types

- ``Mat3``
- ``Mat4``

### Coordinate Conversions

- ``polarToCart(r:theta:)``
- ``cartToPolar(x:y:)``
- ``sphericalToCart(r:theta:phi:)``
- ``cartToSpherical(x:y:z:)``
- ``deg2rad(_:)``
- ``rad2deg(_:)``

### Distance Functions

- ``distance2D(_:_:)``
- ``distance3D(_:_:)``

### Geometric Algorithms

- ``ellipseFitDirect(_:)``
- ``EllipseFitResult``
- ``fitPlane(_:)``
- ``PlaneResult``
