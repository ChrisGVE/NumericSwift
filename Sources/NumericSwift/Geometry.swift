//
//  Geometry.swift
//  NumericSwift
//
//  High-performance 2D/3D geometry using SIMD.
//
//  Licensed under the MIT License.
//

import Foundation
import simd

// MARK: - Type Aliases

/// 2D vector type.
public typealias Vec2 = simd_double2

/// 3D vector type.
public typealias Vec3 = simd_double3

/// 4D vector type.
public typealias Vec4 = simd_double4

/// Quaternion type.
public typealias Quat = simd_quatd

/// 4x4 matrix type.
public typealias Mat4 = simd_double4x4

// MARK: - Vec2 Extensions

extension Vec2 {
    /// Create a Vec2 from an array.
    public init?(_ array: [Double]) {
        guard array.count >= 2 else { return nil }
        self.init(array[0], array[1])
    }

    /// Angle of the vector from positive x-axis.
    public var angle: Double {
        atan2(y, x)
    }

    /// 2D cross product (returns z-component of 3D cross).
    public func cross(_ other: Vec2) -> Double {
        x * other.y - y * other.x
    }

    /// Rotate vector by angle (radians).
    public func rotated(by theta: Double) -> Vec2 {
        let c = cos(theta)
        let s = sin(theta)
        return Vec2(x * c - y * s, x * s + y * c)
    }

    /// Project onto another vector.
    public func projected(onto other: Vec2) -> Vec2 {
        let lenSq = simd_length_squared(other)
        guard lenSq > 0 else { return Vec2.zero }
        return other * (simd_dot(self, other) / lenSq)
    }

    /// Perpendicular vector (90° counter-clockwise).
    public var perpendicular: Vec2 {
        Vec2(-y, x)
    }

    /// Convert to polar coordinates (r, theta).
    public var polar: (r: Double, theta: Double) {
        (simd_length(self), atan2(y, x))
    }

    /// Create from polar coordinates.
    public static func fromPolar(r: Double, theta: Double) -> Vec2 {
        Vec2(r * cos(theta), r * sin(theta))
    }
}

// MARK: - Vec3 Extensions

extension Vec3 {
    /// Create a Vec3 from an array.
    public init?(_ array: [Double]) {
        guard array.count >= 3 else { return nil }
        self.init(array[0], array[1], array[2])
    }

    /// Rotate vector around axis by angle.
    public func rotated(around axis: Vec3, by angle: Double) -> Vec3 {
        let q = Quat(angle: angle, axis: simd_normalize(axis))
        return q.act(self)
    }

    /// Project onto another vector.
    public func projected(onto other: Vec3) -> Vec3 {
        let lenSq = simd_length_squared(other)
        guard lenSq > 0 else { return Vec3.zero }
        return other * (simd_dot(self, other) / lenSq)
    }

    /// Convert to spherical coordinates (r, theta, phi).
    /// theta: azimuthal angle (xy plane), phi: polar angle (from z-axis)
    public var spherical: (r: Double, theta: Double, phi: Double) {
        let r = simd_length(self)
        guard r > 0 else { return (0, 0, 0) }
        let theta = atan2(y, x)
        let phi = acos(z / r)
        return (r, theta, phi)
    }

    /// Create from spherical coordinates.
    public static func fromSpherical(r: Double, theta: Double, phi: Double) -> Vec3 {
        let sinPhi = sin(phi)
        return Vec3(r * sinPhi * cos(theta), r * sinPhi * sin(theta), r * cos(phi))
    }

    /// Convert to cylindrical coordinates (r, theta, z).
    public var cylindrical: (r: Double, theta: Double, z: Double) {
        (sqrt(x * x + y * y), atan2(y, x), z)
    }

    /// Create from cylindrical coordinates.
    public static func fromCylindrical(r: Double, theta: Double, z: Double) -> Vec3 {
        Vec3(r * cos(theta), r * sin(theta), z)
    }
}

// MARK: - Quaternion Extensions

extension Quat {
    /// Create from Euler angles (ZYX order).
    public static func fromEuler(roll: Double, pitch: Double, yaw: Double) -> Quat {
        let cy = cos(yaw * 0.5)
        let sy = sin(yaw * 0.5)
        let cp = cos(pitch * 0.5)
        let sp = sin(pitch * 0.5)
        let cr = cos(roll * 0.5)
        let sr = sin(roll * 0.5)

        return Quat(
            ix: sr * cp * cy - cr * sp * sy,
            iy: cr * sp * cy + sr * cp * sy,
            iz: cr * cp * sy - sr * sp * cy,
            r: cr * cp * cy + sr * sp * sy
        )
    }

    /// Convert to Euler angles (roll, pitch, yaw).
    public var euler: (roll: Double, pitch: Double, yaw: Double) {
        let sinrCosp = 2.0 * (real * imag.x + imag.y * imag.z)
        let cosrCosp = 1.0 - 2.0 * (imag.x * imag.x + imag.y * imag.y)
        let roll = atan2(sinrCosp, cosrCosp)

        let sinp = 2.0 * (real * imag.y - imag.z * imag.x)
        let pitch = abs(sinp) >= 1 ? copysign(.pi / 2, sinp) : asin(sinp)

        let sinyCosp = 2.0 * (real * imag.z + imag.x * imag.y)
        let cosyCosp = 1.0 - 2.0 * (imag.y * imag.y + imag.z * imag.z)
        let yaw = atan2(sinyCosp, cosyCosp)

        return (roll, pitch, yaw)
    }

    /// Convert to axis-angle representation.
    public var axisAngle: (axis: Vec3, angle: Double) {
        let angle = 2.0 * acos(real)
        let s = sqrt(1.0 - real * real)
        let axis: Vec3
        if s < 0.001 {
            axis = Vec3(1, 0, 0)
        } else {
            axis = Vec3(imag.x / s, imag.y / s, imag.z / s)
        }
        return (axis, angle)
    }

    /// Convert to 4x4 rotation matrix.
    public var matrix: Mat4 {
        let x = imag.x, y = imag.y, z = imag.z, w = real
        let xx = x * x, yy = y * y, zz = z * z
        let xy = x * y, xz = x * z, yz = y * z
        let wx = w * x, wy = w * y, wz = w * z

        return Mat4(columns: (
            Vec4(1 - 2*(yy + zz), 2*(xy + wz), 2*(xz - wy), 0),
            Vec4(2*(xy - wz), 1 - 2*(xx + zz), 2*(yz + wx), 0),
            Vec4(2*(xz + wy), 2*(yz - wx), 1 - 2*(xx + yy), 0),
            Vec4(0, 0, 0, 1)
        ))
    }

    /// Dot product of quaternions.
    public func dot(_ other: Quat) -> Double {
        real * other.real + imag.x * other.imag.x + imag.y * other.imag.y + imag.z * other.imag.z
    }
}

// MARK: - Mat4 Extensions

extension Mat4 {
    /// Create translation matrix.
    public static func translation(_ t: Vec3) -> Mat4 {
        Mat4(columns: (
            Vec4(1, 0, 0, 0),
            Vec4(0, 1, 0, 0),
            Vec4(0, 0, 1, 0),
            Vec4(t.x, t.y, t.z, 1)
        ))
    }

    /// Create rotation matrix around X axis.
    public static func rotationX(_ angle: Double) -> Mat4 {
        let c = cos(angle), s = sin(angle)
        return Mat4(columns: (
            Vec4(1, 0, 0, 0),
            Vec4(0, c, s, 0),
            Vec4(0, -s, c, 0),
            Vec4(0, 0, 0, 1)
        ))
    }

    /// Create rotation matrix around Y axis.
    public static func rotationY(_ angle: Double) -> Mat4 {
        let c = cos(angle), s = sin(angle)
        return Mat4(columns: (
            Vec4(c, 0, -s, 0),
            Vec4(0, 1, 0, 0),
            Vec4(s, 0, c, 0),
            Vec4(0, 0, 0, 1)
        ))
    }

    /// Create rotation matrix around Z axis.
    public static func rotationZ(_ angle: Double) -> Mat4 {
        let c = cos(angle), s = sin(angle)
        return Mat4(columns: (
            Vec4(c, s, 0, 0),
            Vec4(-s, c, 0, 0),
            Vec4(0, 0, 1, 0),
            Vec4(0, 0, 0, 1)
        ))
    }

    /// Create rotation matrix around arbitrary axis.
    public static func rotation(angle: Double, axis: Vec3) -> Mat4 {
        let q = Quat(angle: angle, axis: simd_normalize(axis))
        return q.matrix
    }

    /// Create scale matrix.
    public static func scale(_ s: Vec3) -> Mat4 {
        Mat4(diagonal: Vec4(s.x, s.y, s.z, 1))
    }

    /// Apply transformation to a 3D point.
    public func apply(_ point: Vec3) -> Vec3 {
        let v4 = self * Vec4(point.x, point.y, point.z, 1)
        return Vec3(v4.x, v4.y, v4.z)
    }
}

// MARK: - Coordinate Conversions

/// Convert degrees to radians.
///
/// - Parameter degrees: Angle in degrees
/// - Returns: Angle in radians
public func deg2rad(_ degrees: Double) -> Double {
    degrees * .pi / 180.0
}

/// Convert radians to degrees.
///
/// - Parameter radians: Angle in radians
/// - Returns: Angle in degrees
public func rad2deg(_ radians: Double) -> Double {
    radians * 180.0 / .pi
}

/// Convert polar coordinates to Cartesian.
///
/// - Parameters:
///   - r: Radius
///   - theta: Angle in radians
/// - Returns: Tuple (x, y)
public func polarToCart(r: Double, theta: Double) -> (x: Double, y: Double) {
    (r * cos(theta), r * sin(theta))
}

/// Convert Cartesian coordinates to polar.
///
/// - Parameters:
///   - x: X coordinate
///   - y: Y coordinate
/// - Returns: Tuple (r, theta) where theta is in radians
public func cartToPolar(x: Double, y: Double) -> (r: Double, theta: Double) {
    (sqrt(x * x + y * y), atan2(y, x))
}

/// Convert spherical coordinates to Cartesian.
///
/// Uses physics convention:
/// - theta: polar angle (from z-axis) in radians
/// - phi: azimuthal angle (from x-axis in xy-plane) in radians
///
/// - Parameters:
///   - r: Radius
///   - theta: Polar angle (from z-axis) in radians
///   - phi: Azimuthal angle (from x-axis in xy-plane) in radians
/// - Returns: Tuple (x, y, z)
public func sphericalToCart(r: Double, theta: Double, phi: Double) -> (x: Double, y: Double, z: Double) {
    let sinTheta = sin(theta)
    let x = r * sinTheta * cos(phi)
    let y = r * sinTheta * sin(phi)
    let z = r * cos(theta)
    return (x, y, z)
}

/// Convert Cartesian coordinates to spherical.
///
/// Uses physics convention:
/// - theta: polar angle (from z-axis) in radians
/// - phi: azimuthal angle (from x-axis in xy-plane) in radians
///
/// - Parameters:
///   - x: X coordinate
///   - y: Y coordinate
///   - z: Z coordinate
/// - Returns: Tuple (r, theta, phi) where angles are in radians
public func cartToSpherical(x: Double, y: Double, z: Double) -> (r: Double, theta: Double, phi: Double) {
    let r = sqrt(x * x + y * y + z * z)
    let theta = r > 0 ? acos(z / r) : 0
    let phi = atan2(y, x)
    return (r, theta, phi)
}

// MARK: - Geometric Calculations

/// Compute distance between two 2D points.
public func distance2D(_ a: Vec2, _ b: Vec2) -> Double {
    simd_distance(a, b)
}

/// Compute distance between two 3D points.
public func distance3D(_ a: Vec3, _ b: Vec3) -> Double {
    simd_distance(a, b)
}

/// Compute angle between two 2D vectors.
public func angleBetween2D(_ a: Vec2, _ b: Vec2) -> Double {
    let dot = simd_dot(a, b)
    let lenA = simd_length(a)
    let lenB = simd_length(b)
    guard lenA > 0 && lenB > 0 else { return 0 }
    return acos(max(-1, min(1, dot / (lenA * lenB))))
}

/// Compute angle between two 3D vectors.
public func angleBetween3D(_ a: Vec3, _ b: Vec3) -> Double {
    let dot = simd_dot(a, b)
    let lenA = simd_length(a)
    let lenB = simd_length(b)
    guard lenA > 0 && lenB > 0 else { return 0 }
    return acos(max(-1, min(1, dot / (lenA * lenB))))
}

/// Compute convex hull of 2D points using Graham scan.
/// Returns points in counter-clockwise order.
public func convexHull2D(_ points: [Vec2]) -> [Vec2] {
    guard points.count >= 3 else { return points }

    // Find bottom-most point (leftmost if tie)
    var start = points[0]
    var startIdx = 0
    for (i, p) in points.enumerated() {
        if p.y < start.y || (p.y == start.y && p.x < start.x) {
            start = p
            startIdx = i
        }
    }

    var remaining = points
    remaining.remove(at: startIdx)

    // Sort by polar angle
    remaining.sort { a, b in
        let angleA = atan2(a.y - start.y, a.x - start.x)
        let angleB = atan2(b.y - start.y, b.x - start.x)
        if abs(angleA - angleB) < 1e-10 {
            return simd_distance_squared(start, a) < simd_distance_squared(start, b)
        }
        return angleA < angleB
    }

    // Graham scan
    var hull = [start]

    func ccw(_ a: Vec2, _ b: Vec2, _ c: Vec2) -> Double {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }

    for p in remaining {
        while hull.count >= 2 && ccw(hull[hull.count - 2], hull[hull.count - 1], p) <= 0 {
            hull.removeLast()
        }
        hull.append(p)
    }

    return hull
}

/// Test if a point is inside a polygon using ray casting.
public func pointInPolygon(_ point: Vec2, _ polygon: [Vec2]) -> Bool {
    guard polygon.count >= 3 else { return false }

    var inside = false
    var j = polygon.count - 1
    for i in 0..<polygon.count {
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y)) &&
           (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x) {
            inside = !inside
        }
        j = i
    }
    return inside
}

/// Line-line intersection in 2D.
/// Returns nil if lines are parallel.
public func lineIntersection2D(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Vec2? {
    let d1 = p2 - p1
    let d2 = p4 - p3
    let cross = d1.x * d2.y - d1.y * d2.x

    guard abs(cross) > 1e-10 else { return nil }

    let d3 = p3 - p1
    let t = (d3.x * d2.y - d3.y * d2.x) / cross

    return p1 + d1 * t
}

/// Compute area of a triangle.
public func triangleArea2D(_ p1: Vec2, _ p2: Vec2, _ p3: Vec2) -> Double {
    abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2.0
}

/// Compute area of a 3D triangle.
public func triangleArea3D(_ p1: Vec3, _ p2: Vec3, _ p3: Vec3) -> Double {
    simd_length(simd_cross(p2 - p1, p3 - p1)) / 2.0
}

/// Compute centroid of 2D points.
public func centroid2D(_ points: [Vec2]) -> Vec2? {
    guard !points.isEmpty else { return nil }
    var sum = Vec2.zero
    for p in points {
        sum += p
    }
    return sum / Double(points.count)
}

/// Compute centroid of 3D points.
public func centroid3D(_ points: [Vec3]) -> Vec3? {
    guard !points.isEmpty else { return nil }
    var sum = Vec3.zero
    for p in points {
        sum += p
    }
    return sum / Double(points.count)
}

/// Result of circle from 3 points.
public struct CircleResult {
    public let center: Vec2
    public let radius: Double
}

/// Compute circle through 3 points.
/// Returns nil if points are collinear.
public func circleFrom3Points(_ p1: Vec2, _ p2: Vec2, _ p3: Vec2) -> CircleResult? {
    let ax = p1.x, ay = p1.y
    let bx = p2.x, by = p2.y
    let cx = p3.x, cy = p3.y

    let d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    guard abs(d) > 1e-10 else { return nil }

    let ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
    let uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
    let center = Vec2(ux, uy)
    let radius = simd_distance(center, p1)

    return CircleResult(center: center, radius: radius)
}

/// Result of plane from 3 points.
public struct PlaneResult {
    public let normal: Vec3
    public let d: Double  // ax + by + cz + d = 0

    public init(normal: Vec3, d: Double) {
        self.normal = normal
        self.d = d
    }
}

/// Compute plane through 3 points.
public func planeFrom3Points(_ p1: Vec3, _ p2: Vec3, _ p3: Vec3) -> PlaneResult? {
    let v1 = p2 - p1
    let v2 = p3 - p1
    let normal = simd_normalize(simd_cross(v1, v2))
    guard !normal.x.isNaN else { return nil }
    let d = -simd_dot(normal, p1)
    return PlaneResult(normal: normal, d: d)
}

/// Compute signed distance from point to plane.
public func pointPlaneDistance(_ point: Vec3, _ plane: PlaneResult) -> Double {
    simd_dot(plane.normal, point) + plane.d
}

/// Line-plane intersection.
/// Returns nil if line is parallel to plane.
public func linePlaneIntersection(linePoint: Vec3, lineDir: Vec3, plane: PlaneResult) -> Vec3? {
    let denom = simd_dot(plane.normal, lineDir)
    guard abs(denom) > 1e-10 else { return nil }

    let t = -(simd_dot(plane.normal, linePoint) + plane.d) / denom
    return linePoint + lineDir * t
}

/// Plane-plane intersection.
/// Returns a line (point + direction) or nil if parallel.
public func planePlaneIntersection(_ plane1: PlaneResult, _ plane2: PlaneResult) -> (point: Vec3, direction: Vec3)? {
    let direction = simd_cross(plane1.normal, plane2.normal)
    let lenSq = simd_length_squared(direction)
    guard lenSq > 1e-10 else { return nil }

    // Find a point on the intersection line
    let n1 = plane1.normal, d1 = plane1.d
    let n2 = plane2.normal, d2 = plane2.d

    // Solve for point where planes intersect (set one coordinate to 0)
    let absDir = Vec3(abs(direction.x), abs(direction.y), abs(direction.z))
    var point = Vec3.zero

    if absDir.z >= absDir.x && absDir.z >= absDir.y {
        // Set z = 0, solve for x, y
        let det = n1.x * n2.y - n1.y * n2.x
        if abs(det) > 1e-10 {
            point = Vec3(
                (-d1 * n2.y + d2 * n1.y) / det,
                (-n1.x * d2 + n2.x * d1) / det,
                0
            )
        }
    } else if absDir.y >= absDir.x {
        // Set y = 0
        let det = n1.x * n2.z - n1.z * n2.x
        if abs(det) > 1e-10 {
            point = Vec3(
                (-d1 * n2.z + d2 * n1.z) / det,
                0,
                (-n1.x * d2 + n2.x * d1) / det
            )
        }
    } else {
        // Set x = 0
        let det = n1.y * n2.z - n1.z * n2.y
        if abs(det) > 1e-10 {
            point = Vec3(
                0,
                (-d1 * n2.z + d2 * n1.z) / det,
                (-n1.y * d2 + n2.y * d1) / det
            )
        }
    }

    return (point, simd_normalize(direction))
}

/// Result of sphere from 4 points.
public struct SphereResult {
    public let center: Vec3
    public let radius: Double
}

/// Compute sphere through 4 points.
/// Returns nil if points are coplanar.
public func sphereFrom4Points(_ p1: Vec3, _ p2: Vec3, _ p3: Vec3, _ p4: Vec3) -> SphereResult? {
    // Use determinant method
    let a = p2 - p1
    let b = p3 - p1
    let c = p4 - p1

    let det = simd_dot(a, simd_cross(b, c))
    guard abs(det) > 1e-10 else { return nil }

    let a2 = simd_length_squared(a)
    let b2 = simd_length_squared(b)
    let c2 = simd_length_squared(c)

    let u = simd_cross(b, c) * a2 + simd_cross(c, a) * b2 + simd_cross(a, b) * c2
    let center = p1 + u / (2.0 * det)
    let radius = simd_distance(center, p1)

    return SphereResult(center: center, radius: radius)
}

// MARK: - Circle Fitting

/// Result of circle fitting.
public struct CircleFitResult {
    public let center: Vec2
    public let radius: Double
    public let residuals: [Double]
}

/// Fit circle to points using algebraic method (Kasa).
public func circleFitAlgebraic(_ points: [Vec2]) -> CircleFitResult? {
    let n = points.count
    guard n >= 3 else { return nil }

    // Build matrices for Ax = b
    // Equation: x² + y² = 2*cx*x + 2*cy*y + (r² - cx² - cy²)
    var sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumYY = 0.0, sumXY = 0.0
    var sumXXX = 0.0, sumYYY = 0.0, sumXYY = 0.0, sumXXY = 0.0

    for p in points {
        let x = p.x, y = p.y
        sumX += x
        sumY += y
        sumXX += x * x
        sumYY += y * y
        sumXY += x * y
        sumXXX += x * x * x
        sumYYY += y * y * y
        sumXYY += x * y * y
        sumXXY += x * x * y
    }

    let nd = Double(n)

    // Solve 3x3 system
    let a11 = sumXX, a12 = sumXY, a13 = sumX
    let a21 = sumXY, a22 = sumYY, a23 = sumY
    let a31 = sumX, a32 = sumY, a33 = nd

    let b1 = sumXXX + sumXYY
    let b2 = sumXXY + sumYYY
    let b3 = sumXX + sumYY

    // Cramer's rule
    let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
    guard abs(det) > 1e-10 else { return nil }

    let detA = b1 * (a22 * a33 - a23 * a32) - a12 * (b2 * a33 - a23 * b3) + a13 * (b2 * a32 - a22 * b3)
    let detB = a11 * (b2 * a33 - a23 * b3) - b1 * (a21 * a33 - a23 * a31) + a13 * (a21 * b3 - b2 * a31)
    let detC = a11 * (a22 * b3 - b2 * a32) - a12 * (a21 * b3 - b2 * a31) + b1 * (a21 * a32 - a22 * a31)

    let A = detA / det
    let B = detB / det
    let C = detC / det

    let cx = A / 2.0
    let cy = B / 2.0
    let radius = sqrt(C + cx * cx + cy * cy)

    let center = Vec2(cx, cy)

    // Compute residuals
    var residuals = [Double]()
    for p in points {
        let dist = simd_distance(p, center)
        residuals.append(dist - radius)
    }

    return CircleFitResult(center: center, radius: radius, residuals: residuals)
}

/// Fit circle using Taubin method (more robust).
public func circleFitTaubin(_ points: [Vec2]) -> CircleFitResult? {
    let n = points.count
    guard n >= 3 else { return nil }

    // Center data
    var meanX = 0.0, meanY = 0.0
    for p in points {
        meanX += p.x
        meanY += p.y
    }
    meanX /= Double(n)
    meanY /= Double(n)

    // Compute moments
    var Mxx = 0.0, Myy = 0.0, Mxy = 0.0
    var Mxz = 0.0, Myz = 0.0, Mzz = 0.0

    for p in points {
        let xi = p.x - meanX
        let yi = p.y - meanY
        let zi = xi * xi + yi * yi

        Mxx += xi * xi
        Myy += yi * yi
        Mxy += xi * yi
        Mxz += xi * zi
        Myz += yi * zi
        Mzz += zi * zi
    }

    Mxx /= Double(n)
    Myy /= Double(n)
    Mxy /= Double(n)
    Mxz /= Double(n)
    Myz /= Double(n)
    Mzz /= Double(n)

    // Coefficients of characteristic polynomial
    let Mz = Mxx + Myy
    let Cov_xy = Mxx * Myy - Mxy * Mxy
    let Var_z = Mzz - Mz * Mz

    let A3 = 4.0 * Mz
    let A2 = -3.0 * Mz * Mz - Mzz
    let A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    let A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy

    // Newton's method for finding root
    var y = A0
    var x = 0.0

    for _ in 0..<20 {
        let Dy = A1 + x * (2.0 * A2 + x * 3.0 * A3)
        let xnew = x - y / Dy
        if abs(xnew - x) < 1e-12 { break }
        x = xnew
        y = A0 + x * (A1 + x * (A2 + x * A3))
    }

    // Compute parameters
    let det = x * x - x * Mz + Cov_xy
    guard abs(det) > 1e-10 else { return nil }

    let cx = (Mxz * (Myy - x) - Myz * Mxy) / det / 2.0 + meanX
    let cy = (Myz * (Mxx - x) - Mxz * Mxy) / det / 2.0 + meanY
    let radius = sqrt((cx - meanX) * (cx - meanX) + (cy - meanY) * (cy - meanY) + Mz + 2.0 * x)

    let center = Vec2(cx, cy)

    // Compute residuals
    var residuals = [Double]()
    for p in points {
        let dist = simd_distance(p, center)
        residuals.append(dist - radius)
    }

    return CircleFitResult(center: center, radius: radius, residuals: residuals)
}

// MARK: - Sphere Fitting

/// Result of sphere fitting.
public struct SphereFitResult {
    public let center: Vec3
    public let radius: Double
    public let residuals: [Double]
}

/// Fit sphere to points using algebraic method.
public func sphereFitAlgebraic(_ points: [Vec3]) -> SphereFitResult? {
    let n = points.count
    guard n >= 4 else { return nil }

    // Build normal equations: Ax = b
    // x² + y² + z² = 2*cx*x + 2*cy*y + 2*cz*z + (r² - cx² - cy² - cz²)
    var sumX = 0.0, sumY = 0.0, sumZ = 0.0
    var sumXX = 0.0, sumYY = 0.0, sumZZ = 0.0
    var sumXY = 0.0, sumXZ = 0.0, sumYZ = 0.0
    var sumXXX = 0.0, sumYYY = 0.0, sumZZZ = 0.0
    var sumXYY = 0.0, sumXZZ = 0.0
    var sumYXX = 0.0, sumYZZ = 0.0
    var sumZXX = 0.0, sumZYY = 0.0

    for p in points {
        let x = p.x, y = p.y, z = p.z
        sumX += x
        sumY += y
        sumZ += z
        sumXX += x * x
        sumYY += y * y
        sumZZ += z * z
        sumXY += x * y
        sumXZ += x * z
        sumYZ += y * z
        sumXXX += x * x * x
        sumYYY += y * y * y
        sumZZZ += z * z * z
        sumXYY += x * y * y
        sumXZZ += x * z * z
        sumYXX += y * x * x
        sumYZZ += y * z * z
        sumZXX += z * x * x
        sumZYY += z * y * y
    }

    let nd = Double(n)

    // Build 4x4 system
    var A = [[Double]](repeating: [Double](repeating: 0, count: 4), count: 4)
    var b = [Double](repeating: 0, count: 4)

    A[0][0] = sumXX; A[0][1] = sumXY; A[0][2] = sumXZ; A[0][3] = sumX
    A[1][0] = sumXY; A[1][1] = sumYY; A[1][2] = sumYZ; A[1][3] = sumY
    A[2][0] = sumXZ; A[2][1] = sumYZ; A[2][2] = sumZZ; A[2][3] = sumZ
    A[3][0] = sumX;  A[3][1] = sumY;  A[3][2] = sumZ;  A[3][3] = nd

    b[0] = sumXXX + sumXYY + sumXZZ
    b[1] = sumYXX + sumYYY + sumYZZ
    b[2] = sumZXX + sumZYY + sumZZZ
    b[3] = sumXX + sumYY + sumZZ

    // Gaussian elimination
    guard let solution = solveLinearSystem4x4(A, b) else { return nil }

    let cx = solution[0] / 2.0
    let cy = solution[1] / 2.0
    let cz = solution[2] / 2.0
    let radius = sqrt(solution[3] + cx * cx + cy * cy + cz * cz)

    let center = Vec3(cx, cy, cz)

    // Compute residuals
    var residuals = [Double]()
    for p in points {
        let dist = simd_distance(p, center)
        residuals.append(dist - radius)
    }

    return SphereFitResult(center: center, radius: radius, residuals: residuals)
}

/// Solve 4x4 linear system using Gaussian elimination.
private func solveLinearSystem4x4(_ A: [[Double]], _ b: [Double]) -> [Double]? {
    var aug = A
    var rhs = b

    // Forward elimination
    for i in 0..<4 {
        // Find pivot
        var maxIdx = i
        for j in (i+1)..<4 {
            if abs(aug[j][i]) > abs(aug[maxIdx][i]) {
                maxIdx = j
            }
        }

        // Swap rows
        if maxIdx != i {
            aug.swapAt(i, maxIdx)
            rhs.swapAt(i, maxIdx)
        }

        guard abs(aug[i][i]) > 1e-10 else { return nil }

        // Eliminate
        for j in (i+1)..<4 {
            let factor = aug[j][i] / aug[i][i]
            for k in i..<4 {
                aug[j][k] -= factor * aug[i][k]
            }
            rhs[j] -= factor * rhs[i]
        }
    }

    // Back substitution
    var x = [Double](repeating: 0, count: 4)
    for i in stride(from: 3, through: 0, by: -1) {
        var sum = rhs[i]
        for j in (i+1)..<4 {
            sum -= aug[i][j] * x[j]
        }
        x[i] = sum / aug[i][i]
    }

    return x
}

// MARK: - B-Spline

/// Evaluate B-spline basis function.
public func bsplineBasis(i: Int, degree: Int, t: Double, knots: [Double]) -> Double {
    if degree == 0 {
        return (knots[i] <= t && t < knots[i + 1]) ? 1.0 : 0.0
    }

    var result = 0.0

    let denom1 = knots[i + degree] - knots[i]
    if abs(denom1) > 1e-10 {
        result += (t - knots[i]) / denom1 * bsplineBasis(i: i, degree: degree - 1, t: t, knots: knots)
    }

    let denom2 = knots[i + degree + 1] - knots[i + 1]
    if abs(denom2) > 1e-10 {
        result += (knots[i + degree + 1] - t) / denom2 * bsplineBasis(i: i + 1, degree: degree - 1, t: t, knots: knots)
    }

    return result
}

/// Generate uniform knot vector for B-spline.
public func bsplineUniformKnots(n: Int, degree: Int) -> [Double] {
    let m = n + degree + 1
    var knots = [Double](repeating: 0, count: m)

    for i in 0..<m {
        if i <= degree {
            knots[i] = 0.0
        } else if i >= m - degree - 1 {
            knots[i] = 1.0
        } else {
            knots[i] = Double(i - degree) / Double(n - degree)
        }
    }

    return knots
}

/// Evaluate B-spline curve at parameter t.
public func bsplineEvaluate(controlPoints: [Vec2], degree: Int, t: Double, knots: [Double]? = nil) -> Vec2 {
    let n = controlPoints.count
    let actualKnots = knots ?? bsplineUniformKnots(n: n, degree: degree)

    var result = Vec2.zero
    for i in 0..<n {
        let basis = bsplineBasis(i: i, degree: degree, t: t, knots: actualKnots)
        result += controlPoints[i] * basis
    }
    return result
}

/// Evaluate B-spline curve at parameter t (3D).
public func bsplineEvaluate3D(controlPoints: [Vec3], degree: Int, t: Double, knots: [Double]? = nil) -> Vec3 {
    let n = controlPoints.count
    let actualKnots = knots ?? bsplineUniformKnots(n: n, degree: degree)

    var result = Vec3.zero
    for i in 0..<n {
        let basis = bsplineBasis(i: i, degree: degree, t: t, knots: actualKnots)
        result += controlPoints[i] * basis
    }
    return result
}

/// Evaluate B-spline derivative at parameter t.
public func bsplineDerivative(controlPoints: [Vec2], degree: Int, t: Double, knots: [Double]? = nil) -> Vec2 {
    let n = controlPoints.count
    guard degree >= 1, n >= 2 else { return Vec2.zero }

    let actualKnots = knots ?? bsplineUniformKnots(n: n, degree: degree)

    // Compute derivative control points
    var derivCP = [Vec2]()
    for i in 0..<(n - 1) {
        let denom = actualKnots[i + degree + 1] - actualKnots[i + 1]
        if abs(denom) > 1e-10 {
            let dp = (controlPoints[i + 1] - controlPoints[i]) * (Double(degree) / denom)
            derivCP.append(dp)
        } else {
            derivCP.append(Vec2.zero)
        }
    }

    // Evaluate derivative B-spline
    return bsplineEvaluate(controlPoints: derivCP, degree: degree - 1, t: t, knots: Array(actualKnots.dropFirst().dropLast()))
}

// MARK: - Ellipse Fitting

/// Result of ellipse fitting.
public struct EllipseFitResult {
    /// Center x-coordinate.
    public let cx: Double
    /// Center y-coordinate.
    public let cy: Double
    /// Semi-major axis length.
    public let a: Double
    /// Semi-minor axis length.
    public let b: Double
    /// Rotation angle (radians).
    public let theta: Double
    /// Conic coefficients [A, B, C, D, E, F] for Ax² + Bxy + Cy² + Dx + Ey + F = 0.
    public let conic: [Double]
    /// Residuals for each input point.
    public let residuals: [Double]
    /// Root mean square error.
    public let rmse: Double
}

/// Fit ellipse to 2D points using Fitzgibbon's direct least squares method.
/// Guarantees result is always an ellipse (not hyperbola/parabola).
/// - Parameter points: Array of 2D points (minimum 5 required).
/// - Returns: Ellipse fit result, or nil if fitting fails.
public func ellipseFitDirect(points: [Vec2]) -> EllipseFitResult? {
    guard points.count >= 5 else { return nil }

    let n = points.count

    // Center the data for numerical stability
    var meanX = 0.0, meanY = 0.0
    for p in points {
        meanX += p.x
        meanY += p.y
    }
    meanX /= Double(n)
    meanY /= Double(n)

    // Create centered points
    let centered = points.map { ($0.x - meanX, $0.y - meanY) }

    // Compute scatter matrix S = D' * D (6x6)
    var S = [[Double]](repeating: [Double](repeating: 0.0, count: 6), count: 6)

    for p in centered {
        let x = p.0, y = p.1
        let x2 = x * x
        let y2 = y * y
        let xy = x * y

        let d = [x2, xy, y2, x, y, 1.0]

        for i in 0..<6 {
            for j in 0..<6 {
                S[i][j] += d[i] * d[j]
            }
        }
    }

    // Partition S into blocks
    var S1 = [[Double]](repeating: [Double](repeating: 0.0, count: 3), count: 3)
    var S2 = [[Double]](repeating: [Double](repeating: 0.0, count: 3), count: 3)
    var S3 = [[Double]](repeating: [Double](repeating: 0.0, count: 3), count: 3)

    for i in 0..<3 {
        for j in 0..<3 {
            S1[i][j] = S[i][j]
            S2[i][j] = S[i][j + 3]
            S3[i][j] = S[i + 3][j + 3]
        }
    }

    // Constraint matrix C1 for 4ac - b² = 1
    let C1: [[Double]] = [[0, 0, 2], [0, -1, 0], [2, 0, 0]]

    // Compute inverse of S3
    guard let S3inv = invert3x3(S3) else { return nil }

    // Compute T = -S3^(-1) * S2'
    var T = [[Double]](repeating: [Double](repeating: 0.0, count: 3), count: 3)
    for i in 0..<3 {
        for j in 0..<3 {
            for k in 0..<3 {
                T[i][j] -= S3inv[i][k] * S2[j][k]
            }
        }
    }

    // Compute M = S1 + S2 * T
    var M = [[Double]](repeating: [Double](repeating: 0.0, count: 3), count: 3)
    for i in 0..<3 {
        for j in 0..<3 {
            M[i][j] = S1[i][j]
            for k in 0..<3 {
                M[i][j] += S2[i][k] * T[k][j]
            }
        }
    }

    // Compute C1^(-1) * M
    guard let C1inv = invert3x3(C1) else { return nil }

    var C1invM = [[Double]](repeating: [Double](repeating: 0.0, count: 3), count: 3)
    for i in 0..<3 {
        for j in 0..<3 {
            for k in 0..<3 {
                C1invM[i][j] += C1inv[i][k] * M[k][j]
            }
        }
    }

    // Find eigenvalues/eigenvectors
    guard let (eigenvalues, eigenvectors) = eigenDecomposition3x3(C1invM) else { return nil }

    // Find eigenvector for smallest positive eigenvalue with valid ellipse constraint
    var bestIdx = -1
    var bestEig = Double.infinity
    for i in 0..<min(eigenvalues.count, eigenvectors.count) {
        guard eigenvectors[i].count >= 3 else { continue }
        let a = eigenvectors[i][0]
        let b = eigenvectors[i][1]
        let c = eigenvectors[i][2]
        let constraint = 4 * a * c - b * b

        if constraint > 0 && eigenvalues[i] < bestEig && eigenvalues[i] > -1e-10 {
            bestEig = eigenvalues[i]
            bestIdx = i
        }
    }

    guard bestIdx >= 0 else { return nil }

    // Get conic coefficients
    let a1 = [eigenvectors[bestIdx][0], eigenvectors[bestIdx][1], eigenvectors[bestIdx][2]]

    // Recover linear coefficients
    var a2 = [0.0, 0.0, 0.0]
    for i in 0..<3 {
        for j in 0..<3 {
            a2[i] += T[i][j] * a1[j]
        }
    }

    let A = a1[0], B = a1[1], C = a1[2]
    let D0 = a2[0], E0 = a2[1], F0 = a2[2]

    // Un-center
    let D = D0 - 2 * A * meanX - B * meanY
    let E = E0 - 2 * C * meanY - B * meanX
    let F = F0 + A * meanX * meanX + C * meanY * meanY + B * meanX * meanY - D0 * meanX - E0 * meanY

    // Convert to parametric form
    let det = 4 * A * C - B * B
    guard abs(det) > 1e-15 else { return nil }

    let cx = (B * E - 2 * C * D) / det
    let cy = (B * D - 2 * A * E) / det

    let Fc = A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy + F

    // Rotation angle
    let theta: Double
    if abs(A - C) < 1e-15 {
        theta = B > 0 ? Double.pi / 4 : -Double.pi / 4
    } else {
        theta = 0.5 * atan2(B, A - C)
    }

    // Semi-axes
    let cos2t = cos(theta) * cos(theta)
    let sin2t = sin(theta) * sin(theta)
    let sincos = sin(theta) * cos(theta)

    let Ap = A * cos2t + B * sincos + C * sin2t
    let Cp = A * sin2t - B * sincos + C * cos2t

    guard Ap * Fc < 0 && Cp * Fc < 0 else { return nil }

    let semiMajor = sqrt(-Fc / min(Ap, Cp))
    let semiMinor = sqrt(-Fc / max(Ap, Cp))

    var finalTheta = theta
    if Ap > Cp {
        finalTheta += Double.pi / 2
    }

    // Normalize angle
    while finalTheta > Double.pi / 2 { finalTheta -= Double.pi }
    while finalTheta < -Double.pi / 2 { finalTheta += Double.pi }

    // Compute residuals
    var residuals = [Double]()
    var sumResidualsSq = 0.0
    for p in points {
        let val = A * p.x * p.x + B * p.x * p.y + C * p.y * p.y + D * p.x + E * p.y + F
        let grad = sqrt(pow(2 * A * p.x + B * p.y + D, 2) + pow(2 * C * p.y + B * p.x + E, 2))
        let residual = grad > 1e-10 ? val / grad : val
        residuals.append(residual)
        sumResidualsSq += residual * residual
    }

    let rmse = sqrt(sumResidualsSq / Double(n))

    return EllipseFitResult(
        cx: cx, cy: cy, a: semiMajor, b: semiMinor, theta: finalTheta,
        conic: [A, B, C, D, E, F], residuals: residuals, rmse: rmse
    )
}

// MARK: - Cubic Spline Coefficients

/// Cubic spline segment coefficients.
public struct CubicSplineSegment {
    /// Constant term.
    public let a: Double
    /// Linear coefficient.
    public let b: Double
    /// Quadratic coefficient.
    public let c: Double
    /// Cubic coefficient.
    public let d: Double
}

/// Result of cubic spline coefficient computation.
public struct CubicSplineCoeffs {
    /// Knot positions (x values).
    public let knots: [Double]
    /// Values at knots (y values).
    public let values: [Double]
    /// Coefficients for each segment.
    public let coeffs: [CubicSplineSegment]
}

/// Compute cubic spline coefficients using natural boundary conditions.
/// - Parameter points: Array of (x, y) points, must have at least 2 points.
/// - Returns: Spline coefficients, or nil if computation fails.
public func cubicSplineCoeffs(points: [(x: Double, y: Double)]) -> CubicSplineCoeffs? {
    guard points.count >= 2 else { return nil }

    let sortedPoints = points.sorted { $0.x < $1.x }
    let n = sortedPoints.count - 1

    // Handle 2-point case (linear)
    if n == 1 {
        let h = sortedPoints[1].x - sortedPoints[0].x
        let slope = (sortedPoints[1].y - sortedPoints[0].y) / h
        return CubicSplineCoeffs(
            knots: sortedPoints.map { $0.x },
            values: sortedPoints.map { $0.y },
            coeffs: [CubicSplineSegment(a: sortedPoints[0].y, b: slope, c: 0, d: 0)]
        )
    }

    // Calculate interval widths
    var h = [Double](repeating: 0, count: n)
    for i in 0..<n {
        h[i] = sortedPoints[i + 1].x - sortedPoints[i].x
    }

    // Build tridiagonal system for natural cubic spline
    let systemSize = n - 1
    guard systemSize > 0 else { return nil }

    var dl = [Double](repeating: 0, count: systemSize - 1)
    var d = [Double](repeating: 0, count: systemSize)
    var du = [Double](repeating: 0, count: systemSize - 1)
    var b = [Double](repeating: 0, count: systemSize)

    for i in 0..<systemSize {
        let hi = h[i]
        let hi1 = h[i + 1]
        d[i] = 2 * (hi + hi1)

        let di0 = (sortedPoints[i + 1].y - sortedPoints[i].y) / hi
        let di1 = (sortedPoints[i + 2].y - sortedPoints[i + 1].y) / hi1
        b[i] = 6 * (di1 - di0)

        if i > 0 { dl[i - 1] = hi }
        if i < systemSize - 1 { du[i] = hi1 }
    }

    // Solve tridiagonal system using Thomas algorithm
    // Forward sweep
    for i in 1..<systemSize {
        let w = dl[i - 1] / d[i - 1]
        d[i] -= w * du[i - 1]
        b[i] -= w * b[i - 1]
    }

    // Back substitution
    b[systemSize - 1] /= d[systemSize - 1]
    for i in stride(from: systemSize - 2, through: 0, by: -1) {
        b[i] = (b[i] - du[i] * b[i + 1]) / d[i]
    }

    // M[0] = 0 and M[n] = 0 (natural spline)
    var M = [Double](repeating: 0, count: n + 1)
    for i in 0..<systemSize {
        M[i + 1] = b[i]
    }

    // Calculate coefficients
    var coeffs = [CubicSplineSegment]()
    for i in 0..<n {
        let hi = h[i]
        let yi = sortedPoints[i].y
        let yi1 = sortedPoints[i + 1].y
        let mi = M[i]
        let mi1 = M[i + 1]

        let a = yi
        let bi = (yi1 - yi) / hi - hi * (2 * mi + mi1) / 6
        let c = mi / 2
        let di = (mi1 - mi) / (6 * hi)

        coeffs.append(CubicSplineSegment(a: a, b: bi, c: c, d: di))
    }

    return CubicSplineCoeffs(
        knots: sortedPoints.map { $0.x },
        values: sortedPoints.map { $0.y },
        coeffs: coeffs
    )
}

// MARK: - B-Spline Fitting

/// Result of B-spline least squares fitting.
public struct BSplineFitResult {
    /// Fitted control points.
    public let controlPoints: [Vec2]
    /// Knot vector.
    public let knots: [Double]
    /// Spline degree.
    public let degree: Int
    /// Residuals (distances from data points to fitted curve).
    public let residuals: [Double]
    /// Root mean square error.
    public let rmse: Double
    /// Maximum error.
    public let maxError: Double
    /// Parameter values for each data point.
    public let parameters: [Double]
}

/// Result of 3D B-spline least squares fitting.
public struct BSplineFitResult3D {
    /// Fitted control points.
    public let controlPoints: [Vec3]
    /// Knot vector.
    public let knots: [Double]
    /// Spline degree.
    public let degree: Int
    /// Residuals (distances from data points to fitted curve).
    public let residuals: [Double]
    /// Root mean square error.
    public let rmse: Double
    /// Maximum error.
    public let maxError: Double
    /// Parameter values for each data point.
    public let parameters: [Double]
}

/// Parameterization method for B-spline fitting.
public enum BSplineParameterization {
    case uniform
    case chordLength
    case centripetal
}

/// Fit B-spline to 2D data points using least squares.
/// - Parameters:
///   - points: Array of 2D data points.
///   - degree: B-spline degree (1-5).
///   - numControlPoints: Number of control points (must be >= degree + 1).
///   - parameterization: Method for computing parameter values.
/// - Returns: Fit result, or nil if fitting fails.
public func bsplineFit(
    points: [Vec2],
    degree: Int,
    numControlPoints: Int,
    parameterization: BSplineParameterization = .chordLength
) -> BSplineFitResult? {
    let nData = points.count
    guard nData >= numControlPoints, numControlPoints >= degree + 1, degree >= 1, degree <= 5 else {
        return nil
    }

    // Generate parameter values
    var t = [Double](repeating: 0, count: nData)

    switch parameterization {
    case .uniform:
        for i in 0..<nData {
            t[i] = Double(i) / Double(nData - 1)
        }
    case .chordLength, .centripetal:
        var chordLengths = [0.0]
        for i in 1..<nData {
            let dist = simd_distance(points[i], points[i-1])
            let d = parameterization == .centripetal ? sqrt(dist) : dist
            chordLengths.append(chordLengths.last! + d)
        }
        let totalLength = chordLengths.last!
        if totalLength > 1e-15 {
            for i in 0..<nData {
                t[i] = chordLengths[i] / totalLength
            }
        } else {
            for i in 0..<nData {
                t[i] = Double(i) / Double(nData - 1)
            }
        }
    }

    // Generate clamped uniform knot vector
    let m = numControlPoints + degree + 1
    var knots = [Double](repeating: 0, count: m)
    for i in 0..<m {
        if i <= degree {
            knots[i] = 0.0
        } else if i >= m - degree - 1 {
            knots[i] = 1.0
        } else {
            let interior = i - degree
            let numInterior = m - 2 * (degree + 1) + 1
            knots[i] = Double(interior) / Double(numInterior)
        }
    }

    // Build design matrix
    var A = [Double](repeating: 0, count: nData * numControlPoints)
    for i in 0..<nData {
        for j in 0..<numControlPoints {
            A[i + j * nData] = bsplineBasis(i: j, degree: degree, t: t[i], knots: knots)
        }
    }

    // Solve least squares for x and y separately using normal equations
    // A'A * x = A'b
    var AtA = [Double](repeating: 0, count: numControlPoints * numControlPoints)
    for i in 0..<numControlPoints {
        for j in 0..<numControlPoints {
            var sum = 0.0
            for k in 0..<nData {
                sum += A[k + i * nData] * A[k + j * nData]
            }
            AtA[i + j * numControlPoints] = sum
        }
    }

    // Solve for x coordinates
    var Atbx = [Double](repeating: 0, count: numControlPoints)
    for i in 0..<numControlPoints {
        var sum = 0.0
        for k in 0..<nData {
            sum += A[k + i * nData] * points[k].x
        }
        Atbx[i] = sum
    }

    guard let xCoords = solvePositiveDefinite(AtA, Atbx, n: numControlPoints) else { return nil }

    // Solve for y coordinates
    var Atby = [Double](repeating: 0, count: numControlPoints)
    for i in 0..<numControlPoints {
        var sum = 0.0
        for k in 0..<nData {
            sum += A[k + i * nData] * points[k].y
        }
        Atby[i] = sum
    }

    guard let yCoords = solvePositiveDefinite(AtA, Atby, n: numControlPoints) else { return nil }

    // Build control points
    var controlPoints = [Vec2]()
    for j in 0..<numControlPoints {
        controlPoints.append(Vec2(xCoords[j], yCoords[j]))
    }

    // Compute residuals
    var residuals = [Double]()
    var sumResidualsSq = 0.0
    var maxResidual = 0.0

    for i in 0..<nData {
        let fitted = bsplineEvaluate(controlPoints: controlPoints, degree: degree, t: t[i], knots: knots)
        let dist = simd_distance(fitted, points[i])
        residuals.append(dist)
        sumResidualsSq += dist * dist
        maxResidual = max(maxResidual, dist)
    }

    let rmse = sqrt(sumResidualsSq / Double(nData))

    return BSplineFitResult(
        controlPoints: controlPoints,
        knots: knots,
        degree: degree,
        residuals: residuals,
        rmse: rmse,
        maxError: maxResidual,
        parameters: t
    )
}

/// Fit B-spline to 3D data points using least squares.
public func bsplineFit3D(
    points: [Vec3],
    degree: Int,
    numControlPoints: Int,
    parameterization: BSplineParameterization = .chordLength
) -> BSplineFitResult3D? {
    let nData = points.count
    guard nData >= numControlPoints, numControlPoints >= degree + 1, degree >= 1, degree <= 5 else {
        return nil
    }

    // Generate parameter values
    var t = [Double](repeating: 0, count: nData)

    switch parameterization {
    case .uniform:
        for i in 0..<nData {
            t[i] = Double(i) / Double(nData - 1)
        }
    case .chordLength, .centripetal:
        var chordLengths = [0.0]
        for i in 1..<nData {
            let dist = simd_distance(points[i], points[i-1])
            let d = parameterization == .centripetal ? sqrt(dist) : dist
            chordLengths.append(chordLengths.last! + d)
        }
        let totalLength = chordLengths.last!
        if totalLength > 1e-15 {
            for i in 0..<nData {
                t[i] = chordLengths[i] / totalLength
            }
        } else {
            for i in 0..<nData {
                t[i] = Double(i) / Double(nData - 1)
            }
        }
    }

    // Generate knot vector
    let m = numControlPoints + degree + 1
    var knots = [Double](repeating: 0, count: m)
    for i in 0..<m {
        if i <= degree {
            knots[i] = 0.0
        } else if i >= m - degree - 1 {
            knots[i] = 1.0
        } else {
            let interior = i - degree
            let numInterior = m - 2 * (degree + 1) + 1
            knots[i] = Double(interior) / Double(numInterior)
        }
    }

    // Build design matrix
    var A = [Double](repeating: 0, count: nData * numControlPoints)
    for i in 0..<nData {
        for j in 0..<numControlPoints {
            A[i + j * nData] = bsplineBasis(i: j, degree: degree, t: t[i], knots: knots)
        }
    }

    // Build A'A
    var AtA = [Double](repeating: 0, count: numControlPoints * numControlPoints)
    for i in 0..<numControlPoints {
        for j in 0..<numControlPoints {
            var sum = 0.0
            for k in 0..<nData {
                sum += A[k + i * nData] * A[k + j * nData]
            }
            AtA[i + j * numControlPoints] = sum
        }
    }

    // Solve for each coordinate
    var coords = [[Double]](repeating: [], count: 3)
    for dim in 0..<3 {
        var Atb = [Double](repeating: 0, count: numControlPoints)
        for i in 0..<numControlPoints {
            var sum = 0.0
            for k in 0..<nData {
                let val: Double
                switch dim {
                case 0: val = points[k].x
                case 1: val = points[k].y
                default: val = points[k].z
                }
                sum += A[k + i * nData] * val
            }
            Atb[i] = sum
        }
        guard let solution = solvePositiveDefinite(AtA, Atb, n: numControlPoints) else { return nil }
        coords[dim] = solution
    }

    // Build control points
    var controlPoints = [Vec3]()
    for j in 0..<numControlPoints {
        controlPoints.append(Vec3(coords[0][j], coords[1][j], coords[2][j]))
    }

    // Compute residuals
    var residuals = [Double]()
    var sumResidualsSq = 0.0
    var maxResidual = 0.0

    for i in 0..<nData {
        let fitted = bsplineEvaluate3D(controlPoints: controlPoints, degree: degree, t: t[i], knots: knots)
        let dist = simd_distance(fitted, points[i])
        residuals.append(dist)
        sumResidualsSq += dist * dist
        maxResidual = max(maxResidual, dist)
    }

    let rmse = sqrt(sumResidualsSq / Double(nData))

    return BSplineFitResult3D(
        controlPoints: controlPoints,
        knots: knots,
        degree: degree,
        residuals: residuals,
        rmse: rmse,
        maxError: maxResidual,
        parameters: t
    )
}

// MARK: - Private Helper Functions

/// Invert a 3x3 matrix.
private func invert3x3(_ m: [[Double]]) -> [[Double]]? {
    let a = m[0][0], b = m[0][1], c = m[0][2]
    let d = m[1][0], e = m[1][1], f = m[1][2]
    let g = m[2][0], h = m[2][1], i = m[2][2]

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    guard abs(det) > 1e-15 else { return nil }

    let invDet = 1.0 / det

    return [
        [(e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet],
        [(f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet],
        [(d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet]
    ]
}

/// Eigendecomposition of a 3x3 matrix using characteristic polynomial.
private func eigenDecomposition3x3(_ m: [[Double]]) -> (eigenvalues: [Double], eigenvectors: [[Double]])? {
    let a = m[0][0], b = m[0][1], c = m[0][2]
    let d = m[1][0], e = m[1][1], f = m[1][2]
    let g = m[2][0], h = m[2][1], i = m[2][2]

    let trace = a + e + i
    let minor1 = e * i - f * h
    let minor2 = a * i - c * g
    let minor3 = a * e - b * d
    let sumMinors = minor1 + minor2 + minor3
    let det = a * minor1 - b * (d * i - f * g) + c * (d * h - e * g)

    guard let roots = solveCubic(1.0, -trace, sumMinors, -det) else { return nil }

    var eigenvectors = [[Double]]()

    for lambda in roots {
        let A_lambda = [
            [m[0][0] - lambda, m[0][1], m[0][2]],
            [m[1][0], m[1][1] - lambda, m[1][2]],
            [m[2][0], m[2][1], m[2][2] - lambda]
        ]

        let r1 = [A_lambda[0][0], A_lambda[0][1], A_lambda[0][2]]
        let r2 = [A_lambda[1][0], A_lambda[1][1], A_lambda[1][2]]
        let r3 = [A_lambda[2][0], A_lambda[2][1], A_lambda[2][2]]

        var v = cross3(r1, r2)
        var norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

        if norm < 1e-10 {
            v = cross3(r1, r3)
            norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        }

        if norm < 1e-10 {
            v = cross3(r2, r3)
            norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        }

        if norm < 1e-10 {
            v = [1, 0, 0]
            norm = 1
        }

        eigenvectors.append([v[0] / norm, v[1] / norm, v[2] / norm])
    }

    return (roots, eigenvectors)
}

/// Cross product of two 3-vectors.
private func cross3(_ a: [Double], _ b: [Double]) -> [Double] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Solve cubic equation ax³ + bx² + cx + d = 0.
private func solveCubic(_ a: Double, _ b: Double, _ c: Double, _ d: Double) -> [Double]? {
    guard abs(a) > 1e-15 else {
        return solveQuadratic(b, c, d)
    }

    let p = b / a, q = c / a, r = d / a
    let p1 = q - p * p / 3
    let q1 = 2 * p * p * p / 27 - p * q / 3 + r
    let discriminant = q1 * q1 / 4 + p1 * p1 * p1 / 27

    var roots = [Double]()

    // Use Foundation's cbrt for real cube root
    func realCbrt(_ x: Double) -> Double {
        x >= 0 ? pow(x, 1.0/3.0) : -pow(-x, 1.0/3.0)
    }

    if discriminant > 1e-15 {
        let sqrtD = sqrt(discriminant)
        let u = realCbrt(-q1 / 2 + sqrtD)
        let v = realCbrt(-q1 / 2 - sqrtD)
        roots.append(u + v - p / 3)
    } else if discriminant < -1e-15 {
        let rho = sqrt(-p1 * p1 * p1 / 27)
        let theta = acos(-q1 / 2 / rho)
        let m = 2.0 * realCbrt(rho)
        roots.append(m * cos(theta / 3) - p / 3)
        roots.append(m * cos((theta + 2 * Double.pi) / 3) - p / 3)
        roots.append(m * cos((theta + 4 * Double.pi) / 3) - p / 3)
    } else {
        if abs(q1) < 1e-15 {
            roots.append(-p / 3)
        } else {
            let u = realCbrt(-q1 / 2)
            roots.append(2 * u - p / 3)
            roots.append(-u - p / 3)
        }
    }

    return roots.isEmpty ? nil : roots
}

/// Solve quadratic equation ax² + bx + c = 0.
private func solveQuadratic(_ a: Double, _ b: Double, _ c: Double) -> [Double]? {
    guard abs(a) > 1e-15 else {
        guard abs(b) > 1e-15 else { return nil }
        return [-c / b]
    }

    let discriminant = b * b - 4 * a * c
    if discriminant < -1e-15 {
        return []
    } else if discriminant < 1e-15 {
        return [-b / (2 * a)]
    } else {
        let sqrtD = sqrt(discriminant)
        return [(-b + sqrtD) / (2 * a), (-b - sqrtD) / (2 * a)]
    }
}

/// Solve positive definite system using Cholesky decomposition.
private func solvePositiveDefinite(_ A: [Double], _ b: [Double], n: Int) -> [Double]? {
    // Cholesky decomposition: A = L * L^T
    var L = [Double](repeating: 0, count: n * n)

    for i in 0..<n {
        for j in 0...i {
            var sum = 0.0
            if i == j {
                for k in 0..<j {
                    sum += L[j * n + k] * L[j * n + k]
                }
                let val = A[j * n + j] - sum
                guard val > 1e-15 else { return nil }
                L[j * n + j] = sqrt(val)
            } else {
                for k in 0..<j {
                    sum += L[i * n + k] * L[j * n + k]
                }
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j]
            }
        }
    }

    // Forward substitution: L * y = b
    var y = [Double](repeating: 0, count: n)
    for i in 0..<n {
        var sum = b[i]
        for j in 0..<i {
            sum -= L[i * n + j] * y[j]
        }
        y[i] = sum / L[i * n + i]
    }

    // Backward substitution: L^T * x = y
    var x = [Double](repeating: 0, count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        var sum = y[i]
        for j in (i + 1)..<n {
            sum -= L[j * n + i] * x[j]
        }
        x[i] = sum / L[i * n + i]
    }

    return x
}
