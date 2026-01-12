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
