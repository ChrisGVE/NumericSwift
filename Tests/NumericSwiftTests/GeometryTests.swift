//
//  GeometryTests.swift
//  NumericSwift
//
//  Tests for geometry algorithms.
//

import XCTest
import simd
@testable import NumericSwift

final class GeometryTests: XCTestCase {

    // MARK: - Vec2 Tests

    func testVec2FromArray() {
        let v = Vec2([3.0, 4.0])
        XCTAssertNotNil(v)
        XCTAssertEqual(v!.x, 3.0)
        XCTAssertEqual(v!.y, 4.0)

        XCTAssertNil(Vec2([1.0]))
    }

    func testVec2Angle() {
        let v1 = Vec2(1, 0)
        XCTAssertEqual(v1.angle, 0, accuracy: 1e-10)

        let v2 = Vec2(0, 1)
        XCTAssertEqual(v2.angle, .pi / 2, accuracy: 1e-10)

        let v3 = Vec2(-1, 0)
        XCTAssertEqual(v3.angle, .pi, accuracy: 1e-10)
    }

    func testVec2Cross() {
        let v1 = Vec2(1, 0)
        let v2 = Vec2(0, 1)
        XCTAssertEqual(v1.cross(v2), 1.0, accuracy: 1e-10)
        XCTAssertEqual(v2.cross(v1), -1.0, accuracy: 1e-10)
    }

    func testVec2Rotate() {
        let v = Vec2(1, 0)
        let rotated = v.rotated(by: .pi / 2)
        XCTAssertEqual(rotated.x, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.y, 1, accuracy: 1e-10)
    }

    func testVec2Project() {
        let v = Vec2(3, 4)
        let onto = Vec2(1, 0)
        let proj = v.projected(onto: onto)
        XCTAssertEqual(proj.x, 3, accuracy: 1e-10)
        XCTAssertEqual(proj.y, 0, accuracy: 1e-10)
    }

    func testVec2Perpendicular() {
        let v = Vec2(1, 0)
        let perp = v.perpendicular
        XCTAssertEqual(perp.x, 0, accuracy: 1e-10)
        XCTAssertEqual(perp.y, 1, accuracy: 1e-10)
    }

    func testVec2Polar() {
        let v = Vec2(3, 4)
        let polar = v.polar
        XCTAssertEqual(polar.r, 5, accuracy: 1e-10)

        let fromPolar = Vec2.fromPolar(r: 5, theta: 0)
        XCTAssertEqual(fromPolar.x, 5, accuracy: 1e-10)
        XCTAssertEqual(fromPolar.y, 0, accuracy: 1e-10)
    }

    // MARK: - Vec3 Tests

    func testVec3FromArray() {
        let v = Vec3([1.0, 2.0, 3.0])
        XCTAssertNotNil(v)
        XCTAssertEqual(v!.x, 1.0)
        XCTAssertEqual(v!.y, 2.0)
        XCTAssertEqual(v!.z, 3.0)

        XCTAssertNil(Vec3([1.0, 2.0]))
    }

    func testVec3Rotate() {
        let v = Vec3(1, 0, 0)
        let rotated = v.rotated(around: Vec3(0, 0, 1), by: .pi / 2)
        XCTAssertEqual(rotated.x, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.y, 1, accuracy: 1e-10)
        XCTAssertEqual(rotated.z, 0, accuracy: 1e-10)
    }

    func testVec3Project() {
        let v = Vec3(3, 4, 5)
        let onto = Vec3(1, 0, 0)
        let proj = v.projected(onto: onto)
        XCTAssertEqual(proj.x, 3, accuracy: 1e-10)
        XCTAssertEqual(proj.y, 0, accuracy: 1e-10)
        XCTAssertEqual(proj.z, 0, accuracy: 1e-10)
    }

    func testVec3Spherical() {
        let v = Vec3(0, 0, 1)
        let sph = v.spherical
        XCTAssertEqual(sph.r, 1, accuracy: 1e-10)
        XCTAssertEqual(sph.phi, 0, accuracy: 1e-10)  // phi=0 is z-axis

        let fromSph = Vec3.fromSpherical(r: 1, theta: 0, phi: .pi / 2)
        XCTAssertEqual(fromSph.x, 1, accuracy: 1e-10)
        XCTAssertEqual(fromSph.y, 0, accuracy: 1e-10)
        XCTAssertEqual(fromSph.z, 0, accuracy: 1e-10)
    }

    func testVec3Cylindrical() {
        let v = Vec3(3, 4, 5)
        let cyl = v.cylindrical
        XCTAssertEqual(cyl.r, 5, accuracy: 1e-10)
        XCTAssertEqual(cyl.z, 5, accuracy: 1e-10)

        let fromCyl = Vec3.fromCylindrical(r: 5, theta: 0, z: 3)
        XCTAssertEqual(fromCyl.x, 5, accuracy: 1e-10)
        XCTAssertEqual(fromCyl.y, 0, accuracy: 1e-10)
        XCTAssertEqual(fromCyl.z, 3, accuracy: 1e-10)
    }

    // MARK: - Quaternion Tests

    func testQuatFromEuler() {
        let q = Quat.fromEuler(roll: 0, pitch: 0, yaw: .pi / 2)
        let v = Vec3(1, 0, 0)
        let rotated = q.act(v)
        XCTAssertEqual(rotated.x, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.y, 1, accuracy: 1e-10)
        XCTAssertEqual(rotated.z, 0, accuracy: 1e-10)
    }

    func testQuatToEuler() {
        let q = Quat(angle: .pi / 2, axis: Vec3(0, 0, 1))
        let euler = q.euler
        XCTAssertEqual(euler.yaw, .pi / 2, accuracy: 1e-10)
    }

    func testQuatAxisAngle() {
        let q = Quat(angle: .pi / 2, axis: Vec3(0, 0, 1))
        let aa = q.axisAngle
        XCTAssertEqual(aa.angle, .pi / 2, accuracy: 1e-10)
        XCTAssertEqual(aa.axis.z, 1, accuracy: 1e-10)
    }

    func testQuatMatrix() {
        let q = Quat(angle: .pi / 2, axis: Vec3(0, 0, 1))
        let m = q.matrix
        let v = Vec4(1, 0, 0, 1)
        let rotated = m * v
        XCTAssertEqual(rotated.x, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.y, 1, accuracy: 1e-10)
    }

    func testQuatDot() {
        let q1 = Quat(angle: 0, axis: Vec3(0, 0, 1))
        let q2 = Quat(angle: .pi, axis: Vec3(0, 0, 1))
        let dot = q1.dot(q2)
        // cos(0) * cos(pi/2) + ... = 0 for perpendicular quaternions
        XCTAssertTrue(dot.isFinite)
    }

    // MARK: - Mat4 Tests

    func testMat4Translation() {
        let m = Mat4.translation(Vec3(1, 2, 3))
        let p = Vec3(0, 0, 0)
        let translated = m.apply(p)
        XCTAssertEqual(translated.x, 1, accuracy: 1e-10)
        XCTAssertEqual(translated.y, 2, accuracy: 1e-10)
        XCTAssertEqual(translated.z, 3, accuracy: 1e-10)
    }

    func testMat4RotationX() {
        let m = Mat4.rotationX(.pi / 2)
        let p = Vec3(0, 1, 0)
        let rotated = m.apply(p)
        XCTAssertEqual(rotated.x, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.y, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.z, 1, accuracy: 1e-10)
    }

    func testMat4RotationY() {
        let m = Mat4.rotationY(.pi / 2)
        let p = Vec3(1, 0, 0)
        let rotated = m.apply(p)
        XCTAssertEqual(rotated.x, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.y, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.z, -1, accuracy: 1e-10)
    }

    func testMat4RotationZ() {
        let m = Mat4.rotationZ(.pi / 2)
        let p = Vec3(1, 0, 0)
        let rotated = m.apply(p)
        XCTAssertEqual(rotated.x, 0, accuracy: 1e-10)
        XCTAssertEqual(rotated.y, 1, accuracy: 1e-10)
        XCTAssertEqual(rotated.z, 0, accuracy: 1e-10)
    }

    func testMat4Scale() {
        let m = Mat4.scale(Vec3(2, 3, 4))
        let p = Vec3(1, 1, 1)
        let scaled = m.apply(p)
        XCTAssertEqual(scaled.x, 2, accuracy: 1e-10)
        XCTAssertEqual(scaled.y, 3, accuracy: 1e-10)
        XCTAssertEqual(scaled.z, 4, accuracy: 1e-10)
    }

    // MARK: - Geometric Calculation Tests

    func testDistance2D() {
        let d = distance2D(Vec2(0, 0), Vec2(3, 4))
        XCTAssertEqual(d, 5, accuracy: 1e-10)
    }

    func testDistance3D() {
        let d = distance3D(Vec3(0, 0, 0), Vec3(1, 2, 2))
        XCTAssertEqual(d, 3, accuracy: 1e-10)
    }

    func testAngleBetween2D() {
        let angle = angleBetween2D(Vec2(1, 0), Vec2(0, 1))
        XCTAssertEqual(angle, .pi / 2, accuracy: 1e-10)
    }

    func testAngleBetween3D() {
        let angle = angleBetween3D(Vec3(1, 0, 0), Vec3(0, 1, 0))
        XCTAssertEqual(angle, .pi / 2, accuracy: 1e-10)
    }

    func testConvexHull2D() {
        let points = [
            Vec2(0, 0), Vec2(1, 0), Vec2(2, 0),
            Vec2(0, 1), Vec2(1, 1), Vec2(2, 1),
            Vec2(0, 2), Vec2(1, 2), Vec2(2, 2)
        ]
        let hull = convexHull2D(points)

        // Should be 4 corners
        XCTAssertEqual(hull.count, 4)
    }

    func testPointInPolygon() {
        let square = [Vec2(0, 0), Vec2(2, 0), Vec2(2, 2), Vec2(0, 2)]

        XCTAssertTrue(pointInPolygon(Vec2(1, 1), square))
        XCTAssertFalse(pointInPolygon(Vec2(3, 3), square))
    }

    func testLineIntersection2D() {
        // X-axis and Y-axis should intersect at origin
        let intersection = lineIntersection2D(
            p1: Vec2(-1, 0), p2: Vec2(1, 0),
            p3: Vec2(0, -1), p4: Vec2(0, 1)
        )
        XCTAssertNotNil(intersection)
        XCTAssertEqual(intersection!.x, 0, accuracy: 1e-10)
        XCTAssertEqual(intersection!.y, 0, accuracy: 1e-10)

        // Parallel lines should return nil
        let parallel = lineIntersection2D(
            p1: Vec2(0, 0), p2: Vec2(1, 0),
            p3: Vec2(0, 1), p4: Vec2(1, 1)
        )
        XCTAssertNil(parallel)
    }

    func testTriangleArea2D() {
        let area = triangleArea2D(Vec2(0, 0), Vec2(2, 0), Vec2(0, 2))
        XCTAssertEqual(area, 2, accuracy: 1e-10)
    }

    func testTriangleArea3D() {
        let area = triangleArea3D(Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(0, 2, 0))
        XCTAssertEqual(area, 2, accuracy: 1e-10)
    }

    func testCentroid2D() {
        let points = [Vec2(0, 0), Vec2(2, 0), Vec2(2, 2), Vec2(0, 2)]
        let c = centroid2D(points)
        XCTAssertNotNil(c)
        XCTAssertEqual(c!.x, 1, accuracy: 1e-10)
        XCTAssertEqual(c!.y, 1, accuracy: 1e-10)
    }

    func testCentroid3D() {
        let points = [Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)]
        let c = centroid3D(points)
        XCTAssertNotNil(c)
        XCTAssertEqual(c!.x, 0.5, accuracy: 1e-10)
        XCTAssertEqual(c!.y, 0.5, accuracy: 1e-10)
        XCTAssertEqual(c!.z, 0.5, accuracy: 1e-10)
    }

    func testCircleFrom3Points() {
        // Unit circle through (1,0), (0,1), (-1,0)
        let result = circleFrom3Points(Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0))
        XCTAssertNotNil(result)
        XCTAssertEqual(result!.center.x, 0, accuracy: 1e-10)
        XCTAssertEqual(result!.center.y, 0, accuracy: 1e-10)
        XCTAssertEqual(result!.radius, 1, accuracy: 1e-10)
    }

    func testPlaneFrom3Points() {
        // XY plane
        let result = planeFrom3Points(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))
        XCTAssertNotNil(result)
        // Normal should be (0, 0, 1) or (0, 0, -1)
        XCTAssertEqual(abs(result!.normal.z), 1, accuracy: 1e-10)
    }

    func testPointPlaneDistance() {
        let plane = PlaneResult(normal: Vec3(0, 0, 1), d: 0)
        let dist = pointPlaneDistance(Vec3(0, 0, 5), plane)
        XCTAssertEqual(dist, 5, accuracy: 1e-10)
    }

    func testLinePlaneIntersection() {
        let plane = PlaneResult(normal: Vec3(0, 0, 1), d: -1)  // z = 1
        let result = linePlaneIntersection(
            linePoint: Vec3(0, 0, 0),
            lineDir: Vec3(0, 0, 1),
            plane: plane
        )
        XCTAssertNotNil(result)
        XCTAssertEqual(result!.z, 1, accuracy: 1e-10)
    }

    func testPlanePlaneIntersection() {
        // XZ plane and YZ plane should intersect along Z axis
        let plane1 = PlaneResult(normal: Vec3(0, 1, 0), d: 0)
        let plane2 = PlaneResult(normal: Vec3(1, 0, 0), d: 0)
        let result = planePlaneIntersection(plane1, plane2)
        XCTAssertNotNil(result)
        // Direction should be along Z
        XCTAssertEqual(abs(result!.direction.z), 1, accuracy: 1e-10)
    }

    func testSphereFrom4Points() {
        // Unit sphere through 4 points
        let result = sphereFrom4Points(
            Vec3(1, 0, 0), Vec3(-1, 0, 0),
            Vec3(0, 1, 0), Vec3(0, 0, 1)
        )
        XCTAssertNotNil(result)
        XCTAssertEqual(result!.center.x, 0, accuracy: 1e-10)
        XCTAssertEqual(result!.center.y, 0, accuracy: 1e-10)
        XCTAssertEqual(result!.center.z, 0, accuracy: 1e-10)
        XCTAssertEqual(result!.radius, 1, accuracy: 1e-10)
    }

    // MARK: - Circle Fitting Tests

    func testCircleFitAlgebraic() {
        // Generate points on a unit circle
        var points = [Vec2]()
        for i in 0..<8 {
            let theta = Double(i) * .pi / 4
            points.append(Vec2(cos(theta), sin(theta)))
        }

        let result = circleFitAlgebraic(points)
        XCTAssertNotNil(result)
        XCTAssertEqual(result!.center.x, 0, accuracy: 0.01)
        XCTAssertEqual(result!.center.y, 0, accuracy: 0.01)
        XCTAssertEqual(result!.radius, 1, accuracy: 0.01)
    }

    func testCircleFitTaubin() {
        // Generate points on a circle with center (1, 2) and radius 3
        var points = [Vec2]()
        for i in 0..<12 {
            let theta = Double(i) * .pi / 6
            points.append(Vec2(1 + 3 * cos(theta), 2 + 3 * sin(theta)))
        }

        let result = circleFitTaubin(points)
        XCTAssertNotNil(result)
        XCTAssertEqual(result!.center.x, 1, accuracy: 0.01)
        XCTAssertEqual(result!.center.y, 2, accuracy: 0.01)
        XCTAssertEqual(result!.radius, 3, accuracy: 0.01)
    }

    // MARK: - Sphere Fitting Tests

    func testSphereFitAlgebraic() {
        // Generate points on a unit sphere
        var points = [Vec3]()
        points.append(Vec3(1, 0, 0))
        points.append(Vec3(-1, 0, 0))
        points.append(Vec3(0, 1, 0))
        points.append(Vec3(0, -1, 0))
        points.append(Vec3(0, 0, 1))
        points.append(Vec3(0, 0, -1))

        let result = sphereFitAlgebraic(points)
        XCTAssertNotNil(result)
        XCTAssertEqual(result!.center.x, 0, accuracy: 0.01)
        XCTAssertEqual(result!.center.y, 0, accuracy: 0.01)
        XCTAssertEqual(result!.center.z, 0, accuracy: 0.01)
        XCTAssertEqual(result!.radius, 1, accuracy: 0.01)
    }

    // MARK: - B-Spline Tests

    func testBsplineBasis() {
        let knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        let degree = 2
        let n = 3  // number of control points

        // At t=0, first basis should be 1
        XCTAssertEqual(bsplineBasis(i: 0, degree: degree, t: 0, knots: knots), 1.0, accuracy: 1e-10)

        // Sum of basis functions should be 1 for any t in [0, 1] (including endpoints)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            var sum = 0.0
            for i in 0..<n {
                sum += bsplineBasis(i: i, degree: degree, t: t, knots: knots)
            }
            XCTAssertEqual(sum, 1.0, accuracy: 1e-10, "Sum of basis functions at t=\(t) should be 1")
        }

        // At t=1, last basis should be 1
        XCTAssertEqual(bsplineBasis(i: n - 1, degree: degree, t: 1.0, knots: knots), 1.0, accuracy: 1e-10)
    }

    func testBsplineUniformKnots() {
        let knots = bsplineUniformKnots(n: 5, degree: 2)
        XCTAssertEqual(knots.count, 8)  // n + degree + 1

        // First degree+1 knots should be 0
        XCTAssertEqual(knots[0], 0)
        XCTAssertEqual(knots[1], 0)
        XCTAssertEqual(knots[2], 0)

        // Last degree+1 knots should be 1
        XCTAssertEqual(knots[5], 1)
        XCTAssertEqual(knots[6], 1)
        XCTAssertEqual(knots[7], 1)
    }

    func testBsplineEvaluate() {
        let controlPoints = [Vec2(0, 0), Vec2(1, 1), Vec2(2, 0)]
        let degree = 2
        let knots = bsplineUniformKnots(n: 3, degree: degree)

        // At t=0, should be at first control point
        let p0 = bsplineEvaluate(controlPoints: controlPoints, degree: degree, t: 0, knots: knots)
        XCTAssertEqual(p0.x, 0, accuracy: 1e-10)
        XCTAssertEqual(p0.y, 0, accuracy: 1e-10)

        // At t=0.5, should be on the curve
        let p05 = bsplineEvaluate(controlPoints: controlPoints, degree: degree, t: 0.5, knots: knots)
        XCTAssertTrue(p05.x > 0 && p05.x < 2)
    }

    func testBsplineDerivative() {
        let controlPoints = [Vec2(0, 0), Vec2(1, 1), Vec2(2, 0)]
        let degree = 2
        let knots = bsplineUniformKnots(n: 3, degree: degree)

        let deriv = bsplineDerivative(controlPoints: controlPoints, degree: degree, t: 0.5, knots: knots)

        // Derivative should be non-zero
        XCTAssertTrue(simd_length(deriv) > 0)
    }
}
