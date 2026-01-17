//
//  ArraySwiftIntegrationTests.swift
//  NumericSwiftTests
//
//  Tests for ArraySwift integration.
//  These tests are only compiled when NUMERICSWIFT_ARRAYSWIFT is defined.
//
//  Licensed under the MIT License.
//

#if NUMERICSWIFT_ARRAYSWIFT

import XCTest
@testable import NumericSwift
import ArraySwift

final class ArraySwiftIntegrationTests: XCTestCase {

    // MARK: - Matrix ↔ NDArray Conversion Tests

    func testMatrixFromNDArray2D() {
        let arr = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let matrix = LinAlg.Matrix(ndarray: arr)

        XCTAssertNotNil(matrix)
        XCTAssertEqual(matrix?.rows, 2)
        XCTAssertEqual(matrix?.cols, 3)
        XCTAssertEqual(matrix?.data, [1, 2, 3, 4, 5, 6])
    }

    func testMatrixFromNDArray1D() {
        let arr = NDArray([1.0, 2.0, 3.0])
        let matrix = LinAlg.Matrix(ndarray: arr)

        XCTAssertNotNil(matrix)
        XCTAssertEqual(matrix?.rows, 3)
        XCTAssertEqual(matrix?.cols, 1)
        XCTAssertTrue(matrix?.isVector ?? false)
    }

    func testMatrixFromComplexNDArrayReturnsNil() {
        let arr = NDArray.complexArray(shape: [2], real: [1.0, 2.0], imag: [3.0, 4.0])
        let matrix = LinAlg.Matrix(ndarray: arr)

        XCTAssertNil(matrix)
    }

    func testMatrixFromNDArray3DReturnsNil() {
        let arr = NDArray(shape: [2, 2, 2], data: [1, 2, 3, 4, 5, 6, 7, 8])
        let matrix = LinAlg.Matrix(ndarray: arr)

        XCTAssertNil(matrix)
    }

    func testNDArrayFromMatrix() {
        let matrix = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let arr = matrix.toNDArray()

        XCTAssertEqual(arr.shape, [2, 2])
        XCTAssertEqual(arr.real, [1, 2, 3, 4])
    }

    func testNDArrayFromVector() {
        let vector = LinAlg.Matrix([1.0, 2.0, 3.0])
        let arr = vector.toNDArray()

        XCTAssertEqual(arr.shape, [3])
        XCTAssertEqual(arr.real, [1, 2, 3])
    }

    func testRoundTripConversion() {
        let original = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let matrix = original.toMatrix()!
        let roundTrip = matrix.toNDArray()

        XCTAssertEqual(roundTrip.shape, original.shape)
        XCTAssertEqual(roundTrip.real, original.real)
    }

    // MARK: - NDArray Linear Algebra Tests

    func testNDArraySolve() {
        let A = NDArray([[3.0, 1.0], [1.0, 2.0]])
        let b = NDArray([9.0, 8.0])

        let x = A.solve(b)

        XCTAssertNotNil(x)
        // Verify: A * x ≈ b
        // x should be [2, 3]
        XCTAssertEqual(x!.real[0], 2.0, accuracy: 1e-10)
        XCTAssertEqual(x!.real[1], 3.0, accuracy: 1e-10)
    }

    func testNDArrayInverse() {
        let A = NDArray([[1.0, 2.0], [3.0, 4.0]])
        let invA = A.inv()

        XCTAssertNotNil(invA)
        // A * A^-1 should be identity
        let identity = A.dot(invA!)
        XCTAssertEqual(identity.real[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(identity.real[1], 0.0, accuracy: 1e-10)
        XCTAssertEqual(identity.real[2], 0.0, accuracy: 1e-10)
        XCTAssertEqual(identity.real[3], 1.0, accuracy: 1e-10)
    }

    func testNDArrayDeterminant() {
        let A = NDArray([[1.0, 2.0], [3.0, 4.0]])
        let det = A.det()

        XCTAssertNotNil(det)
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        XCTAssertEqual(det!, -2.0, accuracy: 1e-10)
    }

    func testNDArrayRank() {
        let A = NDArray([[1.0, 2.0], [2.0, 4.0]])  // Rank 1 (second row = 2 * first row)
        let rank = A.rank()

        XCTAssertNotNil(rank)
        XCTAssertEqual(rank!, 1)

        let B = NDArray([[1.0, 0.0], [0.0, 1.0]])  // Full rank
        XCTAssertEqual(B.rank()!, 2)
    }

    func testNDArrayTrace() {
        let A = NDArray([[1.0, 2.0], [3.0, 4.0]])
        let trace = A.trace()

        XCTAssertNotNil(trace)
        // trace = 1 + 4 = 5
        XCTAssertEqual(trace!, 5.0, accuracy: 1e-10)
    }

    func testNDArrayNorm() {
        let A = NDArray([[1.0, 2.0], [3.0, 4.0]])
        let norm = A.norm()

        // Frobenius norm = sqrt(1 + 4 + 9 + 16) = sqrt(30)
        XCTAssertEqual(norm, Darwin.sqrt(30.0), accuracy: 1e-10)
    }

    func testNDArrayLU() {
        let A = NDArray([[2.0, 1.0], [1.0, 3.0]])
        let result = A.lu()

        XCTAssertNotNil(result)
        // P * A = L * U
    }

    func testNDArrayQR() {
        let A = NDArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        let result = A.qr()

        XCTAssertNotNil(result)
        XCTAssertEqual(result!.Q.shape[0], 3)
        // R is min(m,n) x n for reduced QR decomposition
        XCTAssertEqual(result!.R.shape, [2, 2])
    }

    func testNDArraySVD() {
        let A = NDArray([[1.0, 2.0], [3.0, 4.0]])
        let result = A.svd()

        XCTAssertNotNil(result)
        XCTAssertEqual(result!.s.size, 2)  // 2 singular values for 2x2 matrix
    }

    func testNDArrayEig() {
        // Symmetric matrix for real eigenvalues
        let A = NDArray([[2.0, 1.0], [1.0, 2.0]])
        let result = A.eig()

        XCTAssertNotNil(result)
        // Eigenvalues of [[2,1],[1,2]] are 3 and 1
        let sortedValues = result!.values.real.sorted()
        XCTAssertEqual(sortedValues[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(sortedValues[1], 3.0, accuracy: 1e-10)
    }

    func testNDArrayCholesky() {
        // Positive definite matrix
        let A = NDArray([[4.0, 2.0], [2.0, 5.0]])
        let L = A.cholesky()

        XCTAssertNotNil(L)
        // Verify L * L^T ≈ A
        let reconstructed = L!.dot(L!.transpose())
        for i in 0..<4 {
            XCTAssertEqual(reconstructed.real[i], A.real[i], accuracy: 1e-10)
        }
    }

    // MARK: - Statistics Functions for NDArray Tests

    func testSumNDArray() {
        let arr = NDArray([1.0, 2.0, 3.0, 4.0, 5.0])
        XCTAssertEqual(NumericSwift.sum(arr), 15.0, accuracy: 1e-10)
    }

    func testMeanNDArray() {
        let arr = NDArray([1.0, 2.0, 3.0, 4.0, 5.0])
        XCTAssertEqual(NumericSwift.mean(arr), 3.0, accuracy: 1e-10)
    }

    func testMedianNDArray() {
        let arr = NDArray([1.0, 3.0, 5.0, 7.0, 9.0])
        XCTAssertEqual(NumericSwift.median(arr), 5.0, accuracy: 1e-10)
    }

    func testVarianceNDArray() {
        let arr = NDArray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        // Population variance
        XCTAssertEqual(NumericSwift.variance(arr, ddof: 0), 4.0, accuracy: 1e-10)
    }

    func testStddevNDArray() {
        let arr = NDArray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        XCTAssertEqual(NumericSwift.stddev(arr, ddof: 0), 2.0, accuracy: 1e-10)
    }

    func testPercentileNDArray() {
        let arr = NDArray([1.0, 2.0, 3.0, 4.0, 5.0])
        XCTAssertEqual(NumericSwift.percentile(arr, 50), 3.0, accuracy: 1e-10)
    }

    func testAminAmaxNDArray() {
        let arr = NDArray([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        XCTAssertEqual(NumericSwift.amin(arr), 1.0)
        XCTAssertEqual(NumericSwift.amax(arr), 9.0)
    }

    func testPtpNDArray() {
        let arr = NDArray([1.0, 5.0, 10.0])
        XCTAssertEqual(NumericSwift.ptp(arr), 9.0, accuracy: 1e-10)
    }

    func testCumsumNDArray() {
        let arr = NDArray([1.0, 2.0, 3.0, 4.0])
        let result = NumericSwift.cumsum(arr)
        XCTAssertEqual(result.real, [1, 3, 6, 10])
    }

    func testCumprodNDArray() {
        let arr = NDArray([1.0, 2.0, 3.0, 4.0])
        let result = NumericSwift.cumprod(arr)
        XCTAssertEqual(result.real, [1, 2, 6, 24])
    }

    func testDiffNDArray() {
        let arr = NDArray([1.0, 3.0, 6.0, 10.0])
        let result = NumericSwift.diff(arr)
        XCTAssertEqual(result.real, [2, 3, 4])
    }

    // MARK: - Utility Functions for NDArray Tests

    func testClipNDArray() {
        let arr = NDArray([1.0, 5.0, 10.0, 15.0])
        let result = NumericSwift.clip(arr, min: 3.0, max: 12.0)
        XCTAssertEqual(result.real, [3, 5, 10, 12])
    }

    func testMathFunctionsNDArray() {
        let arr = NDArray([1.0, 4.0, 9.0])

        let sqrtResult = NumericSwift.sqrtArray(arr)
        XCTAssertEqual(sqrtResult.real[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(sqrtResult.real[1], 2.0, accuracy: 1e-10)
        XCTAssertEqual(sqrtResult.real[2], 3.0, accuracy: 1e-10)

        let absResult = NumericSwift.absArray(NDArray([-1.0, 2.0, -3.0]))
        XCTAssertEqual(absResult.real, [1, 2, 3])
    }

    func testTrigFunctionsNDArray() {
        let arr = NDArray([0.0, Double.pi / 2, Double.pi])

        let sinResult = NumericSwift.sinArray(arr)
        XCTAssertEqual(sinResult.real[0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(sinResult.real[1], 1.0, accuracy: 1e-10)
        XCTAssertEqual(sinResult.real[2], 0.0, accuracy: 1e-10)

        let cosResult = NumericSwift.cosArray(arr)
        XCTAssertEqual(cosResult.real[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(cosResult.real[1], 0.0, accuracy: 1e-10)
        XCTAssertEqual(cosResult.real[2], -1.0, accuracy: 1e-10)
    }
}

#endif // NUMERICSWIFT_ARRAYSWIFT
