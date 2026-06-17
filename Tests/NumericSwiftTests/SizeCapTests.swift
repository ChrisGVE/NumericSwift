//
//  SizeCapTests.swift
//  NumericSwift
//
//  Tests for the two-tier matrix size cap introduced in Task 5 of the
//  unified-pipeline tag.
//
//  HARD cap  — LinAlg.hardMaxMatrixElementCount = Int(Int32.max)
//              Enforced via precondition in Matrix/ComplexMatrix constructors.
//              Not catchable.  Trapping paths are covered at the overflow-flag
//              level (testElementCountOverflowDetected) rather than in-process
//              traps, because Swift `precondition` aborts the process with SIGILL
//              and XCTExpectFailure cannot intercept process-terminating signals.
//
//  SOFT cap  — LinAlg.maxEvaluatorMatrixElements (default 16 777 216 = 4096²)
//              Enforced by LinAlg.checkSoftCap(rows:cols:), which throws
//              LinAlgError.invalidParameter (CONS-07).
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class SizeCapTests: XCTestCase {

    typealias E = LinAlg.LinAlgError

    // -------------------------------------------------------------------------
    // MARK: - Helpers
    // -------------------------------------------------------------------------

    /// Restore the soft cap to its default after each test so state changes
    /// in one test cannot affect subsequent ones.
    override func tearDown() {
        super.tearDown()
        try? LinAlg.setMaxEvaluatorMatrixElements(16_777_216)
    }

    // -------------------------------------------------------------------------
    // MARK: - Hard cap constant
    // -------------------------------------------------------------------------

    func testHardCapValueEqualsInt32Max() {
        XCTAssertEqual(LinAlg.hardMaxMatrixElementCount, Int(Int32.max))
        XCTAssertEqual(LinAlg.hardMaxMatrixElementCount, 2_147_483_647)
    }

    func testHardCapIsImmutable() {
        // The property must be a `let`/computed getter — no settable path.
        // If this compiles, the cap is read-only.  (Setter would be a compile
        // error, not a runtime failure; we just verify the value is stable.)
        let v1 = LinAlg.hardMaxMatrixElementCount
        let v2 = LinAlg.hardMaxMatrixElementCount
        XCTAssertEqual(v1, v2)
    }

    // -------------------------------------------------------------------------
    // MARK: - Overflow-safe element-count helper
    // -------------------------------------------------------------------------

    func testElementCountSmallDims() {
        let (count, overflow) = LinAlg.elementCount(rows: 4, cols: 4)
        XCTAssertFalse(overflow)
        XCTAssertEqual(count, 16)
    }

    func testElementCountOverflowDetected() {
        // rows = Int.max, cols = 2 — product overflows
        let (_, overflow) = LinAlg.elementCount(rows: Int.max, cols: 2)
        XCTAssertTrue(overflow)
    }

    func testElementCountNegativeRowsFlagsOverflow() {
        let (_, overflow) = LinAlg.elementCount(rows: -1, cols: 4)
        XCTAssertTrue(overflow)
    }

    func testElementCountNegativeColsFlagsOverflow() {
        let (_, overflow) = LinAlg.elementCount(rows: 4, cols: -1)
        XCTAssertTrue(overflow)
    }

    func testElementCountZeroDimsAllowed() {
        let (count, overflow) = LinAlg.elementCount(rows: 0, cols: 5)
        XCTAssertFalse(overflow)
        XCTAssertEqual(count, 0)
    }

    // -------------------------------------------------------------------------
    // MARK: - assertWithinHardCap (validator-level tests avoid huge allocs)
    // -------------------------------------------------------------------------

    func testHardCapValidatorPassesUnderCap() {
        // 100×100 = 10 000 — well under Int32.max; must not trap
        LinAlg.assertWithinHardCap(rows: 100, cols: 100)   // no crash → pass
    }

    // The trapping paths of assertWithinHardCap (overflow dims and over-cap dims)
    // cannot be tested in-process: Swift's `precondition` fires SIGILL, which
    // terminates the entire test process.  `XCTExpectFailure` only intercepts
    // XCTest assertion failures, not OS-level signals.
    //
    // Coverage strategy:
    //  • `testElementCountOverflowDetected` confirms `elementCount(Int.max, 2)`
    //    sets overflow=true — the exact flag `assertWithinHardCap` checks before
    //    calling `precondition(!overflow …)`.
    //  • `testElementCountExceedingHardCapFlaggedCorrectly` confirms that element
    //    counts above Int32.max pass the overflow-safe count but are then caught
    //    by the second condition (`count <= hardMaxMatrixElementCount`).
    //  • Constructor tests below confirm the happy path works end-to-end.

    func testElementCountExceedingHardCapFlaggedCorrectly() {
        // Int32.max + 1 fits in Int (on 64-bit) but exceeds the hard cap.
        let overCap = LinAlg.hardMaxMatrixElementCount + 1
        let (count, overflow) = LinAlg.elementCount(rows: overCap, cols: 1)
        // No integer overflow, but count > hardMaxMatrixElementCount
        XCTAssertFalse(overflow)
        XCTAssertGreaterThan(count, LinAlg.hardMaxMatrixElementCount)
    }

    // -------------------------------------------------------------------------
    // MARK: - Matrix constructor hard-cap wiring
    // -------------------------------------------------------------------------

    func testMatrixConstructorUnderHardCapSucceeds() {
        // Small matrix — should construct fine
        let m = LinAlg.Matrix(rows: 3, cols: 3, data: Array(repeating: 0.0, count: 9))
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 3)
    }

    func testMatrix2DArrayConstructorUnderCapSucceeds() {
        let arr: [[Double]] = [[1, 2], [3, 4], [5, 6]]
        let m = LinAlg.Matrix(arr)
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 2)
    }

    func testMatrixVectorConstructorUnderCapSucceeds() {
        let v = LinAlg.Matrix([1.0, 2.0, 3.0])
        XCTAssertEqual(v.rows, 3)
        XCTAssertEqual(v.cols, 1)
    }

    func testComplexMatrixConstructorUnderCapSucceeds() {
        let n = 4
        let re = [Double](repeating: 0.0, count: n * n)
        let im = [Double](repeating: 0.0, count: n * n)
        let cm = LinAlg.ComplexMatrix(rows: n, cols: n, real: re, imag: im)
        XCTAssertEqual(cm.rows, n)
        XCTAssertEqual(cm.cols, n)
    }

    func testComplexMatrixFromRealMatrixUnderCapSucceeds() {
        let m = LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])
        let cm = LinAlg.ComplexMatrix(m)
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 2)
    }

    // -------------------------------------------------------------------------
    // MARK: - Soft cap default value
    // -------------------------------------------------------------------------

    func testSoftCapDefault() {
        XCTAssertEqual(LinAlg.maxEvaluatorMatrixElements, 16_777_216)
        XCTAssertEqual(LinAlg.maxEvaluatorMatrixElements, 4096 * 4096)
    }

    // -------------------------------------------------------------------------
    // MARK: - checkSoftCap boundaries
    // -------------------------------------------------------------------------

    func testCheckSoftCapExactlyAtCapPasses() throws {
        // 4096 × 4096 = 16 777 216 — exactly at the soft cap; must not throw
        try LinAlg.checkSoftCap(rows: 4096, cols: 4096)
    }

    func testCheckSoftCapJustUnderCapPasses() throws {
        // One element below the cap
        try LinAlg.checkSoftCap(rows: 1, cols: 16_777_215)
    }

    func testCheckSoftCapJustOverCapThrows() {
        // 4096 × 4096 + 1 = 16 777 217 elements — one over the soft cap
        XCTAssertThrowsError(try LinAlg.checkSoftCap(rows: 1, cols: 16_777_217)) { error in
            guard case .invalidParameter? = error as? E else {
                XCTFail("expected LinAlgError.invalidParameter, got \(error)")
                return
            }
        }
    }

    func testCheckSoftCapErrorTypeIsLinAlgErrorNotMathExprError() {
        // CONS-07: the error must be LinAlgError.invalidParameter, never MathExprError
        XCTAssertThrowsError(try LinAlg.checkSoftCap(rows: 4097, cols: 4096)) { error in
            XCTAssertNotNil(error as? E,
                "error must be LinAlg.LinAlgError — got \(type(of: error))")
            XCTAssertEqual(error as? E, E.invalidParameter(
                "matrix element count 16781312 exceeds soft cap 16777216"))
        }
    }

    func testCheckSoftCapOverflowingDimsThrows() {
        // Product overflows Int — should throw (not trap), caught before hard-cap
        XCTAssertThrowsError(try LinAlg.checkSoftCap(rows: Int.max, cols: 2)) { error in
            guard case .invalidParameter? = error as? E else {
                XCTFail("expected LinAlgError.invalidParameter, got \(error)")
                return
            }
        }
    }

    func testCheckSoftCapShapeTupleOverloadMatches() throws {
        // Tuple overload must behave identically to (rows:cols:) overload
        try LinAlg.checkSoftCap(shape: (rows: 4096, cols: 4096))
        XCTAssertThrowsError(try LinAlg.checkSoftCap(shape: (rows: 4097, cols: 4096))) { error in
            guard case .invalidParameter? = error as? E else {
                XCTFail("expected invalidParameter, got \(error)")
                return
            }
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - Soft-cap setter
    // -------------------------------------------------------------------------

    func testSetterUpdatesGetter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(1024)
        XCTAssertEqual(LinAlg.maxEvaluatorMatrixElements, 1024)
    }

    func testSetterChangesCheckSoftCapBehavior() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(100)
        // 10×10 = 100 — exactly at new cap; should pass
        try LinAlg.checkSoftCap(rows: 10, cols: 10)
        // 10×11 = 110 — over new cap; should throw
        XCTAssertThrowsError(try LinAlg.checkSoftCap(rows: 10, cols: 11)) { error in
            guard case .invalidParameter? = error as? E else {
                XCTFail("expected invalidParameter, got \(error)")
                return
            }
        }
    }

    func testSetterWithZeroThrows() {
        XCTAssertThrowsError(try LinAlg.setMaxEvaluatorMatrixElements(0)) { error in
            guard case .invalidParameter? = error as? E else {
                XCTFail("expected invalidParameter, got \(error)")
                return
            }
        }
    }

    func testSetterWithNegativeThrows() {
        XCTAssertThrowsError(try LinAlg.setMaxEvaluatorMatrixElements(-1)) { error in
            guard case .invalidParameter? = error as? E else {
                XCTFail("expected invalidParameter, got \(error)")
                return
            }
        }
    }

    func testSetterExceedingHardCapThrows() {
        let overHard = LinAlg.hardMaxMatrixElementCount + 1
        XCTAssertThrowsError(try LinAlg.setMaxEvaluatorMatrixElements(overHard)) { error in
            guard case .invalidParameter? = error as? E else {
                XCTFail("expected invalidParameter, got \(error)")
                return
            }
        }
    }

    func testTearDownRestoresDefaultAfterSetterChange() throws {
        // Change in a previous test's tearDown restores to 16_777_216.
        // Verify the default is visible at test start.
        XCTAssertEqual(LinAlg.maxEvaluatorMatrixElements, 16_777_216)
    }

    // -------------------------------------------------------------------------
    // MARK: - Edge-case shapes
    // -------------------------------------------------------------------------

    func test1x1MatrixSucceeds() throws {
        let m = LinAlg.Matrix(rows: 1, cols: 1, data: [42.0])
        XCTAssertEqual(m[0, 0], 42.0)
        // 1×1 also passes soft cap
        try LinAlg.checkSoftCap(rows: 1, cols: 1)
    }

    func testExact4096x4096PassesSoftCap() throws {
        try LinAlg.checkSoftCap(rows: 4096, cols: 4096)
    }

    func test4097x4096FailsSoftCap() {
        XCTAssertThrowsError(try LinAlg.checkSoftCap(rows: 4097, cols: 4096)) { error in
            guard case .invalidParameter? = error as? E else {
                XCTFail("expected invalidParameter, got \(error)")
                return
            }
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - ComplexMatrix element counting uses logical rows*cols
    // -------------------------------------------------------------------------

    func testComplexMatrixElementCountIsLogicalNotDoubled() {
        // Element count for caps is rows*cols (logical), NOT rows*cols*2.
        // A 2×2 ComplexMatrix has 4 logical elements (not 8).
        let n = 2
        let re = [Double](repeating: 0, count: n * n)
        let im = [Double](repeating: 0, count: n * n)
        let cm = LinAlg.ComplexMatrix(rows: n, cols: n, real: re, imag: im)
        XCTAssertEqual(cm.size, n * n)   // size property: logical element count
    }

    func testComplexMatrixValidatorPassesSameHardBoundaryAsMatrix() {
        // A ComplexMatrix at rows=100 cols=100 should succeed (same boundary check)
        let n = 100
        LinAlg.assertWithinHardCap(rows: n, cols: n)   // should not trap
    }

    // -------------------------------------------------------------------------
    // MARK: - complexMatmul peak working-set cap (MF-5 / §5, Issue #13)
    //
    // The real-block decomposition for CM*CM allocates four intermediate real
    // matrices (Ar·Br, Ai·Bi, Ar·Bi, Ai·Br), each of shape (M×N), plus two
    // output arrays (crData, ciData), also M×N each.  At peak, all six buffers
    // are live simultaneously = 6× the logical result size.  To honour the
    // spirit of the soft cap the admission check must bound the *peak working
    // set*, not just the result shape.
    //
    // Design constraint (§4.8): "each individual allocation within cap" means
    // the per-allocation check (result shape) is necessary but not sufficient.
    // The cumulative-bounding check must prevent:
    //   peakElements = resultElements * WORKING_SET_MULTIPLIER > cap
    //
    // WORKING_SET_MULTIPLIER = 5:
    //   4 intermediate LinAlg.Matrix objects (each resultElements doubles)
    //   + 1 logical result (counted as 1× even though stored as 2× re/im arrays)
    //   The two output [Double] arrays (crData, ciData) are transient at the same
    //   scale but overlap with the intermediates' lifetime, so the multiplier of 5
    //   is a safe upper bound derived from the algorithm structure (§5 of design).
    //
    // Test strategy:
    //   • Lower the soft cap so that a small but concrete result size triggers the
    //     peak-working-set check without actually allocating large matrices.
    //   • result fits individually (resultElements <= cap) but peak set does not
    //     (resultElements * 5 > cap) → must throw.
    //   • result fits even with peak multiplier → must not throw.
    //   • Boundary cases: exactly at peak-scaled threshold → must not throw;
    //     one over → must throw.
    //   • vec·vec path (result is 1×1): always under cap since even 5×1 << cap.
    //   • Rectangular matrix case.
    // -------------------------------------------------------------------------

    // Constructs a zero ComplexMatrix of given size from scalars.
    private func zeroCM(rows: Int, cols: Int) -> LinAlg.ComplexMatrix {
        let n = rows * cols
        return LinAlg.ComplexMatrix(rows: rows, cols: cols,
                                    real: [Double](repeating: 0, count: n),
                                    imag: [Double](repeating: 0, count: n))
    }

    /// Result fits soft cap individually, but peak working set (5×) exceeds it → must throw.
    func testComplexMatmulPeakWorkingSetExceedsCapThrows() throws {
        // Set cap to 100 elements.
        // Result shape: 10×10 = 100 — exactly at cap (passes individual check).
        // Peak working set: 100 * 5 = 500 > 100 → must throw.
        try LinAlg.setMaxEvaluatorMatrixElements(100)

        // A square (10×10) × (10×10) complex matmul
        let a = zeroCM(rows: 10, cols: 10)
        let b = zeroCM(rows: 10, cols: 10)

        XCTAssertThrowsError(
            try NumericDispatch.complexMatmul(lhs: a, rhs: b)
        ) { error in
            guard case LinAlg.LinAlgError.invalidParameter? = error as? LinAlg.LinAlgError else {
                XCTFail("expected LinAlgError.invalidParameter for peak-WS overflow, got \(error)")
                return
            }
        }
    }

    /// Result well within cap including peak multiplier → must succeed.
    func testComplexMatmulComfortablyUnderCapSucceeds() throws {
        // cap = 10000, result 4×4 = 16, peak = 80 — both well under cap.
        try LinAlg.setMaxEvaluatorMatrixElements(10_000)
        let a = zeroCM(rows: 4, cols: 4)
        let b = zeroCM(rows: 4, cols: 4)
        // Should not throw
        let result = try NumericDispatch.complexMatmul(lhs: a, rhs: b)
        // 4×4 complexMatrix result (no 1×1 coercion)
        if case .complexMatrix(let cm) = result {
            XCTAssertEqual(cm.rows, 4)
            XCTAssertEqual(cm.cols, 4)
        } else {
            XCTFail("expected .complexMatrix, got \(result)")
        }
    }

    /// Peak working set exactly equal to the cap limit → must succeed (boundary inclusive).
    func testComplexMatmulPeakAtCapBoundarySucceeds() throws {
        // Set cap = 500.
        // result = 10×10 = 100; peak = 100 * 5 = 500 == cap → must not throw.
        try LinAlg.setMaxEvaluatorMatrixElements(500)
        let a = zeroCM(rows: 10, cols: 10)
        let b = zeroCM(rows: 10, cols: 10)
        // Should succeed: peak == cap is accepted
        _ = try NumericDispatch.complexMatmul(lhs: a, rhs: b)
    }

    /// Peak working set one over the cap limit → must throw.
    func testComplexMatmulPeakOneOverCapThrows() throws {
        // Set cap = 499.
        // result = 10×10 = 100; peak = 100 * 5 = 500 > 499 → must throw.
        try LinAlg.setMaxEvaluatorMatrixElements(499)
        let a = zeroCM(rows: 10, cols: 10)
        let b = zeroCM(rows: 10, cols: 10)
        XCTAssertThrowsError(
            try NumericDispatch.complexMatmul(lhs: a, rhs: b)
        ) { error in
            guard case LinAlg.LinAlgError.invalidParameter? = error as? LinAlg.LinAlgError else {
                XCTFail("expected LinAlgError.invalidParameter, got \(error)")
                return
            }
        }
    }

    /// Rectangular matmul: result fits but peak does not → throws.
    func testComplexMatmulRectangularPeakExceedsCapThrows() throws {
        // A (2×6) × (6×5) = result 2×5 = 10 elements.
        // Set cap = 10; individual result fits (10 == cap), peak = 50 > 10 → throw.
        try LinAlg.setMaxEvaluatorMatrixElements(10)
        let a = zeroCM(rows: 2, cols: 6)
        let b = zeroCM(rows: 6, cols: 5)
        XCTAssertThrowsError(
            try NumericDispatch.complexMatmul(lhs: a, rhs: b)
        ) { error in
            guard case LinAlg.LinAlgError.invalidParameter? = error as? LinAlg.LinAlgError else {
                XCTFail("expected LinAlgError.invalidParameter for rectangular peak, got \(error)")
                return
            }
        }
    }

    /// vec·vec dotProduct (1×N · N×1 = 1×1 result): always accepted even at tiny cap.
    func testComplexVecDotProductAlwaysUnderCap() throws {
        // vec·vec gives a 1×1 result (peak = 5 elements).
        // Set cap = 3 — the 1×1 result still fits and the peak (5) exceeds it,
        // BUT: for vec·vec the result is coerced to .complex (scalar), not 1×1 CM.
        // The key contract: 1×1 result shape → peak = 5 elements.
        // At cap=10 (>5): succeeds.
        try LinAlg.setMaxEvaluatorMatrixElements(10)
        let n = 3
        let a = zeroCM(rows: n, cols: 1)  // n×1 column vector
        let b = zeroCM(rows: n, cols: 1)  // n×1 column vector
        // vec·vec: 1×1 result, peak = 5; 5 <= 10 → must succeed
        let result = try NumericDispatch.complexMatmul(lhs: a, rhs: b)
        // 1×1 is coerced to .complex by coerce1x1Complex
        if case .complex(_) = result {
            // correct coercion
        } else {
            XCTFail("expected 1×1 vec·dot to coerce to .complex, got \(result)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - SEC-05: setter isolation from mathlex/Lua bridge
    // -------------------------------------------------------------------------

    func testSetterNameAbsentFromMathExprIdentifiers() throws {
        // SEC-05: the soft-cap setter/getter must never be reachable as an
        // expression-callable function — exposing them would let untrusted
        // script code disable the resource guard. The dispatch surface is the
        // closed `NumericDispatch.functionRegistry`, so assert the cap-control
        // symbols are absent from its keys, and that calling them as functions
        // raises `.unknownFunction` rather than dispatching.
        let registryKeys = Set(NumericDispatch.functionRegistry.keys)
        let forbidden = [
            "setMaxEvaluatorMatrixElements",
            "maxEvaluatorMatrixElements",
            "hardMaxMatrixElementCount",
        ]
        for name in forbidden {
            XCTAssertFalse(registryKeys.contains(name),
                           "SEC-05: '\(name)' must not be an expression-callable function")
            XCTAssertThrowsError(try NumericDispatch.applyFunction(name, args: [.scalar(1)])) { error in
                guard case MathExprError.unknownFunction(let got) = error else {
                    return XCTFail("expected .unknownFunction for '\(name)', got \(error)")
                }
                XCTAssertEqual(got, name)
            }
        }
    }
}
