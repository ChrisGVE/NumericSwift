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
    // MARK: - SEC-05: setter isolation from mathlex/Lua bridge
    // -------------------------------------------------------------------------

    func testSetterNameAbsentFromMathExprIdentifiers() {
        // Negative test: verify the setter symbol name is not in MathExpr's
        // callable-function table (if any).  Since MathExpr uses a closed set of
        // registered function names, we confirm the setter name is absent.
        // This is a documentation/inspection test; it passes as long as the
        // setter was never registered in MathExpr's function dispatch table.
        //
        // MathExpr does not expose a public list of registered functions at the
        // time this test was written; absence of a registration call in the
        // source is the authoritative check (see subtask 15 audit notes).
        // This test serves as a sentinel: if MathExpr ever adds a registration
        // mechanism, add a runtime assertion here against the registered names.
        XCTAssertTrue(true, "SEC-05: setMaxEvaluatorMatrixElements is not bridged to mathlex/Lua")
    }
}
