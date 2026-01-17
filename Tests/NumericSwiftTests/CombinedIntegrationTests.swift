//
//  CombinedIntegrationTests.swift
//  NumericSwiftTests
//
//  Tests for NumericSwift integration with both ArraySwift and PlotSwift.
//  These tests only run when compiled with both dependencies enabled.
//

#if NUMERICSWIFT_ARRAYSWIFT && NUMERICSWIFT_PLOTSWIFT

import XCTest
@testable import NumericSwift
import ArraySwift
import PlotSwift

final class CombinedIntegrationTests: XCTestCase {

    // MARK: - NDArray to Matrix to Visualization

    func testNDArrayHeatmapVisualization() {
        // Create NDArray data
        let array = NDArray.arange(start: 0, stop: 9).reshape([3, 3])

        // Convert to Matrix
        let matrix = LinAlg.Matrix(ndarray: array)!

        // Visualize
        let ctx = DrawingContext()
        matrix.drawHeatMap(ctx: ctx, x: 0, y: 0, width: 300, height: 300, gradient: ColorGradient.viridis)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    func testNDArrayStatisticsVisualization() {
        // Generate random data using ArraySwift
        let data = NDArray.randn([100])

        // Calculate statistics using NumericSwift via ArraySwift integration
        let dataMean = data.mean()
        let dataStddev = data.std()

        // Create histogram using PlotSwift integration
        let values = (0..<data.shape[0]).map { data[$0] }
        let bins = histogramBins(values, bins: 15)

        // Visualize
        let ctx = DrawingContext()

        // Draw histogram bars
        let barWidth = 450.0 / Double(bins.count)
        let maxCount = bins.map { $0.count }.max() ?? 1
        let scaledHeights = bins.map { Double($0.count) / Double(maxCount) * 300.0 }
        ctx.drawBarChart(scaledHeights, x: 25, y: 50, barWidth: barWidth - 2.0, color: .blue)

        // Overlay normal distribution curve based on computed statistics
        let curve = normalDistributionCurve(mean: dataMean, stddev: dataStddev, points: 100)
        let xMin = values.min() ?? -3.0
        let xMax = values.max() ?? 3.0
        let xRange = xMax - xMin
        let scaledCurve = curve.map { point -> (x: Double, y: Double) in
            let x = 25.0 + (point.x - xMin) / xRange * 450.0
            let y = 50.0 + point.y * dataStddev * Double(data.shape[0]) / Double(maxCount) * 300.0
            return (x, y)
        }
        ctx.drawLinePlot(scaledCurve, color: .red)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    // MARK: - Matrix Operations with Visualization

    func testMatrixDecompositionVisualization() {
        // Create matrix from NDArray
        let array = NDArray([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
        let matrix = LinAlg.Matrix(ndarray: array)!

        // Perform Cholesky decomposition
        guard let chol = LinAlg.cholesky(matrix) else {
            XCTFail("Cholesky decomposition failed")
            return
        }

        // Visualize both matrices side by side
        let ctx = DrawingContext()

        // Original matrix
        matrix.drawHeatMap(ctx: ctx, x: 25, y: 25, width: 250, height: 250, gradient: .diverging)
        matrix.drawGrid(ctx: ctx, x: 25, y: 25, width: 250, height: 250)

        // Cholesky factor
        chol.drawHeatMap(ctx: ctx, x: 425, y: 25, width: 250, height: 250, gradient: .diverging)
        chol.drawGrid(ctx: ctx, x: 425, y: 25, width: 250, height: 250)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    func testEigenvectorVisualization() {
        // 2D rotation matrix (unused - just demonstrating complex eigenvalues)
        let theta = Double.pi / 4  // 45 degrees
        let _ = NDArray([[Foundation.cos(theta), -Foundation.sin(theta)], [Foundation.sin(theta), Foundation.cos(theta)]])

        // Get eigenvalues - use a symmetric matrix for real eigenvalues and eigenvectors
        let symmetric = NDArray([[3, 1], [1, 2]])
        let symMatrix = LinAlg.Matrix(ndarray: symmetric)!

        // eig returns (values: [Double], imagParts: [Double], vectors: Matrix)
        let (eigenvalues, _, eigenvectors) = LinAlg.eig(symMatrix)

        // Visualize eigenvectors
        let ctx = DrawingContext()

        // Draw coordinate axes
        ctx.setStrokeColor(Color(red: 0.8, green: 0.8, blue: 0.8))
        ctx.moveTo(200, 0)
        ctx.lineTo(200, 400)
        ctx.moveTo(0, 200)
        ctx.lineTo(400, 200)
        ctx.strokePath()

        // Draw eigenvectors (scaled for visibility)
        let scale: Double = 100.0
        for i in 0..<2 {
            let vx = eigenvectors[0, i] * scale * eigenvalues[i]
            let vy = eigenvectors[1, i] * scale * eigenvalues[i]

            ctx.setStrokeColor(i == 0 ? .red : .blue)
            ctx.setStrokeWidth(2.0)
            ctx.moveTo(200, 200)
            ctx.lineTo(200.0 + vx, 200.0 - vy)  // Flip y for screen coords
            ctx.strokePath()
        }

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    // MARK: - Statistical Analysis Pipeline

    func testCorrelationHeatmap() {
        // Generate correlated data using ArraySwift
        let n = 50
        let x1 = NDArray.randn([n])
        let x2 = x1 * 0.8 + NDArray.randn([n]) * 0.2  // Correlated with x1
        let x3 = NDArray.randn([n])  // Independent
        let x4 = x1 * -0.5 + NDArray.randn([n]) * 0.5  // Negatively correlated

        // Extract arrays for correlation computation
        let arrays = [x1, x2, x3, x4]
        let vectorArrays = arrays.map { arr -> [Double] in
            (0..<n).map { arr[$0] }
        }

        // Compute correlation matrix
        var corrData: [[Double]] = []
        for i in 0..<4 {
            var row: [Double] = []
            for j in 0..<4 {
                // pearsonr returns TestResult? where statistic is the correlation coefficient
                let r = pearsonr(vectorArrays[i], vectorArrays[j])?.statistic ?? 0
                row.append(r)
            }
            corrData.append(row)
        }
        let corrMatrix = LinAlg.Matrix(corrData)

        // Visualize as heatmap
        let ctx = DrawingContext()
        corrMatrix.drawHeatMap(
            ctx: ctx,
            x: 50,
            y: 50,
            width: 300,
            height: 300,
            gradient: .diverging,
            minValue: -1,
            maxValue: 1
        )
        corrMatrix.drawGrid(ctx: ctx, x: 50, y: 50, width: 300, height: 300)

        // Verify diagonal is 1
        for i in 0..<4 {
            XCTAssertEqual(corrMatrix[i, i], 1.0, accuracy: 1e-10)
        }

        // Verify x1-x2 correlation is high
        XCTAssertGreaterThan(corrMatrix[0, 1], 0.5)

        // Verify x1-x4 correlation is negative
        XCTAssertLessThan(corrMatrix[0, 3], 0)
    }

    // MARK: - Cumulative Statistics Visualization

    func testCumulativeOperationsVisualization() {
        // Time series data
        let n = 100
        let noise = NDArray.randn([n]) * 0.1
        let trend = NDArray.arange(start: 0, stop: Double(n)) * 0.05
        let data = trend + noise

        // Cumulative sum for visualization
        let cumData = data.cumsum()

        // Extract for plotting
        let xValues = (0..<n).map { Double($0) }
        let yValues = (0..<n).map { cumData[$0] }
        let _ = zip(xValues, yValues).map { (x: $0, y: $1) }

        // Visualize
        let ctx = DrawingContext()

        // Scale points to fit canvas
        let scaled = scaleToRange(yValues, min: 50, max: 250)
        let scaledPoints = zip(xValues, scaled).map { (x: $0 * 4.5 + 25.0, y: 300.0 - $1) }

        ctx.drawLinePlot(scaledPoints, color: .blue, lineWidth: 1.5)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    // MARK: - Linear Algebra Visualization

    func testSVDVisualization() {
        // Create a matrix
        let array = NDArray([[3, 2, 2], [2, 3, -2]])
        let matrix = LinAlg.Matrix(ndarray: array)!

        // SVD returns (s: [Double], U: Matrix, Vt: Matrix)
        let (s, U, Vt) = LinAlg.svd(matrix)

        let ctx = DrawingContext()

        // Visualize U matrix
        U.drawHeatMap(ctx: ctx, x: 25, y: 25, width: 150, height: 150, gradient: .diverging, minValue: -1, maxValue: 1)
        U.drawGrid(ctx: ctx, x: 25, y: 25, width: 150, height: 150)

        // Visualize S as diagonal matrix
        let sMatrix = LinAlg.diag(s)
        sMatrix.drawHeatMap(ctx: ctx, x: 225, y: 25, width: 150, height: 150, gradient: .viridis)
        sMatrix.drawGrid(ctx: ctx, x: 225, y: 25, width: 150, height: 150)

        // Visualize Vt matrix
        Vt.drawHeatMap(ctx: ctx, x: 425, y: 25, width: 150, height: 150, gradient: .diverging, minValue: -1, maxValue: 1)
        Vt.drawGrid(ctx: ctx, x: 425, y: 25, width: 150, height: 150)

        // Visualize original matrix for comparison
        matrix.drawHeatMap(ctx: ctx, x: 625, y: 25, width: 150, height: 150, gradient: .diverging)
        matrix.drawGrid(ctx: ctx, x: 625, y: 25, width: 150, height: 150)

        // Verify reconstruction: U @ S @ Vt = original
        var sMatrixFull = LinAlg.zeros(2, 3)
        for i in 0..<min(s.count, 2) {
            sMatrixFull[i, i] = s[i]
        }
        let reconstructed = (U * sMatrixFull) * Vt
        for i in 0..<matrix.rows {
            for j in 0..<matrix.cols {
                XCTAssertEqual(matrix[i, j], reconstructed[i, j], accuracy: 1e-10)
            }
        }
    }

    // MARK: - Scatter Plot with Statistics

    func testScatterPlotWithRegression() {
        // Generate data with linear relationship
        let n = 30
        let xArr = NDArray.linspace(start: 0, stop: 10, num: n)
        let noise = NDArray.randn([n]) * 1.5
        let yArr = xArr * 2 + noise + 1  // y = 2x + 1 + noise

        // Extract values - 1D arrays use subscript with Int directly
        let x = (0..<n).map { xArr[$0] }
        let y = (0..<n).map { yArr[$0] }

        // Linear regression using NumericSwift
        guard let result = ols(y, addConstant(x.map { [$0] })) else {
            XCTFail("OLS regression failed")
            return
        }

        let ctx = DrawingContext()

        // Scale data to canvas
        let xScaled = scaleToRange(x, min: 50, max: 450)
        let yScaled = scaleToRange(y, min: 50, max: 350)
        let points = zip(xScaled, yScaled).map { (x: $0, y: 400.0 - $1) }

        // Draw scatter points
        ctx.drawScatterPlot(points, color: .blue, radius: 4.0)

        // Draw regression line
        let intercept = result.params[0]
        let slope = result.params[1]
        let yMin = intercept + slope * x.min()!
        let yMax = intercept + slope * x.max()!
        let yMinScaled = (yMin - y.min()!) / (y.max()! - y.min()!) * 300 + 50
        let yMaxScaled = (yMax - y.min()!) / (y.max()! - y.min()!) * 300 + 50

        ctx.setStrokeColor(.red)
        ctx.setStrokeWidth(2.0)
        ctx.moveTo(50, 400 - yMinScaled)
        ctx.lineTo(450, 400 - yMaxScaled)
        ctx.strokePath()

        // Verify slope is approximately 2
        XCTAssertEqual(slope, 2.0, accuracy: 0.5)
    }

    // MARK: - Color Mapping with NDArray

    func testColorMappingWithNDArray() {
        // Create 2D function values
        let size = 20
        var values: [[Double]] = []
        for i in 0..<size {
            var row: [Double] = []
            for j in 0..<size {
                let x = Double(i) / Double(size) * 4 - 2  // -2 to 2
                let y = Double(j) / Double(size) * 4 - 2
                row.append(Foundation.sin(x) * Foundation.cos(y))  // 2D wave pattern
            }
            values.append(row)
        }

        // Convert to NDArray and then to Matrix
        let ndarray = NDArray(values)
        let matrix = LinAlg.Matrix(ndarray: ndarray)!

        // Map to colors
        let gradient = ColorGradient.viridis
        let flat = matrix.data
        let colors = gradient.mapValues(flat, min: -1.0, max: 1.0)

        XCTAssertEqual(colors.count, size * size)

        // Verify color mapping range
        // Values near -1 should be dark purple (viridis start)
        // Values near 1 should be yellow (viridis end)

        // Visualize
        let ctx = DrawingContext()
        matrix.drawHeatMap(ctx: ctx, x: 0, y: 0, width: 400, height: 400, gradient: .viridis, minValue: -1, maxValue: 1)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }
}

#endif // NUMERICSWIFT_ARRAYSWIFT && NUMERICSWIFT_PLOTSWIFT
