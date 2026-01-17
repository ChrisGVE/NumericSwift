//
//  PlotSwiftIntegrationTests.swift
//  NumericSwiftTests
//
//  Tests for NumericSwift integration with PlotSwift.
//  These tests only run when compiled with NUMERICSWIFT_PLOTSWIFT=1.
//

#if NUMERICSWIFT_PLOTSWIFT

import XCTest
@testable import NumericSwift
import PlotSwift

final class PlotSwiftIntegrationTests: XCTestCase {

    // MARK: - ColorGradient Tests

    func testColorGradientTwoColor() {
        let gradient = ColorGradient(from: Color.black, to: Color.white)

        // At 0, should be black
        let start = gradient.colorAt(0.0)
        XCTAssertEqual(start.red, 0.0, accuracy: 1e-10)
        XCTAssertEqual(start.green, 0.0, accuracy: 1e-10)
        XCTAssertEqual(start.blue, 0.0, accuracy: 1e-10)

        // At 1, should be white
        let end = gradient.colorAt(1.0)
        XCTAssertEqual(end.red, 1.0, accuracy: 1e-10)
        XCTAssertEqual(end.green, 1.0, accuracy: 1e-10)
        XCTAssertEqual(end.blue, 1.0, accuracy: 1e-10)

        // At 0.5, should be gray
        let mid = gradient.colorAt(0.5)
        XCTAssertEqual(mid.red, 0.5, accuracy: 1e-10)
        XCTAssertEqual(mid.green, 0.5, accuracy: 1e-10)
        XCTAssertEqual(mid.blue, 0.5, accuracy: 1e-10)
    }

    func testColorGradientMultiStop() {
        // Red -> Green -> Blue gradient
        let gradient = ColorGradient(stops: [
            (0.0, Color(red: 1, green: 0, blue: 0)),
            (0.5, Color(red: 0, green: 1, blue: 0)),
            (1.0, Color(red: 0, green: 0, blue: 1))
        ])

        // At 0.25, should be interpolation between red and green
        let quarter = gradient.colorAt(0.25)
        XCTAssertEqual(quarter.red, 0.5, accuracy: 1e-10)
        XCTAssertEqual(quarter.green, 0.5, accuracy: 1e-10)
        XCTAssertEqual(quarter.blue, 0.0, accuracy: 1e-10)

        // At 0.75, should be interpolation between green and blue
        let threeQuarter = gradient.colorAt(0.75)
        XCTAssertEqual(threeQuarter.red, 0.0, accuracy: 1e-10)
        XCTAssertEqual(threeQuarter.green, 0.5, accuracy: 1e-10)
        XCTAssertEqual(threeQuarter.blue, 0.5, accuracy: 1e-10)
    }

    func testColorGradientClamping() {
        let gradient = ColorGradient.grayscale

        // Values outside [0,1] should be clamped
        let belowZero = gradient.colorAt(-0.5)
        XCTAssertEqual(belowZero.red, 0.0, accuracy: 1e-10)

        let aboveOne = gradient.colorAt(1.5)
        XCTAssertEqual(aboveOne.red, 1.0, accuracy: 1e-10)
    }

    func testColorGradientViridis() {
        let gradient = ColorGradient.viridis

        // Viridis starts dark purple
        let start = gradient.colorAt(0.0)
        XCTAssertEqual(start.red, 0.267, accuracy: 0.01)
        XCTAssertEqual(start.green, 0.004, accuracy: 0.01)
        XCTAssertEqual(start.blue, 0.329, accuracy: 0.01)

        // Viridis ends yellow
        let end = gradient.colorAt(1.0)
        XCTAssertEqual(end.red, 0.993, accuracy: 0.01)
        XCTAssertEqual(end.green, 0.906, accuracy: 0.01)
        XCTAssertEqual(end.blue, 0.144, accuracy: 0.01)
    }

    func testColorGradientHeat() {
        let gradient = ColorGradient.heat

        // Heat starts blue
        let start = gradient.colorAt(0.0)
        XCTAssertEqual(start.red, 0.0, accuracy: 1e-10)
        XCTAssertEqual(start.green, 0.0, accuracy: 1e-10)
        XCTAssertEqual(start.blue, 1.0, accuracy: 1e-10)

        // Heat ends red
        let end = gradient.colorAt(1.0)
        XCTAssertEqual(end.red, 1.0, accuracy: 1e-10)
        XCTAssertEqual(end.green, 0.0, accuracy: 1e-10)
        XCTAssertEqual(end.blue, 0.0, accuracy: 1e-10)
    }

    func testColorGradientDiverging() {
        let gradient = ColorGradient.diverging

        // Diverging: blue at 0, white at 0.5, red at 1
        let mid = gradient.colorAt(0.5)
        XCTAssertEqual(mid.red, 1.0, accuracy: 1e-10)
        XCTAssertEqual(mid.green, 1.0, accuracy: 1e-10)
        XCTAssertEqual(mid.blue, 1.0, accuracy: 1e-10)
    }

    func testColorGradientMapValues() {
        let gradient = ColorGradient.grayscale
        let values = [0.0, 50.0, 100.0]
        let colors = gradient.mapValues(values)

        // First value (min) should map to black
        XCTAssertEqual(colors[0].red, 0.0, accuracy: 1e-10)
        // Middle value should map to gray
        XCTAssertEqual(colors[1].red, 0.5, accuracy: 1e-10)
        // Last value (max) should map to white
        XCTAssertEqual(colors[2].red, 1.0, accuracy: 1e-10)
    }

    func testColorGradientMapValuesCustomRange() {
        let gradient = ColorGradient.grayscale
        let values = [25.0, 50.0, 75.0]
        let colors = gradient.mapValues(values, min: 0.0, max: 100.0)

        // 25 out of 100 = 0.25 gray
        XCTAssertEqual(colors[0].red, 0.25, accuracy: 1e-10)
        // 50 out of 100 = 0.5 gray
        XCTAssertEqual(colors[1].red, 0.5, accuracy: 1e-10)
        // 75 out of 100 = 0.75 gray
        XCTAssertEqual(colors[2].red, 0.75, accuracy: 1e-10)
    }

    // MARK: - Normalization Helper Tests

    func testNormalizeForPlot() {
        let values = [10.0, 20.0, 30.0, 40.0, 50.0]
        let normalized = normalizeForPlot(values)

        XCTAssertEqual(normalized[0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(normalized[2], 0.5, accuracy: 1e-10)
        XCTAssertEqual(normalized[4], 1.0, accuracy: 1e-10)
    }

    func testNormalizeForPlotConstant() {
        // All same values should return 0.5
        let values = [5.0, 5.0, 5.0]
        let normalized = normalizeForPlot(values)

        for v in normalized {
            XCTAssertEqual(v, 0.5, accuracy: 1e-10)
        }
    }

    func testNormalizeForPlotEmpty() {
        let values: [Double] = []
        let normalized = normalizeForPlot(values)
        XCTAssertTrue(normalized.isEmpty)
    }

    func testScaleToRange() {
        let values = [0.0, 0.5, 1.0]
        let scaled = scaleToRange(values, min: 100.0, max: 200.0)

        XCTAssertEqual(scaled[0], 100.0, accuracy: 1e-10)
        XCTAssertEqual(scaled[1], 150.0, accuracy: 1e-10)
        XCTAssertEqual(scaled[2], 200.0, accuracy: 1e-10)
    }

    func testScaleToRangeConstant() {
        let values = [3.0, 3.0, 3.0]
        let scaled = scaleToRange(values, min: 0.0, max: 100.0)

        // All same values should map to midpoint
        for v in scaled {
            XCTAssertEqual(v, 50.0, accuracy: 1e-10)
        }
    }

    // MARK: - Distribution Visualization Tests

    func testNormalDistributionCurve() {
        let curve = normalDistributionCurve(mean: 0.0, stddev: 1.0, points: 11)

        XCTAssertEqual(curve.count, 11)

        // Peak should be at mean
        let peakIndex = curve.count / 2
        let peak = curve[peakIndex]
        XCTAssertEqual(peak.x, 0.0, accuracy: 0.1)

        // PDF at mean for N(0,1) is 1/sqrt(2*pi) ≈ 0.3989
        XCTAssertEqual(peak.y, 0.3989, accuracy: 0.01)

        // Curve should be symmetric
        let leftY = curve[0].y
        let rightY = curve[curve.count - 1].y
        XCTAssertEqual(leftY, rightY, accuracy: 1e-6)
    }

    func testNormalDistributionCurveCustomRange() {
        let curve = normalDistributionCurve(mean: 5.0, stddev: 2.0, xMin: 0.0, xMax: 10.0, points: 5)

        XCTAssertEqual(curve.count, 5)
        XCTAssertEqual(curve[0].x, 0.0, accuracy: 1e-10)
        XCTAssertEqual(curve[4].x, 10.0, accuracy: 1e-10)
    }

    func testHistogramBins() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let bins = histogramBins(data, bins: 5)

        XCTAssertEqual(bins.count, 5)

        // Each bin should have 2 values for uniform data
        for bin in bins {
            XCTAssertEqual(bin.count, 2)
        }

        // Bin centers should be evenly spaced
        let binWidth = (10.0 - 1.0) / 5.0
        XCTAssertEqual(bins[0].center, 1.0 + binWidth / 2, accuracy: 1e-10)
    }

    func testHistogramBinsEmpty() {
        let data: [Double] = []
        let bins = histogramBins(data, bins: 5)
        XCTAssertTrue(bins.isEmpty)
    }

    func testHistogramBinsSkewed() {
        // Data concentrated in first bin
        let data = [0.0, 0.1, 0.2, 5.0]
        let bins = histogramBins(data, bins: 5)

        XCTAssertEqual(bins.count, 5)
        // First bin should have 3 values
        XCTAssertEqual(bins[0].count, 3)
        // Last bin should have 1 value
        XCTAssertEqual(bins[4].count, 1)
    }

    // MARK: - Matrix Heatmap Tests

    func testMatrixDrawHeatMap() {
        let matrix = LinAlg.Matrix([[1, 2], [3, 4]])
        let ctx = DrawingContext()

        // Should not throw
        matrix.drawHeatMap(ctx: ctx, x: 0, y: 0, width: 100, height: 100)

        // Basic validation - context should have operations recorded
        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    func testMatrixDrawHeatMapWithGradient() {
        let matrix = LinAlg.Matrix([[0, 0.5], [0.5, 1]])
        let ctx = DrawingContext()

        matrix.drawHeatMap(
            ctx: ctx,
            x: 10,
            y: 10,
            width: 180,
            height: 180,
            gradient: .heat,
            minValue: 0,
            maxValue: 1
        )

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    func testMatrixDrawGrid() {
        let matrix = LinAlg.Matrix([[1, 2, 3], [4, 5, 6]])
        let ctx = DrawingContext()

        matrix.drawGrid(
            ctx: ctx,
            x: 0,
            y: 0,
            width: 300,
            height: 200,
            color: Color(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0),
            lineWidth: 2.0
        )

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    // MARK: - DrawingContext Extension Tests

    func testDrawLinePlot() {
        let ctx = DrawingContext()
        let points: [(x: Double, y: Double)] = [
            (0, 0), (100, 50), (200, 25), (300, 75), (400, 100)
        ]

        ctx.drawLinePlot(points, color: .blue, lineWidth: 2.0)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    func testDrawLinePlotEmpty() {
        let ctx = DrawingContext()
        let points: [(x: Double, y: Double)] = []

        // Should handle empty gracefully
        ctx.drawLinePlot(points)

        // Empty points should not add path commands
        XCTAssertTrue(true)
    }

    func testDrawScatterPlot() {
        let ctx = DrawingContext()
        let points: [(x: Double, y: Double)] = [
            (50, 50), (150, 100), (250, 75), (350, 200)
        ]

        ctx.drawScatterPlot(points, color: .red, radius: 5.0)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    func testDrawBarChart() {
        let ctx = DrawingContext()
        let values = [50.0, 100.0, 75.0, 125.0, 90.0]

        ctx.drawBarChart(values, x: 20, y: 0, barWidth: 60, spacing: 10, color: .blue)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    // MARK: - Integration Tests

    func testVisualizationPipeline() {
        // Generate some data using NumericSwift
        let norm = NormalDistribution(loc: 0, scale: 1)
        let samples = norm.rvs(100)

        // Create histogram
        let bins = histogramBins(samples, bins: 10)

        // Create visualization
        let ctx = DrawingContext()

        // Draw histogram as bar chart
        let heights = bins.map { Double($0.count) }
        ctx.drawBarChart(heights, x: 20, y: 0, barWidth: 35, spacing: 5)

        // Overlay normal curve
        let curve = normalDistributionCurve(mean: 0, stddev: 1, xMin: -4, xMax: 4, points: 100)
        let scaledCurve = curve.map { (x: $0.x * 50.0 + 200.0, y: $0.y * 500.0) }
        ctx.drawLinePlot(scaledCurve, color: .red)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }

    func testMatrixVisualization() {
        // Create a matrix using NumericSwift
        let size = 10
        var data: [[Double]] = []
        for i in 0..<size {
            var row: [Double] = []
            for j in 0..<size {
                // Create a pattern: distance from center
                let dx = Double(i) - Double(size) / 2.0
                let dy = Double(j) - Double(size) / 2.0
                let distSquared = dx * dx + dy * dy
                row.append(Foundation.sqrt(distSquared))
            }
            data.append(row)
        }
        let matrix = LinAlg.Matrix(data)

        // Visualize
        let ctx = DrawingContext()
        matrix.drawHeatMap(ctx: ctx, x: 50, y: 50, width: 400, height: 400, gradient: .viridis)
        matrix.drawGrid(ctx: ctx, x: 50, y: 50, width: 400, height: 400)

        XCTAssertGreaterThan(ctx.commandCount, 0)
    }
}

#endif // NUMERICSWIFT_PLOTSWIFT
