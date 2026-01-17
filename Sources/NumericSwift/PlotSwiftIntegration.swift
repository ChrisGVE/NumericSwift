//
//  PlotSwiftIntegration.swift
//  NumericSwift
//
//  Integration with PlotSwift for data visualization support.
//  This file is only compiled when NUMERICSWIFT_PLOTSWIFT is defined.
//
//  Licensed under the MIT License.
//

#if NUMERICSWIFT_PLOTSWIFT

import Foundation
import PlotSwift

// MARK: - Color Gradient for Data Visualization

/// A color gradient for mapping numerical values to colors.
///
/// Create a gradient and map data values to colors for heat maps or other visualizations:
///
/// ```swift
/// let gradient = ColorGradient.viridis
/// let color = gradient.colorAt(0.5)  // Color at 50% of the gradient
///
/// // Map data to colors
/// let values = [0.0, 0.25, 0.5, 0.75, 1.0]
/// let colors = values.map { gradient.colorAt($0) }
/// ```
public struct ColorGradient: Sendable {
    /// Color stops defining the gradient.
    public let stops: [(position: Double, color: Color)]

    /// Creates a gradient from color stops.
    ///
    /// - Parameter stops: Array of (position, color) tuples. Positions should be in [0, 1].
    public init(stops: [(position: Double, color: Color)]) {
        self.stops = stops.sorted { $0.position < $1.position }
    }

    /// Creates a simple two-color gradient.
    ///
    /// - Parameters:
    ///   - start: Color at position 0
    ///   - end: Color at position 1
    public init(from start: Color, to end: Color) {
        self.stops = [(0.0, start), (1.0, end)]
    }

    /// Returns the interpolated color at the given position.
    ///
    /// - Parameter t: Position in [0, 1]
    /// - Returns: Interpolated color
    public func colorAt(_ t: Double) -> Color {
        let clamped = Swift.max(0, Swift.min(1, t))

        guard stops.count >= 2 else {
            return stops.first?.color ?? Color.black
        }

        // Find surrounding stops
        var lower = stops[0]
        var upper = stops[stops.count - 1]

        for i in 0..<stops.count - 1 {
            if clamped >= stops[i].position && clamped <= stops[i + 1].position {
                lower = stops[i]
                upper = stops[i + 1]
                break
            }
        }

        // Interpolate
        let range = upper.position - lower.position
        let localT = range > 0 ? (clamped - lower.position) / range : 0

        return Color(
            red: lower.color.red + (upper.color.red - lower.color.red) * localT,
            green: lower.color.green + (upper.color.green - lower.color.green) * localT,
            blue: lower.color.blue + (upper.color.blue - lower.color.blue) * localT,
            alpha: lower.color.alpha + (upper.color.alpha - lower.color.alpha) * localT
        )
    }

    /// Maps an array of values to colors.
    ///
    /// - Parameters:
    ///   - values: Array of values
    ///   - min: Minimum value (maps to 0)
    ///   - max: Maximum value (maps to 1)
    /// - Returns: Array of colors
    public func mapValues(_ values: [Double], min: Double? = nil, max: Double? = nil) -> [Color] {
        let minVal = min ?? values.min() ?? 0
        let maxVal = max ?? values.max() ?? 1
        let range = maxVal - minVal

        return values.map { value in
            let t = range > 0 ? (value - minVal) / range : 0.5
            return colorAt(t)
        }
    }

    // MARK: - Predefined Gradients

    /// Viridis colormap (perceptually uniform, colorblind-friendly).
    public static let viridis = ColorGradient(stops: [
        (0.0, Color(red: 0.267, green: 0.004, blue: 0.329)),
        (0.25, Color(red: 0.282, green: 0.140, blue: 0.458)),
        (0.5, Color(red: 0.127, green: 0.566, blue: 0.551)),
        (0.75, Color(red: 0.369, green: 0.789, blue: 0.383)),
        (1.0, Color(red: 0.993, green: 0.906, blue: 0.144))
    ])

    /// Classic heat map gradient (blue -> cyan -> green -> yellow -> red).
    public static let heat = ColorGradient(stops: [
        (0.0, Color(red: 0, green: 0, blue: 1)),
        (0.25, Color(red: 0, green: 1, blue: 1)),
        (0.5, Color(red: 0, green: 1, blue: 0)),
        (0.75, Color(red: 1, green: 1, blue: 0)),
        (1.0, Color(red: 1, green: 0, blue: 0))
    ])

    /// Grayscale gradient.
    public static let grayscale = ColorGradient(from: Color.black, to: Color.white)

    /// Blue-white-red diverging gradient.
    public static let diverging = ColorGradient(stops: [
        (0.0, Color(red: 0.0, green: 0.0, blue: 0.8)),
        (0.5, Color(red: 1.0, green: 1.0, blue: 1.0)),
        (1.0, Color(red: 0.8, green: 0.0, blue: 0.0))
    ])
}

// MARK: - LinAlg.Matrix Extensions for Plotting

extension LinAlg.Matrix {

    /// Creates a heat map visualization of the matrix.
    ///
    /// - Parameters:
    ///   - ctx: Drawing context to draw into
    ///   - x: X position of the heat map
    ///   - y: Y position of the heat map
    ///   - width: Total width of the heat map
    ///   - height: Total height of the heat map
    ///   - gradient: Color gradient to use
    ///   - minValue: Minimum value for color mapping (uses matrix min if nil)
    ///   - maxValue: Maximum value for color mapping (uses matrix max if nil)
    ///
    /// Example:
    /// ```swift
    /// let matrix = LinAlg.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    /// let ctx = DrawingContext()
    /// matrix.drawHeatMap(ctx: ctx, x: 0, y: 0, width: 300, height: 300)
    /// ```
    public func drawHeatMap(
        ctx: DrawingContext,
        x: Double,
        y: Double,
        width: Double,
        height: Double,
        gradient: ColorGradient = .viridis,
        minValue: Double? = nil,
        maxValue: Double? = nil
    ) {
        let minVal = minValue ?? data.min() ?? 0
        let maxVal = maxValue ?? data.max() ?? 1
        let range = maxVal - minVal

        let cellWidth = width / Double(cols)
        let cellHeight = height / Double(rows)

        for row in 0..<rows {
            for col in 0..<cols {
                let value = self[row, col]
                let t = range > 0 ? (value - minVal) / range : 0.5
                let color = gradient.colorAt(t)

                let cellX = x + Double(col) * cellWidth
                let cellY = y + Double(rows - 1 - row) * cellHeight  // Flip Y for math coords

                ctx.setFillColor(color)
                ctx.rect(cellX, cellY, cellWidth, cellHeight)
                ctx.fillPath()
            }
        }
    }

    /// Draws grid lines for a matrix visualization.
    ///
    /// - Parameters:
    ///   - ctx: Drawing context
    ///   - x: X position
    ///   - y: Y position
    ///   - width: Total width
    ///   - height: Total height
    ///   - color: Line color
    ///   - lineWidth: Line width
    public func drawGrid(
        ctx: DrawingContext,
        x: Double,
        y: Double,
        width: Double,
        height: Double,
        color: Color = Color(red: 0.5, green: 0.5, blue: 0.5, alpha: 0.5),
        lineWidth: Double = 1.0
    ) {
        let cellWidth = width / Double(cols)
        let cellHeight = height / Double(rows)

        ctx.setStrokeColor(color)
        ctx.setStrokeWidth(lineWidth)

        // Vertical lines
        for col in 0...cols {
            let lineX = x + Double(col) * cellWidth
            ctx.moveTo(lineX, y)
            ctx.lineTo(lineX, y + height)
        }

        // Horizontal lines
        for row in 0...rows {
            let lineY = y + Double(row) * cellHeight
            ctx.moveTo(x, lineY)
            ctx.lineTo(x + width, lineY)
        }

        ctx.strokePath()
    }
}

// MARK: - Data Normalization Helpers

/// Normalizes an array of values to the range [0, 1].
///
/// - Parameter values: Input values
/// - Returns: Normalized values
public func normalizeForPlot(_ values: [Double]) -> [Double] {
    guard let minVal = values.min(), let maxVal = values.max() else {
        return values
    }
    let range = maxVal - minVal
    if range == 0 { return values.map { _ in 0.5 } }
    return values.map { ($0 - minVal) / range }
}

/// Scales values to fit within a given pixel range.
///
/// - Parameters:
///   - values: Input values
///   - min: Target minimum (e.g., 0 for x-axis)
///   - max: Target maximum (e.g., width for x-axis)
/// - Returns: Scaled values
public func scaleToRange(_ values: [Double], min targetMin: Double, max targetMax: Double) -> [Double] {
    guard let minVal = values.min(), let maxVal = values.max() else {
        return values
    }
    let range = maxVal - minVal
    let targetRange = targetMax - targetMin
    if range == 0 { return values.map { _ in (targetMin + targetMax) / 2 } }
    return values.map { targetMin + (($0 - minVal) / range) * targetRange }
}

// MARK: - Distribution Visualization

/// Generates points for a normal distribution curve.
///
/// - Parameters:
///   - mean: Distribution mean
///   - stddev: Distribution standard deviation
///   - xMin: Minimum x value
///   - xMax: Maximum x value
///   - points: Number of points to generate
/// - Returns: Array of (x, y) tuples
public func normalDistributionCurve(
    mean: Double,
    stddev: Double,
    xMin: Double? = nil,
    xMax: Double? = nil,
    points: Int = 100
) -> [(x: Double, y: Double)] {
    let minX = xMin ?? (mean - 4 * stddev)
    let maxX = xMax ?? (mean + 4 * stddev)
    let step = (maxX - minX) / Double(points - 1)

    let norm = NormalDistribution(loc: mean, scale: stddev)

    return (0..<points).map { i in
        let x = minX + Double(i) * step
        return (x, norm.pdf(x))
    }
}

/// Generates histogram bin counts for data.
///
/// - Parameters:
///   - data: Input data
///   - bins: Number of bins
/// - Returns: Array of (binCenter, count) tuples
public func histogramBins(
    _ data: [Double],
    bins: Int = 10
) -> [(center: Double, count: Int)] {
    guard let minVal = data.min(), let maxVal = data.max() else {
        return []
    }

    let range = maxVal - minVal
    let binWidth = range / Double(bins)

    var counts = [Int](repeating: 0, count: bins)

    for value in data {
        var binIndex = Int((value - minVal) / binWidth)
        if binIndex >= bins { binIndex = bins - 1 }
        if binIndex < 0 { binIndex = 0 }
        counts[binIndex] += 1
    }

    return (0..<bins).map { i in
        let center = minVal + (Double(i) + 0.5) * binWidth
        return (center, counts[i])
    }
}

// MARK: - DrawingContext Convenience Extensions

extension DrawingContext {

    /// Draws a line plot from data points.
    ///
    /// - Parameters:
    ///   - points: Array of (x, y) tuples
    ///   - color: Line color
    ///   - lineWidth: Line width
    public func drawLinePlot(
        _ points: [(x: Double, y: Double)],
        color: Color = .blue,
        lineWidth: Double = 2.0
    ) {
        guard let first = points.first else { return }

        setStrokeColor(color)
        setStrokeWidth(lineWidth)

        moveTo(first.x, first.y)
        for point in points.dropFirst() {
            lineTo(point.x, point.y)
        }
        strokePath()
    }

    /// Draws scatter plot markers at data points.
    ///
    /// - Parameters:
    ///   - points: Array of (x, y) tuples
    ///   - color: Marker fill color
    ///   - radius: Marker radius
    public func drawScatterPlot(
        _ points: [(x: Double, y: Double)],
        color: Color = .blue,
        radius: Double = 3.0
    ) {
        setFillColor(color)

        for point in points {
            circle(cx: point.x, cy: point.y, r: radius)
            fillPath()
        }
    }

    /// Draws a bar chart.
    ///
    /// - Parameters:
    ///   - values: Bar heights
    ///   - x: X position of first bar
    ///   - y: Y position (baseline)
    ///   - barWidth: Width of each bar
    ///   - spacing: Space between bars
    ///   - color: Bar fill color
    public func drawBarChart(
        _ values: [Double],
        x: Double,
        y: Double,
        barWidth: Double,
        spacing: Double = 2.0,
        color: Color = .blue
    ) {
        setFillColor(color)

        for (i, value) in values.enumerated() {
            let barX = x + Double(i) * (barWidth + spacing)
            rect(barX, y, barWidth, value)
            fillPath()
        }
    }
}

#endif // NUMERICSWIFT_PLOTSWIFT
