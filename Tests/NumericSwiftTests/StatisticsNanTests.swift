//
//  StatisticsNanTests.swift
//  NumericSwift
//
//  Tests for NaN-aware statistics functions.
//
//  Licensed under the Apache License, Version 2.0.
//

import Testing

@testable import NumericSwift

// MARK: - nanmean

@Suite("nanmean")
struct NanMeanTests {
  @Test("no NaN matches mean")
  func noNaN() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0]
    #expect(Stats.nanmean(data) == Stats.mean(data))
  }

  @Test("some NaN values ignored")
  func someNaN() {
    let data = [1.0, Double.nan, 3.0, Double.nan, 5.0]
    #expect(Stats.nanmean(data) == Stats.mean([1.0, 3.0, 5.0]))
  }

  @Test("all NaN returns NaN")
  func allNaN() {
    #expect(Stats.nanmean([Double.nan, Double.nan]).isNaN)
  }

  @Test("empty array returns NaN")
  func empty() {
    #expect(Stats.nanmean([]).isNaN)
  }

  @Test("single non-NaN among NaNs")
  func singleAmongNaN() {
    #expect(Stats.nanmean([Double.nan, 7.0, Double.nan]) == 7.0)
  }
}

// MARK: - nanmedian

@Suite("nanmedian")
struct NanMedianTests {
  @Test("no NaN matches median")
  func noNaN() {
    let data = [3.0, 1.0, 4.0, 1.0, 5.0]
    #expect(Stats.nanmedian(data) == Stats.median(data))
  }

  @Test("some NaN values ignored")
  func someNaN() {
    let data = [Double.nan, 2.0, 4.0, Double.nan, 6.0]
    #expect(Stats.nanmedian(data) == Stats.median([2.0, 4.0, 6.0]))
  }

  @Test("all NaN returns NaN")
  func allNaN() {
    #expect(Stats.nanmedian([Double.nan]).isNaN)
  }

  @Test("empty array returns NaN")
  func empty() {
    #expect(Stats.nanmedian([]).isNaN)
  }

  @Test("single non-NaN among NaNs")
  func singleAmongNaN() {
    #expect(Stats.nanmedian([Double.nan, 42.0, Double.nan]) == 42.0)
  }
}

// MARK: - nanvariance

@Suite("nanvariance")
struct NanVarianceTests {
  @Test("no NaN matches variance")
  func noNaN() {
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    #expect(Stats.nanvariance(data) == Stats.variance(data))
  }

  @Test("some NaN values ignored")
  func someNaN() {
    let clean = [2.0, 4.0, 6.0]
    let withNaN = [2.0, Double.nan, 4.0, Double.nan, 6.0]
    #expect(Stats.nanvariance(withNaN) == Stats.variance(clean))
  }

  @Test("ddof forwarded")
  func ddof() {
    let clean = [1.0, 2.0, 3.0]
    let withNaN = [1.0, Double.nan, 2.0, 3.0]
    #expect(Stats.nanvariance(withNaN, ddof: 1) == Stats.variance(clean, ddof: 1))
  }

  @Test("all NaN returns NaN")
  func allNaN() {
    #expect(Stats.nanvariance([Double.nan, Double.nan]).isNaN)
  }

  @Test("empty array returns NaN")
  func empty() {
    #expect(Stats.nanvariance([]).isNaN)
  }
}

// MARK: - nanstd

@Suite("nanstd")
struct NanStdTests {
  @Test("no NaN matches stddev")
  func noNaN() {
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    #expect(Stats.nanstd(data) == Stats.stddev(data))
  }

  @Test("some NaN values ignored")
  func someNaN() {
    let clean = [2.0, 4.0, 6.0]
    let withNaN = [2.0, Double.nan, 4.0, 6.0]
    #expect(Stats.nanstd(withNaN) == Stats.stddev(clean))
  }

  @Test("ddof forwarded")
  func ddof() {
    let clean = [1.0, 2.0, 3.0]
    let withNaN = [1.0, Double.nan, 2.0, 3.0]
    #expect(Stats.nanstd(withNaN, ddof: 1) == Stats.stddev(clean, ddof: 1))
  }

  @Test("all NaN returns NaN")
  func allNaN() {
    #expect(Stats.nanstd([Double.nan]).isNaN)
  }

  @Test("empty array returns NaN")
  func empty() {
    #expect(Stats.nanstd([]).isNaN)
  }
}

// MARK: - nanmin / nanmax

@Suite("nanmin")
struct NanMinTests {
  @Test("no NaN matches amin")
  func noNaN() {
    let data = [3.0, 1.0, 4.0, 1.0, 5.0]
    #expect(Stats.nanmin(data) == Stats.amin(data))
  }

  @Test("some NaN values ignored")
  func someNaN() {
    #expect(Stats.nanmin([Double.nan, 3.0, Double.nan, 1.0]) == 1.0)
  }

  @Test("all NaN returns NaN")
  func allNaN() {
    #expect(Stats.nanmin([Double.nan, Double.nan]).isNaN)
  }

  @Test("empty array returns NaN")
  func empty() {
    #expect(Stats.nanmin([]).isNaN)
  }

  @Test("single non-NaN among NaNs")
  func singleAmongNaN() {
    #expect(Stats.nanmin([Double.nan, -5.0, Double.nan]) == -5.0)
  }
}

@Suite("nanmax")
struct NanMaxTests {
  @Test("no NaN matches amax")
  func noNaN() {
    let data = [3.0, 1.0, 4.0, 1.0, 5.0]
    #expect(Stats.nanmax(data) == Stats.amax(data))
  }

  @Test("some NaN values ignored")
  func someNaN() {
    #expect(Stats.nanmax([Double.nan, 3.0, Double.nan, 7.0]) == 7.0)
  }

  @Test("all NaN returns NaN")
  func allNaN() {
    #expect(Stats.nanmax([Double.nan, Double.nan]).isNaN)
  }

  @Test("empty array returns NaN")
  func empty() {
    #expect(Stats.nanmax([]).isNaN)
  }

  @Test("single non-NaN among NaNs")
  func singleAmongNaN() {
    #expect(Stats.nanmax([Double.nan, 99.0, Double.nan]) == 99.0)
  }
}

// MARK: - nansum

@Suite("nansum")
struct NanSumTests {
  @Test("no NaN matches sum")
  func noNaN() {
    let data = [1.0, 2.0, 3.0]
    #expect(Stats.nansum(data) == sum(data))
  }

  @Test("some NaN values ignored")
  func someNaN() {
    #expect(Stats.nansum([1.0, Double.nan, 2.0, Double.nan, 3.0]) == 6.0)
  }

  @Test("all NaN returns 0")
  func allNaN() {
    // numpy returns 0.0 for nansum of all-NaN arrays
    #expect(Stats.nansum([Double.nan, Double.nan]) == 0.0)
  }

  @Test("empty array returns 0")
  func empty() {
    #expect(Stats.nansum([]) == 0.0)
  }

  @Test("single non-NaN among NaNs")
  func singleAmongNaN() {
    #expect(Stats.nansum([Double.nan, 5.0, Double.nan]) == 5.0)
  }
}

// MARK: - nanpercentile

@Suite("nanpercentile")
struct NanPercentileTests {
  @Test("no NaN matches percentile")
  func noNaN() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0]
    #expect(Stats.nanpercentile(data, 50) == Stats.percentile(data, 50))
  }

  @Test("some NaN values ignored")
  func someNaN() {
    let clean = [1.0, 2.0, 3.0, 4.0, 5.0]
    let withNaN = [1.0, Double.nan, 2.0, 3.0, Double.nan, 4.0, 5.0]
    #expect(Stats.nanpercentile(withNaN, 50) == Stats.percentile(clean, 50))
  }

  @Test("all NaN returns NaN")
  func allNaN() {
    #expect(Stats.nanpercentile([Double.nan, Double.nan], 50).isNaN)
  }

  @Test("empty array returns NaN")
  func empty() {
    #expect(Stats.nanpercentile([], 50).isNaN)
  }

  @Test("single non-NaN among NaNs")
  func singleAmongNaN() {
    #expect(Stats.nanpercentile([Double.nan, 8.0, Double.nan], 50) == 8.0)
  }

  @Test("0th percentile is nanmin")
  func p0() {
    let data = [Double.nan, 3.0, 1.0, Double.nan, 5.0]
    #expect(Stats.nanpercentile(data, 0) == Stats.nanmin(data))
  }

  @Test("100th percentile is nanmax")
  func p100() {
    let data = [Double.nan, 3.0, 1.0, Double.nan, 5.0]
    #expect(Stats.nanpercentile(data, 100) == Stats.nanmax(data))
  }
}
