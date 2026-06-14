// Timing.swift
// Sources/NumericSwiftBench/
//
// Monotonic high-resolution timing primitives for the NumericSwiftBench harness.
//
// Design choices:
//   • Uses clock_gettime_nsec_np(CLOCK_UPTIME_RAW) on Apple platforms — the
//     raw uptime clock ticks at the hardware counter rate and is not adjusted
//     by NTP or wall-clock skew, making it the right choice for short-duration
//     micro-benchmarks.
//   • Warmup: caller specifies how many iterations to discard before recording.
//   • Measurement: caller specifies how many recorded samples to collect.
//   • Aggregation: median is used rather than mean so that a single OS
//     preemption spike cannot skew the reported time.

import Foundation

// MARK: - Timer

/// Monotonic nanosecond timestamp using the raw uptime clock.
///
/// Returns nanoseconds since an unspecified epoch; only differences are
/// meaningful. The raw uptime clock is not adjusted for NTP or sleep,
/// making it suitable for tight performance measurements.
func nowNanoseconds() -> UInt64 {
  clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
}

// MARK: - Median timing driver

/// Measures the median execution time of a closure over repeated samples.
///
/// The first `warmup` invocations are discarded to allow branch-prediction
/// and instruction-cache effects to stabilise. After warmup, `samples`
/// timed invocations are recorded and the median nanosecond duration is
/// returned.
///
/// - Parameters:
///   - warmup: Number of untimed warm-up iterations (default 5).
///   - samples: Number of timed samples to collect (default 31).
///   - body: The workload closure. It must do a non-trivial amount of
///     observable work so the compiler cannot hoist it out of the loop.
///     Use `blackhole(_:)` on results if needed.
/// - Returns: Median execution time in nanoseconds.
@discardableResult
func medianNanoseconds(
  warmup: Int = 5,
  samples: Int = 31,
  body: () -> Void
) -> UInt64 {
  // Warmup phase: run the body without recording.
  for _ in 0..<warmup {
    body()
  }

  // Measurement phase: collect `samples` timed durations.
  var durations = [UInt64](repeating: 0, count: samples)
  for i in 0..<samples {
    let t0 = nowNanoseconds()
    body()
    let t1 = nowNanoseconds()
    durations[i] = t1 - t0
  }

  // Sort in-place and return the middle element (floor median).
  durations.sort()
  return durations[samples / 2]
}

// MARK: - Utility

/// Prevents the compiler from optimising away a value.
///
/// Writes `value` to a volatile-equivalent sink so the result must be
/// materialised. Used inside benchmark bodies to ensure the compiler cannot
/// hoist or dead-code-eliminate the computation being measured.
@inline(never)
func blackhole<T>(_ value: T) {
  _ = value
}
