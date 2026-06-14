// Gate.swift
// Sources/NumericSwiftBench/
//
// Ratio-gate model for the NumericSwiftBench harness.
//
// A Gate measures TWO legs (numerator / denominator) and checks whether their
// ratio satisfies a threshold (ratio ≤ threshold).
//
// Gate states:
//   • PASS    — both legs ran and ratio ≤ threshold
//   • FAIL    — both legs ran and ratio > threshold
//   • PENDING — the numerator leg is a placeholder (unified path not yet
//               implemented). The gate is scaffolded but not yet evaluable.
//               This is the correct Phase-0 state for gates whose "unified"
//               leg does not exist yet. It is NOT treated as PASS.
//   • SKIP    — gate was deliberately excluded from this run (e.g. via CLI).
//   • ERROR   — an unexpected exception occurred during measurement.

import Foundation

// MARK: - Gate state

/// The outcome of evaluating a single performance ratio gate.
public enum GateState: Equatable {
  /// Both legs measured; ratio satisfies the threshold.
  case pass(ratio: Double)
  /// Both legs measured; ratio exceeds the threshold.
  case fail(ratio: Double)
  /// The numerator leg is a stub — unified path not yet implemented.
  /// Reported as PENDING in the summary, NOT as PASS.
  case pending(reason: String)
  /// Gate was skipped for this run.
  case skip(reason: String)
  /// Unexpected error during measurement.
  case error(description: String)

  /// True only when the gate fully passed.
  public var isPassing: Bool {
    if case .pass = self { return true }
    return false
  }

  /// True when the gate is not yet measurable (unified path absent).
  public var isPending: Bool {
    if case .pending = self { return true }
    return false
  }

  /// One-line human-readable label.
  public var label: String {
    switch self {
    case .pass(let r):    return String(format: "PASS   (ratio %.4f)", r)
    case .fail(let r):    return String(format: "FAIL   (ratio %.4f)", r)
    case .pending(let m): return "PENDING (\(m))"
    case .skip(let m):    return "SKIP    (\(m))"
    case .error(let m):   return "ERROR   (\(m))"
    }
  }
}

// MARK: - BenchConfig

/// Timing parameters shared across all gates in a run.
public struct BenchConfig {
  /// Untimed warm-up iterations before each leg.
  public let warmupIterations: Int
  /// Number of timed samples collected per leg; median is reported.
  public let timedSamples: Int
  /// Matrix side dimension used for square-matrix gates.
  public let matrixSize: Int

  public init(
    warmupIterations: Int = 5,
    timedSamples: Int = 31,
    matrixSize: Int = 64
  ) {
    self.warmupIterations = warmupIterations
    self.timedSamples = timedSamples
    self.matrixSize = matrixSize
  }

  /// Override via environment variables for CI tuning.
  ///
  /// Recognised variables:
  ///   BENCH_WARMUP   — warmup iteration count
  ///   BENCH_SAMPLES  — timed sample count
  ///   BENCH_MATSIZE  — matrix side dimension
  public static func fromEnvironment(base: BenchConfig = BenchConfig()) -> BenchConfig {
    let env = ProcessInfo.processInfo.environment
    let warmup = env["BENCH_WARMUP"].flatMap(Int.init) ?? base.warmupIterations
    let samples = env["BENCH_SAMPLES"].flatMap(Int.init) ?? base.timedSamples
    let matSize = env["BENCH_MATSIZE"].flatMap(Int.init) ?? base.matrixSize
    return BenchConfig(
      warmupIterations: warmup,
      timedSamples: samples,
      matrixSize: matSize
    )
  }
}

// MARK: - Gate definition

/// A single performance ratio gate.
///
/// A gate pairs a **denominator leg** (the reference/legacy path) and a
/// **numerator leg** (the path under test, e.g. the unified evaluator).
/// It measures both legs and evaluates `ratio = numerator / denominator`.
/// The gate passes when `ratio ≤ threshold`.
///
/// When the numerator leg is not yet implemented (Phase 0), the gate is
/// created with ``GateDefinition/placeholder(name:threshold:denominator:reason:)``
/// and evaluates to `.pending` — never `.pass`.
public struct GateDefinition {
  /// Unique gate identifier, also used as the display name.
  public let name: String
  /// Gate threshold: the gate passes when `ratio ≤ threshold`.
  public let threshold: Double
  /// Description of the denominator (reference) leg.
  public let denominatorDescription: String
  /// Description of the numerator (test) leg.
  public let numeratorDescription: String
  /// The denominator workload closure.
  public let denominator: () -> Void
  /// The numerator workload closure, or nil when it is a placeholder.
  public let numerator: (() -> Void)?
  /// When non-nil, the numerator is a stub and the gate is PENDING.
  public let pendingReason: String?

  // MARK: Initialisers

  /// Create a gate where both legs are fully implemented.
  public init(
    name: String,
    threshold: Double,
    denominatorDescription: String,
    numeratorDescription: String,
    denominator: @escaping () -> Void,
    numerator: @escaping () -> Void
  ) {
    self.name = name
    self.threshold = threshold
    self.denominatorDescription = denominatorDescription
    self.numeratorDescription = numeratorDescription
    self.denominator = denominator
    self.numerator = numerator
    self.pendingReason = nil
  }

  /// Create a gate where the numerator leg is a placeholder.
  ///
  /// The gate evaluates to `.pending` — not `.pass` — during Phase 0 when
  /// the unified evaluator does not yet exist.
  public static func placeholder(
    name: String,
    threshold: Double,
    denominatorDescription: String,
    numeratorDescription: String,
    denominator: @escaping () -> Void,
    reason: String
  ) -> GateDefinition {
    GateDefinition(
      name: name,
      threshold: threshold,
      denominatorDescription: denominatorDescription,
      numeratorDescription: numeratorDescription + " [PLACEHOLDER — \(reason)]",
      denominator: denominator,
      numerator: nil,
      pendingReason: reason
    )
  }

  // Private memberwise init used by the placeholder factory.
  private init(
    name: String,
    threshold: Double,
    denominatorDescription: String,
    numeratorDescription: String,
    denominator: @escaping () -> Void,
    numerator: (() -> Void)?,
    pendingReason: String?
  ) {
    self.name = name
    self.threshold = threshold
    self.denominatorDescription = denominatorDescription
    self.numeratorDescription = numeratorDescription
    self.denominator = denominator
    self.numerator = numerator
    self.pendingReason = pendingReason
  }
}

// MARK: - Gate result

/// The result of running a single gate.
public struct GateResult {
  public let definition: GateDefinition
  public let denominatorNs: UInt64
  public let numeratorNs: UInt64?
  public let state: GateState

  public var name: String { definition.name }
  public var threshold: Double { definition.threshold }
}

// MARK: - Gate runner

/// Evaluates a ``GateDefinition`` and returns a ``GateResult``.
///
/// - Parameters:
///   - gate: The gate to evaluate.
///   - config: Timing parameters (warmup count, sample count).
/// - Returns: A ``GateResult`` with state `.pass`, `.fail`, or `.pending`.
public func runGate(_ gate: GateDefinition, config: BenchConfig) -> GateResult {
  // If the numerator is a placeholder, report PENDING without timing.
  if let reason = gate.pendingReason {
    // Still time the denominator so developers can see the reference cost.
    let denomNs = medianNanoseconds(
      warmup: config.warmupIterations,
      samples: config.timedSamples,
      body: gate.denominator
    )
    return GateResult(
      definition: gate,
      denominatorNs: denomNs,
      numeratorNs: nil,
      state: .pending(reason: reason)
    )
  }

  // Both legs are real: time them.
  let denomNs = medianNanoseconds(
    warmup: config.warmupIterations,
    samples: config.timedSamples,
    body: gate.denominator
  )
  let numerNs = medianNanoseconds(
    warmup: config.warmupIterations,
    samples: config.timedSamples,
    body: gate.numerator!
  )

  // Protect against zero denominator (degenerate / very fast paths).
  let ratio: Double
  if denomNs == 0 {
    // Treat as infinite overhead — gate fails.
    ratio = Double.infinity
  } else {
    ratio = Double(numerNs) / Double(denomNs)
  }

  let state: GateState = ratio <= gate.threshold ? .pass(ratio: ratio) : .fail(ratio: ratio)
  return GateResult(
    definition: gate,
    denominatorNs: denomNs,
    numeratorNs: numerNs,
    state: state
  )
}
