//
//  WorkbenchCase.swift
//  NumericSwift
//
//  Codable fixture model for the NumericSwift E2E functional workbench.
//
//  Each fixture file lives at:
//    Tests/NumericSwiftTests/Fixtures/workbench/<domain>.json
//
//  The schema mirrors WORKBENCH.md §3 exactly. Oracle numeric values are stored
//  bit-exact via a `UInt64` `bitPattern` field ("bits"), matching the convention
//  established by `LegacySnapshot.json`. This guarantees lossless round-trips for
//  IEEE-754 edge values (NaN, ±Inf, signed zero, subnormals).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - CaseTier

/// Difficulty tier for a workbench fixture case (WORKBENCH.md §2).
public enum CaseTier: String, Codable, Sendable, Equatable {
    /// ~10% of cases. Textbook smoke cases; closed-form answers; sanity floor.
    case trivial
    /// ~80% of cases. Realistic, non-degenerate problems from SciPy / statsmodels suites.
    case hard
    /// ~10% of cases. Degenerate, extreme, or IEEE-754 edge inputs.
    case edge
}

// MARK: - OracleValue

/// A bit-exact oracle value for a workbench fixture.
///
/// `value` is the decoded `Double` for convenience. `bits` is the canonical
/// IEEE-754 bit pattern (`UInt64`) that was frozen at fixture-generation time
/// by the Python oracle. The Swift decoder always constructs the value from
/// `bits`, so there is no double-to-string precision loss.
///
/// ### JSON encoding
///
/// ```json
/// { "value": 1.7724538509055159, "bits": "0x3FFC5BF891B4EF6A" }
/// ```
///
/// `bits` is encoded as a hex string (prefix `0x`) for human readability.
public struct OracleValue: Codable, Sendable, Equatable {

    /// The oracle value, decoded from `bits`.
    public let value: Double

    /// IEEE-754 bit pattern frozen at oracle-generation time.
    public let bits: UInt64

    // MARK: Codable

    private enum CodingKeys: String, CodingKey {
        case value, bits
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // Decode bits as hex string (e.g. "0x3FFC5BF891B4EF6A") or decimal UInt64.
        let bitsRaw = try container.decode(String.self, forKey: .bits)
        let parsed: UInt64
        if bitsRaw.hasPrefix("0x") || bitsRaw.hasPrefix("0X") {
            guard let v = UInt64(bitsRaw.dropFirst(2), radix: 16) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .bits,
                    in: container,
                    debugDescription: "Cannot parse bits hex string: \(bitsRaw)"
                )
            }
            parsed = v
        } else {
            guard let v = UInt64(bitsRaw) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .bits,
                    in: container,
                    debugDescription: "Cannot parse bits string: \(bitsRaw)"
                )
            }
            parsed = v
        }
        self.bits = parsed
        // Reconstruct the Double from the frozen bit pattern — no string rounding.
        self.value = Double(bitPattern: parsed)
        // Ignore any "value" key in JSON (present only for human readability).
        _ = try? container.decode(Double.self, forKey: .value)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(value, forKey: .value)
        // Use %016llX (long long) to correctly format a full 64-bit UInt64.
        try container.encode(String(format: "0x%016llX", bits), forKey: .bits)
    }

    /// Create an OracleValue from a Double at test-authoring time.
    public init(_ value: Double) {
        self.value = value
        self.bits = value.bitPattern
    }
}

// MARK: - PerStrategyTolerance

/// Per-strategy accuracy tolerance for a fixture case.
///
/// The key is a strategy identifier (e.g. `"quad"`, `"romberg"`); the value is
/// the maximum absolute error that strategy is declared to achieve on this case.
/// This is the limitation envelope per WORKBENCH.md §5.
public typealias PerStrategyTolerance = [String: Double]

// MARK: - WorkbenchCase

/// A single workbench fixture case.
///
/// Fixture files are arrays of `WorkbenchCase` objects, one per test case.
/// Each case drives every strategy listed in ``strategies`` and checks the
/// result against ``oracle`` within the per-strategy tolerance from ``tol``.
///
/// ### Example fixture entry
///
/// ```json
/// {
///   "id": "integration.hard.gaussian_bell",
///   "tier": "hard",
///   "inputs": { "a": -5.0, "b": 5.0, "tag": "gaussian_bell" },
///   "oracle": { "value": 1.7724538509055159, "bits": "0x3FFC5BF891B4EF6A" },
///   "source": "scipy.integrate.quad 1.17.1",
///   "strategies": ["quad", "romberg", "simps"],
///   "tol": { "quad": 1e-10, "romberg": 1e-8, "simps": 1e-3 }
/// }
/// ```
///
/// ### Bit-exact decode
///
/// The `oracle.bits` field is always used to reconstruct the Double — the
/// `oracle.value` field is present for human readability only and is ignored
/// by the Swift decoder. See ``OracleValue`` for details.
public struct WorkbenchCase: Codable, Sendable {

    /// Unique case identifier.
    ///
    /// Format: `<domain>.<tier>.<name>` (e.g. `"integration.hard.gaussian_bell"`).
    public let id: String

    /// Difficulty tier.
    public let tier: CaseTier

    /// Domain-specific input parameters.
    ///
    /// Decoded as a raw `[String: AnyCodable]` bag; each ``Strategy`` closure
    /// receives this dictionary and extracts the keys it needs.
    public let inputs: [String: InputValue]

    /// Bit-exact oracle value from the Python reference implementation.
    public let oracle: OracleValue

    /// Citation for the oracle (e.g. `"scipy.integrate.quad 1.17.1"`).
    public let source: String

    /// Strategy identifiers to run for this case.
    public let strategies: [String]

    /// Per-strategy accuracy tolerance.
    ///
    /// Keys match elements of ``strategies``. A missing key is treated as 0.0
    /// (exact match required), which is intentionally strict to surface mistakes
    /// in fixture authoring.
    public let tol: PerStrategyTolerance

    /// Domain extracted from the case id (the part before the first `.`).
    public var domain: String {
        String(id.split(separator: ".").first ?? "unknown")
    }
}

// MARK: - InputValue

/// A JSON value that can appear in the `inputs` dictionary of a ``WorkbenchCase``.
///
/// The workbench fixture inputs are domain-specific and heterogeneous. We
/// decode them into this enum rather than `AnyCodable` to avoid third-party
/// dependencies, while keeping the decoder complete for all JSON scalar types.
public enum InputValue: Codable, Sendable, Equatable {
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([InputValue])
    case null

    public init(from decoder: Decoder) throws {
        let single = try decoder.singleValueContainer()
        if single.decodeNil() {
            self = .null
        } else if let b = try? single.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? single.decode(Int.self) {
            self = .int(i)
        } else if let d = try? single.decode(Double.self) {
            self = .double(d)
        } else if let s = try? single.decode(String.self) {
            self = .string(s)
        } else if let arr = try? single.decode([InputValue].self) {
            self = .array(arr)
        } else {
            throw DecodingError.dataCorruptedError(
                in: single,
                debugDescription: "InputValue: unrecognised JSON type"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var single = encoder.singleValueContainer()
        switch self {
        case .bool(let b): try single.encode(b)
        case .int(let i): try single.encode(i)
        case .double(let d): try single.encode(d)
        case .string(let s): try single.encode(s)
        case .array(let a): try single.encode(a)
        case .null: try single.encodeNil()
        }
    }

    // MARK: Convenience accessors

    public var doubleValue: Double? {
        switch self {
        case .double(let d): return d
        case .int(let i): return Double(i)
        default: return nil
        }
    }

    public var intValue: Int? {
        if case .int(let i) = self { return i }
        return nil
    }

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var arrayValue: [InputValue]? {
        if case .array(let a) = self { return a }
        return nil
    }
}

// MARK: - FixtureFile

/// The top-level structure of a workbench fixture JSON file.
///
/// A fixture file is simply an array of ``WorkbenchCase`` values.
///
/// ### Usage
///
/// ```swift
/// let url = fixturesDir.appendingPathComponent("integration.json")
/// let data = try Data(contentsOf: url)
/// let cases = try JSONDecoder().decode(FixtureFile.self, from: data)
/// ```
public typealias FixtureFile = [WorkbenchCase]
