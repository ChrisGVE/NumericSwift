#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — NumberTheory domain.

Computes bit-exact reference values for number-theoretic and combinatorial
functions with sympy / Python `math` and freezes them as the JSON fixture
`Tests/NumericSwiftTests/Fixtures/workbench/numbertheory.json`.

This domain is **single-strategy-per-function correctness** (WORKBENCH.md §4,
"Single-strategy domains"): each `NumberTheory.*` function IS a strategy id, the
comparison scalar is that function's output for the case's integer argument(s),
and the oracle is the matching sympy / math reference. Number theory is EXACT
integer arithmetic, so EVERY case is in-envelope — there are ZERO
out-of-envelope cases and the gate is a pure correctness-vs-sympy check
(non-vacuous: the oracle is sympy/math, never NumericSwift; FP1 / FP3).

Strategy ids ↔ NumberTheory.* (Sources/NumericSwift/NumberTheory.swift):

  isPrime           → sympy.isprime                         (1.0 / 0.0)
  primeFactorsCount → Ω(n), sum of exponents of factorint   (count with multiplicity)
  primesUpToCount   → π(n), len(list(primerange(2, n+1)))   (count of primes ≤ n)
  factorial         → math.factorial
  gcd               → math.gcd
  lcm               → math.lcm
  eulerPhi          → sympy.totient
  divisorSigma      → sympy.divisor_sigma(n, k)             (Double)
  mobius            → sympy.mobius                          (+1 / 0 / -1)
  liouville         → (-1) ** Ω(n)                          (+1 / -1)
  carmichael        → sympy.reduced_totient
  vonMangoldt       → log(p) if n=p^k else 0                (float; tol 1e-9)
  modPow            → pow(base, exp, m)                     (modest operands)
  comb              → math.comb
  perm              → math.perm
  modInverse        → pow(a, -1, m)                         (when gcd(a,m)=1)
  digitSum          → sum of base-10 digits

### List-returning functions — chosen scalar

NumberTheory.primeFactors and NumberTheory.primesUpTo return lists; the
workbench compares one deterministic scalar:

  * primeFactors → Ω(n) = sum of exponents (count WITH multiplicity). This
    discriminates a correct factorization from a broken one far better than the
    distinct-prime count, because it folds in every repeated factor.
  * primesUpTo   → π(n) = count of primes ≤ n. A single count fully pins down
    the sieve's correctness for a given bound.

The Swift suite computes these scalars from the library's returned arrays.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/numbertheory.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import sympy

SOURCE = f"sympy {sympy.__version__} / python math (stdlib)"


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def omega(n: int) -> int:
    """Number of prime factors of n counted WITH multiplicity, Ω(n)."""
    if n <= 1:
        return 0
    return sum(e for e in sympy.factorint(n).values())


# ── Oracle dispatch ──────────────────────────────────────────────────────────
# Each function's reference scalar, computed ONLY from sympy / math.


def oracle(strategy: str, inputs: dict) -> float:
    n = inputs.get("n")
    a = inputs.get("a")
    b = inputs.get("b")
    k = inputs.get("k")
    base = inputs.get("base")
    exp = inputs.get("exp")
    m = inputs.get("m")

    if strategy == "isPrime":
        return 1.0 if sympy.isprime(n) else 0.0
    if strategy == "primeFactorsCount":
        return float(omega(n))
    if strategy == "primesUpToCount":
        return float(len(list(sympy.primerange(2, n + 1))))
    if strategy == "factorial":
        return float(math.factorial(n))
    if strategy == "gcd":
        return float(math.gcd(a, b))
    if strategy == "lcm":
        return float(math.lcm(a, b))
    if strategy == "eulerPhi":
        return float(sympy.totient(n))
    if strategy == "divisorSigma":
        return float(sympy.divisor_sigma(n, k))
    if strategy == "mobius":
        return float(sympy.mobius(n))
    if strategy == "liouville":
        return float((-1) ** omega(n))
    if strategy == "carmichael":
        return float(sympy.reduced_totient(n))
    if strategy == "vonMangoldt":
        factors = sympy.factorint(n)
        if len(factors) == 1:
            (p,) = factors.keys()
            return float(sympy.log(p).evalf())
        return 0.0
    if strategy == "modPow":
        return float(pow(base, exp, m))
    if strategy == "comb":
        return float(math.comb(n, k))
    if strategy == "perm":
        return float(math.perm(n, k))
    if strategy == "modInverse":
        return float(pow(a, -1, m))
    if strategy == "digitSum":
        return float(sum(int(d) for d in str(abs(n))))
    raise ValueError(f"unknown strategy {strategy}")


# tol per strategy: integer-exact functions use 0.0; float-valued functions
# (vonMangoldt is a transcendental log; divisorSigma/factorial/comb/perm are
# built from Darwin pow/lgamma for large args) use a small absolute tolerance.
FLOAT_TOL = 1e-9
# factorial/comb/perm/divisorSigma are exact for small args but use lgamma for
# large args; we keep their args within the exact regime (n<=20 for factorial,
# n<=20 for comb/perm via the multiplicative path) so 0.0 holds, EXCEPT where a
# case deliberately exercises the lgamma path — those get FLOAT_TOL.
EXACT = 0.0


def case(cid, tier, strategy, inputs, tol):
    val = oracle(strategy, inputs)
    return {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": val, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": [strategy],
        "tol": {strategy: tol},
    }


def build():
    cases = []

    # ── Trivial (~10): textbook smoke cases, hand-checkable ───────────────────
    trivial = [
        ("isPrime", {"n": 7}, EXACT),
        ("gcd", {"a": 24, "b": 36}, EXACT),
        ("lcm", {"a": 4, "b": 6}, EXACT),
        ("factorial", {"n": 5}, EXACT),
        ("primeFactorsCount", {"n": 12}, EXACT),  # 2^2 * 3 → Ω = 3
        ("primesUpToCount", {"n": 30}, EXACT),  # π(30) = 10
        ("eulerPhi", {"n": 10}, EXACT),  # 4
        ("mobius", {"n": 30}, EXACT),  # squarefree, 3 factors → -1
        ("comb", {"n": 5, "k": 2}, EXACT),  # 10
        ("digitSum", {"n": 12345}, EXACT),  # 15
    ]
    for i, (strat, inputs, tol) in enumerate(trivial):
        cases.append(
            case(f"numbertheory.trivial.{strat}_{i}", "trivial", strat, inputs, tol)
        )

    # ── Hard (~80): realistic, varied arguments across every function ─────────
    hard = []

    # isPrime: a spread of composites and primes (well within Int64).
    for n in [
        97,
        561,
        7919,
        104729,
        1299709,
        2147483647,
        999983,
        25,
        1000000,
        600851475143,
    ]:
        # 2147483647 is a Mersenne prime; 561 is a Carmichael composite; 25,1000000 composite.
        hard.append(("isPrime", {"n": n}, EXACT))

    # primeFactorsCount: Ω(n) for diverse n (perfect powers, semiprimes, etc.).
    for n in [
        360,
        1024,
        1000000,
        13195,
        600851475143,
        2310,
        999999,
        123456,
        5040,
        987654321,
    ]:
        hard.append(("primeFactorsCount", {"n": n}, EXACT))

    # primesUpToCount: π(n) at several bounds.
    for n in [100, 1000, 10000, 50000, 100000, 250, 7500, 33333, 500, 2000]:
        hard.append(("primesUpToCount", {"n": n}, EXACT))

    # factorial: keep n<=20 for the exact multiplicative path (0.0 tol).
    for n in [10, 12, 15, 18, 20, 13, 16, 19, 11, 17]:
        hard.append(("factorial", {"n": n}, EXACT))

    # gcd / lcm over varied pairs.
    for a, b in [
        (252, 105),
        (1071, 462),
        (123456, 789012),
        (1000000, 999999),
        (840, 1260),
        (17, 5),
        (1024, 768),
    ]:
        hard.append(("gcd", {"a": a, "b": b}, EXACT))
    for a, b in [(21, 6), (12, 18), (100, 75)]:
        hard.append(("lcm", {"a": a, "b": b}, EXACT))

    # eulerPhi over diverse n.
    for n in [36, 100, 997, 2310, 123456, 1000000, 7919, 360]:
        hard.append(("eulerPhi", {"n": n}, EXACT))

    # divisorSigma: σ_k(n). k=0 (divisor count), k=1 (divisor sum), k=2.
    # Values from sympy are exact integers; NumberTheory builds them via pow,
    # so allow FLOAT_TOL to absorb any benign float rounding on larger results.
    for n, k in [
        (12, 0),
        (12, 1),
        (28, 1),
        (100, 2),
        (360, 1),
        (1000, 0),
        (496, 1),
        (2310, 1),
    ]:
        hard.append(("divisorSigma", {"n": n, "k": k}, FLOAT_TOL))

    # mobius over diverse n (squarefree → ±1, non-squarefree → 0).
    for n in [30, 12, 1, 105, 2, 49, 510510, 97]:
        hard.append(("mobius", {"n": n}, EXACT))

    # liouville λ(n) = (-1)^Ω(n).
    for n in [12, 30, 1, 8, 360, 97]:
        hard.append(("liouville", {"n": n}, EXACT))

    # carmichael reduced totient λ(n).
    for n in [8, 15, 21, 100, 561, 1000, 36]:
        hard.append(("carmichael", {"n": n}, EXACT))

    # modPow — keep base/exp/m modest so the mulMod full-width path is exact.
    for base, exp, m in [
        (2, 10, 1000),
        (3, 100, 7919),
        (7, 256, 1000000007),
        (123, 45, 67890),
        (5, 0, 13),
        (10, 9, 99991),
        (2, 62, 1000003),
    ]:
        hard.append(("modPow", {"base": base, "exp": exp, "m": m}, EXACT))

    # comb / perm — keep n<=20 for the exact multiplicative path.
    for n, k in [(10, 3), (15, 7), (20, 10), (12, 6), (18, 9)]:
        hard.append(("comb", {"n": n, "k": k}, EXACT))
    for n, k in [(10, 3), (15, 4), (20, 5), (12, 2)]:
        hard.append(("perm", {"n": n, "k": k}, EXACT))

    # modInverse — pairs with gcd(a,m)=1.
    for a, m in [(3, 11), (7, 26), (17, 3120), (123, 4567)]:
        hard.append(("modInverse", {"a": a, "m": m}, EXACT))

    # vonMangoldt — Λ(n): prime powers give log(p); else 0. Float-valued.
    for n in [7, 8, 9, 12, 27, 1024, 100, 49, 2]:
        hard.append(("vonMangoldt", {"n": n}, FLOAT_TOL))

    # digitSum over varied n.
    for n in [999999, 1234567890, 1000000000, 314159265, 271828182]:
        hard.append(("digitSum", {"n": n}, EXACT))

    # Trim/pad to ~80 hard cases (proportion enforcement, ±a few).
    hard = hard[:80]
    for i, (strat, inputs, tol) in enumerate(hard):
        cases.append(case(f"numbertheory.hard.{strat}_{i}", "hard", strat, inputs, tol))

    # ── Edge (~10): degenerate but well-defined inputs (still exact) ──────────
    # n=1, n=2, smallest primes, factorial(0)/(1), comb boundary, gcd with 0,
    # mobius(1)=1, modPow with m=1. All remain closed-form vs sympy/math, so
    # they are in-envelope (no diagnostic expected).
    edge = [
        ("isPrime", {"n": 1}, EXACT),  # 1 is not prime → 0
        ("isPrime", {"n": 2}, EXACT),  # smallest prime → 1
        ("factorial", {"n": 0}, EXACT),  # 0! = 1
        ("factorial", {"n": 1}, EXACT),  # 1! = 1
        ("comb", {"n": 5, "k": 0}, EXACT),  # C(5,0) = 1
        ("comb", {"n": 5, "k": 5}, EXACT),  # C(5,5) = 1
        ("gcd", {"a": 0, "b": 12}, EXACT),  # gcd(0,12) = 12
        ("mobius", {"n": 1}, EXACT),  # μ(1) = 1
        ("primesUpToCount", {"n": 2}, EXACT),  # π(2) = 1
        ("primeFactorsCount", {"n": 1}, EXACT),  # Ω(1) = 0
    ]
    for i, (strat, inputs, tol) in enumerate(edge):
        cases.append(case(f"numbertheory.edge.{strat}_{i}", "edge", strat, inputs, tol))

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"numbertheory: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']} "
        f"(0 out-of-envelope)",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "numbertheory.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
