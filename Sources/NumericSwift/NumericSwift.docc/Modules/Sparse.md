# Sparse Linear Algebra

COO and CSR sparse matrices with sparse linear algebra operations.

## Overview

The `Sparse` namespace provides two standard sparse matrix formats and
the operations needed for sparse linear algebra, following the API semantics
of `scipy.sparse`.

Large physical or graph-derived systems are typically sparse: most matrix
entries are zero. Storing only the nonzeros avoids the O(n²) memory cost
of a dense matrix and accelerates matrix–vector products in proportion to
the fill ratio.

Two formats are provided:

- **COO (coordinate)** — ``Sparse/COOMatrix``: three parallel arrays (row
  indices, column indices, values). Convenient for assembly from triplets;
  duplicate (row, col) entries are summed on construction. Convert to CSR
  for arithmetic.
- **CSR (compressed sparse row)** — ``Sparse/CSRMatrix``: `indptr` (row
  pointers, length `rows + 1`), `indices` (column indices), `data`
  (nonzero values). Preferred for matrix–vector products, addition, and
  solving linear systems.

## Building a Sparse Matrix

```swift
import NumericSwift

// Assemble from triplets — duplicate entries are summed
let coo = try Sparse.COOMatrix(rows: 4, cols: 4, triplets: [
    (row: 0, col: 0, value: 4.0),
    (row: 0, col: 1, value: 1.0),
    (row: 1, col: 0, value: 1.0),
    (row: 1, col: 1, value: 3.0),
    (row: 2, col: 2, value: 2.0),
    (row: 3, col: 3, value: 5.0),
])

// Convert to CSR for arithmetic
let csr = coo.toCSR()

// Inspect the dense representation
let dense = csr.toDense()
```

## Sparse Matrix–Vector Product (SpMV)

```swift
let x = [1.0, 2.0, 3.0, 4.0]
let y = try csr.spmv(x)          // y = A·x
```

## Sparse Matrix–Dense Matrix Product (SpMM)

```swift
let B = LinAlg.Matrix(rows: 4, cols: 2, data: [1, 0, 0, 1, 1, 0, 0, 1])
let C = try csr.spmm(B)          // C = A·B  (sparse × dense → dense)
```

## Sparse Arithmetic

```swift
// Element-wise addition (same shape required)
let sum = try csrA.add(csrB)

// Transpose
let At = csr.transpose()
```

## Solving a Sparse Linear System

`Sparse.spsolve(_:_:)` solves `A·x = b` where `A` is a square sparse
matrix. For symmetric positive-definite `A`, the Conjugate Gradient method
is used. For general or non-SPD `A`, the matrix is converted to dense and
solved via ``LinAlg/solve(_:_:)``.

```swift
let b = [1.0, 2.0, 3.0, 4.0]
let x = try Sparse.spsolve(csr, b)   // A·x = b
```

## Error Handling

All fallible operations throw ``Sparse/SparseError``. Pattern-match on the
thrown error to distinguish shape mismatches from singularity:

```swift
do {
    let x = try Sparse.spsolve(csr, b)
} catch Sparse.SparseError.singularMatrix {
    print("no unique solution")
} catch Sparse.SparseError.dimensionMismatch(let msg) {
    print("dimension error:", msg)
}
```

## Deferred Scope

The following items are out of scope for the current release:

- CSC (compressed sparse column) format
- Sparse eigensolvers
- Sparse Cholesky / LU / QR factorizations
- Fancy indexing (row/column slicing)

## Topics

### Namespace

- ``Sparse``

### COO Format

- ``Sparse/COOMatrix``

### CSR Format

- ``Sparse/CSRMatrix``

### Sparse Operations

- ``Sparse/spsolve(_:_:)``

### Errors

- ``Sparse/SparseError``
