# Linear Algebra

Matrix operations, decompositions, and linear system solvers.

## Overview

The LinAlg module provides comprehensive linear algebra functionality inspired by scipy.linalg and numpy.linalg. It includes a `Matrix` type with full arithmetic support, matrix decompositions (LU, QR, SVD, Cholesky), and solvers for linear systems.

## Matrix Creation

```swift
// From 2D array
let A = Matrix([[1, 2], [3, 4]])

// Factory methods
let I = Matrix.identity(3)           // 3x3 identity
let zeros = Matrix.zeros(rows: 2, cols: 3)
let ones = Matrix.ones(rows: 2, cols: 2)
let diag = Matrix.diagonal([1, 2, 3])
```

## Matrix Operations

```swift
let A = Matrix([[1, 2], [3, 4]])
let B = Matrix([[5, 6], [7, 8]])

let sum = A + B           // Element-wise addition
let product = A * B       // Matrix multiplication
let scaled = A * 2.0      // Scalar multiplication
let transposed = A.T      // Transpose
```

## Solving Linear Systems

```swift
let A = Matrix([[4, 3], [6, 3]])
let b = Matrix([[1], [2]])

// Solve Ax = b
let x = solve(A, b)

// Least squares solution
let (x, residuals, rank, s) = lstsq(A, b)
```

## Matrix Decompositions

```swift
let A = Matrix([[4, 3], [6, 3]])

// LU decomposition with pivoting
let (L, U, P) = A.lu()

// QR decomposition
let (Q, R) = A.qr()

// Singular Value Decomposition
let (U, S, Vt) = A.svd()

// Cholesky decomposition (positive definite matrices)
let L = A.cholesky()

// Eigenvalue decomposition
let eigenvalues = A.eigenvalues()
let (values, vectors) = A.eig()
```

## Topics

### Matrix Type

- ``Matrix``

### Matrix Creation

- ``Matrix/identity(_:)``
- ``Matrix/zeros(rows:cols:)``
- ``Matrix/ones(rows:cols:)``
- ``Matrix/diagonal(_:)``

### Matrix Decompositions

- ``Matrix/lu()``
- ``Matrix/qr()``
- ``Matrix/svd()``
- ``Matrix/cholesky()``
- ``Matrix/eigenvalues()``
- ``Matrix/eig()``

### Solvers

- ``solve(_:_:)``
- ``lstsq(_:_:)``
- ``inv(_:)``
- ``det(_:)``
