# Linear Algebra

Matrix operations, decompositions, and linear system solvers.

## Overview

The LinAlg module provides comprehensive linear algebra functionality inspired by scipy.linalg and numpy.linalg. It includes a `Matrix` type with full arithmetic support, matrix decompositions (LU, QR, SVD, Cholesky), and solvers for linear systems.

All functions are accessed through the `LinAlg` namespace.

## Matrix Creation

```swift
// From 2D array
let A = LinAlg.Matrix([[1, 2], [3, 4]])

// Factory methods
let I = LinAlg.eye(3)                    // 3x3 identity
let zeros = LinAlg.zeros(2, 3)           // 2x3 zeros
let ones = LinAlg.ones(2, 2)             // 2x2 ones
let diag = LinAlg.diag([1.0, 2.0, 3.0])  // Diagonal matrix
let range = LinAlg.arange(0, 10, 2)      // [0, 2, 4, 6, 8]
let space = LinAlg.linspace(0, 1, 5)     // [0, 0.25, 0.5, 0.75, 1]
```

## Matrix Operations

```swift
let A = LinAlg.Matrix([[1, 2], [3, 4]])
let B = LinAlg.Matrix([[5, 6], [7, 8]])

let sum = LinAlg.add(A, B)           // Element-wise addition
let product = LinAlg.dot(A, B)       // Matrix multiplication
let scaled = LinAlg.mul(2.0, A)      // Scalar multiplication
let transposed = A.T                  // Transpose

// Using operators
let sum2 = A + B
let product2 = A * B
let scaled2 = 2.0 * A
```

## Solving Linear Systems

```swift
let A = LinAlg.Matrix([[4, 3], [6, 3]])
let b = LinAlg.Matrix([[1], [2]])

// Solve Ax = b
if let x = LinAlg.solve(A, b) {
    print(x)
}

// Least squares solution
if let x = LinAlg.lstsq(A, b) {
    print(x)
}
```

## Matrix Decompositions

```swift
let A = LinAlg.Matrix([[4, 3], [6, 3]])

// LU decomposition with pivoting
let (L, U, P) = LinAlg.lu(A)

// QR decomposition
let (Q, R) = LinAlg.qr(A)

// Singular Value Decomposition
let (s, U, Vt) = LinAlg.svd(A)

// Cholesky decomposition (positive definite matrices)
if let L = LinAlg.cholesky(A) {
    print(L)
}

// Eigenvalue decomposition
let (values, imagParts, vectors) = LinAlg.eig(A)
let (real, imag) = LinAlg.eigvals(A)
```

## Matrix Functions

```swift
// Matrix exponential
let expA = LinAlg.expm(A)

// Matrix logarithm
if let logA = LinAlg.logm(A) {
    print(logA)
}

// Matrix square root
if let sqrtA = LinAlg.sqrtm(A) {
    print(sqrtA)
}
```

## Topics

### Matrix Type

- ``LinAlg/Matrix``
- ``LinAlg/ComplexMatrix``

### Matrix Creation

- ``LinAlg/zeros(_:_:)``
- ``LinAlg/ones(_:_:)``
- ``LinAlg/eye(_:)``
- ``LinAlg/diag(_:)``
- ``LinAlg/arange(_:_:_:)``
- ``LinAlg/linspace(_:_:_:)``

### Matrix Arithmetic

- ``LinAlg/add(_:_:)``
- ``LinAlg/sub(_:_:)``
- ``LinAlg/mul(_:_:)``
- ``LinAlg/dot(_:_:)``
- ``LinAlg/hadamard(_:_:)``
- ``LinAlg/elementDiv(_:_:)``

### Matrix Properties

- ``LinAlg/trace(_:)``
- ``LinAlg/det(_:)``
- ``LinAlg/inv(_:)``
- ``LinAlg/rank(_:)``
- ``LinAlg/cond(_:)``
- ``LinAlg/pinv(_:rcond:)``
- ``LinAlg/norm(_:_:)``

### Matrix Decompositions

- ``LinAlg/lu(_:)``
- ``LinAlg/qr(_:)``
- ``LinAlg/svd(_:)``
- ``LinAlg/cholesky(_:)``
- ``LinAlg/eig(_:)``
- ``LinAlg/eigvals(_:)``

### Solvers

- ``LinAlg/solve(_:_:)``
- ``LinAlg/lstsq(_:_:)``
- ``LinAlg/solveTriangular(_:_:lower:unitDiagonal:)``
- ``LinAlg/choSolve(_:_:)``
- ``LinAlg/luSolve(_:_:_:_:)``

### Matrix Functions

- ``LinAlg/expm(_:)``
- ``LinAlg/logm(_:)``
- ``LinAlg/sqrtm(_:)``
- ``LinAlg/funm(_:_:)``
- ``LinAlg/MatrixFunction``

### Complex Matrix Operations

- ``LinAlg/csolve(_:_:)``
- ``LinAlg/csvd(_:)``
- ``LinAlg/ceig(_:)``
- ``LinAlg/ceigvals(_:)``
- ``LinAlg/cdet(_:)``
- ``LinAlg/cinv(_:)``
