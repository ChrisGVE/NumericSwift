# Mathematical Expressions

Mathematical expression parsing and evaluation.

## Overview

The MathExpr module provides a parser and evaluator for mathematical expressions given as strings. This allows runtime evaluation of formulas.

## Basic Usage

```swift
let expr = MathExpr("2 + 3 * 4")
let result = try expr.evaluate([:])  // 14

// With variables
let expr2 = MathExpr("x^2 + y")
let result2 = try expr2.evaluate(["x": 3, "y": 1])  // 10

// Using functions
let expr3 = MathExpr("sin(pi/2)")
let result3 = try expr3.evaluate([:])  // 1.0
```

## Expression Parser

For repeated evaluation with different variable values:

```swift
let expr = MathExpr("x^2 + 2*x + 1")

// Evaluate with different x values
let y1 = try expr.evaluate(["x": 0])   // 1
let y2 = try expr.evaluate(["x": 1])   // 4
let y3 = try expr.evaluate(["x": 2])   // 9
```

## Supported Operations

### Arithmetic Operators

- `+` Addition
- `-` Subtraction
- `*` Multiplication
- `/` Division
- `^` Exponentiation

### Built-in Functions

```swift
// Trigonometric
MathExpr("sin(x)")
MathExpr("cos(x)")
MathExpr("tan(x)")
MathExpr("asin(x)")
MathExpr("acos(x)")
MathExpr("atan(x)")

// Hyperbolic
MathExpr("sinh(x)")
MathExpr("cosh(x)")
MathExpr("tanh(x)")

// Exponential/Logarithmic
MathExpr("exp(x)")
MathExpr("log(x)")      // Natural log
MathExpr("log10(x)")
MathExpr("log2(x)")

// Power/Root
MathExpr("sqrt(x)")
MathExpr("cbrt(x)")
MathExpr("pow(x, y)")

// Other
MathExpr("abs(x)")
MathExpr("floor(x)")
MathExpr("ceil(x)")
MathExpr("round(x)")
MathExpr("min(x, y)")
MathExpr("max(x, y)")
MathExpr("clamp(x, lo, hi)")
MathExpr("lerp(a, b, t)")
```

### Built-in Constants

```swift
MathExpr("pi")      // 3.14159...
MathExpr("e")       // 2.71828...
MathExpr("tau")     // 2*pi
```

## Expression Utilities

```swift
let expr = MathExpr("x^2 + y")

// Find variables in expression
let vars = expr.findVariables()  // ["x", "y"]

// Substitute values
let substituted = expr.substitute(["x": 3])  // "9 + y"

// Convert back to string
let str = expr.toString()  // "x^2 + y"
```

## Error Handling

```swift
do {
    let expr = MathExpr("1/0")
    let result = try expr.evaluate([:])
} catch MathExprError.divisionByZero {
    print("Cannot divide by zero")
} catch MathExprError.undefinedVariable(let name) {
    print("Variable '\(name)' not defined")
} catch MathExprError.invalidSyntax(let message) {
    print("Syntax error: \(message)")
}
```

## Topics

### Expression Type

- ``MathExprToken``

### Errors

- ``MathExprError``
