# Mathematical Expressions

Mathematical expression parsing and evaluation.

## Overview

The MathExpr module provides a parser and evaluator for mathematical expressions given as strings. This allows runtime evaluation of formulas.

## Basic Usage

```swift
// Simple evaluation
let result = evaluate("2 + 3 * 4")  // 14

// With variables
let result = evaluate("x^2 + y", variables: ["x": 3, "y": 1])  // 10

// Using functions
let result = evaluate("sin(pi/2)")  // 1.0
let result = evaluate("sqrt(2)")    // 1.414...
```

## Expression Parser

For repeated evaluation with different variable values:

```swift
let expr = MathExpression("x^2 + 2*x + 1")

// Evaluate with different x values
let y1 = expr.evaluate(["x": 0])   // 1
let y2 = expr.evaluate(["x": 1])   // 4
let y3 = expr.evaluate(["x": 2])   // 9
```

## Supported Operations

### Arithmetic Operators

- `+` Addition
- `-` Subtraction
- `*` Multiplication
- `/` Division
- `^` Exponentiation
- `%` Modulo

### Built-in Functions

```swift
// Trigonometric
evaluate("sin(x)")
evaluate("cos(x)")
evaluate("tan(x)")
evaluate("asin(x)")
evaluate("acos(x)")
evaluate("atan(x)")

// Exponential/Logarithmic
evaluate("exp(x)")
evaluate("log(x)")      // Natural log
evaluate("log10(x)")
evaluate("log2(x)")

// Power/Root
evaluate("sqrt(x)")
evaluate("cbrt(x)")
evaluate("pow(x, y)")

// Other
evaluate("abs(x)")
evaluate("floor(x)")
evaluate("ceil(x)")
evaluate("round(x)")
evaluate("min(x, y)")
evaluate("max(x, y)")
```

### Built-in Constants

```swift
evaluate("pi")      // 3.14159...
evaluate("e")       // 2.71828...
evaluate("tau")     // 2*pi
```

## Error Handling

```swift
do {
    let result = try MathExpression("1/0").evaluate([:])
} catch MathExprError.divisionByZero {
    print("Cannot divide by zero")
} catch MathExprError.undefinedVariable(let name) {
    print("Variable '\(name)' not defined")
} catch MathExprError.parseError(let message) {
    print("Parse error: \(message)")
}
```

## Topics

### Evaluation

- ``evaluate(_:variables:)``

### Expression Type

- ``MathExpression``
- ``MathExpression/evaluate(_:)``

### Errors

- ``MathExprError``
