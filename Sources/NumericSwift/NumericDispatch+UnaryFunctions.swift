//
//  NumericDispatch+UnaryFunctions.swift
//  NumericSwift
//
//  Unary-operator sub-dispatchers (applyNeg / applyFactorial / applyTransposeUnary).
//
//  Named-function sub-dispatchers (applyTrigFunction, applyExpLogSqrt, etc.)
//  live in NumericDispatch+FunctionDispatchers.swift.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Unary sub-dispatchers

extension NumericDispatch {

    static func applyNeg(operand: NumericValue) throws -> NumericValue {
        switch operand.kind {
        case .scalar:
            return .scalar(-operand.asScalar!)
        case .complex:
            let z = operand.asComplex!
            return .complex(Complex(re: -z.re, im: -z.im))
        case .matrix:
            return .matrix(LinAlg.neg(operand.asMatrix!))
        case .complexMatrix:
            return try evalNegComplexMatrix(cm: operand.asComplexMatrix!)
        }
    }

    static func applyFactorial(operand: NumericValue) throws -> NumericValue {
        switch operand.kind {
        case .scalar:
            let v = operand.asScalar!
            guard v >= 0 else {
                throw MathExprError.invalidArguments(
                    "factorial requires a non-negative argument, got \(v)")
            }
            return .scalar(tgamma(v + 1))
        case .complex:
            // Legacy evalComplexUnary allows factorial for purely real non-negative complex;
            // im=0 and re≥0 delegates to tgamma(re+1), matching MathExpr.swift:391-395.
            let z = operand.asComplex!
            guard z.im == 0, z.re >= 0 else {
                throw MathExprError.invalidArguments(
                    "factorial requires non-negative real argument; "
                    + "got complex \(z.re)+\(z.im)i")
            }
            return .complex(Complex(tgamma(z.re + 1)))
        case .matrix:
            throw MathExprError.invalidArguments(
                "factorial is not defined for matrices")
        case .complexMatrix:
            throw MathExprError.invalidArguments(
                "factorial is not defined for complex matrices")
        }
    }

    static func applyTransposeUnary(operand: NumericValue) throws -> NumericValue {
        switch operand.kind {
        case .scalar:
            return operand           // transpose of a scalar is the scalar itself
        case .complex:
            return operand           // transpose of a complex scalar is the scalar
        case .matrix:
            return .matrix(operand.asMatrix!.T)
        case .complexMatrix:
            return try evalTransposeComplexMatrix(cm: operand.asComplexMatrix!)
        }
    }
}
