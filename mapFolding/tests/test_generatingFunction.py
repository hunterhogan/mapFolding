from __future__ import annotations

from mapFolding.oeis.A400429._A005315 import crunchCoefficients
from mapFolding.oeis.A400429._generatingFunction import reduceFraction, reduceSteps
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.generatingFunctions import (
	parameterRowsGeneratingFunctionCoefficientErrors, parameterRowsGeneratingFunctionCoefficients, parameterRowsGeneratingFunctionErrors,
	parameterRowsGeneratingFunctionFractions, parameterRowsGeneratingFunctionSteps)
import pytest

@pytest.mark.parametrize('次coefficient, numerator, denominator, expected', parameterRowsGeneratingFunctionCoefficients)
def test_crunchCoefficients(次coefficient: int, numerator: tuple[int, ...], denominator: tuple[int, ...], expected: int) -> None:
	assertEqualTo(crunchCoefficients(次coefficient, numerator, denominator), expected,
		crunchCoefficients.__name__, 次coefficient, numerator, denominator)

@pytest.mark.parametrize('次coefficient, numerator, denominator, expected', parameterRowsGeneratingFunctionCoefficientErrors)
def test_crunchCoefficientsError(次coefficient: int, numerator: tuple[int, ...], denominator: tuple[int, ...], expected: type[Exception]) -> None:
	with pytest.raises(expected, match='coefficient index must be nonnegative'):
		crunchCoefficients(次coefficient, numerator, denominator)

@pytest.mark.parametrize('numerator, stepsNumerator, stepsDenominator, expected', parameterRowsGeneratingFunctionFractions)
def test_generatingFunctionReducesFraction(numerator: tuple[int, ...], stepsNumerator: tuple[int, ...], stepsDenominator: tuple[int, ...],
	expected: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
	assertEqualTo(reduceFraction(numerator, stepsNumerator, stepsDenominator), expected,
		reduceFraction.__name__, numerator, stepsNumerator, stepsDenominator)

@pytest.mark.parametrize('numerator, stepsNumerator, stepsDenominator, expected', parameterRowsGeneratingFunctionErrors)
def test_generatingFunctionReducesFractionError(numerator: tuple[int, ...], stepsNumerator: tuple[int, ...], stepsDenominator: tuple[int, ...],
	expected: type[Exception]) -> None:
	with pytest.raises(expected, match='positive integer steps'):
		reduceFraction(numerator, stepsNumerator, stepsDenominator)

@pytest.mark.parametrize('numerator, stepsNumerator, stepsDenominator, expected', parameterRowsGeneratingFunctionSteps)
def test_generatingFunctionReducesSteps(numerator: tuple[int, ...], stepsNumerator: tuple[int, ...], stepsDenominator: tuple[int, ...],
	expected: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> None:
	assertEqualTo(reduceSteps(numerator, stepsNumerator, stepsDenominator), expected,
		reduceSteps.__name__, numerator, stepsNumerator, stepsDenominator)
	assertEqualTo(reduceSteps(*expected), expected,
		reduceSteps.__name__, *expected)

@pytest.mark.parametrize('numerator, stepsNumerator, stepsDenominator, expected', parameterRowsGeneratingFunctionErrors)
def test_generatingFunctionReducesStepsError(numerator: tuple[int, ...], stepsNumerator: tuple[int, ...], stepsDenominator: tuple[int, ...],
	expected: type[Exception]) -> None:
	with pytest.raises(expected, match='positive integer steps'):
		reduceSteps(numerator, stepsNumerator, stepsDenominator)
