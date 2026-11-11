from __future__ import annotations

from functools import partial
from inspect import get_annotations
from itertools import chain, product as CartesianProduct
from mapFolding.oeis import A400429oeisByFormula, boxOfA005315of, byFormula, getMetadata, getTriangleDiagonal, settingsOEIS
from mapFolding.oeis.A400429 import crunchCoefficients
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.A400429oeisByFormula import (
	parametersA005315of, parametersCalculateDiagonal, parametersCrunchCoefficients, parametersCrunchCoefficientsError,
	parametersDiagonalSequence, parametersDiagonalTriangle, parametersTriangleCells)
from typing import get_args, TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from fractions import Fraction
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import OEISid, 形Triangle
	from typing import LiteralString

@pytest.mark.parametrize('oeisID, n, f', tuple(chain.from_iterable(map(  # ruff: ignore[unnecessary-map]
	lambda oeisID: CartesianProduct((oeisID,), filter(getMetadata(oeisID)['offset'].__le__, (getMetadata(oeisID)['offset'], 3, 4, 7))
		, (*get_args(get_args(get_annotations(getattr(byFormula, oeisID), eval_str=True)['f'])[0]), None))
	, filter(partial(hasattr, byFormula), settingsOEIS.boxOfIDs)
))))
def test_byFormula(oeisID: OEISid, n: int, f: LiteralString | None, expected: int) -> None:
	assertEqualTo(getattr(byFormula, oeisID)(n, f), expected, oeisID, n, f)

@pytest.mark.parametrize('oeisID, n', parametersTriangleCells)
def test_fromTriangleCells(triangle: 形Triangle, oeisID: OEISid, n: int, expected: int) -> None:
	assertEqualTo(getattr(A400429oeisByFormula, oeisID)(n, triangle=triangle), expected, oeisID, n)

@pytest.mark.parametrize('oeisID', ('A005315',))
@pytest.mark.parametrize('次A005315of, x, n', parametersA005315of)
def test_boxOfA005315of(次A005315of: int, x: int, oeisID: OEISid, n: int, expected: int) -> None:
	assertEqualTo(boxOfA005315of[次A005315of](x), expected, boxOfA005315of[次A005315of].__name__, x)

@pytest.mark.parametrize('工diagonal, oeisID, n', parametersDiagonalSequence)
def test_工diagonal(工diagonal: CallableFunction[[int], int], oeisID: OEISid, n: int, expected: int) -> None:
	assertEqualTo(工diagonal(n), expected, 工diagonal.__name__, n)

@pytest.mark.parametrize('oeisID', ('A400429',))
@pytest.mark.parametrize('工diagonal, 次diagonal, n', parametersDiagonalTriangle)
def test_工diagonalTriangle(工diagonal: CallableFunction[[int], int], oeisID: OEISid, 次diagonal: int, n: int) -> None:
	assertEqualTo(工diagonal(n), getTriangleDiagonal(oeisID, 次diagonal)[n], 工diagonal.__name__, n)

@pytest.mark.parametrize('calculateDiagonal, oeisID, n', parametersCalculateDiagonal)
def test_calculateDiagonal(calculateDiagonal: CallableFunction[[int], Fraction], oeisID: OEISid, n: int, expected: int) -> None:
	assertEqualTo(calculateDiagonal(n), expected, calculateDiagonal.__name__, n)

@pytest.mark.parametrize('次coefficient, numerator, denominator, expected', parametersCrunchCoefficients)
def test_crunchCoefficients(次coefficient: int, numerator: tuple[int, ...], denominator: tuple[int, ...], expected: int) -> None:
	assertEqualTo(crunchCoefficients(次coefficient, numerator, denominator), expected, crunchCoefficients.__name__, 次coefficient, numerator, denominator)

@pytest.mark.parametrize('次coefficient, numerator, denominator, expected', parametersCrunchCoefficientsError)
def test_crunchCoefficientsError(次coefficient: int, numerator: tuple[int, ...], denominator: tuple[int, ...], expected: type[Exception]) -> None:
	with pytest.raises(expected):
		crunchCoefficients(次coefficient, numerator, denominator)
