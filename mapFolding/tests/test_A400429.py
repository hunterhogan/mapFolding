"""Diagonals of the semi-meander triangle, the A005315 anchors, and rational-series coefficients, `mapFolding.oeis.A400429`.

The stored A400429 rows 2 through 45 are complete, and row 46 is partial, so every domain that reads the triangle stops at 45.
Every value of a sequence is a(n) of its OEIS b-file, which the `expected` fixture reads; exceptions come from the docstring.
"""
from __future__ import annotations

from itertools import chain, product as CartesianProduct, repeat
from mapFolding.oeis import (
	boxOfA005315of, diagonal1, diagonal2, diagonal3, diagonal4, diagonal5, diagonal6, diagonal7, diagonal8, diagonal9, diagonal10, diagonal11)
from mapFolding.oeis.A400429 import crunchCoefficients
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import OEISid, 形Triangle

# `boxOfA005315of[j](2*j)` is A005315(j) and `boxOfA005315of[j](2*j + 1)` is A005315(j + 1), for j = 0 (the base case) through 11.
@pytest.mark.parametrize('oeisID, 次A005315of, x, n', tuple(chain(
	zip(repeat('A005315'), range(12), range(0, 24, 2), range(12))
	, zip(repeat('A005315'), range(12), range(1, 24, 2), range(1, 13))
)))
def test_boxOfA005315of(次A005315of: int, x: int, oeisID: OEISid, n: int, expected: int) -> None:
	assertEqualTo(boxOfA005315of[次A005315of](x), expected, boxOfA005315of[次A005315of].__name__, x)

# The b-files of A400632 and A400633 end at n = 300, so every diagonal stops there.
@pytest.mark.parametrize('diagonal, oeisID, n', tuple(chain(
	CartesianProduct((diagonal2,), ('A400630',), range(4, 301))
	, CartesianProduct((diagonal3,), ('A400631',), range(6, 301))
	, CartesianProduct((diagonal4,), ('A400632',), range(8, 301))
	, CartesianProduct((diagonal5,), ('A400633',), range(10, 301))
	, CartesianProduct((diagonal6,), ('A400634',), range(12, 301))
	, CartesianProduct((diagonal7,), ('A400635',), range(14, 301))
	, CartesianProduct((diagonal8,), ('A400636',), range(16, 301))
	, CartesianProduct((diagonal9,), ('A400637',), range(18, 301))
)))
def test_diagonal(diagonal: CallableFunction[[int], int], oeisID: OEISid, n: int, expected: int) -> None:
	assertEqualTo(diagonal(n), expected, diagonal.__name__, n)

# The triangle is the only reference for every n of D_1, D_10, and D_11 beyond the A005315 anchors.
@pytest.mark.parametrize('diagonal, 次diagonal, n', tuple(chain(
	CartesianProduct((diagonal1,), (1,), range(2, 46))
	, CartesianProduct((diagonal2,), (2,), range(4, 46))
	, CartesianProduct((diagonal3,), (3,), range(6, 46))
	, CartesianProduct((diagonal4,), (4,), range(8, 46))
	, CartesianProduct((diagonal5,), (5,), range(10, 46))
	, CartesianProduct((diagonal6,), (6,), range(12, 46))
	, CartesianProduct((diagonal7,), (7,), range(14, 46))
	, CartesianProduct((diagonal8,), (8,), range(16, 46))
	, CartesianProduct((diagonal9,), (9,), range(18, 46))
	, CartesianProduct((diagonal10,), (10,), range(20, 46))
	, CartesianProduct((diagonal11,), (11,), range(22, 46))
)))
def test_diagonalTriangle(triangle: 形Triangle, diagonal: CallableFunction[[int], int], 次diagonal: int, n: int) -> None:
	"""The diagonal D_j(n) is the cell T(n, floor(n/2) - j + 1), at zero-based column floor(n/2) - j of row n."""
	assertEqualTo(diagonal(n), triangle[n][n // 2 - 次diagonal], diagonal.__name__, n)

# Reference: hand-derived power-series coefficients of N(z)/Q(z).
@pytest.mark.parametrize('次coefficient, numerator, denominator, expected', (
	pytest.param(0, (1,), (1, -1), 1, id='geometricFirst')
	, pytest.param(7, (1,), (1, -1), 1, id='geometric')
	, pytest.param(3, (1,), (1, 1), -1, id='alternating')
	, pytest.param(5, (1,), (1, -2), 32, id='powersOfTwo')
	, pytest.param(10, (1,), (1, -1, -1), 89, id='Fibonacci')
	, pytest.param(2, (1, 2, 3), (1,), 3, id='polynomialLastCoefficient')
	, pytest.param(3, (1, 2, 3), (1,), 0, id='polynomialBeyondDegree')
	, pytest.param(3, (1, 2), (1, -1), 3, id='partialSums')
	, pytest.param(4, (), (1, -1), 0, id='numeratorZero')
))
def test_crunchCoefficients(次coefficient: int, numerator: tuple[int, ...], denominator: tuple[int, ...], expected: int) -> None:
	assertEqualTo(crunchCoefficients(次coefficient, numerator, denominator), expected, crunchCoefficients.__name__, 次coefficient, numerator, denominator)

@pytest.mark.parametrize('次coefficient, numerator, denominator, expected', (pytest.param(-1, (1,), (1, -1), IndexError, id='indexNegative'),))
def test_crunchCoefficientsError(次coefficient: int, numerator: tuple[int, ...], denominator: tuple[int, ...], expected: type[Exception]) -> None:
	with pytest.raises(expected):
		crunchCoefficients(次coefficient, numerator, denominator)
