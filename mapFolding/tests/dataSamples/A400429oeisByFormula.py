"""Parameters for `mapFolding.oeis.A400429oeisByFormula`.

The stored A400429 rows 2 through 45 are complete; row 46 is partial. Every domain below reads only rows through 45.
"""
from __future__ import annotations

from itertools import chain, product as CartesianProduct
from mapFolding.oeis.A400429oeisByFormula import (
	工diagonal1, 工diagonal2, 工diagonal3, 工diagonal4, 工diagonal5, 工diagonal6, 工diagonal7, 工diagonal8, 工diagonal9, 工diagonal10, 工diagonal11)
from mapFolding.oeis.A400429oeisByFormula._integrate_fromTriangleCells import calculateDiagonal2, calculateDiagonal3
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from fractions import Fraction
	from hunterMakesPy import CallableFunction

#======== Triangle-cell identities =====================================

# Reference: the b-file of each OEIS sequence, on the domain of the A400429 formula that the function evaluates.
parametersTriangleCells: tuple[tuple[str, int], ...] = tuple(chain(
	# Sum_{k=1..floor(n/2)} T(n,k) * n = A000136(n).
	CartesianProduct(('A000136',), range(2, 46))
	# Sum_{k=1..floor(n/2)} T(n,k) = A000682(n).
	, CartesianProduct(('A000682',), range(2, 46))
	# T(n,1) = A005315(ceiling(n/2)); `A005315` reads T(2*n,1), and 2*n <= 45.
	, CartesianProduct(('A005315',), range(1, 23))
	# A005316(n) = T(n,1) + (1 - n mod 2) * [T(n,2) - T(n-1,2)]/2, for n >= 5.
	, CartesianProduct(('A005316',), range(5, 46))
	# A076876(n) for even n >= 6; the A076876 b-file ends at n = 40.
	, CartesianProduct(('A076876',), range(6, 41, 2))
	# A006661(n+4) for even n >= 6; `A006661` receives n+4, and the A006661 b-file ends at 40.
	, CartesianProduct(('A006661',), range(10, 41, 2))
	# A077054(n) = [T(2*n+1,1) - T(2*n,2)]/2, for n >= 2, and 2*n+1 <= 45.
	, CartesianProduct(('A077054',), range(2, 23))
	# A077460(n) for even n >= 2 and odd n >= 3; `A077460` reads T(2*n,1), and 2*n <= 45.
	, CartesianProduct(('A077460',), range(2, 23))
))

#======== A005315 anchors =====================================

# Reference: the A005315 b-file. `boxOfA005315of[j](2*j)` = A005315(j) and `boxOfA005315of[j](2*j + 1)` = A005315(j + 1),
# for j = 0 (the base case) through 11.
parametersA005315of: tuple[tuple[int, int, int], ...] = tuple(chain(
	zip(range(12), range(0, 24, 2), range(12), strict=True)
	, zip(range(12), range(1, 24, 2), range(1, 13), strict=True)
))

#======== Diagonals =====================================

# Reference: OEIS A400630 through A400637, the diagonals D_2 through D_9 of A400429, a(n) = T(n, floor(n/2) - j + 1), n >= 2*j.
# The b-files of A400632 and A400633 end at n = 300, so every diagonal stops there.
parametersDiagonalSequence: tuple[tuple[CallableFunction[[int], int], str, int], ...] = tuple(chain(
	CartesianProduct((工diagonal2,), ('A400630',), range(4, 301))
	, CartesianProduct((工diagonal3,), ('A400631',), range(6, 301))
	, CartesianProduct((工diagonal4,), ('A400632',), range(8, 301))
	, CartesianProduct((工diagonal5,), ('A400633',), range(10, 301))
	, CartesianProduct((工diagonal6,), ('A400634',), range(12, 301))
	, CartesianProduct((工diagonal7,), ('A400635',), range(14, 301))
	, CartesianProduct((工diagonal8,), ('A400636',), range(16, 301))
	, CartesianProduct((工diagonal9,), ('A400637',), range(18, 301))
))

# Reference: OEIS A400630 and A400631, the diagonals D_2 and D_3 of A400429, through n = 300.
parametersCalculateDiagonal: tuple[tuple[CallableFunction[[int], Fraction], str, int], ...] = tuple(chain(
	CartesianProduct((calculateDiagonal2,), ('A400630',), range(4, 301))
	, CartesianProduct((calculateDiagonal3,), ('A400631',), range(6, 301))
))

# Reference: the diagonal D_j(n) = T(n, floor(n/2) - j + 1) of the A400429 triangle, for 2*j <= n <= 45.
# This is the only reference for every n of D_1, D_10, and D_11 beyond the A005315 anchors.
parametersDiagonalTriangle: tuple[tuple[CallableFunction[[int], int], int, int], ...] = tuple(chain(
	CartesianProduct((工diagonal1,), (1,), range(2, 46))
	, CartesianProduct((工diagonal2,), (2,), range(4, 46))
	, CartesianProduct((工diagonal3,), (3,), range(6, 46))
	, CartesianProduct((工diagonal4,), (4,), range(8, 46))
	, CartesianProduct((工diagonal5,), (5,), range(10, 46))
	, CartesianProduct((工diagonal6,), (6,), range(12, 46))
	, CartesianProduct((工diagonal7,), (7,), range(14, 46))
	, CartesianProduct((工diagonal8,), (8,), range(16, 46))
	, CartesianProduct((工diagonal9,), (9,), range(18, 46))
	, CartesianProduct((工diagonal10,), (10,), range(20, 46))
	, CartesianProduct((工diagonal11,), (11,), range(22, 46))
))

#======== Rational-series coefficients =====================================

# Reference: hand-derived power-series coefficients of N(z)/Q(z).
parametersCrunchCoefficients: tuple[pytest.ParameterSet, ...] = (
	pytest.param(0, (1,), (1, -1), 1, id='geometric-first')
	, pytest.param(7, (1,), (1, -1), 1, id='geometric')
	, pytest.param(3, (1,), (1, 1), -1, id='alternating')
	, pytest.param(5, (1,), (1, -2), 32, id='powers-of-two')
	, pytest.param(10, (1,), (1, -1, -1), 89, id='Fibonacci')
	, pytest.param(2, (1, 2, 3), (1,), 3, id='polynomial-last-coefficient')
	, pytest.param(3, (1, 2, 3), (1,), 0, id='polynomial-beyond-degree')
	, pytest.param(3, (1, 2), (1, -1), 3, id='partial-sums')
	, pytest.param(4, (), (1, -1), 0, id='zero-numerator')
)

# Reference: the `crunchCoefficients` docstring: a negative index raises `IndexError`.
parametersCrunchCoefficientsError: tuple[pytest.ParameterSet, ...] = (
	pytest.param(-1, (1,), (1, -1), IndexError, id='negative-index'),
)
