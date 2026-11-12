"""Sequence values from the cells of the semi-meander triangle, and the diagonal formulas, `mapFolding.oeis.A400429oeisByFormula`.

The stored A400429 rows 2 through 45 are complete, and row 46 is partial, so every domain that reads the triangle stops at 45.
"""
from __future__ import annotations

from itertools import chain, product as CartesianProduct
from mapFolding.oeis import A400429oeisByFormula
from mapFolding.oeis.A400429oeisByFormula._integrate_fromTriangleCells import calculateDiagonal2, calculateDiagonal3
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from fractions import Fraction
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import OEISid, 形Triangle

@pytest.mark.parametrize('oeisID, n', tuple(chain(
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
)))
def test_fromTriangleCells(triangle: 形Triangle, oeisID: OEISid, n: int, expected: int) -> None:
	assertEqualTo(getattr(A400429oeisByFormula, oeisID)(n, triangle=triangle), expected, oeisID, n)

# The domains stop at n = 300 with the other diagonals in `test_A400429`.
@pytest.mark.parametrize('calculateDiagonal, oeisID, n', tuple(chain(
	CartesianProduct((calculateDiagonal2,), ('A400630',), range(4, 301))
	, CartesianProduct((calculateDiagonal3,), ('A400631',), range(6, 301))
)))
def test_calculateDiagonal(calculateDiagonal: CallableFunction[[int], Fraction], oeisID: OEISid, n: int, expected: int) -> None:
	assertEqualTo(calculateDiagonal(n), expected, calculateDiagonal.__name__, n)
