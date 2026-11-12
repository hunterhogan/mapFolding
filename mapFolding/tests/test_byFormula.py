"""Each formula in `mapFolding.oeis.byFormula`, named by its OEIS sequence, against the OEIS b-file.

Each formula function receives every value of `f` that its annotation declares, and `None`, at its offset and at n = 3, 4, and 7
at or above the offset. `byFormula` is generated from `mapFolding.oeis._symbolic.boxOfRelations`; when a relation changes, change
the parametrization with it.
"""
from __future__ import annotations

from itertools import chain, product as CartesianProduct
from mapFolding.oeis import byFormula
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from typing import LiteralString

@pytest.mark.parametrize('oeisID, n, f', tuple(chain(
	CartesianProduct(('A000136',), (1, 3, 4, 7), ('A000682', None))
	, CartesianProduct(('A000108',), (0, 3, 4, 7), ('A000108', 'A001246', 'A014137', 'A014137 and A401693', None))
	, CartesianProduct(('A000560',), (2, 3, 4, 7), ('A000682', None))
	, CartesianProduct(('A000682',), (1, 3, 4, 7), ('A000136', 'A000560', 'A001010', 'A001010 and A001011', 'A060206', 'A005316 and A223093', 'A000682 and A223094', 'A227167', 'A000682 and A259702', 'A000682 and A301620', 'A000682 and A333971', 'A000682 and A334615', 'A337581', 'A223094', 'A301620', None))
	, CartesianProduct(('A001010',), (1, 3, 4, 7), ('A000682 and A007822', 'A000682 and A001011', None))
	, CartesianProduct(('A001011',), (1, 3, 4, 7), ('A000682 and A001010', None))
	, CartesianProduct(('A001246',), (0, 3, 4, 7), ('A000108', None))
	, CartesianProduct(('A002415',), (0, 3, 4, 7), (None,))
	, CartesianProduct(('A005315',), (0, 3, 4, 7), ('A005316', 'A077054 and A085973', 'A005316, A060206, and A077460', 'A078591', 'A208357', None))
	, CartesianProduct(('A005316',), (0, 3, 4, 7), ('A005315', 'A077054', 'A077014', 'A005315, A060206, and A077460', 'A005316 and A078592', 'A000682 and A223093', 'A217310, A217318, and A227167', None))
	, CartesianProduct(('A006659',), (1, 3, 4, 7), (None,))
	, CartesianProduct(('A007746',), (4, 7), (None,))
	, CartesianProduct(('A007822',), (1, 3, 4, 7), ('A001010', None))
	, CartesianProduct(('A014137',), (0, 3, 4, 7), ('A000108', 'A000108 and A014137', 'A014140', 'A000108, A014137, and A401693', None))
	, CartesianProduct(('A014140',), (0, 3, 4, 7), ('A000108', 'A014137 and A014140', 'A014140 and A401693', None))
	, CartesianProduct(('A045445',), (1, 3, 4, 7), ('A045445', None))
	, CartesianProduct(('A060206',), (0, 3, 4, 7), ('A000682', 'A005315, A005316, and A077460', None))
	, CartesianProduct(('A077014',), (0, 3, 4, 7), ('A005316', None))
	, CartesianProduct(('A077054',), (0, 3, 4, 7), ('A005316', 'A005315 and A085973', None))
	, CartesianProduct(('A077056',), (0, 3, 4, 7), (None,))
	, CartesianProduct(('A077460',), (0, 3, 4, 7), ('A005315, A005316, and A060206', None))
	, CartesianProduct(('A078591',), (0, 3, 4, 7), ('A005315', None))
	, CartesianProduct(('A078592',), (0, 3, 4, 7), ('A005316', None))
	, CartesianProduct(('A085973',), (0, 3, 4, 7), ('A005315 and A077054', None))
	, CartesianProduct(('A208357',), (1, 3, 4, 7), ('A005315', None))
	, CartesianProduct(('A217310',), (1, 3, 4, 7), ('A223093', 'A005316, A217318, and A227167', None))
	, CartesianProduct(('A217318',), (1, 3, 4, 7), ('A223095', 'A005316, A217310, and A227167', None))
	, CartesianProduct(('A223093',), (1, 3, 4, 7), ('A217310', 'A000682 and A005316', 'A223094 and A223095', None))
	, CartesianProduct(('A223094',), (1, 3, 4, 7), ('A000682', 'A223093 and A223095', None))
	, CartesianProduct(('A223095',), (1, 3, 4, 7), ('A217318', 'A223093 and A223094', None))
	, CartesianProduct(('A227167',), (1, 3, 4, 7), ('A000682', 'A005316, A217310, and A217318', None))
	, CartesianProduct(('A259702',), (2, 3, 4, 7), ('A000682', 'A301620', None))
	, CartesianProduct(('A301620',), (1, 3, 4, 7), ('A000682', 'A259702', 'A301620 and A334615', None))
	, CartesianProduct(('A322507',), (1, 3, 4, 7), (None,))
	, CartesianProduct(('A333971',), (2, 3, 4, 7), ('A000682', None))
	, CartesianProduct(('A334615',), (2, 3, 4, 7), ('A000682', 'A301620', None))
	, CartesianProduct(('A337581',), (2, 3, 4, 7), ('A000682', None))
	, CartesianProduct(('A400630',), (4, 7), ('A400630', None))
	, CartesianProduct(('A400631',), (6, 7), (None,))
	, CartesianProduct(('A401693',), (1, 3, 4, 7), ('A014140', 'A000108, A014137, and A401693', None))
)))
def test_byFormula(oeisID: OEISid, n: int, f: LiteralString | None, expected: int) -> None:
	assertEqualTo(getattr(byFormula, oeisID)(n, f), expected, oeisID, n, f)
