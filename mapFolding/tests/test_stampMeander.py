from __future__ import annotations

from archive.permutationMeanders.stamp_meander import doTheNeedful
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid

@pytest.mark.parametrize(
	'oeisID, n, expected',
	[
		pytest.param('A000136', 6, 144, id='stamp-foldings'),
		pytest.param('A000560', 6, 33, id='symmetric-semi-meanders'),
		pytest.param('A000682', 6, 24, id='semi-meanders'),
		pytest.param('A001011', 6, 38, id='unlabeled-stamp-foldings'),
		pytest.param('A005316', 0, 1, id='open-meanders-base'),
		pytest.param('A005316', 6, 14, id='open-meanders'),
		pytest.param('A077055', 0, 1, id='symmetric-open-meanders-base'),
		pytest.param('A077055', 6, 8, id='symmetric-open-meanders'),
		pytest.param('A077055', 7, 13, id='symmetric-open-meanders-odd'),
		pytest.param('A000682', 1, 1, id='semi-meanders-base'),
		pytest.param('A000682', 2, 1, id='semi-meanders-index-shift'),
	],
)
def test_doTheNeedful(oeisID: OEISid, n: int, expected: int) -> None:
	actual: int = doTheNeedful(oeisID, n)

	assert actual == expected, f'doTheNeedful({oeisID=}, {n=}) returned {actual=}, expected {expected=}.'

@pytest.mark.parametrize(
	'oeisID, n, match',
	[
		pytest.param('A999999', 6, 'supports only', id='unsupported-sequence'),
		pytest.param('A000136', 0, 'not defined below', id='stamp-folding-before-offset'),
		pytest.param('A000560', 1, 'not defined below', id='symmetric-semi-meander-before-offset'),
	],
)
def test_doTheNeedful_error(oeisID: OEISid, n: int, match: str) -> None:
	with pytest.raises(ValueError, match=match):
		doTheNeedful(oeisID, n)

def test_doTheNeedful_typeError() -> None:
	with pytest.raises(TypeError, match='integer OEIS index'):
		doTheNeedful('A000136', True)
