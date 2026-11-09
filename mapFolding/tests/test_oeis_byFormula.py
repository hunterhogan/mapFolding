from __future__ import annotations

from functools import partial
from inspect import get_annotations
from itertools import chain, product as CartesianProduct
from mapFolding.oeis import byFormula, getMetadata, settingsOEIS
from mapFolding.tests import assertEqualTo
from typing import get_args, TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from typing import LiteralString

@pytest.mark.parametrize('oeisID, n, f', tuple(chain.from_iterable(map(  # ruff: ignore[unnecessary-map]
	lambda oeisID: CartesianProduct((oeisID,), filter(getMetadata(oeisID)['offset'].__le__, (getMetadata(oeisID)['offset'], 3, 4, 7))
		, (*get_args(get_args(get_annotations(getattr(byFormula, oeisID), eval_str=True)['f'])[0]), None))
	, filter(partial(hasattr, byFormula), settingsOEIS.boxOfIDs)
))))
def test_byFormula(oeisID: OEISid, n: int, f: LiteralString | None, expected: int) -> None:
	assertEqualTo(getattr(byFormula, oeisID)(n, f), expected, oeisID, n, f)
