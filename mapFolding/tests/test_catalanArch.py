"""Arch triangles, `mapFolding.algorithms.catalanArch`.

`doTheNeedful(n)` has no docstring; `easyRun/reduceArches.py` writes its result as row n of A287548, which is the basis of the
expected cells, and its last cell is A000682(n).
"""
from __future__ import annotations

from mapFolding.algorithms.catalanArch import doTheNeedful
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid

@pytest.mark.parametrize('oeisID', ('A287548',))
@pytest.mark.parametrize('n', range(1, 16))
def test_doTheNeedful(n: int, oeisID: OEISid, cellsKnown: list[int]) -> None:
	assertEqualTo(doTheNeedful(n), cellsKnown, doTheNeedful.__name__, n)
