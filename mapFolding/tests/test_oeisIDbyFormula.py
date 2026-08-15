# ruff: file-ignore[undefined-local-with-import-star]
# ruff: file-ignore[suspicious-eval-usage]
from __future__ import annotations

from humpy_cytoolz import concat
from itertools import product as CartesianProduct
from mapFolding.oeis import getMetadata, getValuesKnown
from mapFolding.oeis._byFormulaLookup import *
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.OEISidByFormulaLookup import dictionaryLiterals
from more_itertools import always_iterable
from typing import TYPE_CHECKING
import os
import pytest

if TYPE_CHECKING:
	from _pytest.mark.structures import ParameterSet
	from collections.abc import Iterator
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import OEISid
	from typing import LiteralString

def make_name_n_f(oeisIDAndAnnotation: tuple[LiteralString, str | tuple[str, ...]]) -> CartesianProduct[tuple[LiteralString, int, str]]:
	oeisID: OEISid = oeisIDAndAnnotation[0]
	domainOf_n = range(getMetadata(oeisID)['offset'], getMetadata(oeisID)['valueUnknown'])
	return CartesianProduct((oeisID,), domainOf_n, always_iterable(oeisIDAndAnnotation[1], str))

def makeParameterSet(name_n_f: tuple[str, int, str]) -> ParameterSet:
	name, n, f = name_n_f
	return pytest.param(eval(name), n, f, getValuesKnown(name)[n], id=f'{name}-{f}-{n}')

@pytest.mark.xfail(raises=KeyError, strict=False)
@pytest.mark.parametrize('callableA, n, f, expected', tuple(map(makeParameterSet, concat(map(make_name_n_f, dictionaryLiterals.items())))))
def test_oeisIDbyFormula(callableA: CallableFunction[[int, LiteralString], int], n: int, f: LiteralString, expected: int) -> None:
	assertEqualTo(callableA(n, f), expected, callableA.__name__, n, f)

def make_name_n_fError(oeisIDAndAnnotation: tuple[LiteralString, str | tuple[str, ...]]) -> CartesianProduct[tuple[LiteralString, int, str]]:
	oeisID: OEISid = oeisIDAndAnnotation[0]
	domainOf_f: Iterator[str] = always_iterable(oeisIDAndAnnotation[1], str)
	return CartesianProduct((oeisID,), (getMetadata(oeisID)['valueUnknown'],), domainOf_f)

def makeParameterSetError(name_n_f: tuple[LiteralString, int, str]) -> ParameterSet:
	name, n, f = name_n_f
	return pytest.param(eval(name), n, f, KeyError, id=f"({name}, {n}, '{f}')")

@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="Skipped on GitHub Actions")
@pytest.mark.parametrize('callableA, n, f, expected', tuple(map(makeParameterSetError, concat(map(make_name_n_fError, dictionaryLiterals.items())))))
def test_oeisIDbyFormulaError(callableA: CallableFunction[[int, LiteralString], int], n: int, f: LiteralString, expected: type[BaseException]) -> None:
	with pytest.raises(expected):
		callableA(n, f)
