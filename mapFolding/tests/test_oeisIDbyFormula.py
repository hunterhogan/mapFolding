from __future__ import annotations

from functools import partial
from humpy_cytoolz import concat
from itertools import product as CartesianProduct
from mapFolding._e.tests import messageTestFailure
from mapFolding.oeis import _oeisIDbyFormulaLookup
from mapFolding.oeis._metadata import dictionaryOEIS as dd
from mapFolding.oeis._oeisIDbyFormulaLookup import *  # ruff: ignore[undefined-local-with-import-star]
from mapFolding.tests.dataSamples.OEISidByFormulaLookup import dictionaryLiterals
from more_itertools import always_iterable
from re import fullmatch
from typing import TYPE_CHECKING
import os
import pytest

if TYPE_CHECKING:
	from _pytest.mark.structures import ParameterSet
	from collections.abc import Iterator
	from hunterMakesPy import CallableFunction

def qq(oeisIDAndAnnotation: tuple[str, str | tuple[str, ...]]) -> CartesianProduct[tuple[str, int, str]]:
	oeisID: str = oeisIDAndAnnotation[0]
	domainOf_f: Iterator[str] = always_iterable(oeisIDAndAnnotation[1], str)
	domainOf_n = range(dd[oeisID]['offset'], dd[oeisID]['valueUnknown'])
	return CartesianProduct((oeisID,), domainOf_n, domainOf_f)

def fml(tt: tuple[str, int, str]) -> ParameterSet:
	name, n, f = tt
	return pytest.param(eval(name), n, f, dd[name]['valuesKnown'][n], id=f'{name}-{f}-{n}')  # ruff: ignore[suspicious-eval-usage]

@pytest.mark.xfail(raises=KeyError, strict=False)
@pytest.mark.parametrize('callableA, n, f, expected', tuple(map(fml, concat(map(qq, dictionaryLiterals.items())))))
def test_oeisIDbyFormula(callableA: CallableFunction[[int, str], int], n: int, f: str, expected: int) -> None:
	actual: int = callableA(n, f)

	assert actual == expected, messageTestFailure(actual, expected, callableA.__name__, n, f)

def ee(oeisIDAndAnnotation: tuple[str, str | tuple[str, ...]]) -> CartesianProduct[tuple[str, int, str]]:
	oeisID: str = oeisIDAndAnnotation[0]
	domainOf_f: Iterator[str] = always_iterable(oeisIDAndAnnotation[1], str)
	return CartesianProduct((oeisID,), (dd[oeisID]['valueUnknown'],), domainOf_f)

ww: Iterator[tuple[str, int, str]] = concat(map(ee, dictionaryLiterals.items()))

def bad(tt: tuple[str, int, str]) -> ParameterSet:
	name, n, f = tt
	return pytest.param(eval(name), n, f, KeyError, id=f"({name}, {n}, '{f}')")  # ruff: ignore[suspicious-eval-usage]

@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="Skipped on GitHub Actions")
@pytest.mark.parametrize('callableA, n, f, expected', tuple(map(bad, ww)))
def test_oeisIDbyFormulaError(callableA: CallableFunction[[int, str], int], n: int, f: str, expected: type[BaseException]) -> None:
	with pytest.raises(expected):
		callableA(n, f)

@pytest.mark.parametrize('expected', [pytest.param(set(dictionaryLiterals), id='dictionaryLiterals')])
def test_oeisIDbyFormulaFunctions(expected: set[str]) -> None:
	actual: set[str] = set(filter(partial(fullmatch, r'A[0-9]{6}'), vars(_oeisIDbyFormulaLookup)))

	assert actual == expected, f"The proof module exposed {actual=}; the generated literals described {expected=}."
