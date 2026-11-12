"""The leaves that can appear at each pile of a folding of a 2^n-dimensional map.

A `ChoicesLeaf` reaches the assertion through `getIteratorOfLeaves`, its documented reader.
"""
from __future__ import annotations

from itertools import chain, repeat
from mapFolding.choicesLeaf import getChoicesLeaf, getIteratorOfLeaves, getLookupChoicesLeaf
from mapFolding.dataBaskets import StateElimination
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.maps2上nDimensional import (
	dictionaryChoicesLeafKnown, mapShape2上4Dimensional, mapShape2上5Dimensional, mapShape2上6Dimensional)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from mapFolding.theTypes import UndeterminedPiles

parametersGetChoicesLeaf: dict[str, tuple[tuple[int, ...], int, tuple[int, ...]]] = dict(chain(
	zip(map('mapShape2上4Dimensional-pile{}'.format, dictionaryChoicesLeafKnown[mapShape2上4Dimensional])
		, zip(repeat(mapShape2上4Dimensional), dictionaryChoicesLeafKnown[mapShape2上4Dimensional].keys(), dictionaryChoicesLeafKnown[mapShape2上4Dimensional].values()), strict=True)
	, zip(map('mapShape2上5Dimensional-pile{}'.format, dictionaryChoicesLeafKnown[mapShape2上5Dimensional])
		, zip(repeat(mapShape2上5Dimensional), dictionaryChoicesLeafKnown[mapShape2上5Dimensional].keys(), dictionaryChoicesLeafKnown[mapShape2上5Dimensional].values()), strict=True)
	, zip(map('mapShape2上6Dimensional-pile{}'.format, dictionaryChoicesLeafKnown[mapShape2上6Dimensional])
		, zip(repeat(mapShape2上6Dimensional), dictionaryChoicesLeafKnown[mapShape2上6Dimensional].keys(), dictionaryChoicesLeafKnown[mapShape2上6Dimensional].values()), strict=True)
))

@pytest.mark.parametrize('mapShape, pile, expected', parametersGetChoicesLeaf.values(), ids=parametersGetChoicesLeaf.keys())
def test_getChoicesLeaf(mapShape: tuple[int, ...], pile: int, expected: tuple[int, ...]) -> None:
	assertEqualTo(tuple(getIteratorOfLeaves(getChoicesLeaf(StateElimination(mapShape), pile))), expected, getChoicesLeaf.__name__, mapShape, pile)

@pytest.mark.parametrize('mapShape, expected', (
	pytest.param(mapShape2上4Dimensional, dictionaryChoicesLeafKnown[mapShape2上4Dimensional], id='mapShape2上4Dimensional')
	, pytest.param(mapShape2上5Dimensional, dictionaryChoicesLeafKnown[mapShape2上5Dimensional], id='mapShape2上5Dimensional')
	, pytest.param(mapShape2上6Dimensional, dictionaryChoicesLeafKnown[mapShape2上6Dimensional], id='mapShape2上6Dimensional')
))
def test_getLookupChoicesLeaf(mapShape: tuple[int, ...], expected: dict[int, tuple[int, ...]]) -> None:
	lookupChoicesLeaf: UndeterminedPiles = getLookupChoicesLeaf(StateElimination(mapShape))
	assertEqualTo(dict(zip(lookupChoicesLeaf, map(tuple, map(getIteratorOfLeaves, lookupChoicesLeaf.values())), strict=True)), expected
		, getLookupChoicesLeaf.__name__, mapShape)
