"""The piles at which a leaf, or a group of leaves, can appear in a folding of a 2^n-dimensional map."""
from __future__ import annotations

from itertools import chain, repeat, starmap
from mapFolding import 首一, 首一二, 首二, 首零一, 首零一二, 首零二
from mapFolding.dataBaskets import StateElimination
from mapFolding.leafDomains import (
	getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomainLeaf, getDomain二一零and二一, getDomain二零and二, getDomain首零一二and首一二,
	getDomain首零二and首二, getLookupDomainsLeaves)
from mapFolding.tests import assertEqualTo, messageTestFailure
from mapFolding.tests.dataSamples.maps2上nDimensional import (
	dictionaryLeafDomainKnown, mapShape2上4Dimensional, mapShape2上5Dimensional, mapShape2上6Dimensional)
from more_itertools import all_unique as allUnique吗
from operator import itemgetter
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence
	from hunterMakesPy import CallableFunction

@pytest.mark.parametrize('mapShape, leaf, expected', (parametersGetDomainLeaf := dict(chain(
		zip(map('mapShape2上4Dimensional-leaf{}'.format, dictionaryLeafDomainKnown[mapShape2上4Dimensional])
			, zip(repeat(mapShape2上4Dimensional), dictionaryLeafDomainKnown[mapShape2上4Dimensional]
				, starmap(range, dictionaryLeafDomainKnown[mapShape2上4Dimensional].values())), strict=True)
		, zip(map('mapShape2上5Dimensional-leaf{}'.format, dictionaryLeafDomainKnown[mapShape2上5Dimensional])
			, zip(repeat(mapShape2上5Dimensional), dictionaryLeafDomainKnown[mapShape2上5Dimensional]
				, starmap(range, dictionaryLeafDomainKnown[mapShape2上5Dimensional].values())), strict=True)
		, zip(map('mapShape2上6Dimensional-leaf{}'.format, dictionaryLeafDomainKnown[mapShape2上6Dimensional])
			, zip(repeat(mapShape2上6Dimensional), dictionaryLeafDomainKnown[mapShape2上6Dimensional]
				, starmap(range, dictionaryLeafDomainKnown[mapShape2上6Dimensional].values())), strict=True)
	))).values(), ids=parametersGetDomainLeaf.keys())
def test_getDomainLeaf(mapShape: tuple[int, ...], leaf: int, expected: range) -> None:
	assertEqualTo(getDomainLeaf(StateElimination(mapShape), leaf), expected, getDomainLeaf.__name__, mapShape, leaf)

@pytest.mark.parametrize('mapShape, expected', (
	pytest.param(mapShape2上4Dimensional
		, dict(zip(dictionaryLeafDomainKnown[mapShape2上4Dimensional], starmap(range, dictionaryLeafDomainKnown[mapShape2上4Dimensional].values()), strict=True))
		, id='mapShape2上4Dimensional')
	, pytest.param(mapShape2上5Dimensional
		, dict(zip(dictionaryLeafDomainKnown[mapShape2上5Dimensional], starmap(range, dictionaryLeafDomainKnown[mapShape2上5Dimensional].values()), strict=True))
		, id='mapShape2上5Dimensional')
	, pytest.param(mapShape2上6Dimensional
		, dict(zip(dictionaryLeafDomainKnown[mapShape2上6Dimensional], starmap(range, dictionaryLeafDomainKnown[mapShape2上6Dimensional].values()), strict=True))
		, id='mapShape2上6Dimensional')
))
def test_getLookupDomainsLeaves(mapShape: tuple[int, ...], expected: dict[int, range]) -> None:
	assertEqualTo(getLookupDomainsLeaves(StateElimination(mapShape)), expected, getLookupDomainsLeaves.__name__, mapShape)

@pytest.mark.parametrize('getDomain, mapShape, leaves, projectLeaves', (
	pytest.param(getDomainDimension一, mapShape2上4Dimensional, (3, 2, 首一(4), 首零一(4)), itemgetter(0, 1, 2, 3), id='getDomainDimension一-mapShape2上4Dimensional-leavesAll')
	, pytest.param(getDomainDimension一, mapShape2上5Dimensional, (3, 2, 首一(5), 首零一(5)), itemgetter(0, 1, 2, 3), id='getDomainDimension一-mapShape2上5Dimensional-leavesAll')
	, pytest.param(getDomainDimension一, mapShape2上6Dimensional, (3, 2, 首一(6), 首零一(6)), itemgetter(0, 1, 2, 3), id='getDomainDimension一-mapShape2上6Dimensional-leavesAll')
	, pytest.param(getDomainDimension一, mapShape2上4Dimensional, (3, 首一(4)), itemgetter(0, 2), id='getDomainDimension一-mapShape2上4Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension一, mapShape2上5Dimensional, (3, 首一(5)), itemgetter(0, 2), id='getDomainDimension一-mapShape2上5Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension一, mapShape2上6Dimensional, (3, 首一(6)), itemgetter(0, 2), id='getDomainDimension一-mapShape2上6Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension二, mapShape2上4Dimensional, (6, 7, 5, 4), itemgetter(0, 1, 2, 3), id='getDomainDimension二-mapShape2上4Dimensional-leavesAll')
	, pytest.param(getDomainDimension二, mapShape2上5Dimensional, (6, 7, 5, 4), itemgetter(0, 1, 2, 3), id='getDomainDimension二-mapShape2上5Dimensional-leavesAll')
	, pytest.param(getDomainDimension二, mapShape2上6Dimensional, (6, 7, 5, 4), itemgetter(0, 1, 2, 3), id='getDomainDimension二-mapShape2上6Dimensional-leavesAll')
	, pytest.param(getDomainDimension二, mapShape2上4Dimensional, (6, 5), itemgetter(0, 2), id='getDomainDimension二-mapShape2上4Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension二, mapShape2上5Dimensional, (6, 5), itemgetter(0, 2), id='getDomainDimension二-mapShape2上5Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension二, mapShape2上6Dimensional, (6, 5), itemgetter(0, 2), id='getDomainDimension二-mapShape2上6Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension首二, mapShape2上4Dimensional, (首二(4), 首零二(4), 首零一二(4), 首一二(4)), itemgetter(0, 1, 2, 3), id='getDomainDimension首二-mapShape2上4Dimensional-leavesAll')
	, pytest.param(getDomainDimension首二, mapShape2上5Dimensional, (首二(5), 首零二(5), 首零一二(5), 首一二(5)), itemgetter(0, 1, 2, 3), id='getDomainDimension首二-mapShape2上5Dimensional-leavesAll')
	, pytest.param(getDomainDimension首二, mapShape2上6Dimensional, (首二(6), 首零二(6), 首零一二(6), 首一二(6)), itemgetter(0, 1, 2, 3), id='getDomainDimension首二-mapShape2上6Dimensional-leavesAll')
	, pytest.param(getDomainDimension首二, mapShape2上4Dimensional, (首二(4), 首零一二(4)), itemgetter(0, 2), id='getDomainDimension首二-mapShape2上4Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension首二, mapShape2上5Dimensional, (首二(5), 首零一二(5)), itemgetter(0, 2), id='getDomainDimension首二-mapShape2上5Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension首二, mapShape2上6Dimensional, (首二(6), 首零一二(6)), itemgetter(0, 2), id='getDomainDimension首二-mapShape2上6Dimensional-leavesFirstThird')
	, pytest.param(getDomain二一零and二一, mapShape2上4Dimensional, (7, 6), itemgetter(0, 1), id='getDomain二一零and二一-mapShape2上4Dimensional')
	, pytest.param(getDomain二一零and二一, mapShape2上5Dimensional, (7, 6), itemgetter(0, 1), id='getDomain二一零and二一-mapShape2上5Dimensional')
	, pytest.param(getDomain二一零and二一, mapShape2上6Dimensional, (7, 6), itemgetter(0, 1), id='getDomain二一零and二一-mapShape2上6Dimensional')
	, pytest.param(getDomain二零and二, mapShape2上4Dimensional, (5, 4), itemgetter(0, 1), id='getDomain二零and二-mapShape2上4Dimensional')
	, pytest.param(getDomain二零and二, mapShape2上5Dimensional, (5, 4), itemgetter(0, 1), id='getDomain二零and二-mapShape2上5Dimensional')
	, pytest.param(getDomain二零and二, mapShape2上6Dimensional, (5, 4), itemgetter(0, 1), id='getDomain二零and二-mapShape2上6Dimensional')
	, pytest.param(getDomain首零一二and首一二, mapShape2上4Dimensional, (首零一二(4), 首一二(4)), itemgetter(0, 1), id='getDomain首零一二and首一二-mapShape2上4Dimensional')
	, pytest.param(getDomain首零一二and首一二, mapShape2上5Dimensional, (首零一二(5), 首一二(5)), itemgetter(0, 1), id='getDomain首零一二and首一二-mapShape2上5Dimensional')
	, pytest.param(getDomain首零一二and首一二, mapShape2上6Dimensional, (首零一二(6), 首一二(6)), itemgetter(0, 1), id='getDomain首零一二and首一二-mapShape2上6Dimensional')
	, pytest.param(getDomain首零二and首二, mapShape2上4Dimensional, (首零二(4), 首二(4)), itemgetter(0, 1), id='getDomain首零二and首二-mapShape2上4Dimensional')
	, pytest.param(getDomain首零二and首二, mapShape2上5Dimensional, (首零二(5), 首二(5)), itemgetter(0, 1), id='getDomain首零二and首二-mapShape2上5Dimensional')
	, pytest.param(getDomain首零二and首二, mapShape2上6Dimensional, (首零二(6), 首二(6)), itemgetter(0, 1), id='getDomain首零二and首二-mapShape2上6Dimensional')
))
def test_getDomainCombined(getDomain: CallableFunction[[StateElimination], Sequence[tuple[int, ...]]], mapShape: tuple[int, ...]
	, leaves: tuple[int, ...], projectLeaves: Callable[[tuple[int, ...]], tuple[int, ...]], domainLeavesEnumerated: set[tuple[int, ...]]) -> None:
	"""Each assignment is unique, and the assignments of `leaves`, projected from the assignments, are exactly the enumerated ones."""
	domainCombined: Sequence[tuple[int, ...]] = getDomain(StateElimination(mapShape))
	assert allUnique吗(domainCombined), messageTestFailure(domainCombined, 'unique assignments', getDomain.__name__, mapShape)
	assertEqualTo(set(map(projectLeaves, domainCombined)), domainLeavesEnumerated, getDomain.__name__, mapShape)
