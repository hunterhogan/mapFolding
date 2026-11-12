"""The piles at which a leaf, or a group of leaves, can appear in a folding of a 2^n-dimensional map."""
from __future__ import annotations

from itertools import chain, repeat, starmap
from mapFolding.dataBaskets import StateElimination
from mapFolding.leafDomains import (
	getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomainLeaf, getDomain二一零and二一, getDomain二零and二, getDomain首零一二and首一二,
	getDomain首零二and首二, getLookupDomainsLeaves)
from mapFolding.tests import assertEqualTo, messageTestFailure
from mapFolding.tests.dataSamples import (
	p2上nDimensionalDomain3_2_首一_首零一, p2上nDimensionalDomain3_首一, p2上nDimensionalDomain5_4, p2上nDimensionalDomain6_5,
	p2上nDimensionalDomain6_7_5_4, p2上nDimensionalDomain7_6, p2上nDimensionalDomain首二_首零一二, p2上nDimensionalDomain首二_首零二_首零一二_首一二,
	p2上nDimensionalDomain首零一二_首一二, p2上nDimensionalDomain首零二_首二)
from mapFolding.tests.dataSamples.maps2上nDimensional import (
	dictionaryLeafDomainKnown, mapShape2上4Dimensional, mapShape2上5Dimensional, mapShape2上6Dimensional)
from more_itertools import all_unique as allUnique吗
from operator import itemgetter
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence
	from hunterMakesPy import CallableFunction

parametersGetDomainLeaf: dict[str, tuple[tuple[int, ...], int, range]] = dict(chain(
	zip(map('mapShape2上4Dimensional-leaf{}'.format, dictionaryLeafDomainKnown[mapShape2上4Dimensional])
		, zip(repeat(mapShape2上4Dimensional), dictionaryLeafDomainKnown[mapShape2上4Dimensional]
			, starmap(range, dictionaryLeafDomainKnown[mapShape2上4Dimensional].values())), strict=True)
	, zip(map('mapShape2上5Dimensional-leaf{}'.format, dictionaryLeafDomainKnown[mapShape2上5Dimensional])
		, zip(repeat(mapShape2上5Dimensional), dictionaryLeafDomainKnown[mapShape2上5Dimensional]
			, starmap(range, dictionaryLeafDomainKnown[mapShape2上5Dimensional].values())), strict=True)
	, zip(map('mapShape2上6Dimensional-leaf{}'.format, dictionaryLeafDomainKnown[mapShape2上6Dimensional])
		, zip(repeat(mapShape2上6Dimensional), dictionaryLeafDomainKnown[mapShape2上6Dimensional]
			, starmap(range, dictionaryLeafDomainKnown[mapShape2上6Dimensional].values())), strict=True)
))

@pytest.mark.parametrize('mapShape, leaf, expected', parametersGetDomainLeaf.values(), ids=parametersGetDomainLeaf.keys())
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

markDefectDimension4: pytest.MarkDecorator = pytest.mark.xfail(
	raises=AssertionError, strict=True, reason='Four-dimensional combined domains disagree with observations verified against the folding pickle.')

# Reference: the piles of the leaves in each enumerated folding, extracted by `createDataSamples.makeVerificationDataLeavesDomain`.
@pytest.mark.parametrize('getDomain, mapShape, projectLeavesRecorded, expected', (
	pytest.param(getDomainDimension一, mapShape2上4Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain3_2_首一_首零一.boxOfDomain2上4Dimensional, id='getDomainDimension一-mapShape2上4Dimensional-leavesAll')
	, pytest.param(getDomainDimension一, mapShape2上5Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain3_2_首一_首零一.boxOfDomain2上5Dimensional, id='getDomainDimension一-mapShape2上5Dimensional-leavesAll')
	, pytest.param(getDomainDimension一, mapShape2上6Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain3_2_首一_首零一.boxOfDomain2上6Dimensional, id='getDomainDimension一-mapShape2上6Dimensional-leavesAll')
	, pytest.param(getDomainDimension一, mapShape2上4Dimensional, itemgetter(0, 2), p2上nDimensionalDomain3_首一.boxOfDomain2上4Dimensional, id='getDomainDimension一-mapShape2上4Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension一, mapShape2上5Dimensional, itemgetter(0, 2), p2上nDimensionalDomain3_首一.boxOfDomain2上5Dimensional, id='getDomainDimension一-mapShape2上5Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension一, mapShape2上6Dimensional, itemgetter(0, 2), p2上nDimensionalDomain3_首一.boxOfDomain2上6Dimensional, id='getDomainDimension一-mapShape2上6Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension二, mapShape2上4Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain6_7_5_4.boxOfDomain2上4Dimensional, marks=markDefectDimension4, id='getDomainDimension二-mapShape2上4Dimensional-leavesAll')
	, pytest.param(getDomainDimension二, mapShape2上5Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain6_7_5_4.boxOfDomain2上5Dimensional, id='getDomainDimension二-mapShape2上5Dimensional-leavesAll')
	, pytest.param(getDomainDimension二, mapShape2上6Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain6_7_5_4.boxOfDomain2上6Dimensional, id='getDomainDimension二-mapShape2上6Dimensional-leavesAll')
	, pytest.param(getDomainDimension二, mapShape2上4Dimensional, itemgetter(0, 2), p2上nDimensionalDomain6_5.boxOfDomain2上4Dimensional, marks=markDefectDimension4, id='getDomainDimension二-mapShape2上4Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension二, mapShape2上5Dimensional, itemgetter(0, 2), p2上nDimensionalDomain6_5.boxOfDomain2上5Dimensional, id='getDomainDimension二-mapShape2上5Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension二, mapShape2上6Dimensional, itemgetter(0, 2), p2上nDimensionalDomain6_5.boxOfDomain2上6Dimensional, id='getDomainDimension二-mapShape2上6Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension首二, mapShape2上4Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain首二_首零二_首零一二_首一二.boxOfDomain2上4Dimensional, marks=markDefectDimension4, id='getDomainDimension首二-mapShape2上4Dimensional-leavesAll')
	, pytest.param(getDomainDimension首二, mapShape2上5Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain首二_首零二_首零一二_首一二.boxOfDomain2上5Dimensional, id='getDomainDimension首二-mapShape2上5Dimensional-leavesAll')
	, pytest.param(getDomainDimension首二, mapShape2上6Dimensional, itemgetter(0, 1, 2, 3), p2上nDimensionalDomain首二_首零二_首零一二_首一二.boxOfDomain2上6Dimensional, id='getDomainDimension首二-mapShape2上6Dimensional-leavesAll')
	, pytest.param(getDomainDimension首二, mapShape2上4Dimensional, itemgetter(0, 2), p2上nDimensionalDomain首二_首零一二.boxOfDomain2上4Dimensional, marks=markDefectDimension4, id='getDomainDimension首二-mapShape2上4Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension首二, mapShape2上5Dimensional, itemgetter(0, 2), p2上nDimensionalDomain首二_首零一二.boxOfDomain2上5Dimensional, id='getDomainDimension首二-mapShape2上5Dimensional-leavesFirstThird')
	, pytest.param(getDomainDimension首二, mapShape2上6Dimensional, itemgetter(0, 2), p2上nDimensionalDomain首二_首零一二.boxOfDomain2上6Dimensional, id='getDomainDimension首二-mapShape2上6Dimensional-leavesFirstThird')
	, pytest.param(getDomain二一零and二一, mapShape2上4Dimensional, itemgetter(0, 1), p2上nDimensionalDomain7_6.boxOfDomain2上4Dimensional, marks=markDefectDimension4, id='getDomain二一零and二一-mapShape2上4Dimensional')
	, pytest.param(getDomain二一零and二一, mapShape2上5Dimensional, itemgetter(0, 1), p2上nDimensionalDomain7_6.boxOfDomain2上5Dimensional, id='getDomain二一零and二一-mapShape2上5Dimensional')
	, pytest.param(getDomain二一零and二一, mapShape2上6Dimensional, itemgetter(0, 1), p2上nDimensionalDomain7_6.boxOfDomain2上6Dimensional, id='getDomain二一零and二一-mapShape2上6Dimensional')
	, pytest.param(getDomain二零and二, mapShape2上4Dimensional, itemgetter(0, 1), p2上nDimensionalDomain5_4.boxOfDomain2上4Dimensional, marks=markDefectDimension4, id='getDomain二零and二-mapShape2上4Dimensional')
	, pytest.param(getDomain二零and二, mapShape2上5Dimensional, itemgetter(0, 1), p2上nDimensionalDomain5_4.boxOfDomain2上5Dimensional, id='getDomain二零and二-mapShape2上5Dimensional')
	, pytest.param(getDomain二零and二, mapShape2上6Dimensional, itemgetter(0, 1), p2上nDimensionalDomain5_4.boxOfDomain2上6Dimensional, id='getDomain二零and二-mapShape2上6Dimensional')
	, pytest.param(getDomain首零一二and首一二, mapShape2上4Dimensional, itemgetter(0, 1), p2上nDimensionalDomain首零一二_首一二.boxOfDomain2上4Dimensional, marks=markDefectDimension4, id='getDomain首零一二and首一二-mapShape2上4Dimensional')
	, pytest.param(getDomain首零一二and首一二, mapShape2上5Dimensional, itemgetter(0, 1), p2上nDimensionalDomain首零一二_首一二.boxOfDomain2上5Dimensional, id='getDomain首零一二and首一二-mapShape2上5Dimensional')
	, pytest.param(getDomain首零一二and首一二, mapShape2上6Dimensional, itemgetter(0, 1), p2上nDimensionalDomain首零一二_首一二.boxOfDomain2上6Dimensional, id='getDomain首零一二and首一二-mapShape2上6Dimensional')
	, pytest.param(getDomain首零二and首二, mapShape2上4Dimensional, itemgetter(0, 1), p2上nDimensionalDomain首零二_首二.boxOfDomain2上4Dimensional, marks=markDefectDimension4, id='getDomain首零二and首二-mapShape2上4Dimensional')
	, pytest.param(getDomain首零二and首二, mapShape2上5Dimensional, itemgetter(0, 1), p2上nDimensionalDomain首零二_首二.boxOfDomain2上5Dimensional, id='getDomain首零二and首二-mapShape2上5Dimensional')
	, pytest.param(getDomain首零二and首二, mapShape2上6Dimensional, itemgetter(0, 1), p2上nDimensionalDomain首零二_首二.boxOfDomain2上6Dimensional, id='getDomain首零二and首二-mapShape2上6Dimensional')
))
def test_getDomainCombined(getDomain: CallableFunction[[StateElimination], Sequence[tuple[int, ...]]], mapShape: tuple[int, ...]
	, projectLeavesRecorded: Callable[[tuple[int, ...]], tuple[int, ...]], expected: list[tuple[int, ...]]) -> None:
	"""Each assignment is unique, and the assignments of the recorded leaves are exactly the recorded assignments."""
	domainCombined: Sequence[tuple[int, ...]] = getDomain(StateElimination(mapShape))
	assert allUnique吗(domainCombined), messageTestFailure(domainCombined, 'unique assignments', getDomain.__name__, mapShape)
	assertEqualTo(set(map(projectLeavesRecorded, domainCombined)), set(expected), getDomain.__name__, mapShape)
