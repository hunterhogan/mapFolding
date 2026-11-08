"""Tests for elimination algorithm data functions.

These tests verify the correctness of functions in `mapFolding.leafDomains` and `mapFolding.pileOptions` that compute
leaf domains, pile ranges, and addend dictionaries for map folding elimination algorithms.

The test data is stored in `tests/dataSamples/A001417.py` and supports multiple `mapShape`
configurations. Currently, data exists for:
- (2,)*4 → 16 leaves (2^4-dimensional)
- (2,)*5 → 32 leaves (2^5-dimensional)
- (2,)*6 → 64 leaves (2^6-dimensional)

When adding test data for another `mapShape`, add the data to `A001417.py` and add the `mapShape` to the
parametrization of each test that uses the data.
"""

from __future__ import annotations

from mapFolding.choicesLeaf import getChoicesLeaf, getIteratorOfLeaves, getLookupChoicesLeaf
from mapFolding.creases import getLeavesCreaseAnte, getLeavesCreasePost
from mapFolding.dataBaskets import StateElimination
from mapFolding.leafDomains import (
	getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomainLeaf, getDomain二一零and二一, getDomain二零and二, getDomain首零一二and首一二,
	getDomain首零二and首二, getLookupDomainsLeaves)
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples import (
	A001417, p2上nDimensionalDomain3_2_首一_首零一, p2上nDimensionalDomain3_首一, p2上nDimensionalDomain5_4, p2上nDimensionalDomain6_5,
	p2上nDimensionalDomain6_7_5_4, p2上nDimensionalDomain7_6, p2上nDimensionalDomain首二_首零一二, p2上nDimensionalDomain首二_首零二_首零一二_首一二,
	p2上nDimensionalDomain首零一二_首一二, p2上nDimensionalDomain首零二_首二)
from more_itertools import all_unique as allUnique吗, unique_to_each
from operator import itemgetter
from typing import cast, TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import ChoicesLeaf, Pile

@pytest.mark.parametrize('mapShape', (pytest.param(A001417.mapShape2上4Dimensional, id='2^4-dimensional'), pytest.param(A001417.mapShape2上5Dimensional, id='2^5-dimensional'), pytest.param(A001417.mapShape2上6Dimensional, id='2^6-dimensional')))
def test_getLookupDomainsLeaves(mapShape: tuple[int, ...]) -> None:
	"""Verify getLookupDomainsLeaves against authoritative leaf domain data for all leaves."""
	state: StateElimination = StateElimination(mapShape=mapShape)
	dictionaryLeafDomainsAuthoritativeData: dict[int, tuple[int, int, int]] = A001417.dictionaryLeafDomainKnown[mapShape]

	dictionaryLeafDomainsActual: dict[int, range] = getLookupDomainsLeaves(state)

	assertEqualTo(len(dictionaryLeafDomainsActual), state.totalLeaves, getLookupDomainsLeaves.__name__, mapShape)

	for leaf in range(state.totalLeaves):
		rangeActual: range = dictionaryLeafDomainsActual[leaf]
		startAuthoritativeData, stopAuthoritativeData, stepAuthoritativeData = dictionaryLeafDomainsAuthoritativeData[leaf]
		assertEqualTo(rangeActual.start, startAuthoritativeData, 'getLookupDomainsLeaves.range.start', leaf, mapShape)
		assertEqualTo(rangeActual.stop, stopAuthoritativeData, 'getLookupDomainsLeaves.range.stop', leaf, mapShape)
		assertEqualTo(rangeActual.step, stepAuthoritativeData, 'getLookupDomainsLeaves.range.step', leaf, mapShape)

@pytest.mark.parametrize('mapShape', (pytest.param(A001417.mapShape2上4Dimensional, id='2^4-dimensional'), pytest.param(A001417.mapShape2上5Dimensional, id='2^5-dimensional'), pytest.param(A001417.mapShape2上6Dimensional, id='2^6-dimensional')))
def test_getLookupChoicesLeaf(mapShape: tuple[int, ...]) -> None:
	"""Verify getLookupChoicesLeaf against authoritative pile range data for all piles."""
	state: StateElimination = StateElimination(mapShape=mapShape)
	dictionaryChoicesLeafAuthoritativeData: dict[int, tuple[int, ...]] = A001417.dictionaryChoicesLeafKnown[mapShape]

	dictionaryChoicesLeafActual: dict[Pile, ChoicesLeaf] = getLookupChoicesLeaf(state)

	assertEqualTo(len(dictionaryChoicesLeafActual), state.totalLeaves, getLookupChoicesLeaf.__name__, mapShape)

	for pile in range(state.totalLeaves):
		boxOfLeavesPileActual: tuple[int, ...] = tuple(getIteratorOfLeaves(dictionaryChoicesLeafActual[pile]))
		boxOfLeavesPileAuthoritativeData: tuple[int, ...] = dictionaryChoicesLeafAuthoritativeData[pile]
		assertEqualTo(boxOfLeavesPileActual, boxOfLeavesPileAuthoritativeData, getLookupChoicesLeaf.__name__, pile, mapShape)

@pytest.mark.parametrize('mapShape', (pytest.param(A001417.mapShape2上4Dimensional, id='2^4-dimensional'), pytest.param(A001417.mapShape2上5Dimensional, id='2^5-dimensional'), pytest.param(A001417.mapShape2上6Dimensional, id='2^6-dimensional')))
def test_getDomainLeaf(mapShape: tuple[int, ...]) -> None:
	"""Verify getDomainLeaf against authoritative leaf domain data for all leaves."""
	state: StateElimination = StateElimination(mapShape=mapShape)
	dictionaryLeafDomainsAuthoritativeData: dict[int, tuple[int, int, int]] = A001417.dictionaryLeafDomainKnown[mapShape]

	for leaf in range(state.totalLeaves):
		rangeActual: range = getDomainLeaf(state, leaf)
		startAuthoritativeData, stopAuthoritativeData, stepAuthoritativeData = dictionaryLeafDomainsAuthoritativeData[leaf]
		assertEqualTo(rangeActual.start, startAuthoritativeData, 'getDomainLeaf.range.start', leaf, mapShape)
		assertEqualTo(rangeActual.stop, stopAuthoritativeData, 'getDomainLeaf.range.stop', leaf, mapShape)
		assertEqualTo(rangeActual.step, stepAuthoritativeData, 'getDomainLeaf.range.step', leaf, mapShape)

xfailDimension4: pytest.MarkDecorator = pytest.mark.xfail(
	raises=AssertionError, strict=True, reason='Four-dimensional combined domains disagree with observations verified against the folding pickle.')

@pytest.mark.parametrize('totalDimensions, domainFunction, domainKnown, columnPositions', (
	pytest.param(4, getDomainDimension一, p2上nDimensionalDomain3_2_首一_首零一.boxOfDomain2上4Dimensional, (0, 1, 2, 3), id='dimension-one-4'),
	pytest.param(5, getDomainDimension一, p2上nDimensionalDomain3_2_首一_首零一.boxOfDomain2上5Dimensional, (0, 1, 2, 3), id='dimension-one-5'),
	pytest.param(6, getDomainDimension一, p2上nDimensionalDomain3_2_首一_首零一.boxOfDomain2上6Dimensional, (0, 1, 2, 3), id='dimension-one-6'),
	pytest.param(4, getDomainDimension一, p2上nDimensionalDomain3_首一.boxOfDomain2上4Dimensional, (0, 2), id='dimension-one-pairs-4'),
	pytest.param(5, getDomainDimension一, p2上nDimensionalDomain3_首一.boxOfDomain2上5Dimensional, (0, 2), id='dimension-one-pairs-5'),
	pytest.param(6, getDomainDimension一, p2上nDimensionalDomain3_首一.boxOfDomain2上6Dimensional, (0, 2), id='dimension-one-pairs-6'),
	pytest.param(4, getDomainDimension二, p2上nDimensionalDomain6_7_5_4.boxOfDomain2上4Dimensional, (0, 1, 2, 3), marks=xfailDimension4, id='dimension-two-4'),
	pytest.param(5, getDomainDimension二, p2上nDimensionalDomain6_7_5_4.boxOfDomain2上5Dimensional, (0, 1, 2, 3), id='dimension-two-5'),
	pytest.param(6, getDomainDimension二, p2上nDimensionalDomain6_7_5_4.boxOfDomain2上6Dimensional, (0, 1, 2, 3), id='dimension-two-6'),
	pytest.param(4, getDomainDimension二, p2上nDimensionalDomain6_5.boxOfDomain2上4Dimensional, (0, 2), marks=xfailDimension4, id='dimension-two-pairs-4'),
	pytest.param(5, getDomainDimension二, p2上nDimensionalDomain6_5.boxOfDomain2上5Dimensional, (0, 2), id='dimension-two-pairs-5'),
	pytest.param(6, getDomainDimension二, p2上nDimensionalDomain6_5.boxOfDomain2上6Dimensional, (0, 2), id='dimension-two-pairs-6'),
	pytest.param(4, getDomainDimension首二, p2上nDimensionalDomain首二_首零二_首零一二_首一二.boxOfDomain2上4Dimensional, (0, 1, 2, 3), marks=xfailDimension4, id='head-dimension-two-4'),
	pytest.param(5, getDomainDimension首二, p2上nDimensionalDomain首二_首零二_首零一二_首一二.boxOfDomain2上5Dimensional, (0, 1, 2, 3), id='head-dimension-two-5'),
	pytest.param(6, getDomainDimension首二, p2上nDimensionalDomain首二_首零二_首零一二_首一二.boxOfDomain2上6Dimensional, (0, 1, 2, 3), id='head-dimension-two-6'),
	pytest.param(4, getDomainDimension首二, p2上nDimensionalDomain首二_首零一二.boxOfDomain2上4Dimensional, (0, 2), marks=xfailDimension4, id='head-dimension-two-pairs-4'),
	pytest.param(5, getDomainDimension首二, p2上nDimensionalDomain首二_首零一二.boxOfDomain2上5Dimensional, (0, 2), id='head-dimension-two-pairs-5'),
	pytest.param(6, getDomainDimension首二, p2上nDimensionalDomain首二_首零一二.boxOfDomain2上6Dimensional, (0, 2), id='head-dimension-two-pairs-6'),
	pytest.param(4, getDomain二一零and二一, p2上nDimensionalDomain7_6.boxOfDomain2上4Dimensional, (0, 1), marks=xfailDimension4, id='crease-two-one-4'),
	pytest.param(5, getDomain二一零and二一, p2上nDimensionalDomain7_6.boxOfDomain2上5Dimensional, (0, 1), id='crease-two-one-5'),
	pytest.param(6, getDomain二一零and二一, p2上nDimensionalDomain7_6.boxOfDomain2上6Dimensional, (0, 1), id='crease-two-one-6'),
	pytest.param(4, getDomain二零and二, p2上nDimensionalDomain5_4.boxOfDomain2上4Dimensional, (0, 1), marks=xfailDimension4, id='crease-two-4'),
	pytest.param(5, getDomain二零and二, p2上nDimensionalDomain5_4.boxOfDomain2上5Dimensional, (0, 1), id='crease-two-5'),
	pytest.param(6, getDomain二零and二, p2上nDimensionalDomain5_4.boxOfDomain2上6Dimensional, (0, 1), id='crease-two-6'),
	pytest.param(5, getDomain首零一二and首一二, p2上nDimensionalDomain首零一二_首一二.boxOfDomain2上5Dimensional, (0, 1), id='head-crease-two-one-5'),
	pytest.param(6, getDomain首零一二and首一二, p2上nDimensionalDomain首零一二_首一二.boxOfDomain2上6Dimensional, (0, 1), id='head-crease-two-one-6'),
	pytest.param(5, getDomain首零二and首二, p2上nDimensionalDomain首零二_首二.boxOfDomain2上5Dimensional, (0, 1), id='head-crease-two-5'),
	pytest.param(6, getDomain首零二and首二, p2上nDimensionalDomain首零二_首二.boxOfDomain2上6Dimensional, (0, 1), id='head-crease-two-6'),
))
def test_getDomainLeafsCombined(totalDimensions: int, domainFunction: CallableFunction[[StateElimination], Sequence[tuple[int, ...]]],
	domainKnown: Sequence[tuple[int, ...]], columnPositions: tuple[int, ...]) -> None:
	"""Verify combined domain function against authoritative dataset: completeness, uniqueness, correctness."""
	mapShape: tuple[int, ...] = (2,) * totalDimensions
	state: StateElimination = StateElimination(mapShape=mapShape)
	tuplesDomainActual: tuple[tuple[int, ...], ...] = tuple(domainFunction(state))

	tuplesMissingFromActual, tuplesExtraInActual = unique_to_each(domainKnown,
		map(cast('CallableFunction[[tuple[int, ...]], tuple[int, ...]]', itemgetter(*columnPositions)), tuplesDomainActual))
	tuplesMissingFromActual = tuple(tuplesMissingFromActual)
	tuplesExtraInActual = tuple(tuplesExtraInActual)
	hasAllUnique: bool = allUnique吗(tuplesDomainActual)

	assertEqualTo(hasAllUnique, True, domainFunction.__name__, mapShape)
	assertEqualTo(len(tuplesMissingFromActual), 0, domainFunction.__name__, mapShape, missing=tuplesMissingFromActual)
	assertEqualTo(len(tuplesExtraInActual), 0, domainFunction.__name__, mapShape, extra=tuplesExtraInActual)

@pytest.mark.parametrize('mapShape', (pytest.param(A001417.mapShape2上4Dimensional, id='2^4-dimensional'), pytest.param(A001417.mapShape2上5Dimensional, id='2^5-dimensional'), pytest.param(A001417.mapShape2上6Dimensional, id='2^6-dimensional')))
def test_getChoicesLeaf(mapShape: tuple[int, ...]) -> None:
	"""Verify getChoicesLeaf against authoritative pile range data for all piles."""
	state: StateElimination = StateElimination(mapShape=mapShape)
	dictionaryChoicesLeafAuthoritativeData: dict[int, tuple[int, ...]] = A001417.dictionaryChoicesLeafKnown[mapShape]

	for pile in range(state.totalLeaves):
		boxOfLeavesPileActual: tuple[int, ...] = tuple(getIteratorOfLeaves(getChoicesLeaf(state, pile)))
		boxOfLeavesPileAuthoritativeData: tuple[int, ...] = dictionaryChoicesLeafAuthoritativeData[pile]

		assertEqualTo(boxOfLeavesPileActual, boxOfLeavesPileAuthoritativeData, getChoicesLeaf.__name__, pile, mapShape)

@pytest.mark.parametrize('totalDimensions', (pytest.param(5, id='2^5-dimensional'), pytest.param(6, id='2^6-dimensional')))
@pytest.mark.parametrize('creaseFunction, dictionaryExpectedByMapShape', (
	pytest.param(getLeavesCreasePost, A001417.dictionaryCreasesIncreaseKnown, id='increase')
	, pytest.param(getLeavesCreaseAnte, A001417.dictionaryCreasesDecreaseKnown, id='decrease')
))
def test_getLeavesCrease(creaseFunction: CallableFunction[[StateElimination, int], Iterable[int]], dictionaryExpectedByMapShape: dict[tuple[int, ...], dict[int, list[int]]], totalDimensions: int) -> None:
	mapShape: tuple[int, ...] = (2,) * totalDimensions
	state: StateElimination = StateElimination(mapShape=mapShape)
	dictionaryExpectedByLeaf: dict[int, list[int]] = dictionaryExpectedByMapShape[mapShape]

	for leaf in range(state.totalLeaves):
		boxOfLeavesActual: list[int] = list(creaseFunction(state, leaf))
		boxOfLeavesExpectedSorted: list[int] = dictionaryExpectedByLeaf[leaf]

		assertEqualTo(sorted(boxOfLeavesActual), boxOfLeavesExpectedSorted, creaseFunction.__name__, mapShape, leaf)

		assertEqualTo(allUnique吗(boxOfLeavesActual), True, creaseFunction.__name__, mapShape, leaf)

		for leafPost in boxOfLeavesActual:
			assertEqualTo(0 <= leafPost < state.totalLeaves, True, creaseFunction.__name__, mapShape, leaf, leafPost=leafPost)
			bitFlip: int = leaf ^ leafPost
			assertEqualTo((bitFlip > 0) and ((bitFlip & (bitFlip - 1)) == 0), True, creaseFunction.__name__, mapShape, leaf, leafPost=leafPost, bitFlip=bitFlip)

		boxOfBitFlips: list[int] = [leaf ^ leafPost for leafPost in boxOfLeavesActual]
		assertEqualTo(boxOfBitFlips, sorted(boxOfBitFlips), creaseFunction.__name__, mapShape, leaf)
