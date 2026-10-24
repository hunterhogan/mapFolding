"""Tests for elimination algorithm data functions.

These tests verify the correctness of functions in `mapFolding.leafDomains` and `mapFolding.pileOptions` that compute
leaf domains, pile ranges, and addend dictionaries for map folding elimination algorithms.

The test data is stored in `tests/dataSamples/A001417.py` and supports multiple `mapShape`
configurations. Currently, data exists for:
- (2,)*4 → 16 leaves (2^4-dimensional)
- (2,)*5 → 32 leaves (2^5-dimensional)
- (2,)*6 → 64 leaves (2^6-dimensional)

When adding new test data for additional `mapShape` values, add the data to `A001417.py`
and the tests will automatically pick them up via parametrization.
"""

from __future__ import annotations

from mapFolding.beDRY import getIteratorOfLeaves
from mapFolding.leafDomains import (
	getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomain二一零and二一, getDomain二零and二, getDomain首零一二and首一二, getDomain首零二and首二)
from mapFolding.creases import getLeavesCreaseAnte, getLeavesCreasePost
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples import (
	A001417, p2上nDimensionalDomain3_2_首一_首零一, p2上nDimensionalDomain3_首一, p2上nDimensionalDomain5_4, p2上nDimensionalDomain6_5,
	p2上nDimensionalDomain6_7_5_4, p2上nDimensionalDomain7_6, p2上nDimensionalDomain首二_首零一二, p2上nDimensionalDomain首二_首零二_首零一二_首一二,
	p2上nDimensionalDomain首零一二_首一二, p2上nDimensionalDomain首零二_首二)
from mapFolding.dataBaskets import StateElimination
from mapFolding.leafDomains import getDomainLeaf, getLookupDomainsLeaves
from mapFolding.pileOptions import getChoicesLeaf, getLookupChoicesLeaf
from more_itertools import all_unique as allUnique吗, unique_to_each
from operator import itemgetter
from typing import cast, TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import ChoicesLeaf
	from mapFolding.theTypes import Pile
	from types import ModuleType

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
def test_getLookupDomainsLeaves(mapShape: tuple[int, ...]) -> None:
	"""Verify getLookupDomainsLeaves against authoritative leaf domain data for all leaves."""
	state: StateElimination = StateElimination(mapShape=mapShape)
	dictionaryLeafDomainsAuthoritativeData: dict[int, tuple[int, int, int]] = A001417.dictionaryLeafDomainKnown[mapShape]

	dictionaryLeafDomainsActual: dict[int, range] = getLookupDomainsLeaves(state)

	assertEqualTo(len(dictionaryLeafDomainsActual), state.totalLeaves, 'getLookupDomainsLeaves', mapShape)

	for leaf in range(state.totalLeaves):
		rangeActual: range = dictionaryLeafDomainsActual[leaf]
		startAuthoritativeData, stopAuthoritativeData, stepAuthoritativeData = dictionaryLeafDomainsAuthoritativeData[leaf]
		assertEqualTo(rangeActual.start, startAuthoritativeData, 'getLookupDomainsLeaves.range.start', leaf, mapShape)
		assertEqualTo(rangeActual.stop, stopAuthoritativeData, 'getLookupDomainsLeaves.range.stop', leaf, mapShape)
		assertEqualTo(rangeActual.step, stepAuthoritativeData, 'getLookupDomainsLeaves.range.step', leaf, mapShape)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryChoicesLeafKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryChoicesLeafKnown])
def test_getDictionaryChoicesLeaf(mapShape: tuple[int, ...]) -> None:
	"""Verify getDictionaryChoicesLeaf against authoritative pile range data for all piles."""
	state: StateElimination = StateElimination(mapShape=mapShape)
	dictionaryChoicesLeafAuthoritativeData: dict[int, tuple[int, ...]] = A001417.dictionaryChoicesLeafKnown[mapShape]

	dictionaryChoicesLeafActual: dict[Pile, ChoicesLeaf] = getLookupChoicesLeaf(state)

	assertEqualTo(len(dictionaryChoicesLeafActual), state.totalLeaves, 'getDictionaryChoicesLeaf', mapShape)

	for pile in range(state.totalLeaves):
		boxOfLeavesPileActual: tuple[int, ...] = tuple(getIteratorOfLeaves(dictionaryChoicesLeafActual[pile]))
		boxOfLeavesPileAuthoritativeData: tuple[int, ...] = dictionaryChoicesLeafAuthoritativeData[pile]
		assertEqualTo(boxOfLeavesPileActual, boxOfLeavesPileAuthoritativeData, 'getDictionaryChoicesLeaf', pile, mapShape)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
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

@pytest.mark.parametrize('totalDimensions, domainFunction, moduleAuthoritativeData, columnPositions', (
	pytest.param(4, getDomainDimension一, p2上nDimensionalDomain3_2_首一_首零一, (0, 1, 2, 3), id='dimension-one-4'),
	pytest.param(5, getDomainDimension一, p2上nDimensionalDomain3_2_首一_首零一, (0, 1, 2, 3), id='dimension-one-5'),
	pytest.param(6, getDomainDimension一, p2上nDimensionalDomain3_2_首一_首零一, (0, 1, 2, 3), id='dimension-one-6'),
	pytest.param(4, getDomainDimension一, p2上nDimensionalDomain3_首一, (0, 2), id='dimension-one-pairs-4'),
	pytest.param(5, getDomainDimension一, p2上nDimensionalDomain3_首一, (0, 2), id='dimension-one-pairs-5'),
	pytest.param(6, getDomainDimension一, p2上nDimensionalDomain3_首一, (0, 2), id='dimension-one-pairs-6'),
	pytest.param(4, getDomainDimension二, p2上nDimensionalDomain6_7_5_4, (0, 1, 2, 3), marks=xfailDimension4, id='dimension-two-4'),
	pytest.param(5, getDomainDimension二, p2上nDimensionalDomain6_7_5_4, (0, 1, 2, 3), id='dimension-two-5'),
	pytest.param(6, getDomainDimension二, p2上nDimensionalDomain6_7_5_4, (0, 1, 2, 3), id='dimension-two-6'),
	pytest.param(4, getDomainDimension二, p2上nDimensionalDomain6_5, (0, 2), marks=xfailDimension4, id='dimension-two-pairs-4'),
	pytest.param(5, getDomainDimension二, p2上nDimensionalDomain6_5, (0, 2), id='dimension-two-pairs-5'),
	pytest.param(6, getDomainDimension二, p2上nDimensionalDomain6_5, (0, 2), id='dimension-two-pairs-6'),
	pytest.param(4, getDomainDimension首二, p2上nDimensionalDomain首二_首零二_首零一二_首一二, (0, 1, 2, 3), marks=xfailDimension4, id='head-dimension-two-4'),
	pytest.param(5, getDomainDimension首二, p2上nDimensionalDomain首二_首零二_首零一二_首一二, (0, 1, 2, 3), id='head-dimension-two-5'),
	pytest.param(6, getDomainDimension首二, p2上nDimensionalDomain首二_首零二_首零一二_首一二, (0, 1, 2, 3), id='head-dimension-two-6'),
	pytest.param(4, getDomainDimension首二, p2上nDimensionalDomain首二_首零一二, (0, 2), marks=xfailDimension4, id='headDimensionTwoPairs4'),
	pytest.param(5, getDomainDimension首二, p2上nDimensionalDomain首二_首零一二, (0, 2), id='headDimensionTwoPairs5'),
	pytest.param(6, getDomainDimension首二, p2上nDimensionalDomain首二_首零一二, (0, 2), id='headDimensionTwoPairs6'),
	pytest.param(4, getDomain二一零and二一, p2上nDimensionalDomain7_6, (0, 1), marks=xfailDimension4, id='crease-two-one-4'),
	pytest.param(5, getDomain二一零and二一, p2上nDimensionalDomain7_6, (0, 1), id='crease-two-one-5'),
	pytest.param(6, getDomain二一零and二一, p2上nDimensionalDomain7_6, (0, 1), id='crease-two-one-6'),
	pytest.param(4, getDomain二零and二, p2上nDimensionalDomain5_4, (0, 1), marks=xfailDimension4, id='crease-two-4'),
	pytest.param(5, getDomain二零and二, p2上nDimensionalDomain5_4, (0, 1), id='crease-two-5'),
	pytest.param(6, getDomain二零and二, p2上nDimensionalDomain5_4, (0, 1), id='crease-two-6'),
	pytest.param(5, getDomain首零一二and首一二, p2上nDimensionalDomain首零一二_首一二, (0, 1), id='head-crease-two-one-5'),
	pytest.param(6, getDomain首零一二and首一二, p2上nDimensionalDomain首零一二_首一二, (0, 1), id='head-crease-two-one-6'),
	pytest.param(5, getDomain首零二and首二, p2上nDimensionalDomain首零二_首二, (0, 1), id='head-crease-two-5'),
	pytest.param(6, getDomain首零二and首二, p2上nDimensionalDomain首零二_首二, (0, 1), id='head-crease-two-6'),
))
def test_getDomainLeafsCombined(totalDimensions: int, domainFunction: CallableFunction[[StateElimination], Sequence[tuple[int, ...]]],
	moduleAuthoritativeData: ModuleType, columnPositions: tuple[int, ...]) -> None:
	"""Verify combined domain function against authoritative dataset: completeness, uniqueness, correctness."""
	mapShape: tuple[int, ...] = (2,) * totalDimensions
	state: StateElimination = StateElimination(mapShape=mapShape)
	tuplesDomainActual: tuple[tuple[int, ...], ...] = tuple(domainFunction(state))
	tuplesDomainAuthoritativeData: tuple[tuple[int, ...], ...] = getattr(
		moduleAuthoritativeData, f"boxOfDomain2上{totalDimensions}Dimensional"
	)

	tuplesMissingFromActual, tuplesExtraInActual = unique_to_each(tuplesDomainAuthoritativeData,
		map(cast('CallableFunction[[tuple[int, ...]], tuple[int, ...]]', itemgetter(*columnPositions)), tuplesDomainActual))
	tuplesMissingFromActual = tuple(tuplesMissingFromActual)
	tuplesExtraInActual = tuple(tuplesExtraInActual)
	hasAllUnique: bool = allUnique吗(tuplesDomainActual)

	assertEqualTo(hasAllUnique, True, domainFunction.__name__, mapShape)
	assertEqualTo(len(tuplesMissingFromActual), 0, domainFunction.__name__, mapShape, missing=tuplesMissingFromActual)
	assertEqualTo(len(tuplesExtraInActual), 0, domainFunction.__name__, mapShape, extra=tuplesExtraInActual)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryChoicesLeafKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryChoicesLeafKnown])
def test_getChoicesLeaf(mapShape: tuple[int, ...]) -> None:
	"""Verify getChoicesLeaf against authoritative pile range data for all piles."""
	state: StateElimination = StateElimination(mapShape=mapShape)
	dictionaryChoicesLeafAuthoritativeData: dict[int, tuple[int, ...]] = A001417.dictionaryChoicesLeafKnown[mapShape]

	for pile in range(state.totalLeaves):
		boxOfLeavesPileActual: tuple[int, ...] = tuple(getIteratorOfLeaves(getChoicesLeaf(state, pile)))
		boxOfLeavesPileAuthoritativeData: tuple[int, ...] = dictionaryChoicesLeafAuthoritativeData[pile]

		assertEqualTo(boxOfLeavesPileActual, boxOfLeavesPileAuthoritativeData, 'getChoicesLeaf', pile, mapShape)

@pytest.mark.parametrize("totalDimensions", [5, 6], ids=lambda totalDimensions: f"2^{totalDimensions}-dimensional")
@pytest.mark.parametrize("creaseKind,creaseFunction,dictionaryExpectedByMapShape", [("increase", getLeavesCreasePost, A001417.dictionaryCreasesIncreaseKnown), ("decrease", getLeavesCreaseAnte, A001417.dictionaryCreasesDecreaseKnown)], ids=["increase", "decrease"])
def test_getLeavesCrease(creaseKind: str, creaseFunction: CallableFunction[[StateElimination, int], Iterable[int]], dictionaryExpectedByMapShape: dict[tuple[int, ...], dict[int, list[int]]], totalDimensions: int) -> None:
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
