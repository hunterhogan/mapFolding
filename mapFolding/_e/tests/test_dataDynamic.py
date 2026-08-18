"""Tests for elimination algorithm data functions.

These tests verify the correctness of functions in `mapFolding._e._data` that compute
leaf domains, pile ranges, and addend dictionaries for map folding elimination algorithms.

The test data is stored in `_e/tests/dataSamples/A001417.py` and supports multiple `mapShape`
configurations. Currently, data exists for:
- (2,)*4 → 16 leaves (2^4-dimensional)
- (2,)*5 → 32 leaves (2^5-dimensional)
- (2,)*6 → 64 leaves (2^6-dimensional)

When adding new test data for additional `mapShape` values, add the data to `A001417.py`
and the tests will automatically pick them up via parametrization.
"""

from __future__ import annotations

from mapFolding._e import getChoicesLeaf, getDomainLeaf, getIteratorOfLeaves, getLookupDomainsLeaves
from mapFolding._e._2上nDimensional import (
	getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomain二一零and二一, getDomain二零and二, getDomain首零一二and首一二, getDomain首零二and首二,
	getLeavesCreaseAnte, getLeavesCreasePost)
from mapFolding._e.dataBaskets import StateElimination
from mapFolding._e.pileOptions import getDictionaryChoicesLeaf
from mapFolding._e.tests import assertEqualTo
from mapFolding._e.tests.dataSamples import (
	A001417, p2上nDimensionalDomain3_2_首一_首零一, p2上nDimensionalDomain5_4, p2上nDimensionalDomain6_7_5_4, p2上nDimensionalDomain7_6,
	p2上nDimensionalDomain首二_首零二_首零一二_首一二, p2上nDimensionalDomain首零一二_首一二, p2上nDimensionalDomain首零二_首二)
from more_itertools import all_unique as allUnique吗, unique_to_each
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding._e.theTypes import ChoicesLeaf, Pile
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

	dictionaryChoicesLeafActual: dict[Pile, ChoicesLeaf] = getDictionaryChoicesLeaf(state)

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

@pytest.mark.parametrize("totalDimensions", [5, 6], ids=lambda totalDimensions: f"2^{totalDimensions}-dimensional")
@pytest.mark.parametrize("domainFunction,moduleAuthoritativeData", [
	(getDomainDimension一, p2上nDimensionalDomain3_2_首一_首零一)
	, (getDomainDimension二, p2上nDimensionalDomain6_7_5_4)
	, (getDomainDimension首二, p2上nDimensionalDomain首二_首零二_首零一二_首一二)
	, (getDomain二一零and二一, p2上nDimensionalDomain7_6)
	, (getDomain二零and二, p2上nDimensionalDomain5_4)
	, (getDomain首零一二and首一二, p2上nDimensionalDomain首零一二_首一二)
	, (getDomain首零二and首二, p2上nDimensionalDomain首零二_首二)
], ids=lambda domainFunction: domainFunction.__name__)
def test_getDomainLeafsCombined(domainFunction: CallableFunction[[StateElimination], Sequence[tuple[int, ...]]], moduleAuthoritativeData: ModuleType, totalDimensions: int) -> None:
	"""Verify combined domain function against authoritative dataset: completeness, uniqueness, correctness."""
	mapShape: tuple[int, ...] = (2,) * totalDimensions
	state: StateElimination = StateElimination(mapShape=mapShape)
	tuplesDomainActual: tuple[tuple[int, ...], ...] = tuple(domainFunction(state))
	tuplesDomainAuthoritativeData: tuple[tuple[int, ...], ...] = getattr(
		moduleAuthoritativeData, f"boxOfDomain2上{totalDimensions}Dimensional"
	)

	tuplesMissingFromActual, tuplesExtraInActual = unique_to_each(tuplesDomainAuthoritativeData, tuplesDomainActual)
	tuplesMissingFromActual = tuple(tuplesMissingFromActual)
	tuplesExtraInActual = tuple(tuplesExtraInActual)
	hasAllUnique: bool = allUnique吗(tuplesDomainActual)

	assertEqualTo(hasAllUnique, True, domainFunction.__name__, mapShape)
	assertEqualTo(len(tuplesMissingFromActual), 0, domainFunction.__name__, mapShape)
	assertEqualTo(len(tuplesExtraInActual), 0, domainFunction.__name__, mapShape)

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
