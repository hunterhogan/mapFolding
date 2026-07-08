"""Tests for elimination algorithm data functions.

These tests verify the correctness of functions in `mapFolding._e._data` that compute
leaf domains, pile ranges, and addend dictionaries for map folding elimination algorithms.

The test data is stored in `_e/tests/dataSamples/A001417.py` and supports multiple `mapShape`
configurations. Currently, data exists for:
- (2,)*4 → 16 leaves (2d4)
- (2,)*5 → 32 leaves (2d5)
- (2,)*6 → 64 leaves (2d6)

When adding new test data for additional `mapShape` values, add the data to `A001417.py`
and the tests will automatically pick them up via parametrization.
"""

from __future__ import annotations

from mapFolding._e import (
	getDictionaryLeafDomains, getDictionaryLeafOptions, getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomain二一零and二一,
	getDomain二零and二, getDomain首零一二and首一二, getDomain首零二and首二, getIteratorOfLeaves, getLeafDomain, getLeafOptions, getLeavesCreaseAnte,
	getLeavesCreasePost)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.tests import assertEqualTo
from mapFolding._e.tests.dataSamples import (
	A001417, p2DnDomain3_2_首一_首零一, p2DnDomain5_4, p2DnDomain6_7_5_4, p2DnDomain7_6, p2DnDomain首二_首零二_首零一二_首一二, p2DnDomain首零一二_首一二,
	p2DnDomain首零二_首二)
from more_itertools import all_unique as allUnique吗, unique_to_each
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding._e.theTypes import LeafOptions, Pile
	from types import ModuleType

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
def test_getDictionaryLeafDomains(mapShape: tuple[int, ...]) -> None:
	"""Verify getDictionaryLeafDomains against authoritative leaf domain data for all leaves."""
	state: EliminationState = EliminationState(mapShape=mapShape)
	dictionaryLeafDomainsAuthoritativeData: dict[int, tuple[int, int, int]] = A001417.dictionaryLeafDomainKnown[mapShape]

	dictionaryLeafDomainsActual: dict[int, range] = getDictionaryLeafDomains(state)

	assertEqualTo(len(dictionaryLeafDomainsActual), state.leavesTotal, 'getDictionaryLeafDomains', mapShape)

	for leaf in range(state.leavesTotal):
		rangeActual: range = dictionaryLeafDomainsActual[leaf]
		startAuthoritativeData, stopAuthoritativeData, stepAuthoritativeData = dictionaryLeafDomainsAuthoritativeData[leaf]
		assertEqualTo(rangeActual.start, startAuthoritativeData, 'getDictionaryLeafDomains.range.start', leaf, mapShape)
		assertEqualTo(rangeActual.stop, stopAuthoritativeData, 'getDictionaryLeafDomains.range.stop', leaf, mapShape)
		assertEqualTo(rangeActual.step, stepAuthoritativeData, 'getDictionaryLeafDomains.range.step', leaf, mapShape)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafOptionsKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafOptionsKnown])
def test_getDictionaryLeafOptions(mapShape: tuple[int, ...]) -> None:
	"""Verify getDictionaryLeafOptions against authoritative pile range data for all piles."""
	state: EliminationState = EliminationState(mapShape=mapShape)
	dictionaryLeafOptionsAuthoritativeData: dict[int, tuple[int, ...]] = A001417.dictionaryLeafOptionsKnown[mapShape]

	dictionaryLeafOptionsActual: dict[Pile, LeafOptions] = getDictionaryLeafOptions(state)

	assertEqualTo(len(dictionaryLeafOptionsActual), state.leavesTotal, 'getDictionaryLeafOptions', mapShape)

	for pile in range(state.leavesTotal):
		tupleLeavesPileActual: tuple[int, ...] = tuple(getIteratorOfLeaves(dictionaryLeafOptionsActual[pile]))
		tupleLeavesPileAuthoritativeData: tuple[int, ...] = dictionaryLeafOptionsAuthoritativeData[pile]
		assertEqualTo(tupleLeavesPileActual, tupleLeavesPileAuthoritativeData, 'getDictionaryLeafOptions', pile, mapShape)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
def test_getLeafDomain(mapShape: tuple[int, ...]) -> None:
	"""Verify getLeafDomain against authoritative leaf domain data for all leaves."""
	state: EliminationState = EliminationState(mapShape=mapShape)
	dictionaryLeafDomainsAuthoritativeData: dict[int, tuple[int, int, int]] = A001417.dictionaryLeafDomainKnown[mapShape]

	for leaf in range(state.leavesTotal):
		rangeActual: range = getLeafDomain(state, leaf)
		startAuthoritativeData, stopAuthoritativeData, stepAuthoritativeData = dictionaryLeafDomainsAuthoritativeData[leaf]
		assertEqualTo(rangeActual.start, startAuthoritativeData, 'getLeafDomain.range.start', leaf, mapShape)
		assertEqualTo(rangeActual.stop, stopAuthoritativeData, 'getLeafDomain.range.stop', leaf, mapShape)
		assertEqualTo(rangeActual.step, stepAuthoritativeData, 'getLeafDomain.range.step', leaf, mapShape)

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize("domainFunction,moduleAuthoritativeData", [
	(getDomainDimension一, p2DnDomain3_2_首一_首零一)
	, (getDomainDimension二, p2DnDomain6_7_5_4)
	, (getDomainDimension首二, p2DnDomain首二_首零二_首零一二_首一二)
	, (getDomain二一零and二一, p2DnDomain7_6)
	, (getDomain二零and二, p2DnDomain5_4)
	, (getDomain首零一二and首一二, p2DnDomain首零一二_首一二)
	, (getDomain首零二and首二, p2DnDomain首零二_首二)
], ids=lambda domainFunction: domainFunction.__name__)
def test_getLeafDomainsCombined(domainFunction: CallableFunction[[EliminationState], Sequence[tuple[int, ...]]], moduleAuthoritativeData: ModuleType, dimensionsTotal: int) -> None:
	"""Verify combined domain function against authoritative dataset: completeness, uniqueness, correctness."""
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state: EliminationState = EliminationState(mapShape=mapShape)
	tuplesDomainActual: tuple[tuple[int, ...], ...] = tuple(domainFunction(state))
	tuplesDomainAuthoritativeData: tuple[tuple[int, ...], ...] = getattr(moduleAuthoritativeData, f"listDomain2D{dimensionsTotal}")

	tuplesMissingFromActual, tuplesExtraInActual = unique_to_each(tuplesDomainAuthoritativeData, tuplesDomainActual)
	tuplesMissingFromActual = tuple(tuplesMissingFromActual)
	tuplesExtraInActual = tuple(tuplesExtraInActual)
	hasAllUnique: bool = allUnique吗(tuplesDomainActual)

	assertEqualTo(hasAllUnique, True, domainFunction.__name__, mapShape)
	assertEqualTo(len(tuplesMissingFromActual), 0, domainFunction.__name__, mapShape)
	assertEqualTo(len(tuplesExtraInActual), 0, domainFunction.__name__, mapShape)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafOptionsKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafOptionsKnown])
def test_getLeafOptions(mapShape: tuple[int, ...]) -> None:
	"""Verify getLeafOptions against authoritative pile range data for all piles."""
	state: EliminationState = EliminationState(mapShape=mapShape)
	dictionaryLeafOptionsAuthoritativeData: dict[int, tuple[int, ...]] = A001417.dictionaryLeafOptionsKnown[mapShape]

	for pile in range(state.leavesTotal):
		tupleLeavesPileActual: tuple[int, ...] = tuple(getIteratorOfLeaves(getLeafOptions(state, pile)))
		tupleLeavesPileAuthoritativeData: tuple[int, ...] = dictionaryLeafOptionsAuthoritativeData[pile]

		assertEqualTo(tupleLeavesPileActual, tupleLeavesPileAuthoritativeData, 'getLeafOptions', pile, mapShape)

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize("creaseKind,creaseFunction,dictionaryExpectedByMapShape", [("increase", getLeavesCreasePost, A001417.dictionaryCreasesIncreaseKnown), ("decrease", getLeavesCreaseAnte, A001417.dictionaryCreasesDecreaseKnown)], ids=["increase", "decrease"])
def test_getLeavesCrease(creaseKind: str, creaseFunction: CallableFunction[[EliminationState, int], Iterable[int]], dictionaryExpectedByMapShape: dict[tuple[int, ...], dict[int, list[int]]], dimensionsTotal: int) -> None:
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state: EliminationState = EliminationState(mapShape=mapShape)
	dictionaryExpectedByLeaf: dict[int, list[int]] = dictionaryExpectedByMapShape[mapShape]

	for leaf in range(state.leavesTotal):
		listLeavesActual: list[int] = list(creaseFunction(state, leaf))
		listLeavesExpectedSorted: list[int] = dictionaryExpectedByLeaf[leaf]

		assertEqualTo(sorted(listLeavesActual), listLeavesExpectedSorted, creaseFunction.__name__, mapShape, leaf)

		assertEqualTo(allUnique吗(listLeavesActual), True, creaseFunction.__name__, mapShape, leaf)

		for leafPost in listLeavesActual:
			assertEqualTo(0 <= leafPost < state.leavesTotal, True, creaseFunction.__name__, mapShape, leaf, leafPost=leafPost)
			bitFlip: int = leaf ^ leafPost
			assertEqualTo((bitFlip > 0) and ((bitFlip & (bitFlip - 1)) == 0), True, creaseFunction.__name__, mapShape, leaf, leafPost=leafPost, bitFlip=bitFlip)

		listBitFlips: list[int] = [leaf ^ leafPost for leafPost in listLeavesActual]
		assertEqualTo(listBitFlips, sorted(listBitFlips), creaseFunction.__name__, mapShape, leaf)
