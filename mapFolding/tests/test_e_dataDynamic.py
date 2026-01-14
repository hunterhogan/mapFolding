"""Tests for elimination algorithm data functions.

These tests verify the correctness of functions in `mapFolding._e._data` that compute
leaf domains, pile ranges, and addend dictionaries for map folding elimination algorithms.

The test data is stored in `tests/dataSamples/A001417.py` and supports multiple `mapShape`
configurations. Currently, data exists for:
- (2,)*4 → 16 leaves (2d4)
- (2,)*5 → 32 leaves (2d5)
- (2,)*6 → 64 leaves (2d6)

When adding new test data for additional `mapShape` values, add the data to `A001417.py`
and the tests will automatically pick them up via parametrization.
"""

from collections.abc import Callable, Iterable, Sequence
from mapFolding._e import (
	getDictionaryLeafDomains, getDictionaryPileRanges, getDomainDimension一, getDomainDimension二, getDomainDimension首二,
	getDomain二一零and二一, getDomain二零and二, getDomain首零一二and首一二, getDomain首零二and首二, getLeafDomain, getLeavesCreaseBack,
	getLeavesCreaseNext, getPileRange)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.tests.dataSamples import (
	A001417, p2DnDomain3_2_首一_首零一, p2DnDomain5_4, p2DnDomain6_7_5_4, p2DnDomain7_6, p2DnDomain首二_首零二_首零一二_首一二,
	p2DnDomain首零一二_首一二, p2DnDomain首零二_首二)
from more_itertools import all_unique, unique_to_each
from types import ModuleType
import pytest

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
def test_getDictionaryLeafDomains(mapShape: tuple[int, ...]) -> None:
	"""Verify getDictionaryLeafDomains against authoritative leaf domain data for all leaves."""
	state = EliminationState(mapShape=mapShape)
	dictionaryLeafDomainsAuthoritativeData = A001417.dictionaryLeafDomainKnown[mapShape]

	dictionaryLeafDomainsActual = getDictionaryLeafDomains(state)

	assert len(dictionaryLeafDomainsActual) == state.leavesTotal, (
		f"getDictionaryLeafDomains: dictionary length {len(dictionaryLeafDomainsActual)} != {state.leavesTotal} for {mapShape=}."
	)

	for leaf in range(state.leavesTotal):
		rangeActual = dictionaryLeafDomainsActual[leaf]
		startAuthoritativeData, stopAuthoritativeData, stepAuthoritativeData = dictionaryLeafDomainsAuthoritativeData[leaf]
		assert rangeActual.start == startAuthoritativeData, (
			f"getDictionaryLeafDomains: range.start mismatch at {leaf=} for {mapShape=}. "
			f"Expected {startAuthoritativeData}, got {rangeActual.start}."
		)
		assert rangeActual.stop == stopAuthoritativeData, (
			f"getDictionaryLeafDomains: range.stop mismatch at {leaf=} for {mapShape=}. "
			f"Expected {stopAuthoritativeData}, got {rangeActual.stop}."
		)
		assert rangeActual.step == stepAuthoritativeData, (
			f"getDictionaryLeafDomains: range.step mismatch at {leaf=} for {mapShape=}. "
			f"Expected {stepAuthoritativeData}, got {rangeActual.step}."
		)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryPileRangesKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown])
def test_getDictionaryPileRanges(mapShape: tuple[int, ...]) -> None:
	"""Verify getDictionaryPileRanges against authoritative pile range data for all piles."""
	state = EliminationState(mapShape=mapShape)
	dictionaryPileRangesAuthoritativeData = A001417.dictionaryPileRangesKnown[mapShape]

	dictionaryPileRangesActual = getDictionaryPileRanges(state)

	assert len(dictionaryPileRangesActual) == state.leavesTotal, (
		f"getDictionaryPileRanges: dictionary length {len(dictionaryPileRangesActual)} != {state.leavesTotal} for {mapShape=}."
	)

	for pile in range(state.leavesTotal):
		assert dictionaryPileRangesActual[pile] == dictionaryPileRangesAuthoritativeData[pile], (
			f"getDictionaryPileRanges: mismatch at {pile=} for {mapShape=}. "
			f"Expected {dictionaryPileRangesAuthoritativeData[pile]}, got {dictionaryPileRangesActual[pile]}."
		)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
def test_getLeafDomain(mapShape: tuple[int, ...]) -> None:
	"""Verify getLeafDomain against authoritative leaf domain data for all leaves."""
	state = EliminationState(mapShape=mapShape)
	dictionaryLeafDomainsAuthoritativeData = A001417.dictionaryLeafDomainKnown[mapShape]

	for leaf in range(state.leavesTotal):
		rangeActual = getLeafDomain(state, leaf)
		startAuthoritativeData, stopAuthoritativeData, stepAuthoritativeData = dictionaryLeafDomainsAuthoritativeData[leaf]
		assert rangeActual.start == startAuthoritativeData, (
			f"getLeafDomain: range.start mismatch at {leaf=} for {mapShape=}. "
			f"Expected {startAuthoritativeData}, got {rangeActual.start}."
		)
		assert rangeActual.stop == stopAuthoritativeData, (
			f"getLeafDomain: range.stop mismatch at {leaf=} for {mapShape=}. "
			f"Expected {stopAuthoritativeData}, got {rangeActual.stop}."
		)
		assert rangeActual.step == stepAuthoritativeData, (
			f"getLeafDomain: range.step mismatch at {leaf=} for {mapShape=}. "
			f"Expected {stepAuthoritativeData}, got {rangeActual.step}."
		)

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize("domainFunction,moduleAuthoritativeData", [
	(getDomainDimension一, p2DnDomain3_2_首一_首零一),
	(getDomainDimension二, p2DnDomain6_7_5_4),
	(getDomainDimension首二, p2DnDomain首二_首零二_首零一二_首一二),
	(getDomain二一零and二一, p2DnDomain7_6),
	(getDomain二零and二, p2DnDomain5_4),
	(getDomain首零一二and首一二, p2DnDomain首零一二_首一二),
	(getDomain首零二and首二, p2DnDomain首零二_首二),
], ids=lambda domainFunction: domainFunction.__name__)
def test_getLeafDomainsCombined(domainFunction: Callable[[EliminationState], Sequence[tuple[int, ...]]], moduleAuthoritativeData: ModuleType, dimensionsTotal: int) -> None:
	"""Verify combined domain function against authoritative dataset: completeness, uniqueness, correctness."""
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state = EliminationState(mapShape=mapShape)
	tuplesDomainActual: tuple[tuple[int, ...], ...] = tuple(domainFunction(state))
	tuplesDomainAuthoritativeData: tuple[tuple[int, ...], ...] = getattr(moduleAuthoritativeData, f"listDomain2D{dimensionsTotal}")

	tuplesMissingFromActual, tuplesExtraInActual = unique_to_each(tuplesDomainAuthoritativeData, tuplesDomainActual)
	tuplesMissingFromActual = tuple(tuplesMissingFromActual)
	tuplesExtraInActual = tuple(tuplesExtraInActual)
	hasAllUnique: bool = all_unique(tuplesDomainActual)

	assert hasAllUnique, (
		f"{domainFunction.__name__}: returned duplicate tuples for {mapShape=}."
	)
	assert len(tuplesMissingFromActual) == 0, (
		f"{domainFunction.__name__}: missing {len(tuplesMissingFromActual)} tuples from authoritative data for {mapShape=}. "
		f"Missing samples: {sorted(tuplesMissingFromActual)[:3]}"
	)
	assert len(tuplesExtraInActual) == 0, (
		f"{domainFunction.__name__}: returned {len(tuplesExtraInActual)} tuples not in authoritative data for {mapShape=}. "
		f"Extra samples: {sorted(tuplesExtraInActual)[:3]}"
	)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryPileRangesKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown])
def test_getPileRange(mapShape: tuple[int, ...]) -> None:
	"""Verify getPileRange against authoritative pile range data for all piles."""
	state = EliminationState(mapShape=mapShape)
	dictionaryPileRangesAuthoritativeData: dict[int, tuple[int, ...]] = A001417.dictionaryPileRangesKnown[mapShape]

	for pile in range(state.leavesTotal):
		tupleLeavesPileActual: tuple[int, ...] = tuple(getPileRange(state, pile))
		tupleLeavesPileAuthoritativeData: tuple[int, ...] = dictionaryPileRangesAuthoritativeData[pile]

		assert tupleLeavesPileActual == tupleLeavesPileAuthoritativeData, (
			f"getPileRange: mismatch at {pile=} for {mapShape=}. "
			f"Expected {tupleLeavesPileAuthoritativeData}, got {tupleLeavesPileActual}."
		)

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize(
	"creaseKind,creaseFunction,dictionaryExpectedByMapShape",
	[
		("increase", getLeavesCreaseNext, A001417.dictionaryCreasesIncreaseKnown),
		("decrease", getLeavesCreaseBack, A001417.dictionaryCreasesDecreaseKnown),
	],
	ids=["increase", "decrease"],
)
def test_getLeavesCreaseIterators_match_foldingsTransitions(creaseKind: str, creaseFunction: Callable[[EliminationState, int], Iterable[int]], dictionaryExpectedByMapShape: dict[tuple[int, ...], dict[int, list[int]]], dimensionsTotal: int) -> None:
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state: EliminationState = EliminationState(mapShape=mapShape)
	dictionaryExpectedByLeaf: dict[int, list[int]] = dictionaryExpectedByMapShape[mapShape]

	for leaf in range(state.leavesTotal):
		listLeavesActual: list[int] = list(creaseFunction(state, leaf))
		listLeavesExpectedSorted: list[int] = dictionaryExpectedByLeaf[leaf]

		assert sorted(listLeavesActual) == listLeavesExpectedSorted, (
			f"{creaseFunction.__name__} ({creaseKind}): mismatch for {mapShape=}, {leaf=}. "
			f"Expected(sorted)={listLeavesExpectedSorted}, got(sorted)={sorted(listLeavesActual)}."
		)

		assert all_unique(listLeavesActual), (
			f"{creaseFunction.__name__} ({creaseKind}): duplicates found for {mapShape=}, {leaf=}. "
			f"Actual={listLeavesActual}."
		)

		for leafNext in listLeavesActual:
			assert 0 <= leafNext < state.leavesTotal, (
				f"{creaseFunction.__name__} ({creaseKind}): out-of-range value for {mapShape=}, {leaf=}. "
				f"Got {leafNext}, expected 0 <= leafNext < {state.leavesTotal}."
			)
			bitFlip: int = leaf ^ leafNext
			assert (bitFlip > 0) and ((bitFlip & (bitFlip - 1)) == 0), (
				f"{creaseFunction.__name__} ({creaseKind}): expected one-bit flip for {mapShape=}, {leaf=}. "
				f"Got {leafNext=}, {bitFlip=} (leaf^leafNext)."
			)

		listBitFlips: list[int] = [leaf ^ leafNext for leafNext in listLeavesActual]
		assert listBitFlips == sorted(listBitFlips), (
			f"{creaseFunction.__name__} ({creaseKind}): expected bit flips in increasing dimension order for {mapShape=}, {leaf=}. "
			f"Got bit flips {listBitFlips}."
		)
