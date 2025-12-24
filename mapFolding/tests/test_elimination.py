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

from collections.abc import Callable, Sequence
from mapFolding._e import (
	getDictionaryLeafDomains, getDictionaryPileRanges, getDomainDimension一, getDomainDimension二, getDomainDimension首二,
	getDomain二一零and二一, getDomain二零and二, getDomain首零一二and首一二, getDomain首零二and首二, getLeafDomain, getLeavesCreaseBack,
	getLeavesCreaseNext, getPileRange)
from mapFolding._e.pinning2Dn import (
	pileProcessingOrderDefault, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPiles)
from mapFolding.dataBaskets import EliminationState
from mapFolding.tests.dataSamples import (
	A001417, p2DnDomain3_2_首一_首零一, p2DnDomain5_4, p2DnDomain6_7_5_4, p2DnDomain7_6, p2DnDomain首二_首零二_首零一二_首一二,
	p2DnDomain首零一二_首一二, p2DnDomain首零二_首二)
from numpy.typing import NDArray
from types import ModuleType
import numpy
import pytest

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
def test_getDictionaryLeafDomains(mapShape: tuple[int, ...]) -> None:
	"""Verify getDictionaryLeafDomains returns correct range(start, stop, step) for all leaves."""
	state = EliminationState(mapShape=mapShape)
	dictionaryExpected = A001417.dictionaryLeafDomainKnown[mapShape]

	dictionaryActual = getDictionaryLeafDomains(state)

	assert len(dictionaryActual) == state.leavesTotal, (
		f"getDictionaryLeafDomains: dictionary length mismatch for {mapShape=}. "
		f"Expected {state.leavesTotal}, got {len(dictionaryActual)}."
	)

	for leaf in range(state.leavesTotal):
		rangeActual = dictionaryActual[leaf]
		startExpected, stopExpected, stepExpected = dictionaryExpected[leaf]
		assert rangeActual.start == startExpected, (
			f"getDictionaryLeafDomains: range.start mismatch at {leaf=} for {mapShape=}. "
			f"Expected start={startExpected}, got {rangeActual.start}."
		)
		assert rangeActual.stop == stopExpected, (
			f"getDictionaryLeafDomains: range.stop mismatch at {leaf=} for {mapShape=}. "
			f"Expected stop={stopExpected}, got {rangeActual.stop}."
		)
		assert rangeActual.step == stepExpected, (
			f"getDictionaryLeafDomains: range.step mismatch at {leaf=} for {mapShape=}. "
			f"Expected step={stepExpected}, got {rangeActual.step}."
		)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryPileRangesKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown])
def test_getDictionaryPileRanges(mapShape: tuple[int, ...]) -> None:
	state = EliminationState(mapShape=mapShape)
	dictionaryExpected = A001417.dictionaryPileRangesKnown[mapShape]

	dictionaryActual = getDictionaryPileRanges(state)

	assert len(dictionaryActual) == state.leavesTotal, (
		f"getDictionaryPileRanges: dictionary length mismatch for {mapShape=}. "
		f"Expected {state.leavesTotal}, got {len(dictionaryActual)}."
	)

	for pile in range(state.leavesTotal):
		assert dictionaryActual[pile] == dictionaryExpected[pile], (
			f"getDictionaryPileRanges: mismatch at {pile=} for {mapShape=}. "
			f"Expected {dictionaryExpected[pile]}, got {dictionaryActual[pile]}."
		)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
def test_getLeafDomain(mapShape: tuple[int, ...]) -> None:
	"""Verify getLeafDomain results match getDictionaryLeafDomains for all leaves."""
	state = EliminationState(mapShape=mapShape)
	dictionaryExpected = A001417.dictionaryLeafDomainKnown[mapShape]

	for leaf in range(state.leavesTotal):
		rangeActual = getLeafDomain(state, leaf)
		startExpected, stopExpected, stepExpected = dictionaryExpected[leaf]
		assert rangeActual.start == startExpected, (
			f"getLeafDomain: range.start mismatch at {leaf=} for {mapShape=}. "
			f"Expected start={startExpected}, got {rangeActual.start}."
		)
		assert rangeActual.stop == stopExpected, (
			f"getLeafDomain: range.stop mismatch at {leaf=} for {mapShape=}. "
			f"Expected stop={stopExpected}, got {rangeActual.stop}."
		)
		assert rangeActual.step == stepExpected, (
			f"getLeafDomain: range.step mismatch at {leaf=} for {mapShape=}. "
			f"Expected step={stepExpected}, got {rangeActual.step}."
		)

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize("domainFunction,moduleExpected", [
	(getDomainDimension一, p2DnDomain3_2_首一_首零一),
	(getDomainDimension二, p2DnDomain6_7_5_4),
	(getDomainDimension首二, p2DnDomain首二_首零二_首零一二_首一二),
	(getDomain二一零and二一, p2DnDomain7_6),
	(getDomain二零and二, p2DnDomain5_4),
	(getDomain首零一二and首一二, p2DnDomain首零一二_首一二),
	(getDomain首零二and首二, p2DnDomain首零二_首二),
], ids=lambda domainFunction: domainFunction.__name__)
def test_getLeafDomainsCombined(domainFunction: Callable[[EliminationState], Sequence[tuple[int, ...]]], moduleExpected: ModuleType, dimensionsTotal: int) -> None:
	"""Compare combined domain functions against empirical verification datasets."""
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state = EliminationState(mapShape=mapShape)
	domainActual: tuple[tuple[int, ...], ...] = tuple(domainFunction(state))
	attributeName: str = f"listDomain2D{dimensionsTotal}"
	domainExpected: Sequence[tuple[int, ...]] = getattr(moduleExpected, attributeName)
	setActual: set[tuple[int, ...]] = set(domainActual)
	setExpected: set[tuple[int, ...]] = set(domainExpected)
	missingTuplesFull: list[tuple[int, ...]] = sorted(setExpected.difference(setActual))
	if missingTuplesFull:
		pytest.fail(
			f"{domainFunction.__name__}: combined domain missing {len(missingTuplesFull)} tuples for mapShape={mapShape}. "
			f"Missing samples: {missingTuplesFull[:3]}"
		)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryPileRangesKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown])
def test_getPileRange(mapShape: tuple[int, ...]) -> None:
	"""Verify `getPileRange` results match `getDictionaryPileRanges` for all piles."""
	state = EliminationState(mapShape=mapShape)
	dictionaryExpected = A001417.dictionaryPileRangesKnown[mapShape]

	for pile in range(state.leavesTotal):
		listFromGetPileRange = list(getPileRange(state, pile))
		listFromDictionary = dictionaryExpected[pile]

		assert listFromGetPileRange == listFromDictionary, (
			f"getPileRange inconsistent with getDictionaryPileRanges at {pile=} for {mapShape=}. "
			f"getPileRange returned {listFromGetPileRange}, dictionary has {listFromDictionary}."
		)

@pytest.mark.parametrize("mapShape", [
	(2, 2, 2, 2),
	(2, 2, 2, 2, 2),
	(2, 2, 2, 2, 2, 2),
], ids=["2d4", "2d5", "2d6"])
def test_pileProcessingOrderDefault_all_piles_included(mapShape: tuple[int, ...]) -> None:
	state = EliminationState(mapShape=mapShape)
	pileOrder = pileProcessingOrderDefault(state)
	assert set(pileOrder) >= set(range(state.leavesTotal)), (
		f"pileProcessingOrderDefault: Not all piles in range(state.leavesTotal) are present for {mapShape=}. "
		f"Missing: {set(range(state.leavesTotal)) - set(pileOrder)}"
	)

@pytest.mark.parametrize("mapShape", [
	(2, 2, 2, 2),
	(2, 2, 2, 2, 2),
	(2, 2, 2, 2, 2, 2),
], ids=["2d4", "2d5", "2d6"])
def test_pileProcessingOrderDefault_no_duplicates(mapShape: tuple[int, ...]) -> None:
	state = EliminationState(mapShape=mapShape)
	pileOrder = pileProcessingOrderDefault(state)

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
	seen: set[int] = set()
	for pile in pileOrder:
		assert pile not in seen, (
			f"pileProcessingOrderDefault: Duplicate pile value {pile} for {mapShape=}."
		)
		seen.add(pile)

@pytest.mark.parametrize("mapShape", [
	(2, 2, 2, 2),
	(2, 2, 2, 2, 2),
	(2, 2, 2, 2, 2, 2),
], ids=["2d4", "2d5", "2d6"])
def test_pileProcessingOrderDefault_no_out_of_range(mapShape: tuple[int, ...]) -> None:
	state = EliminationState(mapShape=mapShape)
	pileOrder = pileProcessingOrderDefault(state)
	for pile in pileOrder:
		assert 0 <= pile < state.leavesTotal, (
			f"pileProcessingOrderDefault: Out-of-range pile value {pile} for {mapShape=}. "
			f"Valid range: 0 <= pile < {state.leavesTotal}"
		)

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize("pinningFunction", [pinPiles, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二], ids=lambda pinningFunction: pinningFunction.__name__)
def test_pinningFunctions(loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]], pinningFunction: Callable[[EliminationState], EliminationState], verifyLeavesPinnedAgainstFoldings: Callable[[EliminationState, NDArray[numpy.uint8]], tuple[int, int, int]], dimensionsTotal: int) -> None:
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state = EliminationState(mapShape=mapShape)
	arrayFoldings = loadArrayFoldings(dimensionsTotal)

	state: EliminationState = pinningFunction(state)

	rowsCovered, rowsTotal, countOverlappingDictionaries = verifyLeavesPinnedAgainstFoldings(state, arrayFoldings)

	assert rowsCovered == rowsTotal, f"{pinningFunction.__name__}, {mapShape = }: {rowsCovered}/{rowsTotal} rows covered."
	assert countOverlappingDictionaries == 0, f"{pinningFunction.__name__}, {mapShape = }: {countOverlappingDictionaries = }"

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize(
	"creaseKind,creaseFunction,dictionaryExpectedByMapShape",
	[
		("increase", getLeavesCreaseNext, A001417.dictionaryCreasesIncreaseKnown),
		("decrease", getLeavesCreaseBack, A001417.dictionaryCreasesDecreaseKnown),
	],
	ids=["increase", "decrease"],
)
def test_getListLeavesCreases_matches_foldingsTransitions(
	creaseKind: str,
	creaseFunction: Callable[[EliminationState, int], list[int]],
	dictionaryExpectedByMapShape: dict[tuple[int, ...], dict[int, list[int]]],
	dimensionsTotal: int,
) -> None:
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state: EliminationState = EliminationState(mapShape=mapShape)
	dictionaryExpectedByLeaf: dict[int, list[int]] = dictionaryExpectedByMapShape[mapShape]

	for leaf in range(state.leavesTotal):
		listLeavesActual: list[int] = creaseFunction(state, leaf)
		listLeavesExpectedSorted: list[int] = dictionaryExpectedByLeaf[leaf]

		assert sorted(listLeavesActual) == listLeavesExpectedSorted, (
			f"{creaseFunction.__name__} ({creaseKind}): mismatch for {mapShape=}, {leaf=}. "
			f"Expected(sorted)={listLeavesExpectedSorted}, got(sorted)={sorted(listLeavesActual)}."
		)

		assert len(listLeavesActual) == len(set(listLeavesActual)), (
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
