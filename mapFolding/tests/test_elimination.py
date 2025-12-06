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
	getDomain二一零and二一, getDomain二零and二, getDomain首零一二and首一二, getDomain首零二and首二, getLeafDomain, getPileRange)
from mapFolding._e.pinning2Dn import (
	pinFirstOrder, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesSecondOrder, pinPilesThirdOrder,
	pinPile二)
from mapFolding.dataBaskets import EliminationState
from mapFolding.tests.dataSamples import (
	A001417, p2DnDomain3_2_首一_首零一, p2DnDomain5_4, p2DnDomain6_7_5_4, p2DnDomain7_6, p2DnDomain首二_首零二_首零一二_首一二,
	p2DnDomain首零一二_首一二, p2DnDomain首零二_首二)
from numpy.typing import NDArray
from types import ModuleType
import numpy
import pytest

def verifyPinnedLeavesAgainstFoldings(state: EliminationState, arrayFoldings: NDArray[numpy.uint8]) -> tuple[int, int, int]:
	"""Verify pinned leaves cover all foldings without overlap.

	Parameters
	----------
	state : EliminationState
		State containing listPinnedLeaves to verify.
	arrayFoldings : NDArray[numpy.uint8]
		Known valid foldings to compare against.

	Returns
	-------
	verificationResults : tuple[int, int, int]
		Tuple of (rowsCovered, rowsTotal, countOverlappingDictionaries).
	"""
	rowsTotal: int = int(arrayFoldings.shape[0])
	listMasks: list[numpy.ndarray] = []

	for pinnedLeaves in state.listPinnedLeaves:
		maskMatches: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
		for indexPile, leaf in pinnedLeaves.items():
			maskMatches = maskMatches & (arrayFoldings[:, indexPile] == leaf)
		listMasks.append(maskMatches)

	masksStacked: numpy.ndarray = numpy.column_stack(listMasks)
	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	indicesOverlappedRows: numpy.ndarray = numpy.nonzero(coverageCountPerRow >= 2)[0]

	countOverlappingDictionaries: int = 0
	if indicesOverlappedRows.size > 0:
		for _indexMask, mask in enumerate(listMasks):
			if bool(mask[indicesOverlappedRows].any()):
				countOverlappingDictionaries += 1

	maskUnion = numpy.logical_or.reduce(listMasks)
	rowsCovered: int = int(maskUnion.sum())

	return rowsCovered, rowsTotal, countOverlappingDictionaries

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

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainKnown])
def test_getLeafDomain_consistencyWithDictionary(mapShape: tuple[int, ...]) -> None:
	"""Verify getLeafDomain results match getDictionaryLeafDomains for all leaves."""
	state = EliminationState(mapShape=mapShape)

	dictionaryFromFunction = getDictionaryLeafDomains(state)

	for leaf in range(state.leavesTotal):
		rangeFromGetLeafDomain = getLeafDomain(state, leaf)
		rangeFromDictionary = dictionaryFromFunction[leaf]

		assert rangeFromGetLeafDomain == rangeFromDictionary, (
			f"getLeafDomain inconsistent with getDictionaryLeafDomains at {leaf=} for {mapShape=}. "
			f"getLeafDomain returned {rangeFromGetLeafDomain}, dictionary has {rangeFromDictionary}."
		)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryPileRangesKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown])
def test_getDictionaryPileRanges_completeComparisonKnown(mapShape: tuple[int, ...]) -> None:
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

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryPileRangesKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown])
def test_getPileRange_consistencyWithDictionary(mapShape: tuple[int, ...]) -> None:
	"""Verify `getPileRange` results match `getDictionaryPileRanges` for all piles."""
	state = EliminationState(mapShape=mapShape)

	dictionaryFromFunction = getDictionaryPileRanges(state)

	for pile in range(state.leavesTotal):
		listFromGetPileRange = list(getPileRange(state, pile))
		listFromDictionary = dictionaryFromFunction[pile]

		assert listFromGetPileRange == listFromDictionary, (
			f"getPileRange inconsistent with getDictionaryPileRanges at {pile=} for {mapShape=}. "
			f"getPileRange returned {listFromGetPileRange}, dictionary has {listFromDictionary}."
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
def test_combinedDomains(domainFunction: Callable[[EliminationState], Sequence[tuple[int, ...]]], moduleExpected: ModuleType, dimensionsTotal: int) -> None:
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

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize("pinningFunction", [pinFirstOrder, pinPilesSecondOrder, pinPilesThirdOrder, pinPile二, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二], ids=lambda pinningFunction: pinningFunction.__name__)
def test_pinningFunctions(loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]], pinningFunction: Callable[[EliminationState], EliminationState], dimensionsTotal: int) -> None:
	mapShape = (2,) * dimensionsTotal
	state = EliminationState(mapShape=mapShape)
	arrayFoldings = loadArrayFoldings(dimensionsTotal)

	state = pinningFunction(state)

	rowsCovered, rowsTotal, countOverlappingDictionaries = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

	assert rowsCovered == rowsTotal, f"{pinningFunction.__name__}, {mapShape = }: {rowsCovered}/{rowsTotal} rows covered."

	assert countOverlappingDictionaries == 0, f"{pinningFunction.__name__}, {mapShape = }: {countOverlappingDictionaries = }"
