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

from collections.abc import Callable, Iterable
from mapFolding import packageSettings
from mapFolding._e import (
	getDictionaryAddends4Next, getDictionaryAddends4Prior, getDictionaryLeafDomains, getDictionaryPileRanges,
	getDomain二combined, getLeafDomain, getPileRange)
from mapFolding._e.pinning2Dn import pinByFormula, secondOrderLeavesV2, secondOrderPilings
from mapFolding.dataBaskets import EliminationState
from mapFolding.tests.dataSamples import A001417, p2DnDomain6_7_5_4
from pathlib import Path
from typing import TYPE_CHECKING
import numpy
import pickle
import pytest

if TYPE_CHECKING:
	from numpy.typing import NDArray

class TestGetDictionaryAddends4Next:
	"""Tests for `getDictionaryAddends4Next` function."""

	@pytest.mark.parametrize(
		"mapShape",
		list(A001417.dictionaryAddends4NextKnown),
		ids=[f"mapShape={shape}" for shape in A001417.dictionaryAddends4NextKnown],
	)
	def test_getDictionaryAddends4Next_completeComparisonKnown(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
	) -> None:
		state = makeEliminationState(mapShape)
		dictionaryExpected = A001417.dictionaryAddends4NextKnown[mapShape]

		dictionaryActual = getDictionaryAddends4Next(state)

		assert len(dictionaryActual) == state.leavesTotal, (
			f"getDictionaryAddends4Next: dictionary length mismatch for {mapShape=}. "
			f"Expected {state.leavesTotal}, got {len(dictionaryActual)}."
		)

		for leaf in range(state.leavesTotal):
			assert dictionaryActual[leaf] == dictionaryExpected[leaf], (
				f"getDictionaryAddends4Next: mismatch at {leaf=} for {mapShape=}. "
				f"Expected {dictionaryExpected[leaf]}, got {dictionaryActual[leaf]}."
			)

class TestGetDictionaryAddends4Prior:
	"""Tests for `getDictionaryAddends4Prior` function."""

	@pytest.mark.parametrize(
		"mapShape",
		list(A001417.dictionaryAddends4PriorKnown),
		ids=[f"mapShape={shape}" for shape in A001417.dictionaryAddends4PriorKnown],
	)
	def test_getDictionaryAddends4Prior_completeComparisonKnown(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
	) -> None:
		state = makeEliminationState(mapShape)
		dictionaryExpected = A001417.dictionaryAddends4PriorKnown[mapShape]

		dictionaryActual = getDictionaryAddends4Prior(state)

		assert len(dictionaryActual) == state.leavesTotal, (
			f"getDictionaryAddends4Prior: dictionary length mismatch for {mapShape=}. "
			f"Expected {state.leavesTotal}, got {len(dictionaryActual)}."
		)

		for leaf in range(state.leavesTotal):
			assert dictionaryActual[leaf] == dictionaryExpected[leaf], (
				f"getDictionaryAddends4Prior: mismatch at {leaf=} for {mapShape=}. "
				f"Expected {dictionaryExpected[leaf]}, got {dictionaryActual[leaf]}."
			)

class TestGetDictionaryLeafDomains:
	"""Tests for `getDictionaryLeafDomains` function."""

	@pytest.mark.parametrize(
		"mapShape",
		list(A001417.dictionaryLeafDomainStartKnown),
		ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainStartKnown],
	)
	def test_getDictionaryLeafDomains_startValuesMatch(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
	) -> None:
		state = makeEliminationState(mapShape)
		dictionaryStartExpected = A001417.dictionaryLeafDomainStartKnown[mapShape]

		dictionaryActual = getDictionaryLeafDomains(state)

		assert len(dictionaryActual) == state.leavesTotal, (
			f"getDictionaryLeafDomains: dictionary length mismatch for {mapShape=}. "
			f"Expected {state.leavesTotal}, got {len(dictionaryActual)}."
		)

		for leaf in range(state.leavesTotal):
			rangeActual = dictionaryActual[leaf]
			startExpected = dictionaryStartExpected[leaf]
			assert rangeActual.start == startExpected, (
				f"getDictionaryLeafDomains: range.start mismatch at {leaf=} for {mapShape=}. "
				f"Expected start={startExpected}, got {rangeActual.start}."
			)

	@pytest.mark.parametrize(
		"mapShape",
		list(A001417.dictionaryLeafDomainStopKnown),
		ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainStopKnown],
	)
	def test_getDictionaryLeafDomains_stopValuesMatch(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
	) -> None:
		state = makeEliminationState(mapShape)
		dictionaryStopExpected = A001417.dictionaryLeafDomainStopKnown[mapShape]

		dictionaryActual = getDictionaryLeafDomains(state)

		for leaf in range(state.leavesTotal):
			rangeActual = dictionaryActual[leaf]
			stopExpected = dictionaryStopExpected[leaf]
			assert rangeActual.stop >= stopExpected, (
				f"getDictionaryLeafDomains: range.stop too small at {leaf=} for {mapShape=}. "
				f"Expected stop>={stopExpected}, got {rangeActual.stop}."
			)
			assert rangeActual.stop <= stopExpected + 1, (
				f"getDictionaryLeafDomains: range.stop too large at {leaf=} for {mapShape=}. "
				f"Expected stop<={stopExpected + 1}, got {rangeActual.stop}."
			)

class TestGetLeafDomain:
	"""Tests for `getLeafDomain` function."""

	@pytest.mark.parametrize(
		"mapShape,leaf",
		[
			(mapShape, leaf)
			for mapShape in A001417.dictionaryLeafDomainStartKnown
			for leaf in A001417.leafIndicesSample2d6[:5] if leaf < 2**len(mapShape)
		],
		ids=lambda parameterValue: str(parameterValue),
	)
	def test_getLeafDomain_spotCheckStartValues(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
		leaf: int,
	) -> None:
		state = makeEliminationState(mapShape)
		startExpected = A001417.dictionaryLeafDomainStartKnown[mapShape][leaf]

		rangeActual = getLeafDomain(state, leaf)

		assert rangeActual.start == startExpected, (
			f"getLeafDomain: start mismatch for {leaf=} with {mapShape=}. "
			f"Expected {startExpected}, got {rangeActual.start}."
		)

class TestGetDictionaryPileRanges:
	"""Tests for `getDictionaryPileRanges` function."""

	@pytest.mark.parametrize(
		"mapShape",
		list(A001417.dictionaryPileRangesKnown),
		ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown],
	)
	def test_getDictionaryPileRanges_completeComparisonKnown(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
	) -> None:
		state = makeEliminationState(mapShape)
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

class TestGetPileRange:
	"""Tests for `getPileRange` function."""

	@pytest.mark.parametrize(
		"mapShape,pile",
		[
			(mapShape, pile)
			for mapShape in A001417.dictionaryPileRangesKnown
			for pile in A001417.pileIndicesSample2d4 if pile < 2**len(mapShape)
		],
		ids=lambda parameterValue: str(parameterValue),
	)
	def test_getPileRange_spotCheckValues(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
		pile: int,
	) -> None:
		state = makeEliminationState(mapShape)
		listExpected = A001417.dictionaryPileRangesKnown[mapShape][pile]

		iterableActual: Iterable[int] = getPileRange(state, pile)
		listActual = list(iterableActual)

		assert listActual == listExpected, (
			f"getPileRange: mismatch for {pile=} with {mapShape=}. "
			f"Expected {listExpected}, got {listActual}."
		)

	@pytest.mark.parametrize(
		"mapShape",
		list(A001417.dictionaryPileRangesKnown),
		ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown],
	)
	def test_getPileRange_consistencyWithDictionary(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
	) -> None:
		"""Verify `getPileRange` results match `getDictionaryPileRanges` for all piles."""
		state = makeEliminationState(mapShape)

		dictionaryFromFunction = getDictionaryPileRanges(state)

		for pile in range(state.leavesTotal):
			listFromGetPileRange = list(getPileRange(state, pile))
			listFromDictionary = dictionaryFromFunction[pile]

			assert listFromGetPileRange == listFromDictionary, (
				f"getPileRange inconsistent with getDictionaryPileRanges at {pile=} for {mapShape=}. "
				f"getPileRange returned {listFromGetPileRange}, dictionary has {listFromDictionary}."
			)

# Path to pickled test data
pathDataSamples: Path = Path(f'{packageSettings.pathPackage}/tests/dataSamples')

@pytest.fixture
def loadArrayFoldings() -> Callable[[int], "NDArray[numpy.uint8]"]:
	"""Factory fixture for loading pickled array foldings data.

	Returns
	-------
	loaderFunction : Callable[[int], NDArray[numpy.uint8]]
		Function that loads arrayFoldings for a given dimensionsTotal.
	"""
	def loader(dimensionsTotal: int) -> "NDArray[numpy.uint8]":
		pathFilename = pathDataSamples / f"arrayFoldingsP2d{dimensionsTotal}.pkl"
		arrayFoldings: NDArray[numpy.uint8] = pickle.loads(pathFilename.read_bytes())  # noqa: S301
		return arrayFoldings

	return loader

def verifyPinnedLeavesAgainstFoldings(
	state: EliminationState,
	arrayFoldings: "NDArray[numpy.uint8]",
) -> tuple[int, int, int]:
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

class TestPinByFormula:
	"""Tests for `pinByFormula` function coverage and correctness."""

	@pytest.mark.parametrize(
		"dimensionsTotal",
		[5, 6],
		ids=["2d5_32leaves", "2d6_64leaves"],
	)
	def test_pinByFormula_fullCoverage(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		loadArrayFoldings: Callable[[int], "NDArray[numpy.uint8]"],
		dimensionsTotal: int,
	) -> None:
		mapShape = (2,) * dimensionsTotal
		state = makeEliminationState(mapShape)
		arrayFoldings = loadArrayFoldings(dimensionsTotal)

		state = pinByFormula(state)

		rowsCovered, rowsTotal, _countOverlapping = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

		assert rowsCovered == rowsTotal, (
			f"pinByFormula: incomplete coverage for {mapShape=}. "
			f"Covered {rowsCovered}/{rowsTotal} rows ({100 * rowsCovered / rowsTotal:.2f}%)."
		)

	@pytest.mark.parametrize(
		"dimensionsTotal",
		[5, 6],
		ids=["2d5_32leaves", "2d6_64leaves"],
	)
	def test_pinByFormula_noOverlap(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		loadArrayFoldings: Callable[[int], "NDArray[numpy.uint8]"],
		dimensionsTotal: int,
	) -> None:
		mapShape = (2,) * dimensionsTotal
		state = makeEliminationState(mapShape)
		arrayFoldings = loadArrayFoldings(dimensionsTotal)

		state = pinByFormula(state)

		_rowsCovered, _rowsTotal, countOverlappingDictionaries = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

		assert countOverlappingDictionaries == 0, (
			f"pinByFormula: found {countOverlappingDictionaries} overlapping dictionaries for {mapShape=}. "
			f"Expected 0 overlaps."
		)

class TestSecondOrderLeavesV2:
	"""Tests for `secondOrderLeavesV2` function coverage and correctness."""

	@pytest.mark.parametrize(
		"dimensionsTotal",
		[4, 5, 6],
		ids=["2d4_16leaves", "2d5_32leaves", "2d6_64leaves"],
	)
	def test_secondOrderLeavesV2_fullCoverage(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		loadArrayFoldings: Callable[[int], "NDArray[numpy.uint8]"],
		dimensionsTotal: int,
	) -> None:
		mapShape = (2,) * dimensionsTotal
		state = makeEliminationState(mapShape)
		arrayFoldings = loadArrayFoldings(dimensionsTotal)

		state = secondOrderLeavesV2(state)

		rowsCovered, rowsTotal, _countOverlapping = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

		assert rowsCovered == rowsTotal, (
			f"secondOrderLeavesV2: incomplete coverage for {mapShape=}. "
			f"Covered {rowsCovered}/{rowsTotal} rows ({100 * rowsCovered / rowsTotal:.2f}%)."
		)

	@pytest.mark.parametrize(
		"dimensionsTotal",
		[4, 5, 6],
		ids=["2d4_16leaves", "2d5_32leaves", "2d6_64leaves"],
	)
	def test_secondOrderLeavesV2_noOverlap(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		loadArrayFoldings: Callable[[int], "NDArray[numpy.uint8]"],
		dimensionsTotal: int,
	) -> None:
		mapShape = (2,) * dimensionsTotal
		state = makeEliminationState(mapShape)
		arrayFoldings = loadArrayFoldings(dimensionsTotal)

		state = secondOrderLeavesV2(state)

		_rowsCovered, _rowsTotal, countOverlappingDictionaries = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

		assert countOverlappingDictionaries == 0, (
			f"secondOrderLeavesV2: found {countOverlappingDictionaries} overlapping dictionaries for {mapShape=}. "
			f"Expected 0 overlaps."
		)

class TestSecondOrderPilings:
	"""Tests for `secondOrderPilings` function coverage and correctness."""

	@pytest.mark.parametrize(
		"dimensionsTotal",
		[4, 5, 6],
		ids=["2d4_16leaves", "2d5_32leaves", "2d6_64leaves"],
	)
	def test_secondOrderPilings_fullCoverage(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		loadArrayFoldings: Callable[[int], "NDArray[numpy.uint8]"],
		dimensionsTotal: int,
	) -> None:
		mapShape = (2,) * dimensionsTotal
		state = makeEliminationState(mapShape)
		arrayFoldings = loadArrayFoldings(dimensionsTotal)

		state = secondOrderPilings(state)

		rowsCovered, rowsTotal, _countOverlapping = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

		assert rowsCovered == rowsTotal, (
			f"secondOrderPilings: incomplete coverage for {mapShape=}. "
			f"Covered {rowsCovered}/{rowsTotal} rows ({100 * rowsCovered / rowsTotal:.2f}%)."
		)

	@pytest.mark.parametrize(
		"dimensionsTotal",
		[4, 5, 6],
		ids=["2d4_16leaves", "2d5_32leaves", "2d6_64leaves"],
	)
	def test_secondOrderPilings_noOverlap(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		loadArrayFoldings: Callable[[int], "NDArray[numpy.uint8]"],
		dimensionsTotal: int,
	) -> None:
		mapShape = (2,) * dimensionsTotal
		state = makeEliminationState(mapShape)
		arrayFoldings = loadArrayFoldings(dimensionsTotal)

		state = secondOrderPilings(state)

		_rowsCovered, _rowsTotal, countOverlappingDictionaries = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

		assert countOverlappingDictionaries == 0, (
			f"secondOrderPilings: found {countOverlappingDictionaries} overlapping dictionaries for {mapShape=}. "
			f"Expected 0 overlaps."
		)


class TestCombinedDomains:
	"""Tests for combined domain functions.

	These tests verify that domain functions return domains containing all known valid
	pile position tuples. The domains may be over-inclusive (contain extra tuples beyond
	what is strictly necessary), which is acceptable behavior.

	Test data is stored in `tests/dataSamples/p2DnDomain*.py` files. Each file contains
	a `dictionaryCombinedDomainKnown` mapping mapShape tuples to lists of valid pile
	position tuples.

	To add tests for a new domain function:
	1. Create a data file `tests/dataSamples/p2DnDomain<leaves>.py` with the known data
	2. Import the data module in this test file
	3. Add a test method following the pattern below, referencing the new data module
	"""

	@pytest.mark.parametrize(
		"mapShape",
		list(p2DnDomain6_7_5_4.dictionaryCombinedDomainKnown),
		ids=[f"mapShape={shape}" for shape in p2DnDomain6_7_5_4.dictionaryCombinedDomainKnown],
	)
	def test_getDomain二combined_containsAllKnown(
		self,
		makeEliminationState: Callable[[tuple[int, ...]], EliminationState],
		mapShape: tuple[int, ...],
	) -> None:
		"""Verify getDomain二combined contains all known valid pile position tuples.

		The function may return additional tuples (over-inclusive), which is acceptable.
		This test only verifies that all empirically known valid tuples are present.
		"""
		state = makeEliminationState(mapShape)
		listTuplesKnown = p2DnDomain6_7_5_4.dictionaryCombinedDomainKnown[mapShape]

		domainActual = getDomain二combined(state)
		setDomainActual = set(domainActual)

		listTuplesMissing = [tupleKnown for tupleKnown in listTuplesKnown if tupleKnown not in setDomainActual]

		assert len(listTuplesMissing) == 0, (
			f"getDomain二combined: missing {len(listTuplesMissing)} known tuples for {mapShape=}. "
			f"Missing: {listTuplesMissing[:5]}{'...' if len(listTuplesMissing) > 5 else ''}."
		)

