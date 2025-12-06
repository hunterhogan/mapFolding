# ruff: noqa: ERA001 T201 T203  # noqa: RUF100
from collections.abc import Sequence
from mapFolding._e import getDictionaryPileRanges, howMany0coordinatesAtTail, pileOrigin, 零
from mapFolding._e._data import getDataFrameFoldings
from mapFolding.dataBaskets import EliminationState
from pprint import pprint
from typing import Any
import numpy
import pandas

def getLeafUnconditionalPrecedence(state: EliminationState) -> pandas.DataFrame:
	"""Identify leaves that always precede other leaves across all folding sequences.

	(AI generated docstring)

	Analyzes all valid folding sequences for a given elimination state to find pairs
	of leaves (Earlier, Later) where the Earlier leaf appears at a smaller column
	index than the Later leaf in every single folding sequence without exception.

	The analysis proceeds as follows.
	1. Load sequence data where each row is a folding and columns represent positions.
	2. Build a positions matrix mapping each leaf value to its column index per row.
	3. Construct a comparison cube testing whether each leaf precedes every other leaf.
	4. Reduce across all rows to find pairs where precedence holds universally.

	Parameters
	----------
	state : EliminationState
		The elimination state containing the map shape and dimension information.

	Returns
	-------
	dataframePrecedence : pandas.DataFrame
		A two-column DataFrame with 'Earlier' and 'Later' indicating leaf values
		where the Earlier leaf unconditionally precedes the Later leaf.

	"""
	dataframeSequences: pandas.DataFrame = getDataFrameFoldings(state)
	arraySequences: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = dataframeSequences.to_numpy(dtype=numpy.int16)

	rowsCount: int
	positionsCount: int
	rowsCount, positionsCount = arraySequences.shape
	valueMaximum: int = int(arraySequences.max())
	positionsMatrix: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = numpy.full((rowsCount, valueMaximum + 1), -1, dtype=numpy.int16)

	rowIndices: numpy.ndarray[Any, numpy.dtype[numpy.int32]] = numpy.arange(rowsCount, dtype=numpy.int32)[:, None]
	columnIndices: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = numpy.broadcast_to(numpy.arange(positionsCount, dtype=numpy.int16), (rowsCount, positionsCount))
	positionsMatrix[rowIndices, arraySequences] = columnIndices

	valuesPresentEveryRow: numpy.ndarray[Any, numpy.dtype[numpy.intp]] = numpy.where((positionsMatrix >= 0).all(axis=0))[0]
	positionsAnalyzed: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsMatrix[:, valuesPresentEveryRow]

	comparisonCube: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = positionsAnalyzed[:, :, None] < positionsAnalyzed[:, None, :]
	alwaysEarlierMatrix: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = comparisonCube.all(axis=0)  # pyright: ignore[reportAssignmentType]
	numpy.fill_diagonal(alwaysEarlierMatrix, val=False)

	indicesEarlier: numpy.ndarray[Any, numpy.dtype[numpy.intp]]
	indicesLater: numpy.ndarray[Any, numpy.dtype[numpy.intp]]
	indicesEarlier, indicesLater = numpy.where(alwaysEarlierMatrix)
	dataframePrecedence: pandas.DataFrame = pandas.DataFrame(
		{
			'Earlier': valuesPresentEveryRow[indicesEarlier],
			'Later': valuesPresentEveryRow[indicesLater],
		}
	).sort_values(['Earlier', 'Later']).reset_index(drop=True)

	return dataframePrecedence

def getLeafConditionalPrecedence(state: EliminationState) -> pandas.DataFrame:
	"""Identify precedence relationships that emerge only when a leaf is at its earliest column.

	(AI generated docstring)

	For each leaf, determines the earliest possible column it can occupy based on
	bit structure properties (`bit_count` and `howMany0coordinatesAtTail`). Then
	finds leaves that always precede it in the subset of foldings where that leaf
	is at its earliest column. Excludes relationships already captured by the
	unconditional precedence analysis.

	The formula for the earliest column of a leaf is.
		columnEarliest = leaf.bit_count() + (2^(howMany0coordinatesAtTail(leaf) + 1) - 2)

	Parameters
	----------
	state : EliminationState
		The elimination state containing the map shape and dimension information.
	columnsToExclude : list[int] | None = None
		Column indices (as integers) to exclude from analysis. Pass [0, 1, leavesTotal-1]
		to exclude the trivially-pinned positions.

	Returns
	-------
	dataframeConditionalPrecedence : pandas.DataFrame
		A three-column DataFrame with 'Earlier', 'Later', and 'AtColumn' indicating
		that when 'Later' is at column 'AtColumn', 'Earlier' always precedes it.
		Only includes relationships not already present in unconditional precedence.

	"""
	dataframeSequences: pandas.DataFrame = getDataFrameFoldings(state)
	columnsToExclude: list[int] | None = [pileOrigin, 零, state.pileLast]
	if columnsToExclude is not None: # pyright: ignore[reportUnnecessaryComparison]
		dataframeSequences = dataframeSequences.drop(columns=columnsToExclude)
	arraySequences: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = dataframeSequences.to_numpy(dtype=numpy.int16)

	rowsCount: int
	positionsCount: int
	rowsCount, positionsCount = arraySequences.shape
	valueMaximum: int = int(arraySequences.max())
	positionsMatrix: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = numpy.full((rowsCount, valueMaximum + 1), -1, dtype=numpy.int16)

	rowIndices: numpy.ndarray[Any, numpy.dtype[numpy.int32]] = numpy.arange(rowsCount, dtype=numpy.int32)[:, None]
	columnIndices: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = numpy.broadcast_to(numpy.arange(positionsCount, dtype=numpy.int16), (rowsCount, positionsCount))
	positionsMatrix[rowIndices, arraySequences] = columnIndices

	columnOffset: int = 2 if columnsToExclude is not None and 0 in columnsToExclude and 1 in columnsToExclude else 0 # pyright: ignore[reportUnnecessaryComparison]

	dataframeUnconditional: pandas.DataFrame = getLeafUnconditionalPrecedence(state)
	setUnconditional: set[tuple[Any, Any]] = set(zip(dataframeUnconditional['Earlier'], dataframeUnconditional['Later'], strict=True))

	listConditionalRelationships: list[dict[str, int]] = []

	for leafLater in range(state.leavesTotal):
		columnEarliestOriginal: int = leafLater.bit_count() + (2 ** (howMany0coordinatesAtTail(leafLater) + 1) - 2)
		columnEarliestIndex: int = columnEarliestOriginal - columnOffset

		if columnEarliestIndex < 0:
			continue

		maskRowsAtEarliestColumn: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = (positionsMatrix[:, leafLater] == columnEarliestIndex)

		if not numpy.any(maskRowsAtEarliestColumn):
			continue

		positionsSubset: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsMatrix[maskRowsAtEarliestColumn]

		for leafEarlier in range(state.leavesTotal):
			if leafEarlier == leafLater:
				continue

			positionsOfEarlier: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsSubset[:, leafEarlier]

			isEarlierAlwaysPresentAndPrecedes: bool = bool(numpy.all((positionsOfEarlier >= 0) & (positionsOfEarlier < columnEarliestIndex)))
			if isEarlierAlwaysPresentAndPrecedes and (leafEarlier, leafLater) not in setUnconditional:
				listConditionalRelationships.append({
					'Earlier': leafEarlier,
					'Later': leafLater,
					'AtColumn': columnEarliestOriginal
				})

	dataframeConditionalPrecedence: pandas.DataFrame = pandas.DataFrame(listConditionalRelationships).sort_values(['Later', 'Earlier']).reset_index(drop=True)

	return dataframeConditionalPrecedence

def _getGroupedBy(state: EliminationState, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	dataframeFoldings: pandas.DataFrame = getDataFrameFoldings(state)
	groupedBy: dict[int | tuple[int, ...], list[int]] = dataframeFoldings.groupby(list(groupByLeavesAtPiles))[pileTarget].apply(list).to_dict()
	return {leaves: sorted(set(listLeaves)) for leaves, listLeaves in groupedBy.items()}

def getExcludedLeaves(state: EliminationState, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	return {leaves: sorted(set(getDictionaryPileRanges(state)[pileTarget]).difference(set(listLeaves))) for leaves, listLeaves in _getGroupedBy(state, pileTarget, groupByLeavesAtPiles).items()}

def verifyDomainAgainstKnown(domainComputed: Sequence[tuple[int, ...]], domainKnown: Sequence[tuple[int, ...]], *, printResults: bool = True) -> dict[str, list[tuple[int, ...]]]:
	"""Compare a computed domain against known verification data.

	Parameters
	----------
	domainComputed : Sequence[tuple[int, ...]]
		The domain generated by the function under development.
	domainKnown : Sequence[tuple[int, ...]]
		The empirically extracted domain from verification data (e.g., from `makeVerificationDataLeavesDomain`).
	printResults : bool = True
		Whether to print the comparison results using pprint.

	Returns
	-------
	comparisonResults : dict[str, list[tuple[int, ...]]]
		Dictionary with keys:
		- 'missing': tuples in domainKnown but not in domainComputed (the function fails to generate these)
		- 'surplus': tuples in domainComputed but not in domainKnown (the function generates extra invalid tuples)
		- 'matched': tuples present in both domains

	"""
	setComputed: set[tuple[int, ...]] = set(domainComputed)
	setKnown: set[tuple[int, ...]] = set(domainKnown)

	listMissing: list[tuple[int, ...]] = sorted(setKnown - setComputed)
	listSurplus: list[tuple[int, ...]] = sorted(setComputed - setKnown)
	listMatched: list[tuple[int, ...]] = sorted(setComputed & setKnown)

	comparisonResults: dict[str, list[tuple[int, ...]]] = {
		'missing': listMissing,
		'surplus': listSurplus,
		'matched': listMatched,
	}

	if printResults:
		countComputed: int = len(setComputed)
		countKnown: int = len(setKnown)
		countMissing: int = len(listMissing)
		countSurplus: int = len(listSurplus)
		countMatched: int = len(listMatched)

		print(f"Domain comparison: {countComputed} computed vs {countKnown} known")
		print(f"  Matched: {countMatched} ({100 * countMatched / countKnown:.1f}% of known)")
		# pprint(listMatched, width=140, compact=True)

		if listMissing:
			print(f"  Missing ({countMissing} tuples in known but not in computed):")
			pprint(listMissing, width=140, compact=True)

		if listSurplus:
			print(f"  Surplus ({countSurplus} tuples in computed but not in known):")
			pprint(listSurplus, width=140, compact=True)

		if not listMissing and not listSurplus:
			print("  Perfect match!")

	return comparisonResults

