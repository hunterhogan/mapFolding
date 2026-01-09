# ruff: noqa: ERA001 T201 T203  # noqa: RUF100
# NOTE to AI assistants: this module is not representative of my coding style. Most of it is AI generated, but because it's temporary code, I didn't strictly enforce my usual standards. Do not emulate it.
from gmpy2 import bit_mask
from hunterMakesPy import raiseIfNone
from mapFolding import ansiColorBlackOnCyan, ansiColorReset
from mapFolding._e import (
	dimensionNearestTail, dimensionNearest首, getDictionaryConditionalLeafPredecessors, getLeafDomain, getZ0Z_successor,
	howManyDimensionsHaveOddParity, pileOrigin, 零)
from mapFolding._e._dataDynamic import getDataFrameFoldings
from mapFolding._e.dataBaskets import EliminationState
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
	dataframeSequences: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
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
	dataframeSequences: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
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
		columnEarliestOriginal: int = leafLater.bit_count() + (2 ** (dimensionNearestTail(leafLater) + 1) - 2)
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

def getLeafConditionalPrecedenceAtLastPileOfLeafDomain(state: EliminationState) -> pandas.DataFrame:
	"""Identify precedence relationships that emerge only when a leaf is at the last pile in its domain.

	(AI generated docstring)

	For each leaf, determines the last pile it can occupy within its mathematical
	domain, then finds leaves that always precede it in the subset of foldings
	where that leaf is observed at that last-in-domain pile. Excludes relationships
	already captured by the unconditional precedence analysis.

	The formula for the last pile *in* the domain of a leaf is.
		pileLastOfLeaf = int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf))) - howManyDimensionsHaveOddParity(leaf) + 1

	Parameters
	----------
	state : EliminationState
		The elimination state containing the map shape and dimension information.
	columnsToExclude : list[int] | None = None
		Column indices (as integers) to exclude from analysis. Pass [0, 1, leavesTotal-1]
		to exclude the trivially-pinned positions.

	Returns
	-------
	dataframeConditionalPrecedenceAtLastPile : pandas.DataFrame
		A three-column DataFrame with 'Earlier', 'Later', and 'AtColumn' indicating
		that when 'Later' is at column 'AtColumn' (its last-in-domain pile),
		'Earlier' always precedes it. Only includes relationships not already
		present in unconditional precedence.

	"""
	dataframeSequences: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
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
		pileLastOfLeafOriginal: int = int(bit_mask(state.dimensionsTotal) ^ bit_mask(state.dimensionsTotal - dimensionNearest首(leafLater))) - howManyDimensionsHaveOddParity(leafLater) + 1
		pileLastOfLeafIndex: int = pileLastOfLeafOriginal - columnOffset

		if pileLastOfLeafIndex < 0:
			continue

		maskRowsAtLastPileOfLeaf: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = (positionsMatrix[:, leafLater] == pileLastOfLeafIndex)

		if not numpy.any(maskRowsAtLastPileOfLeaf):
			continue

		positionsSubset: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsMatrix[maskRowsAtLastPileOfLeaf]

		for leafEarlier in range(state.leavesTotal):
			if leafEarlier == leafLater:
				continue

			positionsOfEarlier: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsSubset[:, leafEarlier]

			isEarlierAlwaysPresentAndPrecedes: bool = bool(numpy.all((positionsOfEarlier >= 0) & (positionsOfEarlier < pileLastOfLeafIndex)))
			if isEarlierAlwaysPresentAndPrecedes and (leafEarlier, leafLater) not in setUnconditional:
				listConditionalRelationships.append({
					'Earlier': leafEarlier,
					'Later': leafLater,
					'AtColumn': pileLastOfLeafOriginal
				})

	dataframeConditionalPrecedenceAtLastPile: pandas.DataFrame = pandas.DataFrame(listConditionalRelationships).sort_values(['Later', 'Earlier']).reset_index(drop=True)

	return dataframeConditionalPrecedenceAtLastPile

def getLeafConditionalSuccession(state: EliminationState) -> pandas.DataFrame:
	"""When a leaf is at the last pile in its domain, identify leaves that must come after it."""
	dataframeSequences: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
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

	for leafEarlier in range(state.leavesTotal):
		pileLastOfLeafOriginal: int = int(bit_mask(state.dimensionsTotal) ^ bit_mask(state.dimensionsTotal - dimensionNearest首(leafEarlier))) - howManyDimensionsHaveOddParity(leafEarlier) + 1
		pileLastOfLeafIndex: int = pileLastOfLeafOriginal - columnOffset

		if pileLastOfLeafIndex < 0:
			continue

		maskRowsAtLastPileOfLeaf: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = (positionsMatrix[:, leafEarlier] == pileLastOfLeafIndex)

		if not numpy.any(maskRowsAtLastPileOfLeaf):
			continue

		positionsSubset: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsMatrix[maskRowsAtLastPileOfLeaf]

		for leafLater in range(state.leavesTotal):
			if leafLater == leafEarlier:
				continue

			positionsOfLater: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsSubset[:, leafLater]
			isLaterAlwaysPresentAndFollows: bool = bool(numpy.all((positionsOfLater >= 0) & (pileLastOfLeafIndex < positionsOfLater)))
			if isLaterAlwaysPresentAndFollows and (leafEarlier, leafLater) not in setUnconditional:
				listConditionalRelationships.append({
					'Earlier': leafEarlier,
					'Later': leafLater,
					'AtColumn': pileLastOfLeafOriginal,
				})

	dataframeConditionalSuccession: pandas.DataFrame = pandas.DataFrame(listConditionalRelationships, columns=['Earlier', 'Later', 'AtColumn']).sort_values(['Earlier', 'Later']).reset_index(drop=True)

	return dataframeConditionalSuccession

def getLeafConditionalPrecedenceAcrossLeafDomain(state: EliminationState, leafLater: int) -> pandas.DataFrame:
	dataframeSequences: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
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

	leafDomain: range = getLeafDomain(state, leafLater)

	listConditionalRelationships: list[dict[str, int]] = []
	for pileOfLeafOriginal in leafDomain:
		if pileOfLeafOriginal <= 1:
			continue
		if pileOfLeafOriginal >= state.pileLast:
			continue

		pileOfLeafIndex: int = pileOfLeafOriginal - columnOffset
		if pileOfLeafIndex < 0:
			continue

		maskRowsAtPileOfLeaf: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = (positionsMatrix[:, leafLater] == pileOfLeafIndex)
		if not numpy.any(maskRowsAtPileOfLeaf):
			continue

		positionsSubset: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsMatrix[maskRowsAtPileOfLeaf]
		maskAlwaysEarlier: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = numpy.all((positionsSubset >= 0) & (positionsSubset < pileOfLeafIndex), axis=0)
		maskAlwaysEarlier[leafLater] = False
		indicesEarlier: numpy.ndarray[Any, numpy.dtype[numpy.intp]] = numpy.flatnonzero(maskAlwaysEarlier)

		for leafEarlierCandidate in indicesEarlier.tolist():
			leafEarlier: int = int(leafEarlierCandidate)
			if (leafEarlier, leafLater) in setUnconditional:
				continue
			listConditionalRelationships.append({
				'Earlier': leafEarlier,
				'Later': leafLater,
				'AtColumn': pileOfLeafOriginal,
			})

	dataframeConditionalPrecedenceAcrossDomain: pandas.DataFrame = pandas.DataFrame(listConditionalRelationships, columns=['Earlier', 'Later', 'AtColumn']).sort_values(['AtColumn', 'Earlier']).reset_index(drop=True)
	return dataframeConditionalPrecedenceAcrossDomain

def getLeafConditionalPrecedenceAcrossLeafDomainPileGroups(state: EliminationState, leafLater: int) -> list[list[int]]:
	dataframeConditional: pandas.DataFrame = getLeafConditionalPrecedenceAcrossLeafDomain(state, leafLater)
	pilesSortedUnique: list[int]
	if dataframeConditional.empty:
		pilesSortedUnique = []
	else:
		pilesSortedUnique = sorted({int(pile) for pile in dataframeConditional['AtColumn'].tolist()})

	listPileGroups: list[list[int]] = []
	for pile in pilesSortedUnique:
		if not listPileGroups:
			listPileGroups.append([pile])
		elif pile == listPileGroups[-1][-1] + 2:
			listPileGroups[-1].append(pile)
		else:
			listPileGroups.append([pile])
	return listPileGroups

def getLeafPilesAtDomainEndFromConditionalPrecedenceAcrossLeafDomain(state: EliminationState, leaf: int) -> list[int]:
	listPileGroups: list[list[int]] = getLeafConditionalPrecedenceAcrossLeafDomainPileGroups(state, leaf)
	listPilesAtEnd: list[int] = []
	if listPileGroups:
		listPilesAtEnd = listPileGroups[-1]
	return listPilesAtEnd

def getDictionaryPilesAtDomainEndsFromConditionalPrecedenceAcrossLeafDomain(state: EliminationState, listLeavesAnalyzed: list[int] | None = None) -> dict[int, list[int]]:
	if listLeavesAnalyzed is None:
		leavesExcluded: set[int] = {pileOrigin, 零, state.leavesTotal - 零}
		listLeavesAnalyzed = [leaf for leaf in range(state.leavesTotal) if leaf not in leavesExcluded]

	dictionaryPilesAtDomainEnds: dict[int, list[int]] = {}
	for leaf in listLeavesAnalyzed:
		listPilesAtEnd: list[int] = getLeafPilesAtDomainEndFromConditionalPrecedenceAcrossLeafDomain(state, leaf)
		if listPilesAtEnd:
			dictionaryPilesAtDomainEnds[leaf] = listPilesAtEnd
	return dictionaryPilesAtDomainEnds

if __name__ == '__main__':
	state = EliminationState((2,) * 6)
	# leaf33 is wrong because of step = 4.
	# leaf33 and leaf49 are already known from prior analysis.
	dictionaryPilesAtDomainEnds = getDictionaryPilesAtDomainEndsFromConditionalPrecedenceAcrossLeafDomain(state)
	print(ansiColorBlackOnCyan + 'dictionaryPilesAtDomainEnds' + ansiColorReset)
	pprint(dictionaryPilesAtDomainEnds, width=140)
	pprint(getDictionaryConditionalLeafPredecessors(state), width=380, compact=True)
	pprint(getZ0Z_successor(state), width=380, compact=True)
