# DEVELOPMENT module.
# pyright: reportAssignmentType=false, reportUnnecessaryComparison=false
# ruff: file-ignore[print, p-print]
# ty: ignore[invalid-assignment]
from __future__ import annotations

from gmpy2 import bit_mask
from hunterMakesPy import raiseIfNone
from mapFolding import ansiColorReset, ansiColors
from mapFolding._e import getDomainLeaf, pileOrigin
from mapFolding._e._2上nDimensional import getLeafPredecessors, getLeafSuccessors, 工dimensionTail, 工dimension首零, 工totalDimensionsOdd, 零
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.kitFilesystem import getDataFrameFoldings
from pprint import pprint
from typing import TYPE_CHECKING
import numpy
import pandas

if TYPE_CHECKING:
	from mapFolding._e.theTypes import Leaf, Pile
	from typing import Any

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
	alwaysEarlierMatrix: numpy.ndarray[tuple[Any, ...], numpy.dtype[numpy.bool_]] = comparisonCube.all(axis=0)
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

	Returns
	-------
	dataframeConditionalPrecedence : pandas.DataFrame
		A three-column DataFrame with 'Earlier', 'Later', and 'AtColumn' indicating
		that when 'Later' is at column 'AtColumn', 'Earlier' always precedes it.
		Only includes relationships not already present in unconditional precedence.

	"""
	dataframeSequences: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
	columnsToExclude: list[int] | None = [pileOrigin, 零, state.pileLast]
	if columnsToExclude is not None:
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

	columnOffset: int = 2 if columnsToExclude is not None and 0 in columnsToExclude and 1 in columnsToExclude else 0

	dataframeUnconditional: pandas.DataFrame = getLeafUnconditionalPrecedence(state)
	boxOfUnconditional: set[tuple[Any, Any]] = set(zip(dataframeUnconditional['Earlier'], dataframeUnconditional['Later'], strict=True))

	boxOfConditionalRelationships: list[dict[str, int]] = []

	for leafLater in range(state.totalLeaves):
		columnEarliestOriginal: int = leafLater.bit_count() + (2 ** (工dimensionTail(leafLater) + 1) - 2)
		columnEarliestIndex: int = columnEarliestOriginal - columnOffset

		if columnEarliestIndex < 0:
			continue

		maskRowsAtEarliestColumn: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = (positionsMatrix[:, leafLater] == columnEarliestIndex)

		if not numpy.any(maskRowsAtEarliestColumn):
			continue

		positionsSubset: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsMatrix[maskRowsAtEarliestColumn]

		for leafEarlier in range(state.totalLeaves):
			if leafEarlier == leafLater:
				continue

			positionsOfEarlier: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsSubset[:, leafEarlier]

			isEarlierAlwaysPresentAndPrecedes: bool = bool(numpy.all((positionsOfEarlier >= 0) & (positionsOfEarlier < columnEarliestIndex)))
			if isEarlierAlwaysPresentAndPrecedes and (leafEarlier, leafLater) not in boxOfUnconditional:
				boxOfConditionalRelationships.append({
					'Earlier': leafEarlier,
					'Later': leafLater,
					'AtColumn': columnEarliestOriginal
				})

	dataframeConditionalPrecedence: pandas.DataFrame = pandas.DataFrame(boxOfConditionalRelationships).sort_values(['Later', 'Earlier']).reset_index(drop=True)

	return dataframeConditionalPrecedence

def getLeafConditionalPrecedenceAtLastPileOfLeafDomain(state: EliminationState) -> pandas.DataFrame:
	"""Identify precedence relationships that emerge only when a leaf is at the last pile in its domain.

	(AI generated docstring)

	For each leaf, determines the last pile it can occupy within its mathematical
	domain, then finds leaves that always precede it in the subset of foldings
	where that leaf is observed at that last-in-domain pile. Excludes relationships
	already captured by the unconditional precedence analysis.

	The formula for the last pile *in* the domain of a leaf is.
		pileLastOfLeaf = int(bit_mask(totalDimensions) ^ bit_mask(totalDimensions - 工dimension首零(leaf))) - 工totalDimensionsOdd(leaf) + 1

	Parameters
	----------
	state : EliminationState
		The elimination state containing the map shape and dimension information.

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
	if columnsToExclude is not None:
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

	columnOffset: int = 2 if columnsToExclude is not None and 0 in columnsToExclude and 1 in columnsToExclude else 0

	dataframeUnconditional: pandas.DataFrame = getLeafUnconditionalPrecedence(state)
	boxOfUnconditional: set[tuple[Any, Any]] = set(zip(dataframeUnconditional['Earlier'], dataframeUnconditional['Later'], strict=True))

	boxOfConditionalRelationships: list[dict[str, int]] = []

	for leafLater in range(state.totalLeaves):
		pileLastOfLeafOriginal: int = int(bit_mask(state.totalDimensions) ^ bit_mask(state.totalDimensions - 工dimension首零(leafLater))) - 工totalDimensionsOdd(leafLater) + 1
		pileLastOfLeafIndex: int = pileLastOfLeafOriginal - columnOffset

		if pileLastOfLeafIndex < 0:
			continue

		maskRowsAtLastPileOfLeaf: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = (positionsMatrix[:, leafLater] == pileLastOfLeafIndex)

		if not numpy.any(maskRowsAtLastPileOfLeaf):
			continue

		positionsSubset: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsMatrix[maskRowsAtLastPileOfLeaf]

		for leafEarlier in range(state.totalLeaves):
			if leafEarlier == leafLater:
				continue

			positionsOfEarlier: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsSubset[:, leafEarlier]

			isEarlierAlwaysPresentAndPrecedes: bool = bool(numpy.all((positionsOfEarlier >= 0) & (positionsOfEarlier < pileLastOfLeafIndex)))
			if isEarlierAlwaysPresentAndPrecedes and (leafEarlier, leafLater) not in boxOfUnconditional:
				boxOfConditionalRelationships.append({
					'Earlier': leafEarlier,
					'Later': leafLater,
					'AtColumn': pileLastOfLeafOriginal
				})

	dataframeConditionalPrecedenceAtLastPile: pandas.DataFrame = pandas.DataFrame(boxOfConditionalRelationships).sort_values(['Later', 'Earlier']).reset_index(drop=True)

	return dataframeConditionalPrecedenceAtLastPile

def getLeafConditionalSuccession(state: EliminationState) -> pandas.DataFrame:
	"""When a leaf is at the last pile in its domain, identify leaves that must come after it."""
	dataframeSequences: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
	columnsToExclude: list[int] | None = [pileOrigin, 零, state.pileLast]
	if columnsToExclude is not None:
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

	columnOffset: int = 2 if columnsToExclude is not None and 0 in columnsToExclude and 1 in columnsToExclude else 0

	dataframeUnconditional: pandas.DataFrame = getLeafUnconditionalPrecedence(state)
	boxOfUnconditional: set[tuple[Any, Any]] = set(zip(dataframeUnconditional['Earlier'], dataframeUnconditional['Later'], strict=True))

	boxOfConditionalRelationships: list[dict[str, int]] = []

	for leafEarlier in range(state.totalLeaves):
		pileLastOfLeafOriginal: int = int(bit_mask(state.totalDimensions) ^ bit_mask(state.totalDimensions - 工dimension首零(leafEarlier))) - 工totalDimensionsOdd(leafEarlier) + 1
		pileLastOfLeafIndex: int = pileLastOfLeafOriginal - columnOffset

		if pileLastOfLeafIndex < 0:
			continue

		maskRowsAtLastPileOfLeaf: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] = (positionsMatrix[:, leafEarlier] == pileLastOfLeafIndex)

		if not numpy.any(maskRowsAtLastPileOfLeaf):
			continue

		positionsSubset: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsMatrix[maskRowsAtLastPileOfLeaf]

		for leafLater in range(state.totalLeaves):
			if leafLater == leafEarlier:
				continue

			positionsOfLater: numpy.ndarray[Any, numpy.dtype[numpy.int16]] = positionsSubset[:, leafLater]
			isLaterAlwaysPresentAndFollows: bool = bool(numpy.all((positionsOfLater >= 0) & (pileLastOfLeafIndex < positionsOfLater)))
			if isLaterAlwaysPresentAndFollows and (leafEarlier, leafLater) not in boxOfUnconditional:
				boxOfConditionalRelationships.append({
					'Earlier': leafEarlier,
					'Later': leafLater,
					'AtColumn': pileLastOfLeafOriginal,
				})

	dataframeConditionalSuccession: pandas.DataFrame = pandas.DataFrame(boxOfConditionalRelationships, columns=['Earlier', 'Later', 'AtColumn']).sort_values(['Earlier', 'Later']).reset_index(drop=True)

	return dataframeConditionalSuccession

def getLeafConditionalPrecedenceAcrossLeafDomain(state: EliminationState, leafLater: Leaf) -> pandas.DataFrame:
	dataframeSequences: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
	columnsToExclude: list[Pile] | None = [pileOrigin, 零, state.pileLast]
	if columnsToExclude is not None:
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

	columnOffset: int = 2 if columnsToExclude is not None and 0 in columnsToExclude and 1 in columnsToExclude else 0

	dataframeUnconditional: pandas.DataFrame = getLeafUnconditionalPrecedence(state)
	boxOfUnconditional: set[tuple[Any, Any]] = set(zip(dataframeUnconditional['Earlier'], dataframeUnconditional['Later'], strict=True))

	leafDomain: range = getDomainLeaf(state, leafLater)

	boxOfConditionalRelationships: list[dict[str, int]] = []
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
			leafEarlier: Leaf = int(leafEarlierCandidate)
			if (leafEarlier, leafLater) in boxOfUnconditional:
				continue
			boxOfConditionalRelationships.append({
				'Earlier': leafEarlier,
				'Later': leafLater,
				'AtColumn': pileOfLeafOriginal,
			})

	dataframeConditionalPrecedenceAcrossDomain: pandas.DataFrame = pandas.DataFrame(boxOfConditionalRelationships, columns=['Earlier', 'Later', 'AtColumn']).sort_values(['AtColumn', 'Earlier']).reset_index(drop=True)
	return dataframeConditionalPrecedenceAcrossDomain

def getLeafConditionalPrecedenceAcrossLeafDomainPileGroups(state: EliminationState, leafLater: Leaf) -> list[list[Pile]]:
	dataframeConditional: pandas.DataFrame = getLeafConditionalPrecedenceAcrossLeafDomain(state, leafLater)
	pilesSortedUnique: list[Pile]
	if dataframeConditional.empty:
		pilesSortedUnique = []
	else:
		pilesSortedUnique = sorted({int(pile) for pile in dataframeConditional['AtColumn'].tolist()})

	boxOfPileGroups: list[list[Pile]] = []
	for pile in pilesSortedUnique:
		if not boxOfPileGroups:
			boxOfPileGroups.append([pile])
		elif pile == boxOfPileGroups[-1][-1] + 2:
			boxOfPileGroups[-1].append(pile)
		else:
			boxOfPileGroups.append([pile])
	return boxOfPileGroups

def getLeafPilesAtDomainEndFromConditionalPrecedenceAcrossLeafDomain(state: EliminationState, leaf: Leaf) -> list[Pile]:
	boxOfPileGroups: list[list[Pile]] = getLeafConditionalPrecedenceAcrossLeafDomainPileGroups(state, leaf)
	boxOfPilesAtEnd: list[Pile] = []
	if boxOfPileGroups:
		boxOfPilesAtEnd = boxOfPileGroups[-1]
	return boxOfPilesAtEnd

def getDictionaryPilesAtDomainEndsFromConditionalPrecedenceAcrossLeafDomain(state: EliminationState, boxOfLeavesAnalyzed: list[Leaf] | None = None) -> dict[Leaf, list[Pile]]:
	if boxOfLeavesAnalyzed is None:
		leavesExcluded: set[Leaf] = {pileOrigin, 零, state.totalLeaves - 零}
		boxOfLeavesAnalyzed = [leaf for leaf in range(state.totalLeaves) if leaf not in leavesExcluded]

	dictionaryPilesAtDomainEnds: dict[Leaf, list[Pile]] = {}
	for leaf in boxOfLeavesAnalyzed:
		boxOfPilesAtEnd: list[Pile] = getLeafPilesAtDomainEndFromConditionalPrecedenceAcrossLeafDomain(state, leaf)
		if boxOfPilesAtEnd:
			dictionaryPilesAtDomainEnds[leaf] = boxOfPilesAtEnd
	return dictionaryPilesAtDomainEnds

if __name__ == '__main__':
	state = EliminationState((2,) * 6)
	# leaf33 is wrong because of step = 4.
	# leaf33 and leaf49 are already known from prior analysis.
	dictionaryPilesAtDomainEnds = getDictionaryPilesAtDomainEndsFromConditionalPrecedenceAcrossLeafDomain(state)
	print(ansiColors.BlackOnCyan + 'dictionaryPilesAtDomainEnds' + ansiColorReset)
	pprint(dictionaryPilesAtDomainEnds, width=140)
	pprint(getLeafPredecessors(state), width=380, compact=True)
	pprint(getLeafSuccessors(state), width=380, compact=True)
