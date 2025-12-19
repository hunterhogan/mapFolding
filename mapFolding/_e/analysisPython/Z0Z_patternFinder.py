# ruff: noqa: ERA001 T201 T203  # noqa: RUF100
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from gmpy2 import bit_flip, bit_mask, is_even
from hunterMakesPy import raiseIfNone
from itertools import accumulate
from mapFolding import (
	asciiColorCyan, asciiColorGreen, asciiColorMagenta, asciiColorRed, asciiColorReset, asciiColorYellow, decreasing,
	inclusive, packageSettings)
from mapFolding._e import (
	dimensionNearest首, getDictionaryLeafDomains, getDictionaryPileRanges, getLeafDomain, howMany0coordinatesAtTail,
	howManyDimensionsHaveOddParity, pileOrigin, PinnedLeaves, Z0Z_sumsOfProductsOfDimensionsNearest首, 零, 首零, 首零一三)
from mapFolding._e._dataDynamic import getDataFrameFoldings
from mapFolding._e.pinning2DnAnnex import beansWithoutCornbread
from mapFolding.dataBaskets import EliminationState
from pathlib import Path
from pprint import pprint
from typing import Any
import csv
import numpy
import pandas

@dataclass
class PermutationSpaceStatus:
	listSurplusDictionaries: list[PinnedLeaves]
	maskUnion: numpy.ndarray
	indicesOverlappingRows: numpy.ndarray
	indicesOverlappingLeavesPinned: set[int]
	rowsRequired: int
	rowsTotal: int

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

def _getGroupedBy(state: EliminationState, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
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

def detectPermutationSpaceErrors(arrayFoldings: numpy.ndarray, listPinnedLeaves: Sequence[PinnedLeaves]) -> PermutationSpaceStatus:
	rowsTotal: int = int(arrayFoldings.shape[0])
	listMasks: list[numpy.ndarray] = []
	listSurplusDictionaries: list[PinnedLeaves] = []
	for leavesPinned in listPinnedLeaves:
		maskMatches: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
		for pile, leaf in leavesPinned.items():
			maskMatches = maskMatches & (arrayFoldings[:, pile] == leaf)
		if not bool(maskMatches.any()):
			listSurplusDictionaries.append(leavesPinned)
		listMasks.append(maskMatches)

	if listMasks:
		masksStacked: numpy.ndarray = numpy.column_stack(listMasks)
	else:
		masksStacked = numpy.zeros((rowsTotal, 0), dtype=bool)

	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	maskUnion: numpy.ndarray = coverageCountPerRow > 0
	rowsRequired: int = int(maskUnion.sum())
	indicesOverlappingRows: numpy.ndarray = numpy.flatnonzero(coverageCountPerRow >= 2)
	indicesOverlappingLeavesPinned: set[int] = set()
	if indicesOverlappingRows.size > 0:
		for indexMask, mask in enumerate(listMasks):
			if bool(mask[indicesOverlappingRows].any()):
				indicesOverlappingLeavesPinned.add(indexMask)

	return PermutationSpaceStatus(listSurplusDictionaries, maskUnion, indicesOverlappingRows, indicesOverlappingLeavesPinned, rowsRequired, rowsTotal)

def measureEntropy(state: EliminationState, listLeavesAnalyzed: list[int] | None = None) -> pandas.DataFrame:
	"""Measure the relative entropy and distributional properties of leaves across folding sequences.

	This function analyzes how leaves are distributed across their mathematical domains by comparing
	empirical distributions from actual folding sequences against uniform distributions. The analysis
	uses Shannon entropy normalized by maximum possible entropy to produce comparable measures across
	leaves with different domain sizes.

	Parameters
	----------
	state : EliminationState
		The elimination state containing the map shape and dimension information.
	listLeavesAnalyzed : list[int] | None = None
		Specific leaves to analyze. If None, analyzes all leaves except the trivial ones
		(0, 1, and leavesTotal-1) which always occupy the same pile.

	Returns
	-------
	dataframeEntropy : pandas.DataFrame
		DataFrame with columns:
		- 'leaf': The leaf value being analyzed
		- 'domainSize': Number of possible piles where this leaf can appear
		- 'entropyActual': Shannon entropy of the empirical distribution
		- 'entropyMaximum': Maximum possible entropy (uniform distribution)
		- 'entropyRelative': entropyActual / entropyMaximum (0 to 1)
		- 'concentrationMaximum': Maximum frequency / mean frequency
		- 'bitPattern': Binary representation for easy identification of patterns
		- 'bitCount': Number of 1s in binary representation
		- 'trailingZeros': Number of trailing zeros (power of 2 factor)
		Sorted by entropyRelative descending to show most uniform distributions first.

	Notes
	-----
	The relative entropy metric allows fair comparison between leaves with vastly different domain
	sizes. A value near 1.0 indicates nearly uniform distribution (high entropy, unpredictable),
	while values near 0.0 indicate highly concentrated distribution (low entropy, predictable).

	The concentration metric shows how peaked the distribution is by comparing the most frequent
	position to the mean frequency. Higher values indicate more predictable placement.

	"""
	dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))

	if listLeavesAnalyzed is None:
		leavesExcluded: set[int] = {pileOrigin, 零, state.leavesTotal - 零}
		listLeavesAnalyzed = [leaf for leaf in range(state.leavesTotal) if leaf not in leavesExcluded]

	listEntropyRecords: list[dict[str, Any]] = []

	for leaf in listLeavesAnalyzed:
		domainLeaf: range = getLeafDomain(state, leaf)
		domainSize: int = len(domainLeaf)

		if domainSize == 0:
			continue

		dataframeMelted: pandas.DataFrame = dataframeFoldings[dataframeFoldings == leaf].melt(ignore_index=False)
		dataframeMelted = dataframeMelted.dropna()
		if dataframeMelted.empty:
			continue

		arrayPileCounts: numpy.ndarray = numpy.bincount(dataframeMelted['variable'].astype(int), minlength=state.leavesTotal)
		arrayPileCountsInDomain: numpy.ndarray = arrayPileCounts[list(domainLeaf)]
		arrayFrequencies: numpy.ndarray = arrayPileCountsInDomain / arrayPileCountsInDomain.sum()

		maskNonzero: numpy.ndarray = arrayFrequencies > 0
		entropyActual: float = float(-numpy.sum(arrayFrequencies[maskNonzero] * numpy.log2(arrayFrequencies[maskNonzero])))
		entropyMaximum: float = float(numpy.log2(domainSize))
		entropyRelative: float = entropyActual / entropyMaximum if entropyMaximum > 0 else 0.0

		frequencyMaximum: float = float(arrayFrequencies.max())
		frequencyMean: float = 1.0 / domainSize
		concentrationMaximum: float = frequencyMaximum / frequencyMean if frequencyMean > 0 else 0.0

		listEntropyRecords.append({
			'leaf': leaf,
			'domainSize': domainSize,
			'entropyActual': entropyActual,
			'entropyMaximum': entropyMaximum,
			'entropyRelative': entropyRelative,
			'concentrationMaximum': concentrationMaximum,
			'bitPattern': bin(leaf),
			'bitCount': leaf.bit_count(),
			'trailingZeros': howMany0coordinatesAtTail(leaf),
		})

	return pandas.DataFrame(listEntropyRecords).sort_values('entropyRelative', ascending=False).reset_index(drop=True)

def verifyPinning2Dn(state: EliminationState) -> None:
	arrayFoldings = getDataFrameFoldings(state)
	if arrayFoldings is not None:
		arrayFoldings = arrayFoldings.to_numpy(dtype=numpy.uint8, copy=False)
		pinningCoverage: PermutationSpaceStatus = detectPermutationSpaceErrors(arrayFoldings, state.listPinnedLeaves)

		listDictionaryPinned: list[PinnedLeaves] = pinningCoverage.listSurplusDictionaries
		if listDictionaryPinned:
			print(asciiColorYellow, end='')
			pprint(listDictionaryPinned[0:5], width=140)
		else:
			print(asciiColorGreen, end='')
		print(len(listDictionaryPinned), "surplus dictionaries.")
		print(asciiColorReset, end='')

		pathFilename = Path(f"{packageSettings.pathPackage}/_e/analysisExcel/p2d{state.dimensionsTotal}SurplusDictionaries.csv")

		if listDictionaryPinned:
			with pathFilename.open('w', newline='') as writeStream:
				writerCSV = csv.writer(writeStream)
				listPiles: list[int] = list(range(state.leavesTotal))
				writerCSV.writerow(listPiles)
				for leavesPinned in listDictionaryPinned:
					writerCSV.writerow([leavesPinned.get(pile, '') for pile in listPiles])

		if pinningCoverage.indicesOverlappingLeavesPinned:
			print(f"{asciiColorRed}{len(pinningCoverage.indicesOverlappingLeavesPinned)} overlapping dictionaries", asciiColorReset)
			for indexDictionary in sorted(pinningCoverage.indicesOverlappingLeavesPinned)[0:2]:
				pprint(state.listPinnedLeaves[indexDictionary], width=140)

		beansOrCornbread: Callable[[PinnedLeaves], bool] = beansWithoutCornbread(state)
		listBeans: list[PinnedLeaves] = list(filter(beansOrCornbread, state.listPinnedLeaves))
		if listBeans:
			print(f"{asciiColorMagenta}{len(listBeans)} dictionaries with beans but no cornbread.", asciiColorReset)
			pprint(listBeans[0], width=140)

		maskUnion: numpy.ndarray = pinningCoverage.maskUnion
		rowsRequired: int = pinningCoverage.rowsRequired
		rowsTotal: int = pinningCoverage.rowsTotal
		color = asciiColorReset
		if rowsRequired < rowsTotal:
			color = asciiColorRed
			indicesMissingRows: numpy.ndarray = numpy.flatnonzero(~maskUnion)
			for indexRow in indicesMissingRows[0:2]:
				print(color, arrayFoldings[indexRow, :])
		print(f"{color}Required rows: {rowsRequired}/{rowsTotal}{asciiColorReset}")

if __name__ == '__main__':
	state = EliminationState((2,) * 6)

	from mapFolding._e import getZ0Z_precedence
	# getZ0Z_precedence(state)

	# leaf33 is wrong because of step = 4.
	# leaf33 and leaf49 are already known from prior analysis.
	# dictionaryPilesAtDomainEnds = getDictionaryPilesAtDomainEndsFromConditionalPrecedenceAcrossLeafDomain(state)
	# print(asciiColorCyan + 'dictionaryPilesAtDomainEnds' + asciiColorReset)
	# pprint(dictionaryPilesAtDomainEnds, width=140)
	pprint(getZ0Z_precedence(state), width=380, compact=True)
	# print()
	# print(Z0Z_sumsOfProductsOfDimensionsNearest首(state, 5))
