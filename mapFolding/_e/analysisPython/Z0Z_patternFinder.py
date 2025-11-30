# ruff: noqa: ERA001 T201 T203  # noqa: RUF100
from collections.abc import Callable, Sequence
from gmpy2 import bit_flip, bit_mask, is_odd
from hunterMakesPy import writePython
from itertools import filterfalse, repeat
from mapFolding import exclude, inclusive, packageSettings, reverseLookup
from mapFolding._e import (
	dimensionNearest首, getDictionaryLeafDomains, getDictionaryPileRanges, getLeafDomain, howMany0coordinatesAtTail,
	howManyDimensionsHaveOddParity, leafInSubHyperplane, pileOrigin, ptount, 一, 三, 二, 四, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二,
	首零二)
from mapFolding.beDRY import between, noDuplicates
from mapFolding.dataBaskets import EliminationState
from more_itertools import extract
from pathlib import Path, PurePath
from pprint import pprint
from typing import Any
import csv
import numpy
import pandas

def _getDataFrameFoldings(state: EliminationState) -> pandas.DataFrame:
	pathFilename = Path(f'{packageSettings.pathPackage}/tests/dataSamples/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
	arrayFoldings = pandas.read_pickle(pathFilename)  # noqa: S301
	return pandas.DataFrame(arrayFoldings)

def getDataFrameUnconditionalPrecedence(state: EliminationState, columnsToExclude: list[int] | None = None) -> pandas.DataFrame:
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
	columnsToExclude : list[int] | None = None
		Column indices (as integers) to exclude from analysis. Pass [0, 1, leavesTotal-1]
		to exclude the trivially-pinned positions.

	Returns
	-------
	dataframePrecedence : pandas.DataFrame
		A two-column DataFrame with 'Earlier' and 'Later' indicating leaf values
		where the Earlier leaf unconditionally precedes the Later leaf.

	"""
	dataframeSequences: pandas.DataFrame = _getDataFrameFoldings(state)
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

def getDataFrameConditionalPrecedence(state: EliminationState, columnsToExclude: list[int] | None = None) -> pandas.DataFrame:
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
	dataframeSequences: pandas.DataFrame = _getDataFrameFoldings(state)
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

	dataframeUnconditional: pandas.DataFrame = getDataFrameUnconditionalPrecedence(state, columnsToExclude)
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
	dataframeFoldings: pandas.DataFrame = _getDataFrameFoldings(state)
	groupedBy: dict[int | tuple[int, ...], list[int]] = dataframeFoldings.groupby(list(groupByLeavesAtPiles))[pileTarget].apply(list).to_dict()
	return {leaves: sorted(set(listLeaves)) for leaves, listLeaves in groupedBy.items()}

def getExcludedLeaves(state: EliminationState, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	return {leaves: sorted(set(getDictionaryPileRanges(state)[pileTarget]).difference(set(listLeaves))) for leaves, listLeaves in _getGroupedBy(state, pileTarget, groupByLeavesAtPiles).items()}

def getExcludingDictionary(state: EliminationState, leafExcluder: int) -> dict[int, dict[int, list[int]]] | None:
	"""Get.

	dict[pileExcluder, dict[leaf, listIndicesPilesExcluded]]
	If `leafExcluder` is in `pileExcluder`, then `leaf` is excluded from its domainOfPiles at `listIndicesPilesExcluded`.

	Use `state.leavesTotal` and `state.dimensionsTotal` to dynamically generate a dictionary appropriate for the `mapShape`.
	"""
	excludingDictionary: dict[int, dict[int, list[int]]] | None = None
	dictionaryLeafDomains: dict[int, range] = getDictionaryLeafDomains(state)

	if leafExcluder == 一+零:
		domainOfPilesForLeafExcluder: list[int] = list(dictionaryLeafDomains[leafExcluder])
		sizeDomainExcluder: int = len(domainOfPilesForLeafExcluder)
		excludingDictionary = {pileExcluder: {leaf: [] for leaf in range(state.leavesTotal)} for pileExcluder in domainOfPilesForLeafExcluder}

		for indexDomainPileExcluder, pileExcluder in enumerate(excludingDictionary):
			for leaf in range(3, state.leavesTotal):
				domainOfPilesForLeaf: list[int] = list(dictionaryLeafDomains[leaf])
				sizeDomainLeaf: int = len(domainOfPilesForLeaf)

				if indexDomainPileExcluder in list(filterfalse(lambda index: index % 4 == 3, range(sizeDomainExcluder))):  # noqa: SIM102
					if (一+零 < leaf < 首零(state.dimensionsTotal)) and (ptount(leaf) >= state.dimensionsTotal - 3):
						excludingDictionary[pileExcluder][leaf].extend([-1])

				if is_odd(leaf):
					excludingDictionary[pileExcluder][leaf].extend([indexDomainPileExcluder])
					if 1 < howMany0coordinatesAtTail(leaf - 1) < state.dimensionsTotal - 1:
						excludingDictionary[pileExcluder][leaf].extend([indexDomainPileExcluder + 1])

				if howMany0coordinatesAtTail(leaf) == 1:
					excludingDictionary[pileExcluder][leaf].extend([indexDomainPileExcluder - 1])
					start = 0
					stop = indexDomainPileExcluder - (leafInSubHyperplane(leaf) == 2) - (2 * max(0, leaf.bit_count() - 3))
					excludingDictionary[pileExcluder][leaf].extend([*range(start, stop)])

				if howMany0coordinatesAtTail(leaf) == 2:
					start = 0
					stop = indexDomainPileExcluder - 2 * (leafInSubHyperplane(leaf) == 4) - 2
					excludingDictionary[pileExcluder][leaf].extend([*range(start, stop)])

				if (leaf == 首一(state.dimensionsTotal)):

					if pileExcluder <= 首二(state.dimensionsTotal):
						pass

					elif 首二(state.dimensionsTotal) < pileExcluder < 首一(state.dimensionsTotal):
						excludingDictionary[pileExcluder][leaf].extend([*range(1, sizeDomainLeaf // 2), *range(1 + sizeDomainLeaf // 2, 3 * sizeDomainLeaf // 4)])

					elif pileExcluder == 首一(state.dimensionsTotal):
						excludingDictionary[pileExcluder][leaf].extend([*range(1, sizeDomainLeaf // 2)])

					elif 首一(state.dimensionsTotal) < pileExcluder < 首零(state.dimensionsTotal)-一:
						excludingDictionary[pileExcluder][leaf].extend([*range(3 * sizeDomainLeaf // 4)])

					elif pileExcluder == 首零(state.dimensionsTotal)-一:
						excludingDictionary[pileExcluder][leaf].extend([*range(1, 3 * sizeDomainLeaf // 4)])

					elif pileExcluder == 首零(state.dimensionsTotal):
						excludingDictionary[pileExcluder][leaf].extend([*range(2, sizeDomainLeaf // 2)])

				if leaf == 首零(state.dimensionsTotal) + 零:
					bump: int = 1 - int(pileExcluder.bit_count() == 1)
					howMany: int = state.dimensionsTotal - (pileExcluder.bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern: int = sizeDomainLeaf - onesInBinary

					if pileExcluder == 二:
						excludingDictionary[pileExcluder][leaf].extend([零, 一, 二])

					if 二 < pileExcluder <= 首二(state.dimensionsTotal):
						stop: int = sizeDomainLeaf // 2 - 1
						excludingDictionary[pileExcluder][leaf].extend(range(1, stop))

						aDimensionPropertyNotFullyUnderstood = 5
						for _dimension in range(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
							start: int = 1 + stop
							stop += (stop+1) // 2
							excludingDictionary[pileExcluder][leaf].extend([*range(start, stop)])

						excludingDictionary[pileExcluder][leaf].extend([*range(1 + stop, ImaPattern)])

					if 首二(state.dimensionsTotal) < pileExcluder:
						excludingDictionary[pileExcluder][leaf].extend([*range(1, ImaPattern)])

				def normalizeIndex(index: int, lengthIterable: int) -> int:
					if index < 0:
						index = (index + lengthIterable) % lengthIterable
					return index

				excludingDictionary[pileExcluder][leaf] = sorted(set(map(normalizeIndex, excludingDictionary[pileExcluder][leaf], repeat(sizeDomainLeaf))))

	# else:
	# 	excludingDictionary: dict[int, dict[int, list[int]]] | None = dictionaryExclusions.get(leafExcluder)
	return excludingDictionary

def makeVerificationDataLeavesDomain(listDimensions: Sequence[int], listLeaves: Sequence[int | Callable[[int], int]], pathFilename: PurePath | None = None, settings: dict[str, dict[str, Any]] | None = None) -> PurePath:
	"""Create a Python module containing combined domain data for multiple leaves across multiple mapShapes.

	This function extracts the actual combined domain (the set of valid pile position tuples) for a group of leaves from pickled
	folding data. The data is used for verification in pytest tests comparing computed domains against empirical data.

	The combined domain is a set of tuples where each tuple represents the pile positions for the specified leaves in a valid
	folding. For example, if `listLeaves` is `[4, 5, 6, 7]`, each tuple has 4 elements representing the pile where each of those
	leaves appears in a folding.

	Parameters
	----------
	listDimensions : Sequence[int]
		The dimension counts to process (e.g., `[4, 5, 6]` for 2^4, 2^5, 2^6 leaf maps).
	listLeaves : Sequence[int | Callable[[int], int]]
		The leaves whose combined domain to extract. Elements can be:
		- Integers for absolute leaf indices (e.g., `4`, `5`, `6`, `7`)
		- Callables that take `dimensionsTotal` and return a leaf index (e.g., `首二`, `首零二`)
	pathFilename : PurePath | None = None
		The output file path. If `None`, defaults to `tests/dataSamples/p2DnDomain{leafNames}.py`.
	settings : dict[str, dict[str, Any]] | None = None
		Settings for `writePython` formatter. If `None`, uses defaults.

	Returns
	-------
	pathFilename : PurePath
		The path where the module was written.

	"""
	def resolveLeaf(leafSpec: int | Callable[[int], int], dimensionsTotal: int) -> int:
		return leafSpec(dimensionsTotal) if callable(leafSpec) else leafSpec

	def getLeafName(leafSpec: int | Callable[[int], int]) -> str:
		return leafSpec.__name__ if callable(leafSpec) else str(leafSpec)

	listLeafNames: list[str] = [getLeafName(leafSpec) for leafSpec in listLeaves]
	filenameLeafPart: str = '_'.join(listLeafNames)

	if pathFilename is None:
		pathFilename = Path(f"{packageSettings.pathPackage}/tests/dataSamples/p2DnDomain{filenameLeafPart}.py")
	else:
		pathFilename = Path(pathFilename)

	dictionaryDomainsByDimensions: dict[int, list[tuple[int, ...]]] = {}

	for dimensionsTotal in listDimensions:
		mapShape: tuple[int, ...] = (2,) * dimensionsTotal
		state: EliminationState = EliminationState(mapShape)
		dataframeFoldings: pandas.DataFrame = _getDataFrameFoldings(state)

		listResolvedLeaves: list[int] = [resolveLeaf(leafSpec, dimensionsTotal) for leafSpec in listLeaves]

		listCombinedTuples: list[tuple[int, ...]] = []
		for indexRow in range(len(dataframeFoldings)):
			rowFolding: pandas.Series = dataframeFoldings.iloc[indexRow]
			tuplePiles: tuple[int, ...] = tuple(int(rowFolding[rowFolding == leaf].index[0]) for leaf in listResolvedLeaves)
			listCombinedTuples.append(tuplePiles)

		listUniqueTuples: list[tuple[int, ...]] = sorted(set(listCombinedTuples))
		dictionaryDomainsByDimensions[dimensionsTotal] = listUniqueTuples

	listPythonSource: list[str] = [
		'"""Verification data for combined leaf domains.',
		'',
		'This module contains empirically extracted combined domain data for leaves',
		f'{listLeafNames} across multiple mapShape configurations.',
		'',
		f'Each list is named `list2D{{dimensionsTotal}}Domain{filenameLeafPart}` where `dimensionsTotal`',
		'is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing',
		'valid pile positions for the specified leaves. The tuple order follows the original',
		'leaf argument order.',
		'"""',
		'',
	]

	for dimensionsTotal in sorted(dictionaryDomainsByDimensions):
		variableName: str = f"list2D{dimensionsTotal}Domain{filenameLeafPart}"
		listPythonSource.append(f'{variableName}: list[tuple[int, ...]] = {dictionaryDomainsByDimensions[dimensionsTotal]!r}')
		listPythonSource.append('')

	pythonSource: str = '\n'.join(listPythonSource)
	writePython(pythonSource, pathFilename, settings)

	return pathFilename

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

		if listMissing:
			print(f"  Missing ({countMissing} tuples in known but not in computed):")
			pprint(listMissing, width=140, compact=True)

		if listSurplus:
			print(f"  Surplus ({countSurplus} tuples in computed but not in known):")
			pprint(listSurplus, width=140, compact=True)

		if not listMissing and not listSurplus:
			print("  Perfect match!")

	return comparisonResults

def Z0Z_TESTexcluding2d5(state: EliminationState, leaf: int) -> bool:

	for leafExcluder in range(state.leavesTotal):
		excludingDictionary: dict[int, dict[int, list[int]]] | None = getExcludingDictionary(state, leafExcluder)
		if excludingDictionary is None:
			continue

		if leaf != leafExcluder and leafExcluder in state.pinnedLeaves.values():
			pileExcluder: int = reverseLookup(state.pinnedLeaves, leafExcluder)
			if pileExcluder in excludingDictionary:
				dictionaryIndicesPilesExcluded: dict[int, list[int]] = excludingDictionary[pileExcluder]
				if leaf in dictionaryIndicesPilesExcluded:
					listIndicesPilesExcluded: list[int] = dictionaryIndicesPilesExcluded[leaf]
					domainOfPilesForLeaf: list[int] = list(getLeafDomain(state, leaf))
					listPilesExcluded: list[int] = list(extract(domainOfPilesForLeaf, listIndicesPilesExcluded))
					if state.pile in listPilesExcluded:
						return True

	return False

def analyzeExclusions() -> None:
	"""Analyze.

	Analyze data, and write formulas to be processed by `getExcludingDictionary`.

	Make formulas to create `listIndicesPilesExcluded` as a function of `pileExcluder` and `leaf`.
	`pileExcluder` can be described by its index in `getLeafRange(state, leafExcluder)`. I wonder if the index is a better input variable.
	"""
	dictionaryExclusionsWIP: dict[int, dict[int, dict[int, list[int]]]] = {}

	for leafExcluder in [二+零, 三+零, 三, 四+零, 四+一, 四+三, 四+二, 一+零]:
	# for leafExcluder in [一+零]:
		for dimensions in [6]:
			state = EliminationState((2,) * dimensions)

			dictionaryByPileExcluderFromLeafToPilesExcluded: dict[int, dict[int, list[int]]] = {}
			dictionaryByPileExcluderFromLeafToIndicesPilesExcluded: dict[int, dict[int, list[int]]] = {}

			domainOfPilesForLeafExcluder: list[int] = list(getLeafDomain(state, leafExcluder))

			for indexDomainPileExcluder, pileExcluder in enumerate(domainOfPilesForLeafExcluder):
				dictionaryByPileExcluderFromLeafToPilesExcluded[indexDomainPileExcluder] = {leaf: [pileExcluder] for leaf in range(2, state.leavesTotal)}
				del dictionaryByPileExcluderFromLeafToPilesExcluded[indexDomainPileExcluder][leafExcluder]
				del dictionaryByPileExcluderFromLeafToPilesExcluded[indexDomainPileExcluder][首零(state.dimensionsTotal)]
				for pileTarget in range(2, state.pileLast):
					if pileTarget == pileExcluder:
						continue
					dictionaryExcludedLeaves: dict[int, list[int]] = getExcludedLeaves(state, pileTarget, (pileExcluder,)) # pyright: ignore[reportAssignmentType]
					for leaf in dictionaryExcludedLeaves[leafExcluder]:
						if leaf == leafExcluder:
							continue
						beans, cornbread = 一+零, 一
						if (beans == leafExcluder and cornbread == leaf) or (cornbread == leafExcluder and beans == leaf):
							continue
						beans, cornbread = 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)
						if (beans == leafExcluder and cornbread == leaf) or (cornbread == leafExcluder and beans == leaf):
							continue
						dictionaryByPileExcluderFromLeafToPilesExcluded[indexDomainPileExcluder].setdefault(leaf, []).append(pileTarget)

			for indexDomainPileExcluder, dictionaryLeafToPilesExcluded in dictionaryByPileExcluderFromLeafToPilesExcluded.items():
				dictionaryByPileExcluderFromLeafToIndicesPilesExcluded[indexDomainPileExcluder] = {}
				for leaf, listPilesExcluded in dictionaryLeafToPilesExcluded.items():
					setPilesExcluded: set[int] = set(listPilesExcluded)
					domainOfPilesForLeaf: list[int] = list(getLeafDomain(state, leaf))
					listIndicesPilesExcluded: list[int] = sorted([domainOfPilesForLeaf.index(pile) for pile in domainOfPilesForLeaf if pile in setPilesExcluded])
					dictionaryByPileExcluderFromLeafToIndicesPilesExcluded[indexDomainPileExcluder][leaf] = listIndicesPilesExcluded

			# pprint(dictionaryByPileExcluderFromLeafToIndicesPilesExcluded, width=240)

			pathFilename = Path("/apps/mapFolding/Z0Z_notes/analyzeExcluders2Dn.csv")
			with pathFilename.open('w', newline='') as writeStream:
				writerCSV = csv.writer(writeStream)
				listPiles: list[int] = list(range(首零(state.dimensionsTotal)))
				writerCSV.writerow(['leafExcluder', 'dimensions', 'pileExcluder', 'indexDomainPileExcluder', 'leaf', 'sizeDomainOfPilesForLeaf', *listPiles])
				for indexDomainPileExcluder, dictionaryLeafToIndicesPilesExcluded in dictionaryByPileExcluderFromLeafToIndicesPilesExcluded.items():
					for leaf, listIndicesPilesExcluded in dictionaryLeafToIndicesPilesExcluded.items():
						Z0Z_list: list[int | str] = [''] * 首零(state.dimensionsTotal)
						for index in listIndicesPilesExcluded:
							Z0Z_list[index] = index
						writerCSV.writerow([leafExcluder, dimensions, domainOfPilesForLeafExcluder[indexDomainPileExcluder], indexDomainPileExcluder, leaf, len(list(getLeafDomain(state, leaf))), *Z0Z_list])

			dictionaryExclusionsWIP[leafExcluder] = dictionaryByPileExcluderFromLeafToIndicesPilesExcluded

	pathFilename = Path('/apps/mapFolding/mapFolding/_e/pinning2DnData.py')
	# pathFilename.write_text(f"dictionaryExclusions = {dictionaryExclusionsWIP!r}\n")

if __name__ == '__main__':
	state = EliminationState((2,) * 6)
	# analyzeExclusions()

	# makeVerificationDataLeavesDomain([4,5,6], (首二, 首零二, 首零一二, 首一二))

	from mapFolding.tests.dataSamples import p2DnDomain3_2_首一_首零一

	# verifyDomainAgainstKnown(domainCombined, p2DnDomain3_2_首一_首零一.list2D6Domain3_2_首一_首零一)

