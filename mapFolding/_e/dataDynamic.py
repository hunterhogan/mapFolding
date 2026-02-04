from collections.abc import Callable, Sequence
from cytoolz.dicttoolz import merge
from hunterMakesPy import raiseIfNone
from hunterMakesPy.filesystemToolkit import writePython
from mapFolding import ansiColorReset, ansiColors, packageSettings
from mapFolding._e import getDictionaryPileRanges, getPileRangeOfLeaves, JeanValjean, PermutationSpace
from mapFolding._e.dataBaskets import EliminationState
from pathlib import Path, PurePath
from typing import Any
import pandas
import sys

#======== Specialized tools ===============================

def getDataFrameFoldings(state: EliminationState) -> pandas.DataFrame | None:
	pathFilename: Path = Path(f'{packageSettings.pathPackage}/tests/dataSamples/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
	dataframeFoldings: pandas.DataFrame | None = None
	if pathFilename.exists():
		dataframeFoldings = pandas.DataFrame(pandas.read_pickle(pathFilename))  # noqa: S301
	else:
		message: str = f"{ansiColors.YellowOnBlack}I received {state.dimensionsTotal = }, but I could not find the data at:\n\t{pathFilename!r}.{ansiColorReset}"
		sys.stderr.write(message + '\n')
	return dataframeFoldings

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
		leafSpecName: str = str(leafSpec)
		if callable(leafSpec):
			leafSpecName = getattr(leafSpec, "__name__", leafSpecName)
		return leafSpecName

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
		dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))

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
		'Each list is named `listDomain2D{dimensionsTotal}` where `dimensionsTotal`',
		'is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing',
		'valid pile positions for the specified leaves. The tuple order follows the original',
		'leaf argument order.',
		'"""',
		'',
	]

	for dimensionsTotal in sorted(dictionaryDomainsByDimensions):
		variableName: str = f"listDomain2D{dimensionsTotal}"
		listPythonSource.append(f'{variableName}: list[tuple[int, ...]] = {dictionaryDomainsByDimensions[dimensionsTotal]!r}')
		listPythonSource.append('')

	pythonSource: str = '\n'.join(listPythonSource)
	writePython(pythonSource, pathFilename, settings)

	return pathFilename

# TODO Creation of `permutationSpace2上nDomainDefaults` could possibly be a function. To future proof the performance, I probably want to cache `permutationSpace2上nDomainDefaults`.
def addPileRangesOfLeaves(state: EliminationState) -> EliminationState:
	permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(JeanValjean(getPileRangeOfLeaves(state.leavesTotal, pileRangeOfLeaves)))
								for pile, pileRangeOfLeaves in getDictionaryPileRanges(state).items()}
	state.permutationSpace = merge(permutationSpace2上nDomainDefaults, state.permutationSpace)
	return state

