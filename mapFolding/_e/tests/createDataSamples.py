from __future__ import annotations

from hunterMakesPy import raiseIfNone
from hunterMakesPy.filesystemToolkit import writePython
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.kitFilesystem import getDataFrameFoldings
from mapFolding.theSSOT import settingsPackage
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence
	from typing import Any
	import pandas

def makeVerificationDataLeavesDomain(sequenceTotalDimensions: Sequence[int], boxOfLeaves: Sequence[int | Callable[[int], int]], pathFilename: PurePath | None = None, settings: dict[str, dict[str, Any]] | None = None) -> PurePath:
	"""Create a Python module containing combined domain data for multiple leaves across multiple map shapes.

	This function extracts the actual combined domain (the set of valid pile index tuples) for a group of leaves from pickled
	folding data. The data is used for verification in pytest tests comparing computed domains against empirical data.

	The combined domain is a set of tuples where each tuple represents the pile indices for the specified leaves in a valid
	folding. For example, if `boxOfLeaves` is `[4, 5, 6, 7]`, each tuple has 4 elements representing the pile where each of those
	leaves appears in a folding.

	Parameters
	----------
	sequenceTotalDimensions : Sequence[int]
		The dimension counts to process (e.g., `[4, 5, 6]` for 2^4, 2^5, 2^6 leaf maps).
	boxOfLeaves : Sequence[int | Callable[[int], int]]
		The leaves whose combined domain to extract. Elements can be:
		- Integers for absolute leaf indices (e.g., `4`, `5`, `6`, `7`)
		- Callables that take `totalDimensions` and return a leaf index (e.g., `首二`, `首零二`)
	pathFilename : PurePath | None = None
		The output file path. If `None`, defaults to `_e/tests/dataSamples/p2上nDimensionalDomain{leafNames}.py`.
	settings : dict[str, dict[str, Any]] | None = None
		Settings for `writePython` formatter. If `None`, uses defaults.

	Returns
	-------
	pathFilename : PurePath
		The path where the module was written.

	"""
	def resolveLeaf(leafSpec: int | Callable[[int], int], totalDimensions: int) -> int:
		return leafSpec(totalDimensions) if callable(leafSpec) else leafSpec  # ty: ignore[call-top-callable, invalid-return-type]

	def getLeafName(leafSpec: int | Callable[[int], int]) -> str:
		leafSpecName: str = str(leafSpec)
		if callable(leafSpec):
			leafSpecName = getattr(leafSpec, "__name__", leafSpecName)
		return leafSpecName

	boxOfLeafNames: list[str] = [getLeafName(leafSpec) for leafSpec in boxOfLeaves]
	filenameLeafPart: str = '_'.join(boxOfLeafNames)

	if pathFilename is None:
		pathFilename = Path(f"{settingsPackage.pathPackage}/_e/tests/dataSamples/p2上nDimensionalDomain{filenameLeafPart}.py")
	else:
		pathFilename = Path(pathFilename)

	dictionaryDomainsByDimensions: dict[int, list[tuple[int, ...]]] = {}

	for totalDimensions in sequenceTotalDimensions:
		mapShape: tuple[int, ...] = (2,) * totalDimensions
		state: EliminationState = EliminationState(mapShape)
		dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))

		boxOfResolvedLeaves: list[int] = [resolveLeaf(leafSpec, totalDimensions) for leafSpec in boxOfLeaves]

		boxOfCombinedTuples: list[tuple[int, ...]] = []
		for 次Folding in range(len(dataframeFoldings)):
			seriesFolding: pandas.Series = dataframeFoldings.iloc[次Folding]
			boxOfPiles: tuple[int, ...] = tuple(int(seriesFolding[seriesFolding == leaf].index[0]) for leaf in boxOfResolvedLeaves)
			boxOfCombinedTuples.append(boxOfPiles)

		boxOfUniqueTuples: list[tuple[int, ...]] = sorted(set(boxOfCombinedTuples))
		dictionaryDomainsByDimensions[totalDimensions] = boxOfUniqueTuples

	boxOfPythonSource: list[str] = [
		'"""Verification data for combined leaf domains.',
		'',
		'This module contains empirically extracted combined domain data for leaves',
		f'{boxOfLeafNames} across multiple map-shape configurations.',
		'',
		'Each list is named `boxOfDomain2上{totalDimensions}Dimensional` where `totalDimensions`',  # ruff: ignore[missing-f-string-syntax]
		'is the exponent in the 2^n-dimensional mapShape, and it contains tuples representing',
		'valid pile indices for the specified leaves. The tuple element sequence follows the original',
		'leaf argument order.',
		'"""',
		'',
	]

	for totalDimensions in sorted(dictionaryDomainsByDimensions):
		variableName: str = f"boxOfDomain2上{totalDimensions}Dimensional"
		boxOfPythonSource.extend((f'{variableName}: list[tuple[int, ...]] = {dictionaryDomainsByDimensions[totalDimensions]!r}', ''))

	pythonSource: str = '\n'.join(boxOfPythonSource)
	writePython(pythonSource, pathFilename, settings)

	return pathFilename
