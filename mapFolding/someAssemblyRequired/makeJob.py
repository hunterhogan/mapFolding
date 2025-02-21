from collections.abc import Sequence
from mapFolding import getPathFilenameFoldsTotal, computationState, outfitCountFolds, getAlgorithmSource
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, overload
import pickle

@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[True] , **keywordArguments: str | None) -> Path: ...
@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[False] , **keywordArguments: str | None) -> computationState: ...
def makeStateJob(listDimensions: Sequence[int], *, writeJob: bool = True, **keywordArguments: Any | None) -> computationState | Path:
	"""
	Creates a computation state job for map folding calculations and optionally saves it to disk.

	This function initializes a computation state for map folding calculations based on the given dimensions,
	sets up the initial counting configuration, and can optionally save the state to a pickle file.

	Parameters
	----------
	listDimensions : Sequence[int]
		The dimensions of the map to be folded, typically as [height, width].
	writeJob : bool, optional
		If True, saves the computation state to disk. If False, returns the state object directly.
		Default is True.
	**keywordArguments : Optional[str]
		Additional keyword arguments to be passed to the outfitCountFolds function.

	Returns
	-------
	Union[computationState, Path]
		If writeJob is False, returns the computation state object.
		If writeJob is True, returns the Path object pointing to the saved state file.

	Notes
	-----
	The function creates necessary directories and saves the state as a pickle file
	when writeJob is True. The file is saved in a directory structure based on the map shape.
	"""

	stateUniversal: computationState = outfitCountFolds(listDimensions, **keywordArguments)

	moduleSource: ModuleType = getAlgorithmSource()
	moduleSource.countInitialize(stateUniversal['connectionGraph'], stateUniversal['gapsWhere'], stateUniversal['my'], stateUniversal['track'])

	if not writeJob:
		return stateUniversal

	pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal['mapShape'])
	suffix = pathFilenameChopChop.suffix
	pathJob = Path(str(pathFilenameChopChop)[0:-len(suffix)])
	pathJob.mkdir(parents=True, exist_ok=True)
	pathFilenameJob = pathJob / 'stateJob.pkl'

	pathFilenameJob.write_bytes(pickle.dumps(stateUniversal))
	return pathFilenameJob
