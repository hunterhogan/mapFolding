from collections.abc import Sequence
from mapFolding.beDRY import outfitCountFolds, validateListDimensions, ComputationState
from mapFolding.filesystem import getPathFilenameFoldsTotal
from mapFolding.theSSOT import getAlgorithmSource
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, overload
import pickle

@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[True], **keywordArguments: Any) -> Path: ...
@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[False], **keywordArguments: Any) -> ComputationState: ...
def makeStateJob(listDimensions: Sequence[int], *, writeJob: bool = True, **keywordArguments: Any) -> ComputationState | Path:
	"""
	Creates a computation state job for map folding calculations and optionally saves it to disk.

	This function initializes a computation state for map folding calculations based on the given dimensions,
	sets up the initial counting configuration, and can optionally save the state to a pickle file.

	Parameters:
		listDimensions: List of integers representing the dimensions of the map to be folded.
		writeJob (True): Whether to save the state to disk.
		**keywordArguments: Additional keyword arguments to pass to the computation state initialization.

	Returns:
		stateUniversal|pathFilenameJob: The computation state for the map folding calculations, or
			the path to the saved state file if writeJob is True.
	"""
	mapShape = validateListDimensions(listDimensions)
	stateUniversal = outfitCountFolds(mapShape, **keywordArguments)

	moduleSource: ModuleType = getAlgorithmSource()
	# TODO `countInitialize` is hardcoded
	stateUniversal: ComputationState = moduleSource.countInitialize(stateUniversal)

	if not writeJob:
		return stateUniversal

	pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal.mapShape, None)
	suffix = pathFilenameChopChop.suffix
	pathJob = Path(str(pathFilenameChopChop)[0:-len(suffix)])
	pathJob.mkdir(parents=True, exist_ok=True)
	pathFilenameJob = pathJob / 'stateJob.pkl'

	pathFilenameJob.write_bytes(pickle.dumps(stateUniversal))
	return pathFilenameJob
