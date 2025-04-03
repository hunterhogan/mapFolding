from mapFolding.basecamp import countFolds
from mapFolding.toolboxFilesystem import getPathFilenameFoldsTotal
from mapFolding.beDRY import validateListDimensions
from mapFolding.oeis import getFoldsTotalKnown
from mapFolding.oeis import settingsOEIS, oeisIDfor_n
from mapFolding.someAssemblyRequired.transformationTools import makeInitializedComputationState
from pathlib import Path, PurePosixPath
from tests.conftest import standardizedEqualToCallableReturn, registrarRecordsTmpObject
from types import ModuleType
import importlib.util
import multiprocessing
import threading
from copy import deepcopy
import pytest

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

def test_algorithmSourceParallel(listDimensionsTestParallelization: list[int], useAlgorithmSourceDispatcher: None) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(listDimensionsTestParallelization)), countFolds, listDimensionsTestParallelization, None, 'maximum', None)

def test_algorithmSourceSequential(listDimensionsTestCountFolds: tuple[int, ...], useAlgorithmSourceDispatcher: None) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(listDimensionsTestCountFolds)), countFolds, listDimensionsTestCountFolds)

def test_aOFn_calculate_value(oeisID: str) -> None:
	for n in settingsOEIS[oeisID]['valuesTestValidation']:
		standardizedEqualToCallableReturn(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

def test_syntheticParallel(syntheticDispatcherFixture: None, listDimensionsTestParallelization: list[int]):
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(listDimensionsTestParallelization)), countFolds, listDimensionsTestParallelization, None, 'maximum')

def test_syntheticSequential(syntheticDispatcherFixture: None, listDimensionsTestCountFolds: list[int]) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(listDimensionsTestCountFolds)), countFolds, listDimensionsTestCountFolds)

@pytest.mark.parametrize('pathFilenameTmpTesting', ['.py'], indirect=True)
def test_writeJobNumba(oneTestCuzTestsOverwritingTests: list[int], pathFilenameTmpTesting: Path) -> None:
	from mapFolding.someAssemblyRequired.toolboxNumba import RecipeJob, SpicesJobNumba
	from mapFolding.someAssemblyRequired.synthesizeNumbaJob import makeJobNumba
	mapShape = validateListDimensions(oneTestCuzTestsOverwritingTests)
	state = makeInitializedComputationState(mapShape)

	pathFilenameModule = pathFilenameTmpTesting.absolute()
	pathFilenameFoldsTotal = pathFilenameModule.with_suffix('.foldsTotalTesting')
	registrarRecordsTmpObject(pathFilenameFoldsTotal)

	jobTest = RecipeJob(state
						, pathModule=PurePosixPath(pathFilenameModule.parent)
						, moduleIdentifier=pathFilenameModule.stem
						, pathFilenameFoldsTotal=PurePosixPath(pathFilenameFoldsTotal))
	spices = SpicesJobNumba()
	makeJobNumba(jobTest, spices)

	Don_Lapre_Road_to_Self_Improvement = importlib.util.spec_from_file_location("__main__", pathFilenameModule)
	if Don_Lapre_Road_to_Self_Improvement is None:
		raise ImportError(f"Failed to create module specification from {pathFilenameModule}")
	if Don_Lapre_Road_to_Self_Improvement.loader is None:
		raise ImportError(f"Failed to get loader for module {pathFilenameModule}")
	module = importlib.util.module_from_spec(Don_Lapre_Road_to_Self_Improvement)

	module.__name__ = "__main__"
	Don_Lapre_Road_to_Self_Improvement.loader.exec_module(module)

	standardizedEqualToCallableReturn(str(getFoldsTotalKnown(mapShape)), pathFilenameFoldsTotal.read_text().strip)
