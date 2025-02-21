from tests.conftest import *
import importlib.util
import pytest

def test_algorithmSourceParallel(listDimensionsTestParallelization: list[int], foldsTotalKnown: dict[tuple[int, ...], int], useAlgorithmSourceDispatcher: None) -> None:
	standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestParallelization)], countFolds, listDimensionsTestParallelization, None, 'maximum')

def test_algorithmSourceSequential(listDimensionsTestCountFolds: list[int], foldsTotalKnown: dict[tuple[int, ...], int], useAlgorithmSourceDispatcher: None) -> None:
	standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestCountFolds)], countFolds, listDimensionsTestCountFolds)

def test_aOFn_calculate_value(oeisID: str) -> None:
	for n in settingsOEIS[oeisID]['valuesTestValidation']:
		standardizedEqualTo(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

@pytest.mark.parametrize('pathFilenameTmpTesting', ['.py'], indirect=True)
def test_writeJobNumba(listDimensionsTestCountFolds: list[int], foldsTotalKnown: dict[tuple[int, ...], int], pathFilenameTmpTesting: Path) -> None:
	from mapFolding.syntheticModules import numbaCount
	algorithmSourceHARDCODED: ModuleType = numbaCount
	algorithmSource = algorithmSourceHARDCODED
	callableTargetHARDCODED = 'countSequential'
	callableTarget = callableTargetHARDCODED
	pathFilenameModule = writeJobNumba(listDimensionsTestCountFolds, algorithmSource, callableTarget, pathFilenameWriteJob=pathFilenameTmpTesting.absolute())

	Don_Lapre_Road_to_Self_Improvement = importlib.util.spec_from_file_location("__main__", pathFilenameModule)
	if Don_Lapre_Road_to_Self_Improvement is None:
		raise ImportError(f"Failed to create module specification from {pathFilenameModule}")
	if Don_Lapre_Road_to_Self_Improvement.loader is None:
		raise ImportError(f"Failed to get loader for module {pathFilenameModule}")
	module = importlib.util.module_from_spec(Don_Lapre_Road_to_Self_Improvement)

	module.__name__ = "__main__"
	Don_Lapre_Road_to_Self_Improvement.loader.exec_module(module)

	pathFilenameFoldsTotal = getPathFilenameFoldsTotal(listDimensionsTestCountFolds)
	registrarRecordsTmpObject(pathFilenameFoldsTotal)
	standardizedEqualTo(str(foldsTotalKnown[tuple(listDimensionsTestCountFolds)]), pathFilenameFoldsTotal.read_text().strip)

def test_syntheticParallel(syntheticDispatcherFixture, listDimensionsTestParallelization: list[int], foldsTotalKnown: dict[tuple[int, ...], int]):
	standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestParallelization)], countFolds, listDimensionsTestParallelization, None, 'maximum')

def test_syntheticSequential(syntheticDispatcherFixture, listDimensionsTestCountFolds: list[int], foldsTotalKnown: dict[tuple[int, ...], int]):
	standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestCountFolds)], countFolds, listDimensionsTestCountFolds)
