from typing import List
import importlib.util
from tests.conftest import *
import pytest

def test_algorithmSourceParallel(listDimensionsTestParallelization: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int], useAlgorithmSource: None) -> None:
	standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestParallelization)], countFolds, listDimensionsTestParallelization, None, 'maximum')

def test_algorithmSourceSequential(oeisID: str, useAlgorithmSource: None) -> None:
	n = settingsOEIS[oeisID]['valuesTestValidation'][-1]
	standardizedEqualTo(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

def test_aOFn_calculate_value(oeisID: str) -> None:
	for n in settingsOEIS[oeisID]['valuesTestValidation']:
		standardizedEqualTo(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

@pytest.mark.parametrize('pathFilenameTmpTesting', ['.py'], indirect=True)
def test_Z0Z_(listDimensionsTestCountFolds: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int], pathFilenameTmpTesting):
	from mapFolding.syntheticModules import numba_countSequential
	algorithmSource: ModuleType = numba_countSequential
	pathFilenameModule = writeJobNumba(listDimensionsTestCountFolds, algorithmSource, pathFilenameWriteJob=pathFilenameTmpTesting)

	# Import the module
	Don_Lapre_Road_to_Self_Improvement = importlib.util.spec_from_file_location("__main__", pathFilenameModule)
	if Don_Lapre_Road_to_Self_Improvement is None:
		raise ImportError(f"Failed to create module specification from {pathFilenameModule}")
	if Don_Lapre_Road_to_Self_Improvement.loader is None:
		raise ImportError(f"Failed to get loader for module {pathFilenameModule}")
	module = importlib.util.module_from_spec(Don_Lapre_Road_to_Self_Improvement)

	module.__name__ = "__main__"
	Don_Lapre_Road_to_Self_Improvement.loader.exec_module(module)

	pathFilenameFoldsTotal = getPathFilenameFoldsTotal(listDimensionsTestCountFolds)
	standardizedEqualTo(str(foldsTotalKnown[tuple(listDimensionsTestCountFolds)]), pathFilenameFoldsTotal.read_text().strip)
