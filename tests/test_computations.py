from pathlib import Path
from tests.conftest import *
import importlib.util
import pytest

def test_algorithmSourceParallel(listDimensionsTestParallelization: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int], useAlgorithmSourceDispatcher: None) -> None:
	standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestParallelization)], countFolds, listDimensionsTestParallelization, None, 'maximum')

def test_algorithmSourceSequential(listDimensionsTestCountFolds: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int], useAlgorithmSourceDispatcher: None) -> None:
	standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestCountFolds)], countFolds, listDimensionsTestCountFolds)

def test_aOFn_calculate_value(oeisID: str) -> None:
	for n in settingsOEIS[oeisID]['valuesTestValidation']:
		standardizedEqualTo(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

@pytest.mark.parametrize('pathFilenameTmpTesting', ['.py'], indirect=True)
def test_writeJobNumba(listDimensionsTestCountFolds: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int], pathFilenameTmpTesting):
	from mapFolding.syntheticModules import numba_countSequential
	algorithmSourceHARDCODED: ModuleType = numba_countSequential
	algorithmSource = algorithmSourceHARDCODED
	pathFilenameModule = writeJobNumba(listDimensionsTestCountFolds, algorithmSource, pathFilenameWriteJob=pathFilenameTmpTesting)

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

# def test_makeFlowNumbaOptimized(pathTmpTesting: Path, useThisDispatcher):
# def test_makeFlowNumbaOptimized(useThisDispatcher):
# 	"""To get this to work:
# 	walk_up=True doesn't work on 3.10, so that has to go
# 	the _logical_ import must be in the logical path of the package
# 	fuck python
# 	"""
# 	listCallablesInlineHARDCODED: List[str] = ['countInitialize', 'countParallel', 'countSequential']
# 	listCallablesInline = listCallablesInlineHARDCODED
# 	callableDispatcher = True
# 	algorithmSource = None
# 	relativePathWrite = None
# 	# relativePathWrite = pathTmpTesting.absolute().relative_to(getPathPackage(), walk_up=True)
# 	formatFilenameWrite = "pytest_{callableTarget}"
# 	listSynthesizedModules: List[youOughtaKnow] = makeFlowNumbaOptimized(listCallablesInline, callableDispatcher, algorithmSource, relativePathWrite, formatFilenameWrite)
# 	for stuff in listSynthesizedModules:
# 		registrarRecordsTmpObject(stuff.pathFilenameForMe)
# 		if stuff.callableSynthesized not in listCallablesInline:
# 			dispatcherSynthetic: youOughtaKnow = stuff
# 	if not dispatcherSynthetic: raise FREAKOUT
# 		# dispatcherSynthetic: youOughtaKnow = next(filter(lambda x: x.callableSynthesized not in listCallablesInline, listSynthesizedModules))

# 	# Import the synthetic dispatcher module to get the callable
# 	dispatcherSpec = importlib.util.spec_from_file_location(
# 		dispatcherSynthetic.callableSynthesized,
# 		dispatcherSynthetic.pathFilenameForMe
# 	)
# 	if dispatcherSpec is None:
# 		raise ImportError(f"Failed to create module specification from {dispatcherSynthetic.pathFilenameForMe}")
# 	if dispatcherSpec.loader is None:
# 		raise ImportError(f"Failed to get loader for module {dispatcherSynthetic.pathFilenameForMe}")

# 	dispatcherModule = importlib.util.module_from_spec(dispatcherSpec)
# 	dispatcherSpec.loader.exec_module(dispatcherModule)
# 	callableDispatcherSynthetic = getattr(dispatcherModule, dispatcherSynthetic.callableSynthesized)

# 	useThisDispatcher(callableDispatcherSynthetic)

# 	def test_syntheticSequential(listDimensionsTestCountFolds: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int]):
# 		standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestCountFolds)], countFolds, listDimensionsTestCountFolds)

# 	def test_syntheticParallel(listDimensionsTestParallelization: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int]):
# 		standardizedEqualTo(foldsTotalKnown[tuple(listDimensionsTestParallelization)], countFolds, listDimensionsTestParallelization, None, 'maximum')
