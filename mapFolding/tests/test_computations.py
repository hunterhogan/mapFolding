"""Core computational verification and algorithm validation tests.

(AI generated docstring)

This module validates the mathematical correctness of map folding computations and
serves as the primary testing ground for new computational approaches. It's the most
important module for users who create custom folding algorithms or modify existing ones.

The tests here verify that different computational flows produce identical results,
ensuring mathematical consistency across implementation strategies. This is critical
for maintaining confidence in results as the codebase evolves and new optimization
techniques are added.

Key Testing Areas:
- Flow control validation across different algorithmic approaches
- OEIS sequence value verification against known mathematical results
- Code generation and execution for dynamically created computational modules
- Numerical accuracy and consistency checks

For users implementing new computational methods: use the `test_flowControl` pattern
as a template. It demonstrates how to validate that your algorithm produces results
consistent with the established mathematical foundation.

The `test_writeJobNumba` function shows how to test dynamically generated code,
which is useful if you're working with the code synthesis features of the package.
"""

from mapFolding import countFolds, dictionaryOEIS, dictionaryOEISMapFolding, getFoldsTotalKnown, oeisIDfor_n
from mapFolding.basecamp import NOTcountingFolds
from mapFolding.dataBaskets import MapFoldingState
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2
from mapFolding.someAssemblyRequired.toolkitNumba import parametersNumbaLight
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
from mapFolding.tests.conftest import registrarRecordsTemporaryFilesystemObject, standardizedEqualToCallableReturn
from numba.core.errors import NumbaPendingDeprecationWarning
from pathlib import Path, PurePosixPath
import importlib.util
import multiprocessing
import pytest
import warnings

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

@pytest.mark.parametrize('flow', ['elimination', 'eliminationParallel'])#,  'constraintPropagation']) # ortools requires pandas, which doesn't have Py 3.14 yet.
def test_A000136(flow: str) -> None:
	"""Test A000136 flow options.

	Parameters
	----------
	flow : str
		The computational flow algorithm to validate.

	"""
	oeisID = 'A000136'
	CPUlimit = .5
	oeis_n = 2
	for oeis_n in dictionaryOEISMapFolding[oeisID]['valuesTestValidation']:
		if oeis_n < 2:
			continue

		expected = dictionaryOEISMapFolding[oeisID]['valuesKnown'][oeis_n]

		standardizedEqualToCallableReturn(
			expected
			, NOTcountingFolds, oeisID, oeis_n, flow, CPUlimit
			)

@pytest.mark.parametrize('flow', ['algorithm', 'asynchronous',  'theorem2', 'theorem2Numba', 'theorem2Trimmed'])
def test_A007822(flow: str) -> None:
	"""Test A007822 flow options.

	Parameters
	----------
	flow : str
		The computational flow algorithm to validate.

	"""
	oeisID = 'A007822'
	CPUlimit = .5
	warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
	oeis_n = 2
	for oeis_n in dictionaryOEIS[oeisID]['valuesTestValidation']:
		if oeis_n < 2:
			continue

		expected = dictionaryOEIS[oeisID]['valuesKnown'][oeis_n]

		standardizedEqualToCallableReturn(
			expected
			, NOTcountingFolds, oeisID, oeis_n, flow, CPUlimit
			)

@pytest.mark.parametrize('flow', ['daoOfMapFolding', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'])
def test_countFolds(mapShapeTestCountFolds: tuple[int, ...], flow: str) -> None:
	"""Validate that different computational flows produce valid results.

	(AI generated docstring)

	This is the primary test for ensuring mathematical consistency across different
	algorithmic implementations. When adding a new computational approach, include
	it in the parametrized flow list to verify it produces correct results.

	The test compares the output of each flow against known correct values from
	OEIS sequences, ensuring that optimization techniques don't compromise accuracy.

	Parameters
	----------
	mapShapeTestCountFolds : tuple[int, ...]
		The map shape dimensions to test fold counting for.
	flow : str
		The computational flow algorithm to validate.

	"""
	standardizedEqualToCallableReturn(getFoldsTotalKnown(mapShapeTestCountFolds), countFolds, None, None, None, None, mapShapeTestCountFolds, flow)

@pytest.mark.parametrize('flow', ['matrixMeanders', 'matrixNumPy', 'matrixPandas'])
def test_meanders(oeisIDmeanders: str, flow: str) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values.

	Tests the functions in `mapFolding.algorithms.oeisIDbyFormula` by comparing their
	calculated output against known correct values from the OEIS database for Meanders IDs.

	Parameters
	----------
	oeisIDMeanders : str
		The Meanders OEIS sequence identifier to test calculations for.

	"""
	dictionary = dictionaryOEISMapFolding if oeisIDmeanders in dictionaryOEISMapFolding else dictionaryOEIS
	for n in dictionary[oeisIDmeanders]['valuesTestValidation']:
		standardizedEqualToCallableReturn(
			dictionary[oeisIDmeanders]['valuesKnown'][n]
			, NOTcountingFolds, oeisIDmeanders, n, flow, None
			)

def test_NOTcountingFolds(oeisIDother: str) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values.

	Tests the functions in `mapFolding.algorithms.oeisIDbyFormula` by comparing their
	calculated output against known correct values from the OEIS database for Meanders IDs.

	Parameters
	----------
	oeisIDMeanders : str
		The Meanders OEIS sequence identifier to test calculations for.

	"""
	dictionary = dictionaryOEISMapFolding if oeisIDother in dictionaryOEISMapFolding else dictionaryOEIS
	for n in dictionary[oeisIDother]['valuesTestValidation']:
		standardizedEqualToCallableReturn(
			dictionary[oeisIDother]['valuesKnown'][n]
			, NOTcountingFolds, oeisIDother, n, None, None
			)

def test_oeisIDfor_n(oeisIDmapFolding: str) -> None:
	"""Verify OEIS sequence value calculations against known reference values.

	Tests the `oeisIDfor_n` function by comparing its calculated output against
	known correct values from the OEIS database. This ensures that sequence
	value computations remain mathematically accurate across code changes.

	The test iterates through validation test cases defined in `settingsOEIS`
	for the given OEIS sequence identifier, verifying that each computed value
	matches its corresponding known reference value.

	Parameters
	----------
	oeisID : str
		The OEIS sequence identifier to test calculations for.

	"""
	for n in dictionaryOEISMapFolding[oeisIDmapFolding]['valuesTestValidation']:
		standardizedEqualToCallableReturn(dictionaryOEISMapFolding[oeisIDmapFolding]['valuesKnown'][n], oeisIDfor_n, oeisIDmapFolding, n)

@pytest.mark.parametrize('pathFilename_tmpTesting', ['.py'], indirect=True)
def test_writeJobNumba(oneTestCuzTestsOverwritingTests: tuple[int, ...], pathFilename_tmpTesting: Path) -> None:
	"""Test dynamic code generation and execution for computational modules.

	(AI generated docstring)

	This test validates the package's ability to generate, compile, and execute
	optimized computational code at runtime. It's essential for users working with
	the code synthesis features or implementing custom optimization strategies.

	The test creates a complete computational module, executes it, and verifies
	that the generated code produces mathematically correct results. This pattern
	can be adapted for testing other dynamically generated computational approaches.

	Parameters
	----------
	oneTestCuzTestsOverwritingTests : tuple[int, ...]
		The map shape dimensions for testing code generation.
	pathFilename_tmpTesting : Path
		The temporary file path for generated module testing.

	"""
	from mapFolding.someAssemblyRequired.makeJobTheorem2Numba import makeJobNumba  # noqa: PLC0415
	from mapFolding.someAssemblyRequired.toolkitNumba import SpicesJobNumba  # noqa: PLC0415
	mapShape = oneTestCuzTestsOverwritingTests
	state = transitionOnGroupsOfFolds(MapFoldingState(mapShape))

	pathFilenameModule = pathFilename_tmpTesting.absolute()
	pathFilenameFoldsTotal = pathFilenameModule.with_suffix('.foldsTotalTesting')
	registrarRecordsTemporaryFilesystemObject(pathFilenameFoldsTotal)

	jobTest = RecipeJobTheorem2(state
						, pathModule=PurePosixPath(pathFilenameModule.parent)
						, moduleIdentifier=pathFilenameModule.stem
						, pathFilenameFoldsTotal=PurePosixPath(pathFilenameFoldsTotal))
	spices = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	makeJobNumba(jobTest, spices)

	Don_Lapre_Road_to_Self_Improvement = importlib.util.spec_from_file_location("__main__", pathFilenameModule)
	if Don_Lapre_Road_to_Self_Improvement is None:
		message = f"Failed to create module specification from {pathFilenameModule}"
		raise ImportError(message)
	if Don_Lapre_Road_to_Self_Improvement.loader is None:
		message = f"Failed to get loader for module {pathFilenameModule}"
		raise ImportError(message)
	module = importlib.util.module_from_spec(Don_Lapre_Road_to_Self_Improvement)

	module.__name__ = "__main__"
	Don_Lapre_Road_to_Self_Improvement.loader.exec_module(module)

	standardizedEqualToCallableReturn(str(getFoldsTotalKnown(oneTestCuzTestsOverwritingTests)), pathFilenameFoldsTotal.read_text(encoding="utf-8").strip)
