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

from mapFolding import countFolds, getFoldsTotalKnown, oeisIDfor_n
from mapFolding.dataBaskets import MapFoldingState
from mapFolding.oeis import dictionaryOEIS
from mapFolding.someAssemblyRequired.makeAllModules import parametersNumbaLight
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2
from mapFolding.syntheticModules.initializeCount import initializeGroupsOfFolds
from mapFolding.tests.conftest import registrarRecordsTemporaryFilesystemObject, standardizedEqualToCallableReturn
from pathlib import Path, PurePosixPath
from typing import Literal
import importlib.util
import multiprocessing
import pytest

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

@pytest.mark.parametrize('flow', ['daoOfMapFolding', 'theorem2', 'theorem2Trimmed', 'theorem2numba'])
def test_flowControl(mapShapeTestCountFolds: tuple[int, ...], flow: Literal['daoOfMapFolding', 'theorem2', 'theorem2numba']) -> None:
	"""Validate that different computational flows produce identical results.

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
	flow : Literal['daoOfMapFolding', 'theorem2', 'theorem2numba']
		The computational flow algorithm to validate.

	"""
	standardizedEqualToCallableReturn(getFoldsTotalKnown(mapShapeTestCountFolds), countFolds, None, None, None, None, mapShapeTestCountFolds, None, None, flow)

def test_aOFn_calculate_value(oeisID: str) -> None:
	"""Verify OEIS sequence value calculations against known reference values.

	(AI generated docstring)

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
	for n in dictionaryOEIS[oeisID]['valuesTestValidation']:
		standardizedEqualToCallableReturn(dictionaryOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

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
	state = initializeGroupsOfFolds(MapFoldingState(mapShape))

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
