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

from mapFolding import (
	countFolds, dictionaryOEIS, dictionaryOEISMapFolding, eliminateFolds, getFoldsTotalKnown, oeisIDfor_n)
from mapFolding.basecamp import NOTcountingFolds
from mapFolding.dataBaskets import MapFoldingState
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2
from mapFolding.someAssemblyRequired.toolkitNumba import parametersNumbaLight
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
from mapFolding.tests.conftest import (
	mapShapeFromScenario, registrarRecordsTemporaryFilesystemObject, standardizedEqualToCallableReturn, TestScenario)
from numba.core.errors import NumbaPendingDeprecationWarning
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
import importlib.util
import multiprocessing
import pytest
import warnings

if TYPE_CHECKING:
	from importlib.machinery import ModuleSpec
	from mapFolding import MetadataOEISid, MetadataOEISidMapFolding

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

def test_A007822(scenarioA007822: TestScenario) -> None:
	"""Test A007822 flow options.

	Parameters
	----------
	a007822Scenario : TestScenario
		Scenario describing the OEIS index and flow to validate.

	"""
	oeisID: str = scenarioA007822.oeisID
	warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
	expected: int = dictionaryOEIS[oeisID]['valuesKnown'][scenarioA007822.n]
	standardizedEqualToCallableReturn(
		expected,
		NOTcountingFolds,
		oeisID,
		scenarioA007822.n,
		scenarioA007822.flow,
		scenarioA007822.CPUlimit,
	)

def test_countFolds(scenarioCountFolds: TestScenario) -> None:
	"""Validate that different computational flows produce valid results.

	(AI generated docstring)

	This is the primary test for ensuring mathematical consistency across different
	algorithmic implementations. When adding a new computational approach, include
	it in the parametrized flow list to verify it produces correct results.

	The test compares the output of each flow against known correct values from
	OEIS sequences, ensuring that optimization techniques don't compromise accuracy.

	Parameters
	----------
	countFoldsScenario : TestScenario
		Scenario describing the OEIS index and flow to validate.

	"""
	mapShape: tuple[int, ...] = mapShapeFromScenario(scenarioCountFolds)
	expected: int = dictionaryOEISMapFolding[scenarioCountFolds.oeisID]['valuesKnown'][scenarioCountFolds.n]
	standardizedEqualToCallableReturn(expected, countFolds, None, None, None, scenarioCountFolds.CPUlimit, mapShape, scenarioCountFolds.flow)

def test_eliminateFolds(scenarioEliminateFolds: TestScenario) -> None:
	"""Validate `eliminateFolds` and different flows produce valid results.

	Parameters
	----------
	eliminateFoldsScenario : TestScenario
		Scenario describing the OEIS index and flow to validate.
	"""
	mapShape: tuple[int, ...] = mapShapeFromScenario(scenarioEliminateFolds)
	state = None
	pathLikeWriteFoldsTotal: None = None
	expected: int = dictionaryOEISMapFolding[scenarioEliminateFolds.oeisID]['valuesKnown'][scenarioEliminateFolds.n]
	standardizedEqualToCallableReturn(expected, eliminateFolds, mapShape, state, pathLikeWriteFoldsTotal, scenarioEliminateFolds.CPUlimit, scenarioEliminateFolds.flow)

def test_meanders(scenarioMeanders: TestScenario) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values.

	Tests the functions in `mapFolding.algorithms.oeisIDbyFormula` by comparing their
	calculated output against known correct values from the OEIS database for Meanders IDs.

	Parameters
	----------
	meandersScenario : TestScenario
		Scenario describing the OEIS index and flow to validate.

	"""
	dictionaryCurrent: dict[str, MetadataOEISidMapFolding] | dict[str, MetadataOEISid] = dictionaryOEISMapFolding if scenarioMeanders.oeisID in dictionaryOEISMapFolding else dictionaryOEIS
	expected: int = dictionaryCurrent[scenarioMeanders.oeisID]['valuesKnown'][scenarioMeanders.n]
	standardizedEqualToCallableReturn(
		expected,
		NOTcountingFolds,
		scenarioMeanders.oeisID,
		scenarioMeanders.n,
		scenarioMeanders.flow,
		None,
	)

def test_NOTcountingFolds(scenarioOeisFormula: TestScenario) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values.

	Tests the functions in `mapFolding.algorithms.oeisIDbyFormula` by comparing their
	calculated output against known correct values from the OEIS database for Meanders IDs.

	Parameters
	----------
	formulaScenario : TestScenario
		Scenario describing the OEIS index evaluated via formula dispatch.

	"""
	dictionaryCurrent: dict[str, MetadataOEISidMapFolding] | dict[str, MetadataOEISid] = dictionaryOEISMapFolding if scenarioOeisFormula.oeisID in dictionaryOEISMapFolding else dictionaryOEIS
	expected: int = dictionaryCurrent[scenarioOeisFormula.oeisID]['valuesKnown'][scenarioOeisFormula.n]
	standardizedEqualToCallableReturn(
		expected,
		NOTcountingFolds,
		scenarioOeisFormula.oeisID,
		scenarioOeisFormula.n,
		scenarioOeisFormula.flow,
		None,
	)

def test_oeisIDfor_n(scenarioOeisValue: TestScenario) -> None:
	"""Verify OEIS sequence value calculations against known reference values.

	Tests the `oeisIDfor_n` function by comparing its calculated output against
	known correct values from the OEIS database. This ensures that sequence
	value computations remain mathematically accurate across code changes.

	The test iterates through validation test cases defined in `settingsOEIS`
	for the given OEIS sequence identifier, verifying that each computed value
	matches its corresponding known reference value.

	Parameters
	----------
	oeisValueScenario : TestScenario
		Scenario describing the OEIS index validated through the public interface.

	"""
	expected = dictionaryOEISMapFolding[scenarioOeisValue.oeisID]['valuesKnown'][scenarioOeisValue.n]
	standardizedEqualToCallableReturn(expected, oeisIDfor_n, scenarioOeisValue.oeisID, scenarioOeisValue.n)

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
	mapShape: tuple[int, ...] = oneTestCuzTestsOverwritingTests
	state: MapFoldingState = transitionOnGroupsOfFolds(MapFoldingState(mapShape))

	pathFilenameModule: Path = pathFilename_tmpTesting.absolute()
	pathFilenameFoldsTotal: Path = pathFilenameModule.with_suffix('.foldsTotalTesting')
	registrarRecordsTemporaryFilesystemObject(pathFilenameFoldsTotal)

	jobTest = RecipeJobTheorem2(state
						, pathModule=PurePosixPath(pathFilenameModule.parent)
						, moduleIdentifier=pathFilenameModule.stem
						, pathFilenameFoldsTotal=PurePosixPath(pathFilenameFoldsTotal))
	spices = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	makeJobNumba(jobTest, spices)

	Don_Lapre_Road_to_Self_Improvement: ModuleSpec | None = importlib.util.spec_from_file_location("__main__", pathFilenameModule)
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
