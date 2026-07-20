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

from __future__ import annotations

from hunterMakesPy import raiseIfNone
from itertools import product as CartesianProduct
from mapFolding.basecamp import countFolds
from mapFolding.dataBaskets import MapFoldingState
from mapFolding.oeis import countingMeanders, dictionaryOEIS, dictionaryOEISMapFolding, getFoldsTotalKnown, oeisIDfor_n
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2
from mapFolding.someAssemblyRequired.toolkitNumba import parametersNumbaLight
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
from mapFolding.tests import assertEqualTo, messageTestFailure
from mapFolding.tests.conftest import registrarRecordsTemporaryFilesystemObject
from numba.core.errors import NumbaPendingDeprecationWarning
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
import importlib.util
import pytest
import warnings

if TYPE_CHECKING:
	from importlib.machinery import ModuleSpec
	from mapFolding import MetadataOEISid, MetadataOEISidMapFolding
	from os import PathLike
	from types import ModuleType

@pytest.mark.parametrize(
	'oeisID, n, flow, CPUlimit'
	, [
		pytest.param('A007822', 4, 'algorithm', 0.5, id='algorithm')
		, pytest.param('A007822', 4, 'asynchronous', 0.5, id='asynchronous')
		, pytest.param('A007822', 4, 'theorem2', 0.5, id='theorem2')
		, pytest.param('A007822', 4, 'theorem2Numba', 0.5, id='theorem2Numba')
		, pytest.param('A007822', 4, 'theorem2Trimmed', 0.5, id='theorem2Trimmed')
	]
)
def test_A007822(oeisID: str, n: int, flow: str, CPUlimit: float) -> None:
	"""Test A007822 flow options.

	Parameters
	----------
	oeisID : str
		OEIS identifier to validate.
	n : int
		Sequence index to validate.
	flow : str
		Computation flow to validate.
	CPUlimit : float
		CPU limit for the computation.

	"""
	pathLikeWriteFoldsTotal: PathLike[str] | None = None
	warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
	expected: int = dictionaryOEIS[oeisID]['valuesKnown'][n]
	actual: int = countingMeanders(oeisID, n, flow, pathLikeWriteFoldsTotal, CPUlimit=CPUlimit)
	assertEqualTo(actual, expected, countingMeanders.__name__, oeisID, n, flow, pathLikeWriteFoldsTotal, CPUlimit)

@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize('oeisID, n, flow'
	, [
		*[
			pytest.param(oeisID, n, flow, id=f'{oeisID}::n{n}::{flow}')
			for oeisID, nValues in (
				('A000136', (2, 3, 12))
				, ('A001415', (2, 3, 9))
				, ('A001416', (2, 4, 6))
				, ('A001417', (3, 5))
				, ('A001418', (4,))
				, ('A195646', (2,))
			)
			for n, flow in CartesianProduct(nValues, ('numba', 'theorem2Codon', 'theorem2Numba'))
		]
		, *[
			pytest.param(oeisID, n, flow, id=f'{oeisID}::n{n}::{flow}')
			for oeisID, nValues in (
				('A000136', (2, 3, 12))
				, ('A001415', (2, 3, 6))
				, ('A001416', (2, 4))
				, ('A001417', (3,))
				, ('A001418', (3,))
				, ('A195646', (1,))
			)
			for n, flow in CartesianProduct(nValues, ('daoOfMapFolding', 'theorem2', 'theorem2Trimmed'))
		]
	]
)
def test_countFolds(oeisID: str, n: int, flow: str, CPUlimit: float | None) -> None:
	"""Validate that different computational flows produce valid results.

	(AI generated docstring)

	This is the primary test for ensuring mathematical consistency across different
	algorithmic implementations. When adding a new computational approach, include
	it in the parametrized flow list to verify it produces correct results.

	The test compares the output of each flow against known correct values from
	OEIS sequences, ensuring that optimization techniques don't compromise accuracy.

	Parameters
	----------
	oeisID : str
		OEIS identifier to validate.
	n : int
		Sequence index to validate.
	flow : str
		Computation flow to validate.
	CPUlimit : float | None
		CPU limit for the computation.

	"""
	if flow == 'theorem2Codon' and importlib.util.find_spec('codon') is None:
		pytest.skip('codon-jit is not installed')

	mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)
	expected: int = dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]
	actual: int = countFolds(None, None, None, CPUlimit=CPUlimit, mapShape=mapShape, flow=flow)
	assertEqualTo(actual, expected, countFolds.__name__, None, None, None, CPUlimit, mapShape, flow)

@pytest.mark.parametrize('oeisID', ('A000682', 'A005316'))
@pytest.mark.parametrize('n, flow', (*CartesianProduct((2, 29), ('matrixNumPy', 'matrixPandas')), (3, 'matrixMeanders'), (10, 'matrixMeanders')))
def test_meanders(oeisID: str, n: int, flow: str) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values.

	Tests the functions in `mapFolding.algorithms.oeisIDbyFormula` by comparing their
	calculated output against known correct values from the OEIS database for Meanders IDs.

	Parameters
	----------
	oeisID : str
		OEIS identifier to validate.
	n : int
		Sequence index to validate.
	flow : str
		Computation flow to validate.

	"""
	dictionaryCurrent: dict[str, MetadataOEISidMapFolding] | dict[str, MetadataOEISid] = dictionaryOEISMapFolding if oeisID in dictionaryOEISMapFolding else dictionaryOEIS
	expected: int = dictionaryCurrent[oeisID]['valuesKnown'][n]
	actual: int = countingMeanders(oeisID, n, flow, None)
	assertEqualTo(actual, expected, countingMeanders.__name__, oeisID, n, flow, None)

@pytest.mark.parametrize(
	'oeisID, n'
	, [
		pytest.param('A000560', 3, id='A000560::n3')
		, pytest.param('A000682', 3, id='A000682::n3')
		, pytest.param('A001010', 3, id='A001010::n3')
		, pytest.param('A001011', 3, id='A001011::n3')
		, pytest.param('A005315', 3, id='A005315::n3')
		, pytest.param('A005316', 3, id='A005316::n3')
		, pytest.param('A007822', 3, id='A007822::n3')
		, pytest.param('A060206', 3, id='A060206::n3')
		, pytest.param('A077460', 3, id='A077460::n3')
		, pytest.param('A078591', 3, id='A078591::n3')
		, pytest.param('A086345', 3, id='A086345::n3')
		, pytest.param('A178961', 3, id='A178961::n3')
		, pytest.param('A223094', 3, id='A223094::n3')
		, pytest.param('A259702', 3, id='A259702::n3')
		, pytest.param('A301620', 3, id='A301620::n3')
	]
)
def test_countingMeanders(oeisID: str, n: int) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values.

	Tests the functions in `mapFolding.algorithms.oeisIDbyFormula` by comparing their
	calculated output against known correct values from the OEIS database for Meanders IDs.

	Parameters
	----------
	oeisID : str
		OEIS identifier to validate.
	n : int
		Sequence index to validate.

	"""
	dictionaryCurrent: dict[str, MetadataOEISidMapFolding] | dict[str, MetadataOEISid] = dictionaryOEISMapFolding if oeisID in dictionaryOEISMapFolding else dictionaryOEIS
	expected: int = dictionaryCurrent[oeisID]['valuesKnown'][n]
	actual: int = countingMeanders(oeisID, n, None, None)
	assertEqualTo(actual, expected, countingMeanders.__name__, oeisID, n, None, None)

@pytest.mark.parametrize(
	'oeisID, n'
	, [
		pytest.param('A000136', 3, id='A000136::n3')
		, pytest.param('A001415', 3, id='A001415::n3')
		, pytest.param('A001416', 3, id='A001416::n3')
		, pytest.param('A001417', 3, id='A001417::n3')
		, pytest.param('A001418', 3, id='A001418::n3')
		, pytest.param('A195646', 2, id='A195646::n2')
	]
)
def test_oeisIDfor_n(oeisID: str, n: int) -> None:
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
		OEIS identifier to validate.
	n : int
		Sequence index to validate.

	"""
	expected: int = dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]
	actual: int = oeisIDfor_n(oeisID, n)
	assertEqualTo(actual, expected, oeisIDfor_n.__name__, oeisID, n)

@pytest.fixture
def oneTestCuzTestsOverwritingTests() -> tuple[int, ...]:  # Note to AI: DO NOT CHANGE THIS IDENTIFIER NAME.
	"""Return one deterministic map shape suitable for code generation tests."""
	return dictionaryOEISMapFolding['A000136']['getMapShape'](3)

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
	from mapFolding.someAssemblyRequired.makeJobTheorem2Numba import makeJobNumba  # ruff:ignore[import-outside-top-level]
	from mapFolding.someAssemblyRequired.toolkitNumba import SpicesJobNumba  # ruff:ignore[import-outside-top-level]

	mapShape: tuple[int, ...] = oneTestCuzTestsOverwritingTests
	state: MapFoldingState = transitionOnGroupsOfFolds(MapFoldingState(mapShape))

	pathFilenameModule: Path = pathFilename_tmpTesting.absolute()
	pathFilenameFoldsTotal: Path = pathFilenameModule.with_suffix('.foldsTotalTesting')
	registrarRecordsTemporaryFilesystemObject(pathFilenameFoldsTotal)

	jobTest = RecipeJobTheorem2(state, pathModule=PurePosixPath(pathFilenameModule.parent), moduleIdentifier=pathFilenameModule.stem
		, pathFilenameFoldsTotal=PurePosixPath(pathFilenameFoldsTotal), foldsTotalMultiplier=state.leavesTotal)
	spices = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	makeJobNumba(jobTest, spices)

	Don_Lapre_Road_to_Self_Improvement: ModuleSpec = raiseIfNone(importlib.util.spec_from_file_location('__main__', pathFilenameModule))
	module: ModuleType = importlib.util.module_from_spec(Don_Lapre_Road_to_Self_Improvement)

	module.__name__ = '__main__'
	loader = Don_Lapre_Road_to_Self_Improvement.loader
	assert loader is not None, messageTestFailure(loader, 'a module loader', 'importlib.util.spec_from_file_location', '__main__', pathFilenameModule)
	loader.exec_module(module)

	expected: str = str(getFoldsTotalKnown(oneTestCuzTestsOverwritingTests))
	actual: str = pathFilenameFoldsTotal.read_text(encoding='utf-8').strip()
	assertEqualTo(actual, expected, 'Path.read_text', pathFilenameFoldsTotal, encoding='utf-8')
