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
from mapFolding.basecamp import countFolds, countFoldsSymmetric, countMeanders
from mapFolding.dataBaskets import MapFoldingState
from mapFolding.oeis import getTotalFoldsKnown, getValuesKnown, makeMapShape, oeisIDfor_n
from mapFolding.someAssemblyRequired.numba.kitNumba import parametersNumbaLight, SpicesJobNumba
from mapFolding.someAssemblyRequired.numba.makeJobTheorem2Numba import makeJobNumba
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
from mapFolding.tests import assertEqualTo, messageTestFailure
from mapFolding.tests.conftest import registrarRecordsTemporaryFilesystemObject
from numba.core.errors import NumbaPendingDeprecationWarning
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
import importlib.util
import pytest
import warnings

if TYPE_CHECKING:
	from importlib.machinery import ModuleSpec
	from mapFolding.theTypes import OEISid, 形KeywordArgumentsCount
	from os import PathLike
	from pathlib import Path
	from types import ModuleType
	from typing import LiteralString

@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize('oeisID, n, flow'
	, [*[pytest.param(oeisID, n, flow, id=f'{flow},{oeisID}({n})') for oeisID, nValues in (
			('A000136', (2, 3, 12))
			, ('A001415', (2, 3, 9))
			, ('A001416', (2, 4, 6))
			, ('A001417', (3, 5))
			, ('A001418', (4,))
			, ('A195646', (2,))) for n, flow in CartesianProduct(nValues
				, ('numba', 'theorem2Numba')
		)]
		, *[pytest.param(oeisID, n, flow, id=f'{flow},{oeisID}({n})') for oeisID, nValues in (
			('A000136', (2, 3, 12))
			, ('A001415', (2, 3, 6))
			, ('A001416', (2, 4))
			, ('A001417', (3,))
			, ('A001418', (3,))
			, ('A195646', (1,))) for n, flow in CartesianProduct(nValues
				, ('daoOfMapFolding', 'theorem2', 'theorem2Trimmed')
)]])
def test_countFolds(oeisID: OEISid, n: int, flow: LiteralString, CPUlimit: float | None) -> None:
	"""Validate that different computational flows produce valid results."""
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	expected: int = getValuesKnown(oeisID)[n]
	actual: int = countFolds(mapShape, flow, CPUlimit=CPUlimit)
	assertEqualTo(actual, expected, countFolds.__name__, mapShape, flow)

@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize('n, flow'
	, [*[pytest.param(n, flow, id=f'{flow}, {n}') for n, flow in CartesianProduct((2, 7), ('numba', 'theorem2Numba'))]
	, *[pytest.param(n, flow, id=f'{flow}, {n}') for n, flow in CartesianProduct((2, 5), ('algorithm', 'theorem2', 'theorem2Trimmed')
)]])
def test_countFoldsSymmetric(n: int, flow: LiteralString, CPUlimit: float) -> None:
	"""Test foldsSymmetric flow options."""
	oeisID: LiteralString = 'A007822'
	pathLikeWrite: PathLike[str] | None = None
	warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	expected: int = getValuesKnown(oeisID)[n]
	actual: int = countFoldsSymmetric(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit)
	assertEqualTo(actual, expected, countFoldsSymmetric.__name__, n, flow)

# FIXME Run the numpy/pandas tests in series because they cause namespace problems.
@pytest.mark.parametrize('n, flow, kind', (
	(30, 'matrixNumPy', 'semi'), (3, 'matrixMeanders', 'meanders'), (20, 'matrixPandas', 'meanders'), (10, 'matrixMeanders', 'semi')
))
def test_meanders(kind: LiteralString, n: int, flow: LiteralString) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values."""
	fml = {'semi': 'A000682', 'meanders': 'A005316'}
	expected: int = getValuesKnown(fml[kind])[n]
	actual: int = countMeanders(kind, n, flow, None)
	assertEqualTo(actual, expected, countMeanders.__name__, kind, n, flow, None)

# TODO Make param that isn't stoopid.
@pytest.mark.parametrize(
	'oeisID, n, f, keywordArguments'
	, [
		pytest.param('A000136', 3, '', {'flow': 'daoOfMapFolding'}, id='A000136,countFolds')
		, pytest.param('A001415', 3, '', {'flow': 'daoOfMapFolding'}, id='A001415,countFolds')
		, pytest.param('A001416', 3, '', {'flow': 'daoOfMapFolding'}, id='A001416,countFolds')
		, pytest.param('A001417', 3, '', {'flow': 'daoOfMapFolding'}, id='A001417,countFolds')
		, pytest.param('A001418', 3, '', {'flow': 'daoOfMapFolding'}, id='A001418,countFolds')
		, pytest.param('A195646', 2, '', {'flow': 'daoOfMapFolding'}, id='A195646,countFolds')
		, pytest.param('A000682', 3, '', {'flow': 'matrixMeanders'}, id='A000682,countMeanders')
		, pytest.param('A005316', 3, '', {'flow': 'matrixMeanders'}, id='A005316,countMeanders')
		, pytest.param('A007822', 3, '', {'flow': 'algorithm'}, id='foldsSymmetric,countFoldsSymmetric')
	]
)
def test_oeisIDfor_n(oeisID: OEISid, n: int, f: LiteralString, keywordArguments: 形KeywordArgumentsCount) -> None:
	"""Verify OEIS sequence value calculations against known reference values."""
	expected: int = getValuesKnown(oeisID)[n]
	actual: int = oeisIDfor_n(oeisID, n, f, **keywordArguments)
	assertEqualTo(actual, expected, oeisIDfor_n.__name__, oeisID, n, f, **keywordArguments)

# TODO Make param that isn't stoopid.
@pytest.mark.parametrize(
	'oeisID, f'
	, [
		pytest.param('A000560', '', id='A000560')
		, pytest.param('A000136', 'A000682', id='A000136,A000682')
		, pytest.param('A000136', 'A000560', id='A000136,A000560')
		, pytest.param('A000682', 'A000560', id='A000682,A000560')
		, pytest.param('A000682', 'A301620', id='A000682,A301620')
		, pytest.param('A000682', 'A259689', id='A000682,A259689')
		, pytest.param('A000682', 'A000136', id='A000682,A000136')
		, pytest.param('A000682', 'A223094', id='A000682,A223094')
		, pytest.param('A001010', 'A000682 and A007822', id='A001010,A000682-and-A007822')
		, pytest.param('A001010', 'A001011 and A000136', id='A001010,A001011-and-A000136')
		, pytest.param('A223094', 'A000136 and A000682', id='A223094,A000136-and-A000682')
		, pytest.param('A223094', 'A223094 and A000682', id='A223094,A223094-and-A000682')
		, pytest.param('A223094', 'A000682', id='A223094,A000682')
		, pytest.param('A259689', '', id='A259689')
		, pytest.param('A001011', '', id='A001011')
		, pytest.param('A005315', '', id='A005315')
		, pytest.param('A060206', '', id='A060206')
		, pytest.param('A077460', '', id='A077460')
		, pytest.param('A078591', '', id='A078591')
		, pytest.param('A301620', '', id='A301620')
		, pytest.param('A301620', 'A259689', id='A301620,A259689')
	]
)
@pytest.mark.parametrize(
	'oeis_n'
	, [pytest.param(0, id='offset'), pytest.param(2, id='offsetPlus2'), pytest.param(5, id='offsetPlus5')]
	, indirect=True
)
def test_oeisIDfor_n_byFormula(oeisID: OEISid, oeis_n: int, f: LiteralString) -> None:
	expected: int = getValuesKnown(oeisID)[oeis_n]
	actual: int = oeisIDfor_n(oeisID, oeis_n, f=f)
	assertEqualTo(actual, expected, oeisIDfor_n.__name__, oeisID, oeis_n, f=f)

@pytest.fixture
def oneTestCuzTestsOverwritingTests() -> tuple[int, ...]:  # Note to AI: DO NOT CHANGE THIS IDENTIFIER NAME.
	"""Return one deterministic map shape suitable for code generation tests."""
	return makeMapShape('A000136', 3)

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
	mapShape: tuple[int, ...] = oneTestCuzTestsOverwritingTests
	state: MapFoldingState = transitionOnGroupsOfFolds(MapFoldingState(mapShape))

	pathFilenameModule: Path = pathFilename_tmpTesting.absolute()
	pathFilenameTotalFolds: Path = pathFilenameModule.with_suffix('.totalFoldsTesting')
	registrarRecordsTemporaryFilesystemObject(pathFilenameTotalFolds)

	jobTest = RecipeJobTheorem2(state, pathModule=PurePosixPath(pathFilenameModule.parent), moduleIdentifier=pathFilenameModule.stem
		, pathFilenameTotalFolds=PurePosixPath(pathFilenameTotalFolds), totalFoldsMultiplier=state.totalLeaves)
	spices = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	makeJobNumba(jobTest, spices)

	Don_Lapre_Road_to_Self_Improvement: ModuleSpec = raiseIfNone(importlib.util.spec_from_file_location('__main__', pathFilenameModule))
	module: ModuleType = importlib.util.module_from_spec(Don_Lapre_Road_to_Self_Improvement)

	module.__name__ = '__main__'
	loader = Don_Lapre_Road_to_Self_Improvement.loader
	assert loader is not None, messageTestFailure(loader, 'a module loader', 'importlib.util.spec_from_file_location', '__main__', pathFilenameModule)
	loader.exec_module(module)

	expected: str = str(getTotalFoldsKnown(oneTestCuzTestsOverwritingTests) or 0)
	actual: str = pathFilenameTotalFolds.read_text(encoding='utf-8').strip()
	assertEqualTo(actual, expected, 'Path.read_text', pathFilenameTotalFolds, encoding='utf-8')
