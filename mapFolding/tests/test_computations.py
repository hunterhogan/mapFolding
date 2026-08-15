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
from mapFolding.oeis import getTotalFoldsKnown, getValuesKnown, makeMapShape
from mapFolding.kitAST.numba.kitNumba import parametersNumbaLight, SpicesJobNumba
from mapFolding.kitAST.numba.makeJob import makeJobNumba
from mapFolding.kitAST.RecipeJob import RecipeJobTheorem2
from mapFolding.synthesized.initializeState import transitionOnGroupsOfFolds
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
	from mapFolding.theTypes import OEISid
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
				, ('numba', 'theorem2Numba', 'daoOfMapFoldingNumba')
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
	, [*[pytest.param(n, flow, id=f'{flow}, {n}') for n, flow in CartesianProduct((2, 7), ('numba', 'theorem2Numba', 'algorithmNumba'))]
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
