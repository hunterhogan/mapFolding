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

from itertools import product as CartesianProduct
from mapFolding.basecamp import countFolds, countFoldsSymmetric, countMeanders, enumerateSequences
from mapFolding.dataBaskets import StateMapFolding
from mapFolding.kitAST.numba.kitNumba import parametersNumbaLight, SpicesJobNumba
from mapFolding.kitAST.numba.makeJob import makeJobNumba
from mapFolding.kitAST.RecipeJob import RecipeJobTheorem2
from mapFolding.kitFilesystem import readAlbum, writeAlbum
from mapFolding.oeis import getTotalFoldsKnown, getValuesKnown, makeMapShape
from mapFolding.synthesized.initializeState import transitionOnGroupsOfFolds
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.sequenceAlbums import albumSemi4, parametersEnumeration, parametersErrors
from numba.core.errors import NumbaPendingDeprecationWarning
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
import pytest
import runpy
import warnings

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import Folding, OEISid
	from os import PathLike
	from typing import Literal, LiteralString

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
	, [
		*[pytest.param(n, flow, id=f'{flow}, {n}') for n, flow in CartesianProduct((2, 7), ('numba', 'theorem2Numba', 'algorithmNumba'))],
		*[pytest.param(n, flow, id=f'{flow}, {n}') for n, flow in CartesianProduct((2, 5), ('algorithm', 'theorem2', 'theorem2Trimmed')
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

@pytest.mark.parametrize('n, flow, kind', (
	*[pytest.param(n, flow, kind, id=f'{flow}-{kind}-{n}') for flow, kind, n in CartesianProduct(
			('matrixMeanders', 'matrixPandas', 'matrixNumPy')
			, ('semi', 'meanders')
			, (1, 2, 3, 11, 20, 30))
	]
	, *[pytest.param(n, flow, kind, id=f'{flow}-{kind}-{n}') for flow, kind, n in CartesianProduct(
			('prune', 'pruneNumPy', 'prunePandas')
			, ('semi', 'meanders')
			, (1, 2, 3, 22, 23, 30))
	]
))
def test_meanders(kind: LiteralString, n: int, flow: LiteralString, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values."""
	monkeypatch.chdir(tmp_path)
	fml = {'semi': 'A000682', 'meanders': 'A005316'}
	expected: int = getValuesKnown(fml[kind])[n]
	state = countMeanders(kind, n, flow, None)
	actual: int = state.totalMeanders
	assertEqualTo(actual, expected, countMeanders.__name__, kind, n, flow, None)

@pytest.mark.parametrize('mapShape', [pytest.param((2, 4), id='p2x4')])
def test_writeJobNumba(mapShape: tuple[int, ...], pathRootJobDEFAULTTesting: Path) -> None:
	"""Test dynamic code generation and execution for computational modules.

	(AI generated docstring)

	This test validates the package's ability to generate, compile, and execute optimized
	computational code at runtime. It's essential for users working with the code synthesis features
	or implementing custom optimization strategies.

	The test creates a complete computational module, executes it, and verifies that the generated
	code produces mathematically correct results. This pattern can be adapted for testing other
	dynamically generated computational approaches.

	Parameters
	----------
	mapShape : tuple[int, ...]
		The map shape dimensions for testing code generation.
	pathRootJobDEFAULTTesting : Path
		The pytest-managed job directory.

	"""
	state: StateMapFolding = transitionOnGroupsOfFolds(StateMapFolding(mapShape))

	pathFilenameModule: Path = pathRootJobDEFAULTTesting / 'jobNumba.py'
	pathFilenameTotalFolds: Path = pathFilenameModule.with_suffix('.totalFoldsTesting')

	recipeJobTheorem2 = RecipeJobTheorem2(state, pathModule=PurePosixPath(pathFilenameModule.parent), identifierModule=pathFilenameModule.stem
		, pathFilenameTotalFolds=PurePosixPath(pathFilenameTotalFolds), totalFoldsMultiplier=state.totalLeaves)
	spicesJobNumba = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	makeJobNumba(recipeJobTheorem2, spicesJobNumba)

	runpy.run_path(str(pathFilenameModule), run_name='__main__')

	expected: str = str(getTotalFoldsKnown(mapShape))
	actual: str = pathFilenameTotalFolds.read_text(encoding='utf-8').strip()
	assertEqualTo(actual, expected, 'Path.read_text', pathFilenameTotalFolds, encoding='utf-8')

@pytest.mark.parametrize('suffix', ('.album', '.csv'))
@pytest.mark.parametrize('pathLikeWrite', (None, Path(), Path('nested/sequences.album')))
@pytest.mark.parametrize('n, kind, flow, CPUlimit, oeisID', parametersEnumeration)
def test_enumerateSequences(
	n: int, kind: LiteralString | Literal['stamps', 'stamps blank', 'meanders', 'meanders symmetric', 'semi', 'semi-meanders symmetric'] | None,
	flow: Literal['permutations', 'bilateral', 'bilateralConcurrent'] | None,
	pathLikeWrite: PathLike[str] | None, *, CPUlimit: ConcurrencyLimit, suffix: str, expected: int,
	pathRootJobDEFAULTTesting: Path
) -> None:
	if pathLikeWrite == Path():
		pathLikeWrite = pathRootJobDEFAULTTesting
	actual: list[Folding] | int = enumerateSequences(n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
	if flow in {'bilateral', 'bilateralConcurrent'}:
		assertEqualTo(actual, expected, enumerateSequences.__name__, n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
	elif isinstance(actual, list):
		assertEqualTo(len(actual), expected, enumerateSequences.__name__, n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
		assertEqualTo(len(set(actual)), expected, enumerateSequences.__name__, n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
	if pathLikeWrite is None:
		assertEqualTo(tuple(pathRootJobDEFAULTTesting.iterdir()), (), enumerateSequences.__name__, n, kind, flow, pathLikeWrite)
	else:
		pathFilenameAlbum: Path = (
			pathRootJobDEFAULTTesting / f"{kind or 'semi'}_{n}{suffix}"
			if pathLikeWrite == pathRootJobDEFAULTTesting else pathRootJobDEFAULTTesting / pathLikeWrite
		)
		if isinstance(actual, int):
			assertEqualTo(pathFilenameAlbum.read_text(encoding='utf-8'), str(expected), enumerateSequences.__name__, n, kind, flow, pathLikeWrite)
		elif n == 0 and kind == 'meanders':
			assertEqualTo(pathFilenameAlbum.read_text(encoding='utf-8'), '\n', enumerateSequences.__name__, n, kind, flow, pathLikeWrite)
		else:
			assertEqualTo(readAlbum(pathFilenameAlbum), tuple(actual), enumerateSequences.__name__, n, kind, flow, pathLikeWrite)

@pytest.mark.parametrize('suffix', ('.album',))
@pytest.mark.parametrize('n, kind, flow, pathLikeWrite, CPUlimit, expected', parametersErrors)
def test_enumerateSequencesError(
	n: int, kind: LiteralString | Literal['stamps', 'stamps blank', 'meanders', 'meanders symmetric', 'semi', 'semi-meanders symmetric'] | None,
	flow: Literal['permutations', 'bilateral', 'bilateralConcurrent'] | None,
	pathLikeWrite: PathLike[str] | None, *, CPUlimit: ConcurrencyLimit, suffix: str, expected: type[Exception],
	pathRootJobDEFAULTTesting: Path
) -> None:
	if pathLikeWrite is not None:
		pathLikeWrite = pathRootJobDEFAULTTesting / pathLikeWrite
		writeAlbum(albumSemi4, pathLikeWrite)
	with pytest.raises(expected):
		enumerateSequences(n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
	if pathLikeWrite is not None:
		assertEqualTo(readAlbum(Path(pathLikeWrite)), tuple(albumSemi4), enumerateSequences.__name__, n, kind, flow, pathLikeWrite)
