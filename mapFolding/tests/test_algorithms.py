"""Count, enumerate, and synthesize map foldings with each algorithm and compare to known values.

Each test function targets one entry point: `countFolds`, `countFoldsSymmetric`, `countMeanders`,
`eliminateFolds`, `enumerateSequences`, `insertion.doTheNeedful`, or `makeJobNumba`. Where the entry
point selects an algorithm with `flow`, adding the new `flow` to the existing rows validates it.
"""
from __future__ import annotations

from functools import partial
from itertools import chain
from mapFolding import leafOrigin, pileOrigin
from mapFolding.algorithms import insertion
from mapFolding.algorithms.iff import foldingValid吗 as diagramValid吗
from mapFolding.basecamp import countFolds, countFoldsSymmetric, countMeanders, eliminateFolds, enumerateSequences
from mapFolding.beDRY import getTotalLeaves
from mapFolding.dataBaskets import StateElimination, StateMapFolding
from mapFolding.kitAST.numba.kitNumba import parametersNumbaDefault
from mapFolding.kitAST.numba.makeJob import makeJobNumba
from mapFolding.kitAST.RecipeJob import RecipeJobTheorem2
from mapFolding.kitFilesystem import readAlbum, readText
from mapFolding.oeis import getTotalFoldsKnown, makeMapShape
from mapFolding.pinIt import (
	pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零,
	pin首beans)
from mapFolding.synthesized.initializeState import transitionOnGroupsOfFolds
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.algorithmsData import (
	parametersCountFolds, parametersCountFoldsSymmetric, parametersCountMeanders, parametersEliminateFolds, parametersEliminateFoldsErrors)
from mapFolding.tests.dataSamples.sequenceAlbums import parametersEnumeration, parametersEnumerationErrors
from operator import itemgetter, lt, methodcaller
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
import pytest
import runpy

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction, ConcurrencyLimit
	from mapFolding.theTypes import Folding, OEISid
	from os import PathLike
	from typing import Literal, LiteralString

@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize('oeisID, n, flow, computationDivisions', parametersCountFolds)
def test_countFolds(oeisID: OEISid, n: int, flow: LiteralString, pathLikeWrite: PathLike[str] | None, CPUlimit: ConcurrencyLimit
	, computationDivisions: int | str | None, expected: int) -> None:
	"""Validate that different computational flows produce valid results."""
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	actual: int = countFolds(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit, computationDivisions=computationDivisions)
	assertEqualTo(actual, expected, countFolds.__name__, mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit, computationDivisions=computationDivisions)

@pytest.mark.parametrize('CPUlimit, computationDivisions, expected', (
	pytest.param(None, {'wrong': 'value'}, ValueError, id='computationDivisions-mapping')
	, pytest.param({'invalid': True}, 'cpu', TypeError, id='CPUlimit-mapping')
	, pytest.param(['weird'], 'cpu', TypeError, id='CPUlimit-list')
))
@pytest.mark.parametrize('flow', ('',))
@pytest.mark.parametrize('mapShape', (pytest.param((1, 3), id='A000136(3)'), pytest.param((2, 3), id='A001415(3)')))
def test_countFoldsError(mapShape: tuple[int, ...], flow: LiteralString, pathLikeWrite: PathLike[str] | None, CPUlimit: ConcurrencyLimit
	, computationDivisions: int | str | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		countFolds(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit, computationDivisions=computationDivisions)

@pytest.mark.filterwarnings('ignore::numba.core.errors.NumbaPendingDeprecationWarning')
@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize('oeisID', ('A007822',))
@pytest.mark.parametrize('n, flow', parametersCountFoldsSymmetric)
def test_countFoldsSymmetric(oeisID: OEISid, n: int, flow: LiteralString, pathLikeWrite: PathLike[str] | None, CPUlimit: ConcurrencyLimit
	, expected: int) -> None:
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	actual: int = countFoldsSymmetric(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit)
	assertEqualTo(actual, expected, countFoldsSymmetric.__name__, mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit)

@pytest.mark.parametrize('kind, oeisID', (pytest.param('semi', 'A000682', id='semi'), pytest.param('meanders', 'A005316', id='meanders')))
@pytest.mark.parametrize('flow, n', parametersCountMeanders)
def test_countMeanders(kind: LiteralString, n: int, flow: LiteralString, pathLikeWrite: PathLike[str] | None, oeisID: OEISid, expected: int) -> None:
	"""Verify Meanders OEIS sequence value calculations against known reference values."""
	actual: int = countMeanders(kind, n, flow, pathLikeWrite).totalMeanders
	assertEqualTo(actual, expected, countMeanders.__name__, kind, n, flow, pathLikeWrite)

@pytest.mark.parametrize('oeisID, n, flow, CPUlimit', parametersEliminateFolds)
def test_eliminateFolds(oeisID: OEISid, n: int, pathLikeWrite: PathLike[str] | None, CPUlimit: ConcurrencyLimit, flow: str, expected: int) -> None:
	"""Validate `eliminateFolds` and different flows produce valid results."""
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	actual: int = eliminateFolds(mapShape, None, pathLikeWrite, CPUlimit=CPUlimit, flow=flow)
	assertEqualTo(actual, expected, eliminateFolds.__name__, mapShape, None, pathLikeWrite, CPUlimit=CPUlimit, flow=flow)

@pytest.mark.parametrize('oeisID, n, flow, CPUlimit, expected', parametersEliminateFoldsErrors)
def test_eliminateFoldsError(oeisID: OEISid, n: int, pathLikeWrite: PathLike[str] | None, CPUlimit: ConcurrencyLimit, flow: str
	, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		eliminateFolds(makeMapShape(oeisID, n), None, pathLikeWrite, CPUlimit=CPUlimit, flow=flow)

@pytest.mark.parametrize('function', (pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPile零Ante首零, pin首beans))
# @pytest.mark.parametrize("n", [4, 5], ids=lambda n: f"2^{n}-dimensional")
@pytest.mark.parametrize('n', (pytest.param(4, id='2^4-dimensional'),))
@pytest.mark.parametrize('oeisID', ('A001417',))
@pytest.mark.parametrize('flow', ('crease',))
# @pytest.mark.parametrize("flow", ["crease", "constraintPropagation"])
def test_eliminateFoldsPinnedState(function: CallableFunction[..., StateElimination], oeisID: OEISid, n: int, CPUlimit: ConcurrencyLimit, flow: str
	, expected: int) -> None:
	"""Validate `eliminateFolds` after applying state-only pinning functions to `A001417`."""
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	statePinned: StateElimination = function(StateElimination(mapShape), CPUlimit=CPUlimit)
	actual: int = eliminateFolds(mapShape, statePinned, None, CPUlimit=CPUlimit, flow=flow)
	assertEqualTo(actual, expected, eliminateFolds.__name__, function.__name__, mapShape, CPUlimit=CPUlimit, flow=flow)

@pytest.mark.parametrize('pileDepth', (pytest.param(2, id='pileDepth=2'), pytest.param(3, id='pileDepth=3'), pytest.param(4, id='pileDepth=4')))
# @pytest.mark.parametrize("n", [4, 5], ids=lambda n: f"2^{n}-dimensional")
@pytest.mark.parametrize('n', (pytest.param(4, id='2^4-dimensional'),))
@pytest.mark.parametrize('oeisID', ('A001417',))
@pytest.mark.parametrize('flow', ('crease',))
# @pytest.mark.parametrize("flow", ["crease", "constraintPropagation"])
def test_eliminateFoldsPinPilesAtEnds(pileDepth: int, oeisID: OEISid, n: int, CPUlimit: ConcurrencyLimit, flow: str, expected: int) -> None:
	"""Validate `eliminateFolds` after applying `pinPilesAtEnds` with several pile depths."""
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	statePinned: StateElimination = pinPilesAtEnds(StateElimination(mapShape), pileDepth, CPUlimit=CPUlimit)
	actual: int = eliminateFolds(mapShape, statePinned, None, CPUlimit=CPUlimit, flow=flow)
	assertEqualTo(actual, expected, eliminateFolds.__name__, pinPilesAtEnds.__name__, mapShape, CPUlimit=CPUlimit, flow=flow, pileDepth=pileDepth)

@pytest.mark.parametrize('suffix', ('.album', '.csv'))
@pytest.mark.parametrize('pathLikeWrite', (None, Path(), Path('nested/sequences.album')), indirect=True)
@pytest.mark.parametrize('n, kind, flow, CPUlimit, oeisID', parametersEnumeration)
def test_enumerateSequences(
	n: int, kind: LiteralString | Literal['stamps', 'stamps blank', 'meanders', 'meanders symmetric', 'semi', 'semi-meanders symmetric'] | None,
	flow: Literal['permutations', 'bilateral', 'bilateralConcurrent'] | None,
	pathLikeWrite: Path | None, *, CPUlimit: ConcurrencyLimit, suffix: str, expected: int, pathRootJobDEFAULT: Path
) -> None:
	actual: list[Folding] | int = enumerateSequences(n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
	if isinstance(actual, int):
		assertEqualTo(actual, expected, enumerateSequences.__name__, n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
	else:
		assertEqualTo(len(actual), expected, enumerateSequences.__name__, n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
		assertEqualTo(len(set(actual)), expected, enumerateSequences.__name__, n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
	if pathLikeWrite is None:
		assertEqualTo(tuple(pathRootJobDEFAULT.iterdir()), (), enumerateSequences.__name__, n, kind, flow, pathLikeWrite)
	else:
		(pathFilenameAlbum,) = filter(Path.is_file, pathRootJobDEFAULT.rglob('*'))
		if isinstance(actual, int):
			assertEqualTo(readText(pathFilenameAlbum), str(expected), enumerateSequences.__name__, n, kind, flow, pathLikeWrite)
		elif n == 0 and kind == 'meanders':
			assertEqualTo(readText(pathFilenameAlbum), '\n', enumerateSequences.__name__, n, kind, flow, pathLikeWrite)
		else:
			assertEqualTo(readAlbum(pathFilenameAlbum), tuple(actual), enumerateSequences.__name__, n, kind, flow, pathLikeWrite)

@pytest.mark.parametrize('suffix', ('.album',))
@pytest.mark.parametrize('n, kind, flow, pathFilename, CPUlimit, expected', parametersEnumerationErrors, indirect=['pathFilename'])
def test_enumerateSequencesError(
	n: int, kind: LiteralString | Literal['stamps', 'stamps blank', 'meanders', 'meanders symmetric', 'semi', 'semi-meanders symmetric'] | None,
	flow: Literal['permutations', 'bilateral', 'bilateralConcurrent'] | None,
	pathFilename: Path, *, CPUlimit: ConcurrencyLimit, suffix: str, expected: type[Exception]
) -> None:
	with pytest.raises(expected):
		enumerateSequences(n, kind, flow, pathFilename, CPUlimit=CPUlimit, suffix=suffix)

@pytest.mark.parametrize('CPUlimit', (1, 2))
@pytest.mark.parametrize('mapShape, expected, totalRepresentatives, albumKnown', (
	pytest.param((1,), 'p1.album', 1, None, id='one-leaf')
	, pytest.param((2,), 'p2.album', 1, (2,), id='two-leaves')
	, pytest.param((1, 2), 'p1x2.album', 1, (1, 2), id='two-leaf-strip')
	, pytest.param((1, 5), 'p1x5.album', 5, (1, 5), id='strip')
	, pytest.param((5,), 'p5.album', 5, (1, 5), id='single-dimension')
	, pytest.param((2, 2), 'p2x2.album', 1, (2, 2), id='A001417-2')
	, pytest.param((2, 2, 2), 'p2x2x2.album', 6, (2, 2, 2), id='A001417-3')
	, pytest.param((2, 3), 'p2x3.album', 5, None, id='A001415')
	, pytest.param((3, 2), 'p3x2.album', 5, None, id='A001416')
	, pytest.param((2, 3, 2), 'p2x3x2.album', 102, None, id='mixed-three-dimensions')
	, pytest.param((2, 2, 2, 2), 'p2x2x2x2.album', 144, (2, 2, 2, 2), id='A001417-4')
	, pytest.param((3, 3), 'p3x3.album', 76, None, id='A001418')
), indirect=['albumKnown'])
def test_insertion(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit, expected: str, totalRepresentatives: int, albumKnown: tuple[Folding, ...]
	, pathAlbum: Path) -> None:
	pathFilenameAlbum: Path = insertion.doTheNeedful(mapShape, CPUlimit)
	album: tuple[Folding, ...] = readAlbum(pathFilenameAlbum)
	totalLeaves: int = getTotalLeaves(mapShape)

	assertEqualTo(pathFilenameAlbum.name, expected, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(tuple(pathAlbum.iterdir()), (pathFilenameAlbum,), insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(len(album), totalRepresentatives, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(len(set(album)), len(album), insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(len, album)), {totalLeaves}, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(frozenset, album)), {frozenset(range(totalLeaves))}, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(itemgetter(pileOrigin), album)), {leafOrigin}, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	if 3 <= totalLeaves:
		assertEqualTo(all(map(lt, map(methodcaller('index', 1), album), map(methodcaller('index', 2), album), strict=True)), True, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(all(map(partial(diagramValid吗, mapShape=mapShape), album)), True, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(albumKnown).difference(album), set(), insertion.doTheNeedful.__name__, mapShape, CPUlimit)

@pytest.mark.parametrize('CPUlimit', (1, 2))
@pytest.mark.parametrize('mapShape, expected, pathAlbum, albumKnown', (
	pytest.param((), ValueError, None, None, id='no-dimensions')
	, pytest.param((-2, 3), ValueError, None, None, id='negative-dimension')
	, pytest.param((0,), ValueError, None, None, id='zero-leaves')
	, pytest.param((2, 0), ValueError, None, None, id='empty-map')
	, pytest.param((2, 2, 2), FileExistsError, (2, 2, 2), (2, 2, 2), id='existing-album')
), indirect=['pathAlbum', 'albumKnown'])
def test_insertionError(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit, expected: type[Exception], albumKnown: tuple[Folding, ...]
	, pathAlbum: Path) -> None:
	with pytest.raises(expected):
		insertion.doTheNeedful(mapShape, CPUlimit)
	assertEqualTo(tuple(chain.from_iterable(map(readAlbum, pathAlbum.iterdir()))), albumKnown, insertion.doTheNeedful.__name__, mapShape, CPUlimit)

@pytest.mark.parametrize('mapShape', [pytest.param((2, 4), id='p2x4')])
@pytest.mark.parametrize('useNumbaProgressBar', (False, True))
def test_makeJobNumba(mapShape: tuple[int, ...], useNumbaProgressBar: bool, pathRootJobDEFAULT: Path) -> None:
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
	pathRootJobDEFAULT : Path
		The pytest-managed job directory.

	"""
	state: StateMapFolding = transitionOnGroupsOfFolds(StateMapFolding(mapShape))

	pathFilenameModule: Path = pathRootJobDEFAULT / 'jobNumba.py'
	pathFilenameTotalFolds: Path = pathFilenameModule.with_suffix('.totalFoldsTesting')

	recipeJobTheorem2 = RecipeJobTheorem2(state, pathModule=PurePosixPath(pathFilenameModule.parent), identifierModule=pathFilenameModule.stem
		, pathFilenameTotalFolds=PurePosixPath(pathFilenameTotalFolds), totalFoldsMultiplier=state.totalLeaves)
	makeJobNumba(recipeJobTheorem2, useNumbaProgressBar=useNumbaProgressBar, parametersNumba=parametersNumbaDefault)

	runpy.run_path(str(pathFilenameModule), run_name='__main__')

	expected: str = str(getTotalFoldsKnown(mapShape))
	actual: str = readText(pathFilenameTotalFolds).strip()
	assertEqualTo(actual, expected, readText.__name__, pathFilenameTotalFolds)
