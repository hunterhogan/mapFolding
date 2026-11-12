"""Count and enumerate foldings and meanders with each algorithm, and compare the results to OEIS.

Every count's expected value is a(n) of its OEIS sequence, which the `expected` fixture reads from the stored b-file, and every
exception comes from the targets' docstrings. Each `flow` that selects an algorithm is a dimension, so a `flow` added to a
parametrization meets every map there.
"""
from __future__ import annotations

from collections import Counter
from functools import partial
from importlib.util import find_spec
from itertools import chain, product as CartesianProduct
from mapFolding.basecamp import countFolds, countFoldsSymmetric, countMeanders, eliminateFolds, enumerateSequences
from mapFolding.dataBaskets import StateElimination
from mapFolding.kitFilesystem import readAlbum, readText
from mapFolding.oeis import makeMapShape
from mapFolding.pinIt import (
	pin3beans2, pinLeaf首零Plus零, pinLeavesDimension0, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension零,
	pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零, pin首beans)
from mapFolding.tests import assertEqualTo, messageTestFailure
from mapFolding.tests.dataSamples.albums import (
	albumMeanders3, albumMeanders4, albumMeandersSymmetric4, albumSemi4, albumSemi6, albumSemi7, albumSemiSymmetric4, albumSemiSymmetric7,
	albumStamps4, albumStampsBlank4, contentsAlbumSemi4)
from pathlib import Path
from typing import Literal, LiteralString, TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from _pytest.mark.structures import ParameterSet
	from collections.abc import Callable
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import Folding, OEISid

boxOfFlowsNumba: tuple[str, ...] = ('numba', 'theorem2Numba', 'daoOfMapFoldingNumba')
boxOfFlowsPython: tuple[str, ...] = ('daoOfMapFolding', 'theorem2', 'theorem2Trimmed')

# Each sequence starts at its smallest index above the offset. The Python flows run about a hundred times slower, so they stop sooner.
@pytest.mark.parametrize('oeisID, n, flow', tuple(chain(
	CartesianProduct(('A000136',), (2, 3, 12), boxOfFlowsNumba)
	, CartesianProduct(('A001415',), (2, 3, 9), boxOfFlowsNumba)
	, CartesianProduct(('A001416',), (2, 4, 6), boxOfFlowsNumba)
	, CartesianProduct(('A001417',), (3, 5), boxOfFlowsNumba)
	, CartesianProduct(('A001418',), (4,), boxOfFlowsNumba)
	, CartesianProduct(('A195646',), (2,), boxOfFlowsNumba)
	, CartesianProduct(('A000136',), (2, 3, 12), boxOfFlowsPython)
	, CartesianProduct(('A001415',), (2, 3, 6), boxOfFlowsPython)
	, CartesianProduct(('A001416',), (2, 4), boxOfFlowsPython)
	, CartesianProduct(('A001417',), (3,), boxOfFlowsPython)
	, CartesianProduct(('A001418',), (3,), boxOfFlowsPython)
	, CartesianProduct(('A195646',), (1,), boxOfFlowsPython)
)))
def test_countFolds(oeisID: OEISid, n: int, flow: LiteralString, expected: int) -> None:
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	assertEqualTo(countFolds(mapShape, flow), expected, countFolds.__name__, mapShape, flow)

@pytest.mark.parametrize('oeisID, n, computationDivisions', (('A001417', 5, 'maximum'), ('A001415', 6, 'cpu'), ('A001416', 4, 3)))
def test_countFoldsDivided(oeisID: OEISid, n: int, computationDivisions: int | str, expected: int) -> None:
	"""A task count above one selects the parallel implementation regardless of `flow`."""
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	assertEqualTo(countFolds(mapShape, computationDivisions=computationDivisions), expected, countFolds.__name__, mapShape
		, computationDivisions=computationDivisions)

parametersFoldsWritten: tuple[tuple[str, int], ...] = (('A000136', 5), ('A001415', 3))

boxOfPathLikeWrite: tuple[ParameterSet, ...] = (
	pytest.param('.', id='directoryExisting'), pytest.param('nested/sequences.album', id='pathFilenameRelative'))

@pytest.mark.parametrize('pathLikeWrite', boxOfPathLikeWrite, indirect=True)
@pytest.mark.parametrize('oeisID, n', parametersFoldsWritten)
def test_countFoldsWrite(pathRootJobDEFAULT: Path, oeisID: OEISid, n: int, pathLikeWrite: Path, expected: int) -> None:
	"""The only file in the job directory holds the count as decimal text."""
	countFolds(makeMapShape(oeisID, n), None, pathLikeWrite)
	assertEqualTo(tuple(map(readText, filter(Path.is_file, pathRootJobDEFAULT.rglob('*')))), (str(expected),), countFolds.__name__
		, makeMapShape(oeisID, n), None, pathLikeWrite)

@pytest.mark.parametrize('CPUlimit, computationDivisions, expected', (
	pytest.param(None, {'wrong': 'value'}, ValueError, id='computationDivisionsMapping')
	, pytest.param({'invalid': True}, 'cpu', TypeError, id='CPUlimitMapping')
	, pytest.param(['weird'], 'cpu', TypeError, id='CPUlimitList')
))
@pytest.mark.parametrize('mapShape', (pytest.param((1, 3), id='p1x3'), pytest.param((2, 3), id='p2x3')))
def test_countFoldsError(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit, computationDivisions: int | str | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		countFolds(mapShape, CPUlimit=CPUlimit, computationDivisions=computationDivisions)

@pytest.mark.filterwarnings('ignore::numba.core.errors.NumbaPendingDeprecationWarning')
@pytest.mark.parametrize('oeisID, n, flow', tuple(chain(
	CartesianProduct(('A007822',), (2, 7), ('numba', 'theorem2Numba', 'algorithmNumba'))
	, CartesianProduct(('A007822',), (2, 5), ('algorithm', 'theorem2', 'theorem2Trimmed'))
)))
def test_countFoldsSymmetric(oeisID: OEISid, n: int, flow: LiteralString, expected: int) -> None:
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	assertEqualTo(countFoldsSymmetric(mapShape, flow), expected, countFoldsSymmetric.__name__, mapShape, flow)

markDefectLengthened: pytest.MarkDecorator = pytest.mark.xfail(raises=UserWarning, strict=True
	, reason='`getTotalBuckets` allocates 32 rows of `dataframeAnalyzed` at boundary 27, `recordArcCodes` needs 33, and it warns when it lengthens the allocation.')

# The known defect marks one combination of `flow`, `n`, and `kind`, so the combinations are enumerated rather than stacked.
@pytest.mark.parametrize('flow, n, kind, oeisID', tuple(chain(
	CartesianProduct(('matrixMeanders', 'matrixPandas', 'matrixNumPy'), (1, 2, 3, 11, 20, 30), ('semi',), ('A000682',))
	, CartesianProduct(('matrixMeanders', 'matrixPandas', 'matrixNumPy'), (1, 2, 3, 11, 20, 30), ('meanders',), ('A005316',))
	, CartesianProduct(('prune', 'pruneNumPy'), (1, 2, 3, 22, 23, 30), ('semi',), ('A000682',))
	, CartesianProduct(('prunePandas',), (1, 2, 3, 22, 23), ('semi',), ('A000682',))
	, (pytest.param('prunePandas', 30, 'semi', 'A000682', marks=markDefectLengthened),)
	, CartesianProduct(('prune', 'pruneNumPy', 'prunePandas'), (1, 2, 3, 22, 23, 30), ('meanders',), ('A005316',))
)))
def test_countMeanders(kind: LiteralString, n: int, flow: LiteralString, oeisID: OEISid, expected: int) -> None:
	assertEqualTo(countMeanders(kind, n, flow).totalMeanders, expected, countMeanders.__name__, kind, n, flow)

markOrtoolsMissing: pytest.MarkDecorator = pytest.mark.skipif(find_spec('ortools') is None, reason='constraintPropagation requires ortools.')

@pytest.mark.parametrize('oeisID, n, flow, CPUlimit', (
	pytest.param('A001417', 4, 'crease', 0.99, id='A001417-4-crease')
	, pytest.param('A000136', 3, 'elimination', 0.99, id='A000136-3-elimination')
	, pytest.param('A001415', 3, 'elimination', 0.99, id='A001415-3-elimination')
	, pytest.param('A001416', 2, 'elimination', 0.99, id='A001416-2-elimination')
	, pytest.param('A001417', 3, 'elimination', 0.99, id='A001417-3-elimination')
	, pytest.param('A001418', 2, 'elimination', 0.99, id='A001418-2-elimination')
	, pytest.param('A195646', 1, 'elimination', 0.99, id='A195646-1-elimination')
	, pytest.param('A000136', 1, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A000136-1-constraintPropagation')
	, pytest.param('A000136', 2, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A000136-2-constraintPropagation')
	, pytest.param('A000136', 5, 'constraintPropagation', 0.99, marks=markOrtoolsMissing, id='A000136-5-constraintPropagation')
	, pytest.param('A001415', 0, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A001415-0-constraintPropagation')
	, pytest.param('A001415', 1, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A001415-1-constraintPropagation')
	, pytest.param('A001415', 5, 'constraintPropagation', 0.99, marks=markOrtoolsMissing, id='A001415-5-constraintPropagation')
	, pytest.param('A001416', 0, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A001416-0-constraintPropagation')
	, pytest.param('A001416', 1, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A001416-1-constraintPropagation')
	, pytest.param('A001416', 4, 'constraintPropagation', 0.99, marks=markOrtoolsMissing, id='A001416-4-constraintPropagation')
	, pytest.param('A001417', 1, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A001417-1-constraintPropagation')
	, pytest.param('A001417', 4, 'constraintPropagation', 0.99, marks=markOrtoolsMissing, id='A001417-4-constraintPropagation')
	, pytest.param('A001418', 1, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A001418-1-constraintPropagation')
	, pytest.param('A001418', 2, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A001418-2-constraintPropagation')
	, pytest.param('A001418', 3, 'constraintPropagation', 0.99, marks=markOrtoolsMissing, id='A001418-3-constraintPropagation')
	, pytest.param('A195646', 1, 'constraintPropagation', 1, marks=markOrtoolsMissing, id='A195646-1-constraintPropagation')
	, pytest.param('A195646', 2, 'constraintPropagation', 0.99, marks=markOrtoolsMissing, id='A195646-2-constraintPropagation')
))
def test_eliminateFolds(oeisID: OEISid, n: int, CPUlimit: ConcurrencyLimit, flow: LiteralString, expected: int) -> None:
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	assertEqualTo(eliminateFolds(mapShape, CPUlimit=CPUlimit, flow=flow), expected, eliminateFolds.__name__, mapShape, CPUlimit=CPUlimit, flow=flow)

@pytest.mark.parametrize('pathLikeWrite', boxOfPathLikeWrite, indirect=True)
@pytest.mark.parametrize('oeisID, n', parametersFoldsWritten)
def test_eliminateFoldsWrite(pathRootJobDEFAULT: Path, oeisID: OEISid, n: int, pathLikeWrite: Path, expected: int) -> None:
	"""The only file in the job directory holds the count as decimal text."""
	eliminateFolds(makeMapShape(oeisID, n), None, pathLikeWrite, CPUlimit=1, flow='elimination')
	assertEqualTo(tuple(map(readText, filter(Path.is_file, pathRootJobDEFAULT.rglob('*')))), (str(expected),), eliminateFolds.__name__
		, makeMapShape(oeisID, n), None, pathLikeWrite, CPUlimit=1, flow='elimination')

# `'crease'` requires `(2,) * n` for n >= 4.
@pytest.mark.parametrize('oeisID, n, flow, CPUlimit, expected', (
	pytest.param('A001415', 3, 'crease', 0.99, NotImplementedError, id='A001415-3-crease')
	, pytest.param('A001417', 3, 'crease', 0.99, NotImplementedError, id='A001417-3-crease')
	, pytest.param('A001417', 0, 'constraintPropagation', 1, ValueError, marks=markOrtoolsMissing, id='A001417-0-constraintPropagation')
	, pytest.param('A195646', 0, 'constraintPropagation', 1, ValueError, marks=markOrtoolsMissing, id='A195646-0-constraintPropagation')
))
def test_eliminateFoldsError(oeisID: OEISid, n: int, CPUlimit: ConcurrencyLimit, flow: LiteralString, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		eliminateFolds(makeMapShape(oeisID, n), CPUlimit=CPUlimit, flow=flow)

@pytest.mark.parametrize('oeisID, n', (('A001417', 4),))
@pytest.mark.parametrize('pin', (
	pytest.param(pinPilesAtEnds, id='pinPilesAtEnds')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=2), id='pinPilesAtEnds-pileDepth2')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=3), id='pinPilesAtEnds-pileDepth3')
	, pytest.param(pinPile零Ante首零, id='pinPile零Ante首零')
	, pytest.param(pinLeavesDimension0, id='pinLeavesDimension0')
	, pytest.param(pinLeaf首零Plus零, id='pinLeaf首零Plus零')
	, pytest.param(pinLeavesDimension零, id='pinLeavesDimension零')
	, pytest.param(pinLeavesDimension一, id='pinLeavesDimension一')
	, pytest.param(pinLeavesDimensions0零一, id='pinLeavesDimensions0零一')
	, pytest.param(pinLeavesDimension二, id='pinLeavesDimension二')
	, pytest.param(pinLeavesDimension首二, id='pinLeavesDimension首二')
	, pytest.param(pin3beans2, id='pin3beans2')
	, pytest.param(pin首beans, id='pin首beans')
))
def test_eliminateFoldsPinned(pin: Callable[..., StateElimination], oeisID: OEISid, n: int, expected: int) -> None:
	"""Relation: eliminating the foldings of a pinned state counts every folding of the map."""
	state: StateElimination = pin(StateElimination(makeMapShape(oeisID, n)), CPUlimit=0.99)
	assertEqualTo(eliminateFolds(state.mapShape, state, CPUlimit=0.99, flow='crease'), expected, eliminateFolds.__name__, 'pin', state.mapShape)

parametersEnumerateSequencesWritten: tuple[ParameterSet, ...] = (
	pytest.param(1, 'meanders', 'A005316', id='1-meanders')
	, pytest.param(3, 'meanders', 'A005316', id='3-meanders')
	, pytest.param(4, 'meanders', 'A005316', id='4-meanders')
	, pytest.param(0, 'meanders symmetric', 'A077055', id='0-meandersSymmetric')
	, pytest.param(1, 'meanders symmetric', 'A077055', id='1-meandersSymmetric')
	, pytest.param(4, 'meanders symmetric', 'A077055', id='4-meandersSymmetric')
	, pytest.param(1, None, 'A000682', id='1-None')
	, pytest.param(4, None, 'A000682', id='4-None')
	, pytest.param(2, 'semi', 'A000682', id='2-semi')
	, pytest.param(4, 'semi', 'A000682', id='4-semi')
	, pytest.param(6, 'semi', 'A000682', id='6-semi')
	, pytest.param(7, 'semi', 'A000682', id='7-semi')
	, pytest.param(2, 'semi-meanders symmetric', 'A000560', id='2-semiMeandersSymmetric')
	, pytest.param(4, 'semi-meanders symmetric', 'A000560', id='4-semiMeandersSymmetric')
	, pytest.param(7, 'semi-meanders symmetric', 'A000560', id='7-semiMeandersSymmetric')
	, pytest.param(1, 'stamps', 'A000136', id='1-stamps')
	, pytest.param(4, 'stamps', 'A000136', id='4-stamps')
	, pytest.param(1, 'stamps blank', 'A001011', id='1-stampsBlank')
	, pytest.param(4, 'stamps blank', 'A001011', id='4-stampsBlank')
)

# `readAlbum` skips empty records, so the album of zero meanders, which holds one empty folding, is not among the written albums.
@pytest.mark.parametrize('flow', (None, 'permutations'))
@pytest.mark.parametrize('n, kind, oeisID', (pytest.param(0, 'meanders', 'A005316', id='0-meanders'), *parametersEnumerateSequencesWritten))
def test_enumerateSequences(pathRootJobDEFAULT: Path, n: int, kind: LiteralString | None, flow: Literal['permutations'] | None, oeisID: OEISid, expected: int) -> None:
	"""The album has a(n) distinct foldings, and `pathLikeWrite=None` writes no file."""
	album: list[Folding] | int = enumerateSequences(n, kind, flow)
	assert isinstance(album, list), messageTestFailure(type(album), list, enumerateSequences.__name__, n, kind, flow)
	assertEqualTo(len(album), expected, enumerateSequences.__name__, n, kind, flow)
	assertEqualTo(len(set(album)), expected, enumerateSequences.__name__, n, kind, flow)
	assertEqualTo(tuple(pathRootJobDEFAULT.iterdir()), (), enumerateSequences.__name__, n, kind, flow)

@pytest.mark.parametrize('flow', (None, 'permutations'))
@pytest.mark.parametrize('n, kind, expected', (
	pytest.param(3, 'meanders', albumMeanders3, id='3-meanders')
	, pytest.param(4, 'meanders', albumMeanders4, id='4-meanders')
	, pytest.param(4, 'meanders symmetric', albumMeandersSymmetric4, id='4-meandersSymmetric')
	, pytest.param(4, 'semi', albumSemi4, id='4-semi')
	, pytest.param(6, 'semi', albumSemi6, id='6-semi')
	, pytest.param(7, 'semi', albumSemi7, id='7-semi')
	, pytest.param(4, 'semi-meanders symmetric', albumSemiSymmetric4, id='4-semiMeandersSymmetric')
	, pytest.param(7, 'semi-meanders symmetric', albumSemiSymmetric7, id='7-semiMeandersSymmetric')
	, pytest.param(4, 'stamps', albumStamps4, id='4-stamps')
	, pytest.param(4, 'stamps blank', albumStampsBlank4, id='4-stampsBlank')
))
def test_enumerateSequencesAlbum(n: int, kind: LiteralString, flow: Literal['permutations'] | None, expected: list[Folding]) -> None:
	"""The album holds the recorded foldings. The docstring promises no order, so the albums are compared as collections."""
	album: list[Folding] | int = enumerateSequences(n, kind, flow)
	assert isinstance(album, list), messageTestFailure(type(album), list, enumerateSequences.__name__, n, kind, flow)
	assertEqualTo(Counter(album), Counter(expected), enumerateSequences.__name__, n, kind, flow)

# The bilateral flows count only semi-meanders and symmetric semi-meanders.
parametersEnumerateSequencesBilateral: tuple[ParameterSet, ...] = (
	pytest.param(1, 'semi', 'A000682', id='1-semi')
	, pytest.param(2, 'semi', 'A000682', id='2-semi')
	, pytest.param(4, 'semi', 'A000682', id='4-semi')
	, pytest.param(7, 'semi', 'A000682', id='7-semi')
	, pytest.param(2, 'semi-meanders symmetric', 'A000560', id='2-semiMeandersSymmetric')
	, pytest.param(4, 'semi-meanders symmetric', 'A000560', id='4-semiMeandersSymmetric')
	, pytest.param(7, 'semi-meanders symmetric', 'A000560', id='7-semiMeandersSymmetric')
)

boxOfFlowsBilateral: tuple[ParameterSet, ...] = (
	pytest.param('bilateral', None, id='bilateral')
	, pytest.param('bilateralConcurrent', 1, id='bilateralConcurrent-1')
	, pytest.param('bilateralConcurrent', 2, id='bilateralConcurrent-2')
)

@pytest.mark.parametrize('flow, CPUlimit', boxOfFlowsBilateral)
@pytest.mark.parametrize('n, kind, oeisID', parametersEnumerateSequencesBilateral)
def test_enumerateSequencesBilateral(n: int, kind: LiteralString, flow: Literal['bilateral', 'bilateralConcurrent'], CPUlimit: ConcurrencyLimit, oeisID: OEISid, expected: int) -> None:
	assertEqualTo(enumerateSequences(n, kind, flow, CPUlimit=CPUlimit), expected, enumerateSequences.__name__, n, kind, flow, CPUlimit=CPUlimit)

@pytest.mark.parametrize('suffix', ('.album', '.csv'))
@pytest.mark.parametrize('pathLikeWrite', boxOfPathLikeWrite, indirect=True)
@pytest.mark.parametrize('flow', (None, 'permutations'))
@pytest.mark.parametrize('n, kind, oeisID', parametersEnumerateSequencesWritten)
def test_enumerateSequencesWriteAlbum(pathRootJobDEFAULT: Path, n: int, kind: LiteralString | None, flow: Literal['permutations'] | None, pathLikeWrite: Path, suffix: str, oeisID: OEISid) -> None:
	"""Relation: the only file in the job directory holds the returned album."""
	album: list[Folding] | int = enumerateSequences(n, kind, flow, pathLikeWrite, suffix=suffix)
	assert isinstance(album, list), messageTestFailure(type(album), list, enumerateSequences.__name__, n, kind, flow)
	assertEqualTo(tuple(map(readAlbum, filter(Path.is_file, pathRootJobDEFAULT.rglob('*')))), (tuple(album),), enumerateSequences.__name__
		, n, kind, flow, pathLikeWrite, suffix=suffix)

@pytest.mark.parametrize('suffix', ('.album', '.csv'))
@pytest.mark.parametrize('pathLikeWrite', boxOfPathLikeWrite, indirect=True)
@pytest.mark.parametrize('flow, CPUlimit', boxOfFlowsBilateral)
@pytest.mark.parametrize('n, kind, oeisID', parametersEnumerateSequencesBilateral)
def test_enumerateSequencesWriteCount(pathRootJobDEFAULT: Path, n: int, kind: LiteralString, flow: Literal['bilateral', 'bilateralConcurrent'], pathLikeWrite: Path, CPUlimit: ConcurrencyLimit, suffix: str
	, oeisID: OEISid, expected: int) -> None:
	"""The only file in the job directory holds the count as decimal text, whatever its suffix."""
	enumerateSequences(n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)
	assertEqualTo(tuple(map(readText, filter(Path.is_file, pathRootJobDEFAULT.rglob('*')))), (str(expected),), enumerateSequences.__name__
		, n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit, suffix=suffix)

@pytest.mark.parametrize('n, kind, flow, pathFilename, CPUlimit, expected', (
	pytest.param(6, 'unknown', 'permutations', None, None, KeyError, id='kindUnknown-permutations')
	, pytest.param(4, '', 'permutations', None, None, KeyError, id='kindEmpty-permutations')
	, pytest.param(4, 'semi', 'permutations', contentsAlbumSemi4, None, FileExistsError, id='albumExists-permutations')
	, pytest.param(4, 'semi', 'bilateral', contentsAlbumSemi4, None, FileExistsError, id='albumExists-bilateral')
	, pytest.param(4, 'semi', 'bilateralConcurrent', contentsAlbumSemi4, 2, FileExistsError, id='albumExists-bilateralConcurrent')
	, pytest.param(4, 'stamps', 'bilateral', None, None, KeyError, id='stamps-bilateral')
	, pytest.param(4, 'stamps blank', 'bilateral', None, None, KeyError, id='stampsBlank-bilateral')
	, pytest.param(4, 'meanders', 'bilateral', None, None, KeyError, id='meanders-bilateral')
	, pytest.param(4, 'meanders symmetric', 'bilateral', None, None, KeyError, id='meandersSymmetric-bilateral')
	, pytest.param(4, 'stamps', 'bilateralConcurrent', None, 2, KeyError, id='stamps-bilateralConcurrent')
	, pytest.param(4, 'stamps blank', 'bilateralConcurrent', None, 2, KeyError, id='stampsBlank-bilateralConcurrent')
	, pytest.param(4, 'meanders', 'bilateralConcurrent', None, 2, KeyError, id='meanders-bilateralConcurrent')
	, pytest.param(4, 'meanders symmetric', 'bilateralConcurrent', None, 2, KeyError, id='meandersSymmetric-bilateralConcurrent')
), indirect=['pathFilename'])
def test_enumerateSequencesError(n: int, kind: LiteralString, flow: Literal['permutations', 'bilateral', 'bilateralConcurrent'], pathFilename: Path, CPUlimit: ConcurrencyLimit, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		enumerateSequences(n, kind, flow, pathFilename, CPUlimit=CPUlimit)
