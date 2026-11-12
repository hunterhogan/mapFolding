"""Albums of representative foldings from the insertion algorithm, `mapFolding.algorithms.insertion.doTheNeedful`.

Expected values and exceptions come from the targets' docstrings unless a comment names another basis.
"""
from __future__ import annotations

from functools import partial
from itertools import chain, repeat
from mapFolding import leafOrigin, pileOrigin
from mapFolding.algorithms import insertion
from mapFolding.algorithms.iff import foldingValid吗
from mapFolding.beDRY import getTotalLeaves
from mapFolding.kitFilesystem import readAlbum
from mapFolding.tests import assertEqualTo
from operator import itemgetter
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import Folding
	from pathlib import Path

# Reference for `expected`: the docstring of `mapFolding.kitFilesystem.makeFilenameFolds`, with the suffix `.album`.
# Accepted recording for `totalRepresentatives`, stored before the 2026-11 reorganization.
@pytest.mark.parametrize('CPUlimit', (1, 2))
@pytest.mark.parametrize('pathAlbum', (None,), indirect=True, ids=['albumAbsent'])
@pytest.mark.parametrize('mapShape, expected, totalRepresentatives, albumKnown', (
	pytest.param((1,), 'p1.album', 1, None, id='p1')
	, pytest.param((2,), 'p2.album', 1, (2,), id='p2')
	, pytest.param((1, 2), 'p1x2.album', 1, (1, 2), id='p1x2')
	, pytest.param((1, 5), 'p1x5.album', 5, (1, 5), id='p1x5')
	, pytest.param((5,), 'p5.album', 5, (1, 5), id='p5')
	, pytest.param((2, 2), 'p2x2.album', 1, (2, 2), id='p2x2')
	, pytest.param((2, 2, 2), 'p2x2x2.album', 6, (2, 2, 2), id='p2x2x2')
	, pytest.param((2, 3), 'p2x3.album', 5, (2, 3), id='p2x3')
	, pytest.param((2, 4), 'p2x4.album', 20, (2, 4), id='p2x4')
	, pytest.param((2, 5), 'p2x5.album', 99, (2, 5), id='p2x5')
	, pytest.param((2, 6), 'p2x6.album', 438, (2, 6), id='p2x6')
	, pytest.param((3, 2), 'p3x2.album', 5, None, id='p3x2')
	, pytest.param((3, 4), 'p3x4.album', 648, (3, 4), id='p3x4')
	, pytest.param((2, 3, 2), 'p2x3x2.album', 102, None, id='p2x3x2')
	, pytest.param((2, 2, 2, 2), 'p2x2x2x2.album', 144, (2, 2, 2, 2), id='p2x2x2x2')
	, pytest.param((3, 3), 'p3x3.album', 76, (3, 3), id='p3x3')
), indirect=['albumKnown'])
def test_insertion(pathAlbum: Path, albumKnown: tuple[Folding, ...], mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit, expected: str
	, totalRepresentatives: int) -> None:
	"""The album directory holds only the new album, whose distinct representatives are valid foldings of every leaf.

	By the docstring, each representative has the origin leaf at the origin pile, and leaf one precedes leaf two in it. A map of
	fewer than three leaves has the leaves of `(1, 2)` that exist. The new album contains every folding of the stored album of
	`mapShape` named by `albumKnown`.
	"""
	pathFilenameAlbum: Path = insertion.doTheNeedful(mapShape, CPUlimit)
	album: tuple[Folding, ...] = readAlbum(pathFilenameAlbum)
	totalLeaves: int = getTotalLeaves(mapShape)

	assertEqualTo(pathFilenameAlbum.name, expected, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(tuple(pathAlbum.iterdir()), (pathFilenameAlbum,), insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(len(set(album)), totalRepresentatives, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(len(album), totalRepresentatives, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(frozenset, album)), {frozenset(range(totalLeaves))}, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(itemgetter(pileOrigin), album)), {leafOrigin}, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(tuple, map(filter, repeat((1, 2).__contains__), album))), {(1, 2)[:totalLeaves - 1]}, insertion.doTheNeedful.__name__
		, mapShape, CPUlimit)
	assertEqualTo(all(map(partial(foldingValid吗, mapShape=mapShape), album)), True, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(albumKnown).difference(album), set(), insertion.doTheNeedful.__name__, mapShape, CPUlimit)

@pytest.mark.parametrize('CPUlimit', (1, 2))
@pytest.mark.parametrize('mapShape, pathAlbum, albumKnown, expected', (
	pytest.param((), None, None, ValueError, id='noDimensions')
	, pytest.param((-2, 3), None, None, ValueError, id='negativeDimension')
	, pytest.param((0,), None, None, ValueError, id='zeroLeaves')
	, pytest.param((2, 0), None, None, ValueError, id='p2x0')
	, pytest.param((2, 2, 2), (2, 2, 2), (2, 2, 2), FileExistsError, id='albumExists')
), indirect=['pathAlbum', 'albumKnown'])
def test_insertionError(pathAlbum: Path, albumKnown: tuple[Folding, ...], mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit
	, expected: type[Exception]) -> None:
	"""The call raises, and the album directory still holds exactly the stored album that `pathAlbum` copied into it."""
	with pytest.raises(expected):
		insertion.doTheNeedful(mapShape, CPUlimit)
	assertEqualTo(tuple(chain.from_iterable(map(readAlbum, pathAlbum.iterdir()))), albumKnown, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
