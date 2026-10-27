from __future__ import annotations

from functools import partial
from mapFolding import leafOrigin, pileOrigin
from mapFolding.algorithms import insertion
from mapFolding.algorithms.iff import foldingValid吗 as diagramValid吗
from mapFolding.beDRY import getTotalLeaves
from mapFolding.kitFilesystem import makePathFilenameFolds, readAlbum, writeAlbum
from mapFolding.tests import assertEqualTo
from mapFolding.theSSOT import settingsPackage
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import Folding


@pytest.mark.parametrize(
	'mapShape, expected, totalFoldsKnown, filenameKnown',
	[
		pytest.param((1,), Path('p1.album'), 1, None, id='one-leaf'),
		pytest.param((2,), Path('p2.album'), 2, 'p2.album', id='two-leaves'),
		pytest.param((1, 2), Path('p1x2.album'), 2, 'p1x2.album', id='two-leaf-strip'),
		pytest.param((1, 5), Path('p1x5.album'), 50, 'p1x5.album', id='strip'),
		pytest.param((5,), Path('p5.album'), 50, 'p1x5.album', id='single-dimension'),
		pytest.param((2, 2), Path('p2x2.album'), 8, 'p2x2.album', id='A001417-2'),
		pytest.param((2, 2, 2), Path('p2x2x2.album'), 96, 'p2x2x2.album', id='A001417-3'),
		pytest.param((2, 3), Path('p2x3.album'), 60, None, id='A001415'),
		pytest.param((3, 2), Path('p3x2.album'), 60, None, id='A001416'),
		pytest.param((2, 3, 2), Path('p2x3x2.album'), 2448, None, id='mixed-three-dimensions'),
		pytest.param((2, 2, 2, 2), Path('p2x2x2x2.album'), 4608, 'p2x2x2x2.album', id='A001417-4'),
		pytest.param((3, 3), Path('p3x3.album'), 1368, None, id='A001418'),
	],
)
@pytest.mark.parametrize('CPUlimit', [1, 2])
def test_doTheNeedful(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit, expected: Path, totalFoldsKnown: int,
	filenameKnown: str | None, path_tmpTesting: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	albumKnown: tuple[Folding, ...] | None = readAlbum(settingsPackage.pathAlbum / filenameKnown) if filenameKnown is not None else None
	monkeypatch.setattr(settingsPackage, 'pathAlbum', path_tmpTesting)
	expected = path_tmpTesting / expected
	pathFilenameAlbum: Path = insertion.doTheNeedful(mapShape, CPUlimit)
	album: tuple[Folding, ...] = readAlbum(pathFilenameAlbum)
	totalLeaves: int = getTotalLeaves(mapShape)
	totalRepresentatives: int = 1 if totalLeaves < 3 else totalFoldsKnown // (2 * totalLeaves)

	assertEqualTo(pathFilenameAlbum, expected, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(tuple(path_tmpTesting.iterdir()), (expected,), insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(len(album), totalRepresentatives, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(len(set(album)), len(album), insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(len, album)), {totalLeaves}, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(frozenset, album)), {frozenset(range(totalLeaves))}, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(set(map(itemgetter(pileOrigin), album)), {leafOrigin}, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	if 3 <= totalLeaves:
		assertEqualTo(tuple(filter(lambda diagram: diagram.index(1) >= diagram.index(2), album)), (), insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	assertEqualTo(all(map(partial(diagramValid吗, mapShape=mapShape), album)), True, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	if albumKnown is not None:
		assertEqualTo(set(albumKnown).difference(album), set(), insertion.doTheNeedful.__name__, mapShape, CPUlimit)


@pytest.mark.parametrize('mapShape, expected', [
	pytest.param((), ValueError, id='no-dimensions'),
	pytest.param((-2, 3), ValueError, id='negative-dimension'),
	pytest.param((0,), ValueError, id='zero-leaves'),
	pytest.param((2, 0), ValueError, id='empty-map'),
	pytest.param((2, 2, 2), FileExistsError, id='existing-album'),
])
@pytest.mark.parametrize('CPUlimit', [1, 2])
def test_doTheNeedfulError(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit, expected: type[Exception], path_tmpTesting: Path,
	monkeypatch: pytest.MonkeyPatch) -> None:
	albumKnown: tuple[Folding, ...] = readAlbum(settingsPackage.pathAlbum / 'p2x2x2.album')
	monkeypatch.setattr(settingsPackage, 'pathAlbum', path_tmpTesting)
	pathFilenameAlbum: Path = makePathFilenameFolds(mapShape, path_tmpTesting, suffix='.album')
	if expected is FileExistsError:
		writeAlbum(albumKnown, pathFilenameAlbum)
	with pytest.raises(expected):
		insertion.doTheNeedful(mapShape, CPUlimit)
	if expected is FileExistsError:
		assertEqualTo(readAlbum(pathFilenameAlbum), albumKnown, insertion.doTheNeedful.__name__, mapShape, CPUlimit)
	else:
		assertEqualTo(tuple(path_tmpTesting.iterdir()), (), insertion.doTheNeedful.__name__, mapShape, CPUlimit)
