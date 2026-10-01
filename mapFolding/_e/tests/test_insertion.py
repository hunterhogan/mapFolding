from __future__ import annotations

from mapFolding._e.algorithms import insertion
from mapFolding._e.algorithms.iff import foldingValid吗
from mapFolding.beDRY import getTotalLeaves
from mapFolding.kitFilesystem import makePathFilenameFolds, readAlbum, writeAlbum
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.parametrize(
	'mapShape, expected',
	[
		pytest.param((1, 5), 50, id='strip'),
		pytest.param((5,), 50, id='single-dimension'),
		pytest.param((2, 3), 60, id='A001415'),
		pytest.param((3, 2), 60, id='A001416'),
		pytest.param((2, 3, 2), 2448, id='mixed-three-dimensions'),
		pytest.param((2, 2, 2, 2), 4608, id='A001417'),
		pytest.param((3, 3), 1368, id='A001418'),
	],
)
@pytest.mark.parametrize('CPUlimit', [1, 2])
def test_doTheNeedful(mapShape: tuple[int, ...], CPUlimit: int, expected: int, path_tmpTesting: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(insertion, 'pathAlbum', path_tmpTesting)
	pathFilenameAlbum: Path = insertion.doTheNeedful(mapShape, CPUlimit)
	album = readAlbum(pathFilenameAlbum)

	assert pathFilenameAlbum == makePathFilenameFolds(mapShape, path_tmpTesting, suffix='.album')
	assert list(path_tmpTesting.iterdir()) == [pathFilenameAlbum]
	assert len(album) * 2 * getTotalLeaves(mapShape) == expected
	assert len(set(album)) == len(album)
	assert next(filter(lambda folding: folding[0] != 0 or folding.index(1) >= folding.index(2) or not foldingValid吗(folding, mapShape), album), None) is None


@pytest.mark.parametrize('mapShape, expected', [
	pytest.param((2, 0), ValueError, id='empty-map'),
	pytest.param((2, 2), FileExistsError, id='existing-album'),
])
@pytest.mark.parametrize('CPUlimit', [1])
def test_doTheNeedfulError(mapShape: tuple[int, ...], CPUlimit: int, expected: type[Exception], path_tmpTesting: Path,
	monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(insertion, 'pathAlbum', path_tmpTesting)
	if expected is FileExistsError:
		writeAlbum(((0, 1, 3, 2),), makePathFilenameFolds(mapShape, path_tmpTesting, suffix='.album'))
	with pytest.raises(expected):
		insertion.doTheNeedful(mapShape, CPUlimit)
