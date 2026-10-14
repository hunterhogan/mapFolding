from __future__ import annotations

from itertools import chain, repeat
from mapFolding._e.algorithms.insertion import pathAlbum
from mapFolding._e.algorithms.nishat import foldingValid吗
from mapFolding.kitFilesystem import makePathFilenameFolds, streamAlbum
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from mapFolding.theTypes import Folding

@pytest.mark.parametrize('folding, mapShape, expected', tuple(chain(
	zip(streamAlbum(makePathFilenameFolds((1, 2), pathAlbum, suffix='.album')), repeat((1, 2)), repeat(True), strict=False),
	zip(streamAlbum(makePathFilenameFolds((1, 5), pathAlbum, suffix='.album')), repeat((1, 5)), repeat(True), strict=False),
	zip(streamAlbum(makePathFilenameFolds((1, 8), pathAlbum, suffix='.album')), repeat((1, 8)), repeat(True), strict=False),
	zip(streamAlbum(makePathFilenameFolds((2, 2), pathAlbum, suffix='.album')), repeat((2, 2)), repeat(True), strict=False),
	zip(streamAlbum(makePathFilenameFolds((2, 2, 2), pathAlbum, suffix='.album')), repeat((2, 2, 2)), repeat(True), strict=False),
	zip(streamAlbum(makePathFilenameFolds((2, 2, 2, 2), pathAlbum, suffix='.album')), repeat((2, 2, 2, 2)), repeat(True), strict=False),
	zip(streamAlbum(makePathFilenameFolds((2, 2, 2, 2, 2), pathAlbum, suffix='.album')), repeat((2, 2, 2, 2, 2)), repeat(True), strict=False),
)))
def test_foldingValid吗(folding: Folding, mapShape: tuple[int, ...], expected: bool) -> None:
	foldingValid: bool = foldingValid吗(folding, mapShape)
	assert foldingValid is expected, f'{foldingValid=}, {expected=}, {folding=}, {mapShape=}.'
