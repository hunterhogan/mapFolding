from __future__ import annotations

from mapFolding.dataStructures import completeTriangle, makeLookupDiagonal, sequenceToTriangle, triangleToSequence
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.triangleData import sequenceGaps, triangleDense, triangleGaps, triangleSparse
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping, Sequence

@pytest.mark.parametrize('triangle, 次diagonal, fromRight, rowLength, expected', (
	pytest.param(triangleDense, 1, True, int, {2: 13, 3: 23}, id='dense-right-partial'),
	pytest.param(triangleDense, 1, False, None, {2: 11, 3: 17, 5: 29}, id='dense-left'),
	pytest.param(triangleDense, 1, True, None, {2: 13, 3: 23, 5: 29}, id='stored-right'),
	pytest.param(triangleSparse, 1, True, int, {2: 13, 3: 23, 5: 31}, id='sparse-right'),
	pytest.param(triangleSparse, 2, True, int, {2: 11}, id='sparse-missing'),
	pytest.param(triangleSparse, 1, False, None, {2: 11, 3: 17}, id='sparse-left'),
	pytest.param(triangleSparse, 0, True, int, {}, id='zero-position'),
))
def test_makeLookupDiagonal(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], 次diagonal: int,
	fromRight: bool, rowLength: Callable[[int], int] | None, expected: dict[int, int]) -> None:
	assertEqualTo(makeLookupDiagonal(triangle, 次diagonal, fromRight=fromRight, rowLength=rowLength), expected,
		makeLookupDiagonal.__name__, triangle, 次diagonal, fromRight=fromRight, rowLength=rowLength)

@pytest.mark.parametrize('triangle, 次diagonal, fromRight, rowLength, expected', (
	pytest.param(triangleDense, -1, True, None, ValueError, id='negative-position'),
	pytest.param(triangleSparse, 1, True, None, TypeError, id='unknown-right-edge'),
))
def test_makeLookupDiagonalError(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], 次diagonal: int,
	fromRight: bool, rowLength: Callable[[int], int] | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		makeLookupDiagonal(triangle, 次diagonal, fromRight=fromRight, rowLength=rowLength)

@pytest.mark.parametrize('triangleTerms, rowLength, expected', (
	pytest.param(triangleSparse, int, {2: (11, 13)}, id='missing-cells'),
	pytest.param({2: {0: 11, 1: 13}}, int, {}, id='wrong-columns'),
	pytest.param({2: {2: 0, 1: 11}}, int, {2: (11, 0)}, id='zero-is-known'),
))
def test_completeTriangle(triangleTerms: Mapping[int, Mapping[int, int]], rowLength: Callable[[int], int],
	expected: dict[int, tuple[int, ...]]) -> None:
	assertEqualTo(completeTriangle(triangleTerms, rowLength=rowLength), expected, completeTriangle.__name__, triangleTerms)

@pytest.mark.parametrize('triangle, rowStart, offset, rowLength, expected', (
	pytest.param(triangleGaps, 1, 1, int, sequenceGaps, id='sparse-indices'),
	pytest.param({2: (11,), 3: (13, 17)}, 2, 7, None, {7: 11, 8: 13, 9: 17}, id='shifted-origin'),
))
def test_triangleToSequence(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], rowStart: int, offset: int,
	rowLength: Callable[[int], int] | None, expected: dict[int, int]) -> None:
	assertEqualTo(triangleToSequence(triangle, rowStart=rowStart, offset=offset, rowLength=rowLength), expected,
		triangleToSequence.__name__, triangle)

@pytest.mark.parametrize('triangle, rowStart, offset, rowLength, expected', (
	pytest.param({2: (11, 13, 17)}, 1, 1, int, ValueError, id='too-long'),
	pytest.param({2: {0: 11}}, 1, 1, int, ValueError, id='zero-column'),
	pytest.param({0: (11,)}, 1, 1, int, ValueError, id='before-origin'),
))
def test_triangleToSequenceError(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], rowStart: int, offset: int,
	rowLength: Callable[[int], int] | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		triangleToSequence(triangle, rowStart=rowStart, offset=offset, rowLength=rowLength)

@pytest.mark.parametrize('sequence, rowLengths, rowStart, offset, expected', (
	pytest.param(sequenceGaps, None, 1, 1, triangleGaps, id='missing-indices'),
	pytest.param({}, None, 1, 1, {}, id='empty'),
))
def test_sequenceToTriangle(sequence: Mapping[int, int], rowLengths: Iterable[int] | None, rowStart: int, offset: int,
	expected: dict[int, dict[int, int]]) -> None:
	assertEqualTo(sequenceToTriangle(sequence, rowLengths, rowStart, offset), expected, sequenceToTriangle.__name__, sequence)
