from __future__ import annotations

from mapFolding.kitFilesystem import readText
from mapFolding.oeis import (
	getMetadata, getOEISids, getTotalFoldsKnown, getTriangle, getTriangleDiagonal, getValuesKnown, OEIS_for_n, oeisIDfor_n, updateTriangle,
	updateValuesKnown)
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.tests import assertEqualTo, messageTestFailure
from mapFolding.tests.dataSamples.oeisData import parametersOEISidFor_n
from mapFolding.tests.dataSamples.triangleData import contents_bFile, sequenceHalf, triangleHalf
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch
import pytest
import re as regex
import sys

if TYPE_CHECKING:
	from _pytest.capture import CaptureResult
	from collections.abc import Mapping
	from mapFolding.theTypes import OEISid, 形KeywordArgumentsCount, 形Triangle
	from typing import LiteralString

@pytest.mark.parametrize('oeisID, n, f, flow', parametersOEISidFor_n)
def test_oeisIDfor_n(oeisID: OEISid, n: int, f: LiteralString | None, flow: LiteralString | None, expected: int) -> None:
	keywordArgumentsCount: 形KeywordArgumentsCount = {}
	if flow is not None:
		keywordArgumentsCount['flow'] = flow
	assertEqualTo(oeisIDfor_n(oeisID, n, f, **keywordArgumentsCount), expected, oeisIDfor_n.__name__, oeisID, n, f, **keywordArgumentsCount)

@pytest.mark.parametrize('oeisID, n, f, keywordArgumentsCount, expected', (
	pytest.param('A000136', -5, None, {}, ValueError, id='negative-n')
	, pytest.param('A000136', 'foo', None, {}, ValueError, id='non-integer-string')
	, pytest.param('A000136', 1.5, None, {}, ValueError, id='non-integer-float')
	, pytest.param('A001418', 0, None, {}, ArithmeticError, id='below-offset')
	, pytest.param('A999999', 3, None, {}, KeyError, id='unknown-identifier')
	, pytest.param('A204352', 3, None, {}, ValueError, id='unimplemented-sequence')
))
def test_oeisIDfor_nError(oeisID: OEISid, n: int, f: LiteralString | None, keywordArgumentsCount: 形KeywordArgumentsCount, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		oeisIDfor_n(oeisID, n, f, **keywordArgumentsCount)

@pytest.mark.parametrize('oeisID, expected', (
	pytest.param('A001415', 'A001415', id='uppercase')
	, pytest.param('a001415', 'A001415', id='lowercase')
	, pytest.param(' A001415 ', 'A001415', id='surrounding-spaces')
	, pytest.param('\ta001415\n', 'A001415', id='surrounding-whitespace')
))
def test_formatOEISid(oeisID: OEISid, expected: OEISid) -> None:
	assertEqualTo(formatOEISid(oeisID), expected, formatOEISid.__name__, oeisID)

@pytest.mark.parametrize('mapShape, expected', (
	pytest.param((1, 3), 6, id='strip')
	, pytest.param((2, 3), 60, id='rectangle')
	, pytest.param((3, 3), 1368, id='square')
	, pytest.param((999, 999), None, id='unknown-shape')
))
def test_getTotalFoldsKnown(mapShape: tuple[int, ...], expected: int | None) -> None:
	assertEqualTo(getTotalFoldsKnown(mapShape), expected, getTotalFoldsKnown.__name__, mapShape)

@pytest.mark.parametrize('expected', (None,))
def test_getOEISids(expected: None, oeisID: OEISid, capsys: pytest.CaptureFixture[str]) -> None:
	assertEqualTo(getOEISids(), expected, getOEISids.__name__)
	actual: CaptureResult[str] = capsys.readouterr()
	assertEqualTo(actual.err, '', getOEISids.__name__)
	assertEqualTo(oeisID in actual.out, True, getOEISids.__name__, oeisID)
	assertEqualTo(getMetadata(oeisID)['description'] in actual.out, True, getOEISids.__name__, oeisID)
	assertEqualTo('Available OEIS sequences:' in actual.out, True, getOEISids.__name__)
	assertEqualTo('Usage examples:' in actual.out, True, getOEISids.__name__)
	assertEqualTo('OEIS_for_n A001415 6' in actual.out, True, getOEISids.__name__)
	assertEqualTo("oeisIDfor_n('A001415', 6)" in actual.out, True, getOEISids.__name__)

@pytest.mark.parametrize('argv, expected', (
	pytest.param(('OEIS_for_n', 'A000136', '3'), 6, id='strip')
	, pytest.param(('OEIS_for_n', 'A001415', '3'), 60, id='rectangle')
	, pytest.param(('OEIS_for_n', 'A001415', '6'), 10512, id='documented-example')
	, pytest.param(('OEIS_for_n', 'A000560', '3'), 2, id='formula')
	, pytest.param(('OEIS_for_n', 'A001418', '1'), 1, id='offset')
))
def test_OEIS_for_n(expected: int, argv: tuple[str, ...], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(sys, 'argv', list(argv))
	OEIS_for_n()
	actual: CaptureResult[str] = capsys.readouterr()
	assertEqualTo(actual.err, '', OEIS_for_n.__name__, argv)
	assert regex.fullmatch(rf'{expected} distinct folding patterns\.\nTime elapsed: \d+\.\d{{3}} seconds\n', actual.out), messageTestFailure(actual.out, expected, OEIS_for_n.__name__, argv)

@pytest.mark.parametrize('argv, expected', (
	pytest.param(('OEIS_for_n',), (2, '', r'(?s).*the following arguments are required.*'), id='missing-arguments')
	, pytest.param(('OEIS_for_n', 'A001415'), (2, '', r'(?s).*the following arguments are required.*'), id='missing-n')
	, pytest.param(('OEIS_for_n', 'A999999', '3'), (1, '', "Error: 'A999999'\n"), id='unknown-identifier')
	, pytest.param(('OEIS_for_n', 'A001415', '-5'), (1, '', r'(?s)Error: .*non-negative integer.*'), id='negative-n')
	, pytest.param(('OEIS_for_n', 'A001415', 'abc'), (2, '', r'(?s).*invalid int value.*'), id='non-integer-n')
	, pytest.param(('OEIS_for_n', 'A001418', '0'), (1, '', r'(?s)Error: .*not defined.*offset.*'), id='below-offset')
	, pytest.param(('OEIS_for_n', '--help'), (0, r'(?s).*Available OEIS sequences:.*Usage examples:.*', ''), id='help')
))
def test_OEIS_for_nError(expected: tuple[int, str, str], argv: tuple[str, ...], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(sys, 'argv', list(argv))
	with pytest.raises(SystemExit) as exception:
		OEIS_for_n()
	actual: CaptureResult[str] = capsys.readouterr()
	assertEqualTo(exception.value.code, expected[0], OEIS_for_n.__name__, argv)
	assert regex.fullmatch(expected[1], actual.out), messageTestFailure(actual.out, expected[1], OEIS_for_n.__name__, argv)
	assert regex.fullmatch(expected[2], actual.err), messageTestFailure(actual.err, expected[2], OEIS_for_n.__name__, argv)

@pytest.mark.parametrize('oeisID', ('A400429', ' a400429 '))
@pytest.mark.parametrize('pathTerms, valuesKnown, sequence, expected', (
	pytest.param(None, dict[int, int](), {2: 11}, '2 11\n', id='absent'),
	pytest.param(None, {2: 11}, {2: 37, 3: 13}, '3 13\n', id='retain-memory'),
	pytest.param('  # primes\n2 11\n', {2: 11}, {3: 13}, '  # primes\n2 11\n3 13\n', id='indented-comment'),
	pytest.param('', dict[int, int](), {2: 11}, '2 11\n', id='empty'),
	pytest.param('# prime observations\n', dict[int, int](), {2: 11}, '# prime observations\n2 11\n', id='comments-only'),
	pytest.param(contents_bFile, {2: 701, 3: 709}, {2: 719, 4: 727}, '# prime observations\n2 701\n3 709\n4 727\n', id='retain-known'),
	pytest.param(contents_bFile, {2: 701, 3: 709}, {2: 733}, contents_bFile, id='no-additions'),
), indirect=['pathTerms'])
def test_updateValuesKnown(oeisID: OEISid, sequence: Mapping[int, int], pathTerms: Path, valuesKnown: Mapping[int, int], expected: str) -> None:
	with patch.dict(getMetadata(oeisID)), patch.dict(getValuesKnown(oeisID), valuesKnown, clear=True):
		known = getValuesKnown(oeisID)
		pathFilename: Path = updateValuesKnown(oeisID, sequence, pathTerms)
		assertEqualTo(pathFilename.relative_to(pathTerms), Path('files/A400/b400429.txt'), updateValuesKnown.__name__, oeisID, sequence)
		assertEqualTo(readText(pathFilename), expected, updateValuesKnown.__name__, oeisID, sequence)
		assert known is getMetadata(oeisID)['valuesKnown']
		assertEqualTo(getMetadata(oeisID)['valueUnknown'], max(known) + 1, updateValuesKnown.__name__, oeisID, sequence)

@pytest.mark.parametrize('oeisID, triangle, expected', (
	pytest.param('A400429', triangleHalf, {2: 503, 3: 509, 4: 521, 5: 523, 6: 541}, id='half-length-rows'),
))
def test_updateTriangle(oeisID: OEISid, triangle: 形Triangle, pathTerms: Path, expected: dict[int, int]) -> None:
	with patch.dict(getMetadata(oeisID)), patch.dict(getValuesKnown(oeisID), clear=True):
		pathFilename: Path = updateTriangle(oeisID, triangle, pathTerms)
		assertEqualTo(pathFilename.relative_to(pathTerms), Path('files/A400/b400429.txt'), updateTriangle.__name__, oeisID, triangle)
		assertEqualTo(getValuesKnown(oeisID), expected, updateTriangle.__name__, oeisID, triangle)

@pytest.mark.parametrize('valuesKnown, expected', (
	pytest.param(sequenceHalf, {2: [601], 3: [607], 4: [613, 617], 5: [619]}, id='partial-row'),
	pytest.param({2: 11, 5: 19}, {2: [11]}, id='first-gap'),
))
@pytest.mark.parametrize('oeisID', ('A400429',))
def test_getTriangle(oeisID: OEISid, expected: 形Triangle, valuesKnown: Mapping[int, int]) -> None:
	with patch.dict(getValuesKnown(oeisID), valuesKnown, clear=True):
		assertEqualTo(getTriangle(oeisID), expected, getTriangle.__name__, oeisID)

@pytest.mark.parametrize('valuesKnown', (sequenceHalf,))
@pytest.mark.parametrize('oeisID, 次diagonal, expected', (
	pytest.param('A400429', 1, {2: 601, 3: 607, 4: 617}, id='last-cells'),
	pytest.param('A400429', 2, {4: 613, 5: 619}, id='partial-row'),
))
def test_getTriangleDiagonal(oeisID: OEISid, 次diagonal: int, expected: dict[int, int], valuesKnown: Mapping[int, int]) -> None:
	with patch.dict(getValuesKnown(oeisID), valuesKnown, clear=True):
		assertEqualTo(getTriangleDiagonal(oeisID, 次diagonal), expected, getTriangleDiagonal.__name__, oeisID, 次diagonal)
