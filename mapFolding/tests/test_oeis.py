"""OEIS sequence values, the command line, triangle layouts, and local b-file updates, `mapFolding.oeis`.

Every value of a sequence is a(n) of its OEIS b-file, which the `expected` fixture reads. Other expected values and exceptions
come from the targets' docstrings unless a comment names another basis.
"""
from __future__ import annotations

from itertools import chain, product as CartesianProduct
from mapFolding.kitFilesystem import readText
from mapFolding.oeis import (
	getMetadata, getOEISids, getTotalFoldsKnown, getTriangle, getTriangleDiagonal, OEIS_for_n, oeisIDfor_n, updateTriangle, updateValuesKnown)
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.tests import assertEqualTo, messageTestFailure
from mapFolding.tests.dataSamples.triangles import sequenceHalf, triangleHalf
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
import re as regex

if TYPE_CHECKING:
	from _pytest.capture import CaptureResult
	from collections.abc import Mapping
	from mapFolding.theTypes import OEISid, 形Triangle
	from typing import LiteralString

# Each sequence meets its offset and n = 3.
@pytest.mark.parametrize('oeisID, n, f', tuple(chain(
	CartesianProduct(('A000136',), (1, 3), ('A000560', 'A001011 and A001010', 'A223094 and A000682', 'A223095, A077014, and A000682', 'A227167', 'A000682'))
	, CartesianProduct(('A000560',), (2, 3), ('A000136', 'A000682'))
	, CartesianProduct(('A000682',), (1, 3), ('A000560', 'A001010', 'A223094', 'A259689', 'A301620', 'A337581', 'A077460, A005316, and A223093'
		, 'A223093 and A005316', 'A223093 and A077014', 'A000136, A077014, and A223095', 'A223094 and A000682', 'A259702 and A000682'
		, 'A333971 and A000682', 'A334615 and A000682', 'A060206 and A000560', 'A000136'))
	, CartesianProduct(('A001010',), (1, 3), ('A001011 and A000682', 'A007822 and A000682'))
	, CartesianProduct(('A001011',), (1, 3), ('A001010 and A000682',))
	, CartesianProduct(('A005315',), (0, 3), ('A077460, A005316, and A060206', 'A078591', 'A085973 and A077054', 'A208357', 'A005316'))
	, CartesianProduct(('A005316',), (0, 3), ('A077014', 'A077054 and A005315', 'A077460, A005315, and A060206', 'A078592 and A005315'
		, 'A227167, A217310, and A217318', 'A000682 and A223093'))
	, CartesianProduct(('A007822',), (1, 3), ('A001010',))
	, CartesianProduct(('A060206',), (0, 3), ('A077460, A005315, and A005316', 'A000682'))
	, CartesianProduct(('A077014',), (0, 3), ('A000682 and A223093', 'A223095 and A000682', 'A005316'))
	, CartesianProduct(('A077054',), (0, 3), ('A085973 and A005315', 'A005316'))
	, CartesianProduct(('A077460',), (0, 3), ('A005315, A005316, and A060206', 'A000682 and A005316'))
	, CartesianProduct(('A078591',), (0, 3), ('A005316', 'A005315'))
	, CartesianProduct(('A078592',), (0, 3), ('A005316',))
	, CartesianProduct(('A085973',), (0, 3), ('A077054 and A005315', 'A005316'))
	, CartesianProduct(('A208357',), (1, 3), ('A005315', 'A005316'))
	, CartesianProduct(('A217310',), (1, 3), ('A223093', 'A227167, A217318, and A005316', 'A000682 and A005316'))
	, CartesianProduct(('A217318',), (1, 3), ('A223095', 'A227167, A217310, and A005316', 'A005316 and A000682'))
	, CartesianProduct(('A223093',), (1, 3), ('A217310', 'A223094 and A223095', 'A000682 and A077014', 'A000682 and A005316'))
	, CartesianProduct(('A223094',), (1, 3), ('A223094 and A000682', 'A223093 and A223095', 'A000682'))
	, CartesianProduct(('A223095',), (1, 3), ('A077014, and A000682', 'A223094 and A223093', 'A217318', 'A000682 and A005316'))
	, CartesianProduct(('A227167',), (1, 3), ('A217310, A217318, and A005316', 'A000136'))
	, CartesianProduct(('A259689',), (2, 3), ('A000682',))
	, CartesianProduct(('A259702',), (2, 3), ('A301620', 'A000682'))
	, CartesianProduct(('A301620',), (1, 3), ('A334615, A301620, and A000682', 'A259689', 'A259702', 'A000682'))
	, CartesianProduct(('A333971',), (2, 3), ('A000682',))
	, CartesianProduct(('A334615',), (2, 3), ('A000560', 'A001010', 'A259702', 'A337581', 'A227167', 'A223094', 'A301620', 'A005316 and A223093'
		, 'A005316 and A217310', 'A000682'))
	, CartesianProduct(('A337581',), (2, 3), ('A000682',))
	, CartesianProduct(('A000108', 'A001246', 'A001263', 'A001405', 'A002057', 'A002415', 'A002694', 'A006659', 'A007746', 'A008828', 'A014137', 'A014140', 'A060972', 'A401693')
		, (7,), (None,))
	, CartesianProduct(('A053121', 'A259974', 'A380369'), (10,), (None,))
)))
def test_oeisIDfor_n(oeisID: OEISid, n: int, f: LiteralString | None, expected: int) -> None:
	assertEqualTo(oeisIDfor_n(oeisID, n, f), expected, oeisIDfor_n.__name__, oeisID, n, f)

boxOfFlowsMapFolding: tuple[str, ...] = ('daoOfMapFolding', 'daoOfMapFoldingNumba', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed')
boxOfFlowsMeanders: tuple[str, ...] = ('matrixMeanders', 'matrixNumPy', 'matrixPandas', 'prune', 'pruneNumPy', 'prunePandas')
boxOfFlowsSymmetric: tuple[str, ...] = ('algorithm', 'algorithmNumba', 'asynchronous', 'theorem2', 'theorem2Numba', 'theorem2Trimmed')

@pytest.mark.parametrize('oeisID, n, flow', tuple(chain(
	CartesianProduct(('A000136',), (1, 3), boxOfFlowsMapFolding)
	, CartesianProduct(('A001415', 'A001416'), (0, 3), boxOfFlowsMapFolding)
	, CartesianProduct(('A001417', 'A195646'), (0,), boxOfFlowsMapFolding)
	, CartesianProduct(('A001418',), (1,), boxOfFlowsMapFolding)
	, CartesianProduct(('A000682',), (1, 3), boxOfFlowsMeanders)
	, CartesianProduct(('A005316',), (0, 3), boxOfFlowsMeanders)
	, CartesianProduct(('A007822',), (1, 3), boxOfFlowsSymmetric)
)))
def test_oeisIDfor_nFlow(oeisID: OEISid, n: int, flow: LiteralString, expected: int) -> None:
	assertEqualTo(oeisIDfor_n(oeisID, n, flow=flow), expected, oeisIDfor_n.__name__, oeisID, n, flow=flow)

@pytest.mark.parametrize('oeisID, n, expected', (
	pytest.param('A000136', -5, ValueError, id='negative')
	, pytest.param('A000136', 'NE', ValueError, id='stringNE')
	, pytest.param('A000136', 1.5, ValueError, id='fraction')
	, pytest.param('A001418', 0, ArithmeticError, id='belowOffset')
	, pytest.param('A999999', 3, KeyError, id='unregistered')
	, pytest.param('A204352', 3, ValueError, id='noFormula')
))
def test_oeisIDfor_nError(oeisID: OEISid, n: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		oeisIDfor_n(oeisID, n)

@pytest.mark.parametrize('oeisID, expected', (
	pytest.param('A001415', 'A001415', id='uppercase')
	, pytest.param('a001415', 'A001415', id='lowercase')
	, pytest.param(' A001415 ', 'A001415', id='spaces')
	, pytest.param('\ta001415\n', 'A001415', id='whitespace')
))
def test_formatOEISid(oeisID: OEISid, expected: OEISid) -> None:
	assertEqualTo(formatOEISid(oeisID), expected, formatOEISid.__name__, oeisID)

# Reference: the OEIS b-files of A000136(3), A001415(3), and A001418(3); no sequence knows a 999x999 map.
@pytest.mark.parametrize('mapShape, expected', (
	pytest.param((1, 3), 6, id='p1x3')
	, pytest.param((2, 3), 60, id='p2x3')
	, pytest.param((3, 3), 1368, id='p3x3')
	, pytest.param((999, 999), None, id='p999x999')
))
def test_getTotalFoldsKnown(mapShape: tuple[int, ...], expected: int | None) -> None:
	assertEqualTo(getTotalFoldsKnown(mapShape), expected, getTotalFoldsKnown.__name__, mapShape)

@pytest.mark.parametrize('oeisID', (
	'A000108', 'A000136', 'A000560', 'A000682', 'A001010', 'A001011', 'A001246', 'A001263', 'A001405', 'A001415', 'A001416', 'A001417', 'A001418'
	, 'A002057', 'A002415', 'A002694', 'A005315', 'A005316', 'A006657', 'A006658', 'A006659', 'A006660', 'A006661', 'A006662', 'A007746', 'A007822', 'A008828'
	, 'A014137', 'A014140', 'A045445', 'A053121', 'A060206', 'A060972', 'A076876', 'A077014', 'A077054', 'A077055', 'A077056', 'A077460'
	, 'A078591', 'A078592', 'A085973', 'A195646', 'A204352', 'A208357', 'A217310', 'A217318', 'A223093', 'A223094', 'A223095', 'A227167'
	, 'A259689', 'A259698', 'A259700', 'A259701', 'A259702', 'A259703', 'A259974', 'A260785', 'A287548', 'A301620', 'A322507', 'A331499'
	, 'A333971', 'A334615', 'A337581', 'A339179', 'A380367', 'A380369', 'A400429', 'A400630', 'A400631', 'A400632', 'A400633', 'A400634'
	, 'A400635', 'A400636', 'A400637', 'A401693'
))
def test_getOEISids(capsys: pytest.CaptureFixture[str], oeisID: OEISid) -> None:
	"""Standard output names each registered sequence with its description, and the usage examples."""
	getOEISids()
	printed: CaptureResult[str] = capsys.readouterr()
	assertEqualTo(printed.err, '', getOEISids.__name__)
	assertEqualTo(oeisID in printed.out, True, getOEISids.__name__, oeisID)
	assertEqualTo(getMetadata(oeisID)['description'] in printed.out, True, getOEISids.__name__, oeisID)
	assertEqualTo(tuple(map(printed.out.__contains__, ('Available OEIS sequences:', 'Usage examples:', 'OEIS_for_n A001415 6', "oeisIDfor_n('A001415', 6)")))
		, (True, True, True, True), getOEISids.__name__)

# Reference: A000136(3), A001415(3), the documented example A001415(6), A000560(3), and A001418(1) in the OEIS b-files.
@pytest.mark.parametrize('argv, expected', (
	pytest.param(('OEIS_for_n', 'A000136', '3'), 6, id='A000136-3')
	, pytest.param(('OEIS_for_n', 'A001415', '3'), 60, id='A001415-3')
	, pytest.param(('OEIS_for_n', 'A001415', '6'), 10512, id='A001415-6')
	, pytest.param(('OEIS_for_n', 'A000560', '3'), 2, id='A000560-3')
	, pytest.param(('OEIS_for_n', 'A001418', '1'), 1, id='A001418-1')
), indirect=['argv'])
def test_OEIS_for_n(capsys: pytest.CaptureFixture[str], argv: list[str], expected: int) -> None:
	OEIS_for_n()
	printed: CaptureResult[str] = capsys.readouterr()
	assertEqualTo(printed.err, '', OEIS_for_n.__name__, argv)
	assert regex.fullmatch(rf'{expected} distinct folding patterns\.\nTime elapsed: \d+\.\d{{3}} seconds\n', printed.out), messageTestFailure(printed.out, expected, OEIS_for_n.__name__, argv)

# Reference: `argparse` exits with code 2 and its usage error, or with code 0 after `--help`; `OEIS_for_n` exits with code 1 and
# its message.
@pytest.mark.parametrize('argv, exitCode, patternOut, patternErr', (
	pytest.param(('OEIS_for_n',), 2, '', r'(?s).*the following arguments are required.*', id='noArguments')
	, pytest.param(('OEIS_for_n', 'A001415'), 2, '', r'(?s).*the following arguments are required.*', id='nMissing')
	, pytest.param(('OEIS_for_n', 'A999999', '3'), 1, '', "Error: 'A999999'\n", id='unregistered')
	, pytest.param(('OEIS_for_n', 'A001415', '-5'), 1, '', r'(?s)Error: .*non-negative integer.*', id='negative')
	, pytest.param(('OEIS_for_n', 'A001415', 'NE'), 2, '', r'(?s).*invalid int value.*', id='stringNE')
	, pytest.param(('OEIS_for_n', 'A001418', '0'), 1, '', r'(?s)Error: .*not defined.*offset.*', id='belowOffset')
	, pytest.param(('OEIS_for_n', '--help'), 0, r'(?s).*Available OEIS sequences:.*Usage examples:.*', '', id='help')
), indirect=['argv'])
def test_OEIS_for_nError(capsys: pytest.CaptureFixture[str], argv: list[str], exitCode: int, patternOut: str, patternErr: str) -> None:
	with pytest.raises(SystemExit) as exitInformation:
		OEIS_for_n()
	printed: CaptureResult[str] = capsys.readouterr()
	assertEqualTo(exitInformation.value.code, exitCode, OEIS_for_n.__name__, argv)
	assert regex.fullmatch(patternOut, printed.out), messageTestFailure(printed.out, patternOut, OEIS_for_n.__name__, argv)
	assert regex.fullmatch(patternErr, printed.err), messageTestFailure(printed.err, patternErr, OEIS_for_n.__name__, argv)

contents_bFile: str = '# prime observations\n2 701\n3 709\n'
pathFilenameRelative_bFileA400429: Path = Path('files/A400/b400429.txt')

@pytest.mark.parametrize('oeisID', ('A400429', pytest.param(' a400429 ', id='a400429Padded')))
@pytest.mark.parametrize('pathTerms, valuesKnown, sequence, expected', (
	pytest.param(None, {}, {2: 11}, '2 11\n', id='absent')
	, pytest.param(None, {2: 11}, {2: 37, 3: 13}, '3 13\n', id='knownValueRetained')
	, pytest.param('  # primes\n2 11\n', {2: 11}, {3: 13}, '  # primes\n2 11\n3 13\n', id='indentedComment')
	, pytest.param('', {}, {2: 11}, '2 11\n', id='empty')
	, pytest.param('# prime observations\n', {}, {2: 11}, '# prime observations\n2 11\n', id='commentsOnly')
	, pytest.param(contents_bFile, {2: 701, 3: 709}, {2: 719, 4: 727}, '# prime observations\n2 701\n3 709\n4 727\n', id='fileValueRetained')
	, pytest.param(contents_bFile, {2: 701, 3: 709}, {2: 733}, contents_bFile, id='noAddition')
), indirect=['pathTerms', 'valuesKnown'])
def test_updateValuesKnown(pathTerms: Path, valuesKnown: dict[int, int], oeisID: OEISid, sequence: Mapping[int, int], expected: str) -> None:
	"""The b-file holds the expected terms, and the shared known values, updated in place, set the first unknown index."""
	pathFilename: Path = updateValuesKnown(oeisID, sequence, pathTerms)
	assertEqualTo(pathFilename.relative_to(pathTerms), pathFilenameRelative_bFileA400429, updateValuesKnown.__name__, oeisID, sequence)
	assertEqualTo(readText(pathFilename), expected, updateValuesKnown.__name__, oeisID, sequence)
	assert valuesKnown is getMetadata(oeisID)['valuesKnown'], messageTestFailure(getMetadata(oeisID)['valuesKnown'], 'the same dictionary', updateValuesKnown.__name__, oeisID, sequence)
	assertEqualTo(getMetadata(oeisID)['valueUnknown'], max(valuesKnown) + 1, updateValuesKnown.__name__, oeisID, sequence)

# Reference: hand-derived from the A400429 layout: row n holds floor(n/2) cells, and a(2) is T(2, 1).
@pytest.mark.parametrize('oeisID', ('A400429',))
@pytest.mark.parametrize('pathTerms, valuesKnown', ((None, {}),), indirect=True, ids=['empty'])
@pytest.mark.parametrize('triangle, expected', (pytest.param(triangleHalf, {2: 503, 3: 509, 4: 521, 5: 523, 6: 541}, id='triangleHalf'),))
def test_updateTriangle(pathTerms: Path, valuesKnown: dict[int, int], oeisID: OEISid, triangle: 形Triangle, expected: dict[int, int]) -> None:
	pathFilename: Path = updateTriangle(oeisID, triangle, pathTerms)
	assertEqualTo(pathFilename.relative_to(pathTerms), pathFilenameRelative_bFileA400429, updateTriangle.__name__, oeisID, triangle)
	assertEqualTo(valuesKnown, expected, updateTriangle.__name__, oeisID, triangle)

# Reference: hand-derived from the A400429 layout: consecutive values fill rows of floor(n/2) cells from n = 2; the last row may be
# partial; a gap in n ends the triangle.
@pytest.mark.parametrize('oeisID', ('A400429',))
@pytest.mark.parametrize('valuesKnown, expected', (
	pytest.param(sequenceHalf, {2: [601], 3: [607], 4: [613, 617], 5: [619]}, id='sequenceHalf')
	, pytest.param({2: 11, 5: 19}, {2: [11]}, id='gapAfterFirst')
), indirect=['valuesKnown'])
def test_getTriangle(valuesKnown: dict[int, int], oeisID: OEISid, expected: 形Triangle) -> None:
	assertEqualTo(getTriangle(oeisID), expected, getTriangle.__name__, oeisID)

# Reference: hand-derived from `sequenceHalf` and the definition D_j(n) = T(n, floor(n/2) - j + 1), for each row that has the cell.
@pytest.mark.parametrize('oeisID', ('A400429',))
@pytest.mark.parametrize('valuesKnown', (sequenceHalf,), indirect=True, ids=['sequenceHalf'])
@pytest.mark.parametrize('次diagonal, expected', (pytest.param(1, {2: 601, 3: 607, 4: 617}, id='1'), pytest.param(2, {4: 613, 5: 619}, id='2')))
def test_getTriangleDiagonal(valuesKnown: dict[int, int], oeisID: OEISid, 次diagonal: int, expected: dict[int, int]) -> None:
	assertEqualTo(getTriangleDiagonal(oeisID, 次diagonal), expected, getTriangleDiagonal.__name__, oeisID, 次diagonal)
