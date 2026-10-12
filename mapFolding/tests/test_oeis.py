from __future__ import annotations

from itertools import chain, product as CartesianProduct, starmap
from mapFolding.oeis import getMetadata, getOEISids, getTotalFoldsKnown, OEIS_for_n, oeisIDfor_n
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.tests import assertEqualTo, messageTestFailure
from operator import itemgetter
from typing import TYPE_CHECKING
import pytest
import re as regex

if TYPE_CHECKING:
	from _pytest.capture import CaptureResult
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import OEISid, 形KeywordArgumentsCount
	from typing import LiteralString

parametrizeOEISidByFormula: pytest.MarkDecorator = pytest.mark.parametrize('oeisID, f', (
	pytest.param('A000136', 'A000560')
	, pytest.param('A000136', 'A001011 and A001010')
	, pytest.param('A000136', 'A223094 and A000682')
	, pytest.param('A000136', 'A223095, A077014, and A000682')
	, pytest.param('A000136', 'A227167')
	, pytest.param('A000136', 'A000682')
	, pytest.param('A000560', 'A000136')
	, pytest.param('A000560', 'A000682')
	, pytest.param('A000682', 'A000560')
	, pytest.param('A000682', 'A001010')
	, pytest.param('A000682', 'A223094')
	, pytest.param('A000682', 'A259689')
	, pytest.param('A000682', 'A301620')
	, pytest.param('A000682', 'A337581')
	, pytest.param('A000682', 'A077460, A005316, and A223093')
	, pytest.param('A000682', 'A223093 and A005316')
	, pytest.param('A000682', 'A223093 and A077014')
	, pytest.param('A000682', 'A000136, A077014, and A223095')
	, pytest.param('A000682', 'A223094 and A000682')
	, pytest.param('A000682', 'A259702 and A000682')
	, pytest.param('A000682', 'A333971 and A000682')
	, pytest.param('A000682', 'A334615 and A000682')
	, pytest.param('A000682', 'A060206 and A000560')
	, pytest.param('A000682', 'A000136')
	, pytest.param('A001010', 'A001011 and A000682')
	, pytest.param('A001010', 'A007822 and A000682')
	, pytest.param('A001011', 'A001010 and A000682')
	, pytest.param('A005315', 'A077460, A005316, and A060206')
	, pytest.param('A005315', 'A078591')
	, pytest.param('A005315', 'A085973 and A077054')
	, pytest.param('A005315', 'A208357')
	, pytest.param('A005315', 'A005316')
	, pytest.param('A005316', 'A077014')
	, pytest.param('A005316', 'A077054 and A005315')
	, pytest.param('A005316', 'A077460, A005315, and A060206')
	, pytest.param('A005316', 'A078592 and A005315')
	, pytest.param('A005316', 'A227167, A217310, and A217318')
	, pytest.param('A005316', 'A000682 and A223093')
	, pytest.param('A007822', 'A001010')
	, pytest.param('A060206', 'A077460, A005315, and A005316')
	, pytest.param('A060206', 'A000682')
	, pytest.param('A077014', 'A000682 and A223093')
	, pytest.param('A077014', 'A223095 and A000682')
	, pytest.param('A077014', 'A005316')
	, pytest.param('A077054', 'A085973 and A005315')
	, pytest.param('A077054', 'A005316')
	, pytest.param('A077460', 'A005315, A005316, and A060206')
	, pytest.param('A077460', 'A000682 and A005316')
	, pytest.param('A078591', 'A005316')
	, pytest.param('A078591', 'A005315')
	, pytest.param('A078592', 'A005316')
	, pytest.param('A085973', 'A077054 and A005315')
	, pytest.param('A085973', 'A005316')
	, pytest.param('A208357', 'A005315')
	, pytest.param('A208357', 'A005316')
	, pytest.param('A217310', 'A223093')
	, pytest.param('A217310', 'A227167, A217318, and A005316')
	, pytest.param('A217310', 'A000682 and A005316')
	, pytest.param('A217318', 'A223095')
	, pytest.param('A217318', 'A227167, A217310, and A005316')
	, pytest.param('A217318', 'A005316 and A000682')
	, pytest.param('A223093', 'A217310')
	, pytest.param('A223093', 'A223094 and A223095')
	, pytest.param('A223093', 'A000682 and A077014')
	, pytest.param('A223093', 'A000682 and A005316')
	, pytest.param('A223094', 'A223094 and A000682')
	, pytest.param('A223094', 'A223093 and A223095')
	, pytest.param('A223094', 'A000682')
	, pytest.param('A223095', 'A077014, and A000682')
	, pytest.param('A223095', 'A223094 and A223093')
	, pytest.param('A223095', 'A217318')
	, pytest.param('A223095', 'A000682 and A005316')
	, pytest.param('A227167', 'A217310, A217318, and A005316')
	, pytest.param('A227167', 'A000136')
	, pytest.param('A259689', 'A000682')
	, pytest.param('A259702', 'A301620')
	, pytest.param('A259702', 'A000682')
	, pytest.param('A301620', 'A334615, A301620, and A000682')
	, pytest.param('A301620', 'A259689')
	, pytest.param('A301620', 'A259702')
	, pytest.param('A301620', 'A000682')
	, pytest.param('A333971', 'A000682')
	, pytest.param('A334615', 'A000560')
	, pytest.param('A334615', 'A001010')
	, pytest.param('A334615', 'A259702')
	, pytest.param('A334615', 'A337581')
	, pytest.param('A334615', 'A227167')
	, pytest.param('A334615', 'A223094')
	, pytest.param('A334615', 'A301620')
	, pytest.param('A334615', 'A005316 and A223093')
	, pytest.param('A334615', 'A005316 and A217310')
	, pytest.param('A334615', 'A000682')
	, pytest.param('A337581', 'A000682')
))

parametrizeOEISfor_n: pytest.MarkDecorator = pytest.mark.parametrize('oeisID, n, f, keywordArguments', tuple(chain(
	chain.from_iterable(
		CartesianProduct((parameterRow.values[0],)
			, (getMetadata(parameterRow.values[0])['offset'], 3)
			, (parameterRow.values[1],), (None,))
			for parameterRow in parametrizeOEISidByFormula.args[1])

	, chain.from_iterable(starmap(CartesianProduct, (
		(('A000136',), (getMetadata('A000136')['offset'], 3), (None,), ('daoOfMapFolding', 'daoOfMapFoldingNumba', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'))
		, (('A001415',), (getMetadata('A001415')['offset'], 3), (None,), ('daoOfMapFolding', 'daoOfMapFoldingNumba', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'))
		, (('A001416',), (getMetadata('A001416')['offset'], 3), (None,), ('daoOfMapFolding', 'daoOfMapFoldingNumba', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'))
		, (('A001417',), (getMetadata('A001417')['offset'],), (None,), ('daoOfMapFolding', 'daoOfMapFoldingNumba', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'))
		, (('A001418',), (getMetadata('A001418')['offset'],), (None,), ('daoOfMapFolding', 'daoOfMapFoldingNumba', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'))
		, (('A195646',), (getMetadata('A195646')['offset'],), (None,), ('daoOfMapFolding', 'daoOfMapFoldingNumba', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'))
		, (('A000682',), (getMetadata('A000682')['offset'], 3), (None,), ('matrixMeanders', 'matrixNumPy', 'matrixPandas', 'matrixPolars', 'prune', 'pruneNumPy', 'prunePandas'))
		, (('A005316',), (getMetadata('A005316')['offset'], 3), (None,), ('matrixMeanders', 'matrixNumPy', 'matrixPandas', 'matrixPolars', 'prune', 'pruneNumPy', 'prunePandas'))
		, (('A007822',), (getMetadata('A007822')['offset'], 3), (None,), ('algorithm', 'algorithmNumba', 'asynchronous', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'))
	)))
)), indirect=('keywordArguments',))

@pytest.mark.parametrize('oeisID, n, f', tuple(filter(
	itemgetter(2), map(itemgetter(0, 1, 2), parametrizeOEISfor_n.args[1])
)))
def test_byFormula(callableA: CallableFunction[[int, LiteralString | None], int], n: int, f: LiteralString | None, expected: int) -> None:
	assertEqualTo(callableA(n, f), expected, callableA.__name__, n, f)

@parametrizeOEISfor_n
def test_oeisIDfor_n(oeisID: OEISid, n: int, f: LiteralString | None, keywordArguments: 形KeywordArgumentsCount, expected: int) -> None:
	assertEqualTo(oeisIDfor_n(oeisID, n, f, **keywordArguments), expected, oeisIDfor_n.__name__, oeisID, n, f, **keywordArguments)

@pytest.mark.parametrize('oeisID, n, f, keywordArguments, expected', (
	pytest.param('A000136', -5, None, {}, ValueError, id='negative-n')
	, pytest.param('A000136', 'foo', None, {}, ValueError, id='non-integer-string')
	, pytest.param('A000136', 1.5, None, {}, ValueError, id='non-integer-float')
	, pytest.param('A001418', 0, None, {}, ArithmeticError, id='below-offset')
	, pytest.param('A999999', 3, None, {}, KeyError, id='unknown-identifier')
	, pytest.param('A000108', 3, None, {}, ValueError, id='unimplemented-sequence')
))
def test_oeisIDfor_nError(oeisID: OEISid, n: int, f: LiteralString | None, keywordArguments: 形KeywordArgumentsCount, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		oeisIDfor_n(oeisID, n, f, **keywordArguments)

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
), indirect=('argv',))
def test_OEIS_for_n(expected: int, argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]) -> None:
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
), indirect=('argv',))
def test_OEIS_for_nError(expected: tuple[int, str, str], argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit) as exception:
		OEIS_for_n()
	actual: CaptureResult[str] = capsys.readouterr()
	assertEqualTo(exception.value.code, expected[0], OEIS_for_n.__name__, argv)
	assert regex.fullmatch(expected[1], actual.out), messageTestFailure(actual.out, expected[1], OEIS_for_n.__name__, argv)
	assert regex.fullmatch(expected[2], actual.err), messageTestFailure(actual.err, expected[2], OEIS_for_n.__name__, argv)
