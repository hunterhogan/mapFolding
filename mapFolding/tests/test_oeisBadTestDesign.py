"""OEIS (Online Encyclopedia of Integer Sequences) integration testing.

This module validates the package's integration with OEIS, ensuring that sequence
identification, value retrieval, and caching mechanisms work correctly. The OEIS
connection provides the mathematical foundation that validates computational results
against established mathematical knowledge.

These tests verify both the technical aspects of OEIS integration (network requests,
caching, error handling) and the mathematical correctness of sequence identification
and value mapping.

Key Testing Areas:
- OEIS sequence ID validation and normalization
- Network request handling and error recovery
- Local caching of sequence data for offline operation
- Command-line interface for OEIS sequence queries
- Mathematical consistency between local computations and OEIS values

The caching tests are particularly important for users working in environments with
limited network access, as they ensure the package can operate effectively offline
once sequence data has been retrieved.

Network error handling tests verify graceful degradation when OEIS is unavailable,
which is crucial for maintaining package reliability in production environments.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from mapFolding.oeis import getMetadata, getOEISids, getTotalFoldsKnown, getValuesKnown, OEIS_for_n, oeisIDfor_n, oeisIDsImplemented
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.tests import assertEqualTo, messageTestFailure
from typing import LiteralString, TYPE_CHECKING
import io
import pytest
import random
import re as regex
import unittest.mock

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence
	from mapFolding.theTypes import OEISid, 形KeywordArgumentsCount
	from typing import Any

def test_formatOEISid(oeisID: OEISid) -> None:
	actual: str = formatOEISid(oeisID)
	assertEqualTo(actual, oeisID, formatOEISid.__name__, oeisID)
	actual: str = formatOEISid(oeisID.lower())
	assertEqualTo(actual, oeisID, formatOEISid.__name__, oeisID.lower())
	actual: str = formatOEISid(oeisID.swapcase())
	assertEqualTo(actual, oeisID, formatOEISid.__name__, oeisID.swapcase())

@pytest.mark.parametrize(('mapShape', 'expected'), (pytest.param((999, 999), None, id='mapShapeNotFound'),))
def test_getTotalFoldsKnown(mapShape: tuple[int, ...], expected: int | None) -> None:
	actual: int | None = getTotalFoldsKnown(mapShape)
	assertEqualTo(actual, expected, getTotalFoldsKnown.__name__, mapShape)

# TODO Make param that isn't stoopid.
@pytest.mark.parametrize(
	'oeisID, n, f, keywordArguments'
	, [
		pytest.param('A000136', 3, '', {'flow': 'daoOfMapFolding'}, id='A000136,countFolds')
		, pytest.param('A001415', 3, '', {'flow': 'daoOfMapFolding'}, id='A001415,countFolds')
		, pytest.param('A001416', 3, '', {'flow': 'daoOfMapFolding'}, id='A001416,countFolds')
		, pytest.param('A001417', 3, '', {'flow': 'daoOfMapFolding'}, id='A001417,countFolds')
		, pytest.param('A001418', 3, '', {'flow': 'daoOfMapFolding'}, id='A001418,countFolds')
		, pytest.param('A195646', 2, '', {'flow': 'daoOfMapFolding'}, id='A195646,countFolds')
		, pytest.param('A000682', 3, '', {'flow': 'matrixMeanders'}, id='A000682,countMeanders')
		, pytest.param('A005316', 3, '', {'flow': 'matrixMeanders'}, id='A005316,countMeanders')
		, pytest.param('A007822', 3, '', {'flow': 'algorithm'}, id='foldsSymmetric,countFoldsSymmetric')
	]
)
def test_oeisIDfor_n(oeisID: OEISid, n: int, f: LiteralString, keywordArguments: 形KeywordArgumentsCount) -> None:
	"""Verify OEIS sequence value calculations against known reference values."""
	expected: int = getValuesKnown(oeisID)[n]
	actual: int = oeisIDfor_n(oeisID, n, f, **keywordArguments)
	assertEqualTo(actual, expected, oeisIDfor_n.__name__, oeisID, n, f, **keywordArguments)

# TODO Make param that isn't stoopid.
@pytest.mark.parametrize(
	'oeisID, f'
	, [
		pytest.param('A000560', '', id='A000560')
		, pytest.param('A000136', 'A000682', id='A000136,A000682')
		, pytest.param('A000136', 'A000560', id='A000136,A000560')
		, pytest.param('A000682', 'A000560', id='A000682,A000560')
		, pytest.param('A000682', 'A301620', id='A000682,A301620')
		, pytest.param('A000682', 'A259689', id='A000682,A259689')
		, pytest.param('A000682', 'A000136', id='A000682,A000136')
		, pytest.param('A000682', 'A223094', id='A000682,A223094')
		, pytest.param('A001010', 'A000682 and A007822', id='A001010,A000682-and-A007822')
		, pytest.param('A001010', 'A001011 and A000136', id='A001010,A001011-and-A000136')
		, pytest.param('A223094', 'A000136 and A000682', id='A223094,A000136-and-A000682')
		, pytest.param('A223094', 'A223094 and A000682', id='A223094,A223094-and-A000682')
		, pytest.param('A223094', 'A000682', id='A223094,A000682')
		, pytest.param('A259689', '', id='A259689')
		, pytest.param('A001011', '', id='A001011')
		, pytest.param('A005315', '', id='A005315')
		, pytest.param('A060206', '', id='A060206')
		, pytest.param('A077460', '', id='A077460')
		, pytest.param('A078591', '', id='A078591')
		, pytest.param('A301620', '', id='A301620')
		, pytest.param('A301620', 'A259689', id='A301620,A259689')
	]
)
@pytest.mark.parametrize(
	'oeis_n'
	, [pytest.param(0, id='offset'), pytest.param(2, id='offsetPlus2'), pytest.param(5, id='offsetPlus5')]
	, indirect=True
)
def test_oeisIDfor_n_byFormula(oeisID: OEISid, oeis_n: int, f: LiteralString) -> None:
	expected: int = getValuesKnown(oeisID)[oeis_n]
	actual: int = oeisIDfor_n(oeisID, oeis_n, f=f)
	assertEqualTo(actual, expected, oeisIDfor_n.__name__, oeisID, oeis_n, f=f)

parameters_test_aOFn_invalid_n = [(-random.randint(1, 100), 'randomNegative'), ('foo', 'string'), (1.5, 'float')]
badValues, badValuesIDs = zip(*parameters_test_aOFn_invalid_n, strict=True)

@pytest.mark.parametrize('badN', badValues, ids=badValuesIDs)
def test_oeisIDfor_nError(oeisID_1random: LiteralString, badN: Any, expected: type[ValueError] = ValueError) -> None:
	"""Check that negative or non-integer n raises ValueError."""
	with pytest.raises(expected) as exception:
		oeisIDfor_n(oeisID_1random, badN)
	assertEqualTo(type(exception.value), expected, oeisIDfor_n.__name__, oeisID_1random, badN)

@pytest.mark.parametrize('oeisID,n', [('A001418', 0)])
def test_oeisIDfor_nErrorA001418(oeisID: OEISid, n: int, expected: type[ArithmeticError] = ArithmeticError) -> None:
	with pytest.raises(expected) as exception:
		oeisIDfor_n(oeisID, n)
	assertEqualTo(type(exception.value), expected, oeisIDfor_n.__name__, oeisID, n)

#===== Command Line Interface Tests =====

def testHelpText() -> None:
	"""Test that help text is complete and examples are valid."""
	outputStream = io.StringIO()
	with redirect_stdout(outputStream):
		getOEISids()

	helpText = outputStream.getvalue()

	# Verify content
	for oeisID in oeisIDsImplemented:
		assertEqualTo(oeisID in helpText, True, getOEISids.__name__, oeisID)
		assertEqualTo(getMetadata(oeisID)['description'] in helpText, True, getOEISids.__name__, oeisID)

	# Extract and verify examples

	cliMatch = regex.search(r'OEIS_for_n (\w+) (\d+)', helpText)
	pythonMatch = regex.search(r"oeisIDfor_n\('(\w+)', (\d+)\)", helpText)

	assert cliMatch is not None, messageTestFailure(cliMatch, 'a CLI example match', getOEISids.__name__)
	assert pythonMatch is not None, messageTestFailure(pythonMatch, 'a Python example match', getOEISids.__name__)
	oeisID, n = pythonMatch.groups()
	n = int(n)

	# Verify CLI and Python examples use same values
	assertEqualTo(cliMatch.groups(), (oeisID, str(n)), getOEISids.__name__)

	# Verify the example works
	# TODO Why is a str not a str?
	expectedValue = oeisIDfor_n(oeisID, n)  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]

	# Test CLI execution of the example
	with unittest.mock.patch('sys.argv', ['OEIS_for_n', oeisID, str(n)]):
		outputStream = io.StringIO()
		with redirect_stdout(outputStream):
			OEIS_for_n()
		actual: int = int(outputStream.getvalue().strip().split()[0])
		assertEqualTo(actual, expectedValue, OEIS_for_n.__name__, oeisID, n)

def standardizedSystemExit(expected: str | int | Sequence[int], functionTarget: Callable[..., Any], *arguments: Any) -> None:
	"""Template for tests expecting SystemExit.

	Parameters
	----------
	expected : str | int | Sequence[int]
		Exit code expectation:
		- "error": any non-zero exit code
		- "nonError": specifically zero exit code
		- int: exact exit code match
		- Sequence[int]: exit code must be one of these values
	functionTarget : Callable[..., Any]
		The function to test.
	arguments : Any
		Arguments to pass to the function.

	"""
	with pytest.raises(SystemExit) as exitInfo:
		functionTarget(*arguments)

	exitCode = exitInfo.value.code
	functionName: str = getattr(functionTarget, "__name__", functionTarget.__class__.__name__)

	if expected == "error":
		assert exitCode != 0, messageTestFailure(exitCode, "a non-zero exit code", functionName, *arguments)
	elif expected == "nonError":
		assertEqualTo(exitCode, 0, functionName, *arguments)
	elif isinstance(expected, (list, tuple)):
		assert exitCode in expected, messageTestFailure(exitCode, expected, functionName, *arguments)
	else:
		assertEqualTo(exitCode, expected, functionName, *arguments)

def testCLI_InvalidInputs() -> None:
	"""Test CLI error handling."""
	testCases = [
		(['OEIS_for_n'], 'missing arguments')
		, (['OEIS_for_n', 'A999999', '1'], 'invalid OEIS ID')
		, (['OEIS_for_n', 'A001415', '-1'], 'negative n')
		, (['OEIS_for_n', 'A001415', 'abc'], 'non-integer n')
	]

	for arguments, _testID in testCases:
		with unittest.mock.patch('sys.argv', arguments):
			standardizedSystemExit('error', OEIS_for_n)

def testCLI_HelpFlag() -> None:
	"""Verify --help output contains required information."""
	with unittest.mock.patch('sys.argv', ['OEIS_for_n', '--help']):
		outputStream = io.StringIO()
		with redirect_stdout(outputStream):
			standardizedSystemExit('nonError', OEIS_for_n)

		helpOutput = outputStream.getvalue()
		assertEqualTo('Available OEIS sequences:' in helpOutput, True, OEIS_for_n.__name__, '--help')
		assertEqualTo('Usage examples:' in helpOutput, True, OEIS_for_n.__name__, '--help')
		assertEqualTo(all(oeisID in helpOutput for oeisID in oeisIDsImplemented), True, OEIS_for_n.__name__, '--help')
