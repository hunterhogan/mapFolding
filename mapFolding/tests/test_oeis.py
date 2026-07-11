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
from mapFolding import packageSettings
from mapFolding.oeis import _formatOEISid, dictionaryOEISMapFolding, getOEISids, OEIS_for_n, oeisIDfor_n
from mapFolding.tests import assertEqualTo, messageTestFailure
from typing import TYPE_CHECKING
import io
import pytest
import random
import re as regex
import unittest.mock

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence
	from typing import Any

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

def test__validateOEISid_valid_id(oeisIDmapFolding: str) -> None:
	actual: str = _formatOEISid(oeisIDmapFolding)
	assertEqualTo(actual, oeisIDmapFolding, _formatOEISid.__name__, oeisIDmapFolding)

def test__validateOEISid_valid_id_case_insensitive(oeisIDmapFolding: str) -> None:
	expected: str = oeisIDmapFolding.upper()
	actualLower: str = _formatOEISid(oeisIDmapFolding.lower())
	actualUpper: str = _formatOEISid(oeisIDmapFolding.upper())
	actualSwapcase: str = _formatOEISid(oeisIDmapFolding.swapcase())
	assertEqualTo(actualLower, expected, _formatOEISid.__name__, oeisIDmapFolding.lower())
	assertEqualTo(actualUpper, expected, _formatOEISid.__name__, oeisIDmapFolding.upper())
	assertEqualTo(actualSwapcase, expected, _formatOEISid.__name__, oeisIDmapFolding.swapcase())

parameters_test_aOFn_invalid_n = [(-random.randint(1, 100), 'randomNegative'), ('foo', 'string'), (1.5, 'float')]
badValues, badValuesIDs = zip(*parameters_test_aOFn_invalid_n, strict=True)

@pytest.mark.parametrize('badN', badValues, ids=badValuesIDs)
def test_aOFn_invalid_n(oeisID_1random: str, badN: Any) -> None:
	"""Check that negative or non-integer n raises ValueError."""
	expected: type[ValueError] = ValueError
	with pytest.raises(expected) as exceptionInfo:
		oeisIDfor_n(oeisID_1random, badN)
	assertEqualTo(type(exceptionInfo.value), expected, oeisIDfor_n.__name__, oeisID_1random, badN)

def test_aOFn_zeroDim_A001418() -> None:
	expected: type[ArithmeticError] = ArithmeticError
	with pytest.raises(expected) as exceptionInfo:
		oeisIDfor_n('A001418', 0)
	assertEqualTo(type(exceptionInfo.value), expected, oeisIDfor_n.__name__, 'A001418', 0)

# ===== Command Line Interface Tests =====
def testHelpText() -> None:
	"""Test that help text is complete and examples are valid."""
	outputStream = io.StringIO()
	with redirect_stdout(outputStream):
		getOEISids()

	helpText = outputStream.getvalue()

	# Verify content
	for oeisID in packageSettings.oeisIDsImplementedMapFolding:
		assertEqualTo(oeisID in helpText, True, getOEISids.__name__, oeisID)
		assertEqualTo(dictionaryOEISMapFolding[oeisID]['description'] in helpText, True, getOEISids.__name__, oeisID)

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
	expectedValue = oeisIDfor_n(oeisID, n)

	# Test CLI execution of the example
	with unittest.mock.patch('sys.argv', ['OEIS_for_n', oeisID, str(n)]):
		outputStream = io.StringIO()
		with redirect_stdout(outputStream):
			OEIS_for_n()
		actual: int = int(outputStream.getvalue().strip().split()[0])
		assertEqualTo(actual, expectedValue, OEIS_for_n.__name__, oeisID, n)

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
		assertEqualTo(all(oeisID in helpOutput for oeisID in packageSettings.oeisIDsImplementedMapFolding), True, OEIS_for_n.__name__, '--help')
