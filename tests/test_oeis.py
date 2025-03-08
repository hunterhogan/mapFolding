from contextlib import redirect_stdout
from mapFolding.oeis import oeisIDfor_n, getOEISids, clearOEIScache, getOEISidValues, OEIS_for_n, oeisIDsImplemented, settingsOEIS, validateOEISid
from pathlib import Path
from tests.conftest import standardizedEqualToCallableReturn, standardizedSystemExit
from typing import Any, NoReturn
from urllib.error import URLError
import io
import pytest
import random
import re as regex
import unittest.mock
import urllib.request

@pytest.mark.parametrize("badID", ["A999999", "  A999999  ", "A999999extra"])
def test__validateOEISid_invalid_id(badID: str) -> None:
	standardizedEqualToCallableReturn(KeyError, validateOEISid, badID)

def test__validateOEISid_partially_valid(oeisID_1random: str) -> None:
	standardizedEqualToCallableReturn(KeyError, validateOEISid, f"{oeisID_1random}extra")

def test__validateOEISid_valid_id(oeisID: str) -> None:
	standardizedEqualToCallableReturn(oeisID, validateOEISid, oeisID)

def test__validateOEISid_valid_id_case_insensitive(oeisID: str) -> None:
	standardizedEqualToCallableReturn(oeisID.upper(), validateOEISid, oeisID.lower())
	standardizedEqualToCallableReturn(oeisID.upper(), validateOEISid, oeisID.upper())
	standardizedEqualToCallableReturn(oeisID.upper(), validateOEISid, oeisID.swapcase())

parameters_test_aOFn_invalid_n = [
	(-random.randint(1, 100), "randomNegative"),
	("foo", "string"),
	(1.5, "float")
]
badValues, badValuesIDs = zip(*parameters_test_aOFn_invalid_n)
@pytest.mark.parametrize("badN", badValues, ids=badValuesIDs)
def test_aOFn_invalid_n(oeisID_1random: str, badN: Any) -> None:
	"""Check that negative or non-integer n raises ValueError."""
	standardizedEqualToCallableReturn(ValueError, oeisIDfor_n, oeisID_1random, badN)

def test_aOFn_zeroDim_A001418() -> None:
	standardizedEqualToCallableReturn(ArithmeticError, oeisIDfor_n, 'A001418', 0)

# ===== OEIS Cache Tests =====
@pytest.mark.parametrize("cacheExists", [True, False])
@unittest.mock.patch('pathlib.Path.exists')
@unittest.mock.patch('pathlib.Path.unlink')
def test_clearOEIScache(mock_unlink: unittest.mock.MagicMock, mock_exists: unittest.mock.MagicMock, cacheExists: bool) -> None:
	"""Test OEIS cache clearing with both existing and non-existing cache."""
	mock_exists.return_value = cacheExists
	clearOEIScache()

	if cacheExists:
		# Each OEIS ID has two cache files
		expected_calls = len(settingsOEIS) * 2
		assert mock_unlink.call_count == expected_calls
		mock_unlink.assert_has_calls([unittest.mock.call(missing_ok=True)] * expected_calls)
	else:
		mock_exists.assert_called_once()
		mock_unlink.assert_not_called()

def testNetworkError(monkeypatch: pytest.MonkeyPatch, pathCacheTesting: Path) -> None:
	"""Test network error handling."""
	def mockUrlopen(*args: Any, **kwargs: Any) -> NoReturn:
		raise URLError("Network error")

	monkeypatch.setattr(urllib.request, 'urlopen', mockUrlopen)
	standardizedEqualToCallableReturn(URLError, getOEISidValues, next(iter(settingsOEIS)))

# ===== Command Line Interface Tests =====
def testHelpText() -> None:
	"""Test that help text is complete and examples are valid."""
	outputStream = io.StringIO()
	with redirect_stdout(outputStream):
		getOEISids()

	helpText = outputStream.getvalue()

	# Verify content
	for oeisID in oeisIDsImplemented:
		assert oeisID in helpText
		assert settingsOEIS[oeisID]['description'] in helpText

	# Extract and verify examples

	cliMatch = regex.search(r'OEIS_for_n (\w+) (\d+)', helpText)
	pythonMatch = regex.search(r"oeisIDfor_n\('(\w+)', (\d+)\)", helpText)

	assert cliMatch and pythonMatch, "Help text missing examples"
	oeisID, n = pythonMatch.groups()
	n = int(n)

	# Verify CLI and Python examples use same values
	assert cliMatch.groups() == (oeisID, str(n)), "CLI and Python examples inconsistent"

	# Verify the example works
	expectedValue = oeisIDfor_n(oeisID, n)

	# Test CLI execution of the example
	with unittest.mock.patch('sys.argv', ['OEIS_for_n', oeisID, str(n)]):
		outputStream = io.StringIO()
		with redirect_stdout(outputStream):
			OEIS_for_n()
		standardizedEqualToCallableReturn(expectedValue, lambda: int(outputStream.getvalue().strip().split()[0]))

def testCLI_InvalidInputs() -> None:
	"""Test CLI error handling."""
	testCases = [
		(['OEIS_for_n'], "missing arguments"),
		(['OEIS_for_n', 'A999999', '1'], "invalid OEIS ID"),
		(['OEIS_for_n', 'A001415', '-1'], "negative n"),
		(['OEIS_for_n', 'A001415', 'abc'], "non-integer n"),
	]

	for arguments, _testID in testCases:
		with unittest.mock.patch('sys.argv', arguments):
			standardizedSystemExit("error", OEIS_for_n)

def testCLI_HelpFlag() -> None:
	"""Verify --help output contains required information."""
	with unittest.mock.patch('sys.argv', ['OEIS_for_n', '--help']):
		outputStream = io.StringIO()
		with redirect_stdout(outputStream):
			standardizedSystemExit("nonError", OEIS_for_n)

		helpOutput = outputStream.getvalue()
		assert "Available OEIS sequences:" in helpOutput
		assert "Usage examples:" in helpOutput
		assert all(oeisID in helpOutput for oeisID in oeisIDsImplemented)
