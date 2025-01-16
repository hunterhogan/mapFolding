from tests.conftest import *
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import io
import os
import pathlib
import pytest
import random
import re as regex
import unittest
import unittest.mock
import urllib.error
import urllib.request

def test_aOFn_calculate_value(oeisID: str):
    for n in settingsOEIS[oeisID]['valuesTestValidation']:
        standardComparison(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

@pytest.mark.parametrize("badID", ["A999999", "  A999999  ", "A999999extra"])
def test__validateOEISid_invalid_id(badID: str):
    standardComparison(KeyError, _validateOEISid, badID)

def test__validateOEISid_partially_valid(oeisID_1random: str):
    standardComparison(KeyError, _validateOEISid, f"{oeisID_1random}extra")

def test__validateOEISid_valid_id(oeisID: str):
    standardComparison(oeisID, _validateOEISid, oeisID)

def test__validateOEISid_valid_id_case_insensitive(oeisID: str):
    standardComparison(oeisID.upper(), _validateOEISid, oeisID.lower())
    standardComparison(oeisID.upper(), _validateOEISid, oeisID.upper())
    standardComparison(oeisID.upper(), _validateOEISid, oeisID.swapcase())

parameters_test_aOFn_invalid_n = [
    # (2, "ok"), # test the test template
    (-random.randint(1, 100), "randomNegative"),
    ("foo", "string"),
    (1.5, "float")
]
badValues, badValuesIDs = zip(*parameters_test_aOFn_invalid_n)
@pytest.mark.parametrize("badN", badValues, ids=badValuesIDs)
def test_aOFn_invalid_n(oeisID_1random: str, badN):
    """Check that negative or non-integer n raises ValueError."""
    standardComparison(ValueError, oeisIDfor_n, oeisID_1random, badN)

def test_aOFn_zeroDim_A001418():
    standardComparison(ArithmeticError, oeisIDfor_n, 'A001418', 0)

# ===== OEIS Cache Tests =====
@pytest.mark.parametrize("cacheExists", [True, False])
@unittest.mock.patch('pathlib.Path.exists')
@unittest.mock.patch('pathlib.Path.unlink')
def test_clearOEIScache(mock_unlink: unittest.mock.MagicMock, mock_exists: unittest.mock.MagicMock, cacheExists: bool):
    """Test OEIS cache clearing with both existing and non-existing cache."""
    mock_exists.return_value = cacheExists
    clearOEIScache()

    if cacheExists:
        assert mock_unlink.call_count == len(settingsOEIS)
        mock_unlink.assert_has_calls([unittest.mock.call(missing_ok=True)] * len(settingsOEIS))
    else:
        mock_exists.assert_called_once()
        mock_unlink.assert_not_called()

@pytest.mark.parametrize("scenarioCache", ["miss", "expired", "invalid"])
def testCacheScenarios(pathCacheTesting: pathlib.Path, oeisID_1random: str, scenarioCache: str) -> None:
    """Test cache scenarios: missing file, expired file, and invalid file."""

    def setupCacheExpired(pathCache: pathlib.Path, oeisID: str) -> None:
        pathCache.write_text("# Old cache content")
        oldModificationTime = datetime.now() - timedelta(days=30)
        os.utime(pathCache, times=(oldModificationTime.timestamp(), oldModificationTime.timestamp()))

    def setupCacheInvalid(pathCache: pathlib.Path, oeisID: str) -> None:
        pathCache.write_text("Invalid content")

    if scenarioCache == "miss":
        standardCacheTest(settingsOEIS[oeisID_1random]['valuesKnown'], None, oeisID_1random, pathCacheTesting)
    elif scenarioCache == "expired":
        standardCacheTest(settingsOEIS[oeisID_1random]['valuesKnown'], setupCacheExpired, oeisID_1random, pathCacheTesting)
    else:
        standardCacheTest(settingsOEIS[oeisID_1random]['valuesKnown'], setupCacheInvalid, oeisID_1random, pathCacheTesting)

def testInvalidFileContent(pathCacheTesting: pathlib.Path, oeisID_1random: str):
    pathFilenameCache = pathCacheTesting / _getFilenameOEISbFile(oeisID=oeisID_1random)

    # Write invalid content to cache
    pathFilenameCache.write_text("# A999999\n1 1\n2 2\n")
    modificationTimeOriginal = pathFilenameCache.stat().st_mtime

    # Function should detect invalid content, fetch fresh data, and update cache
    OEISsequence = _getOEISidValues(oeisID_1random)

    # Verify the function succeeded
    assert OEISsequence is not None
    # Verify cache was updated (modification time changed)
    assert pathFilenameCache.stat().st_mtime > modificationTimeOriginal
    # Verify cache now contains correct sequence ID
    assert f"# {oeisID_1random}" in pathFilenameCache.read_text()

def testParseContentErrors():
    """Test invalid content parsing."""
    standardComparison(ValueError, _parseBFileOEIS, "Invalid content\n1 2\n", 'A001415')

def testExtraComments(pathCacheTesting: pathlib.Path, oeisID_1random: str):
    pathFilenameCache = pathCacheTesting / _getFilenameOEISbFile(oeisID=oeisID_1random)

    # Write content with extra comment lines
    contentWithExtraComments = f"""# {oeisID_1random}
# Normal place for comment line 1
# Abnormal comment line
1 2
2 4
3 6
# Another comment in the middle
4 8
5 10"""
    pathFilenameCache.write_text(contentWithExtraComments)

    OEISsequence = _getOEISidValues(oeisID_1random)
    # Verify sequence values are correct despite extra comments
    standardComparison(2, lambda d: d[1], OEISsequence)  # First value
    standardComparison(8, lambda d: d[4], OEISsequence)  # Value after mid-sequence comment
    standardComparison(10, lambda d: d[5], OEISsequence)  # Last value

def testNetworkError(monkeypatch: pytest.MonkeyPatch, pathCacheTesting: pathlib.Path):
    """Test network error handling."""
    def mockUrlopen(*args, **kwargs):
        raise urllib.error.URLError("Network error")

    monkeypatch.setattr(urllib.request, 'urlopen', mockUrlopen)
    standardComparison(urllib.error.URLError, _getOEISidValues, next(iter(settingsOEIS)))

# ===== Command Line Interface Tests =====
def testHelpText():
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
        standardComparison(expectedValue, lambda: int(outputStream.getvalue().strip().split()[0]))

def testCLI_InvalidInputs():
    """Test CLI error handling."""
    testCases = [
        (['OEIS_for_n'], "missing arguments"),
        (['OEIS_for_n', 'A999999', '1'], "invalid OEIS ID"),
        (['OEIS_for_n', 'A001415', '-1'], "negative n"),
        (['OEIS_for_n', 'A001415', 'abc'], "non-integer n"),
    ]

    for arguments, testID in testCases:
        with unittest.mock.patch('sys.argv', arguments):
            expectSystemExit("error", OEIS_for_n)

def testCLI_HelpFlag():
    """Verify --help output contains required information."""
    with unittest.mock.patch('sys.argv', ['OEIS_for_n', '--help']):
        outputStream = io.StringIO()
        with redirect_stdout(outputStream):
            expectSystemExit("nonError", OEIS_for_n)

        helpOutput = outputStream.getvalue()
        assert "Available OEIS sequences:" in helpOutput
        assert "Usage examples:" in helpOutput
        assert all(oeisID in helpOutput for oeisID in oeisIDsImplemented)
