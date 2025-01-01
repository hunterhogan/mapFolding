from .conftest import expectError, compareValues
from mapFolding import oeis
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from typing import get_args
import io
import os
import pytest
import random
import urllib.error
import urllib.request

def test_aOFn_calculate_value(oeisID):
    for n in oeis.settingsOEISsequences[oeisID]['valuesTestValidation']:
        compareValues(oeis.settingsOEISsequences[oeisID]['valuesKnown'][n], oeis.oeisSequence_aOFn, oeisID, n)

@pytest.mark.parametrize("badID", ["A999999", "  A999999  ", "A999999extra"])
def test__validateOEISid_invalid_id(badID):
    """Check that invalid or unknown IDs raise KeyError."""
    expectError(KeyError, oeis._validateOEISid, badID)

def test__validateOEISid_partially_valid(oeisIDrandom):
    expectError(KeyError, oeis._validateOEISid, f"{oeisIDrandom}extra")

def test__validateOEISid_valid_id(oeisID):
    compareValues(oeisID, oeis._validateOEISid, oeisID)

def test__validateOEISid_valid_id_case_insensitive(oeisID):
    compareValues(oeisID.upper(), oeis._validateOEISid, oeisID.lower())
    compareValues(oeisID.upper(), oeis._validateOEISid, oeisID.upper())
    compareValues(oeisID.upper(), oeis._validateOEISid, oeisID.swapcase())

parameters_test_aOFn_invalid_n = [
    # (2, "ok"), # test the test template
    (-random.randint(1, 100), "randomNegative"),
    ("foo", "string"),
    (1.5, "float")
]
badValues, badValuesIDs = zip(*parameters_test_aOFn_invalid_n)
@pytest.mark.parametrize("badN", badValues, ids=badValuesIDs)
def test_aOFn_invalid_n(oeisIDrandom, badN):
    """Check that negative or non-integer n raises ValueError."""
    expectError(ValueError, oeis.oeisSequence_aOFn, oeisIDrandom, badN)

def test_aOFn_zeroDim_A001418():
    with pytest.raises(ArithmeticError):
        oeis.oeisSequence_aOFn('A001418', 0)

# ===== Cache-related Tests =====
def testCacheMiss(pathCacheTesting, oeisIDrandom):
    pathFilenameCache = pathCacheTesting / oeis._formatFilenameCache.format(oeisID=oeisIDrandom)
    
    assert not pathFilenameCache.exists()
    OEISsequence = oeis._getOEISidValues(oeisIDrandom)
    assert OEISsequence is not None
    assert pathFilenameCache.exists()

def testCacheExpired(pathCacheTesting, oeisIDrandom):
    pathFilenameCache = pathCacheTesting / oeis._formatFilenameCache.format(oeisID=oeisIDrandom)
    pathFilenameCache.write_text("# Old cache content")
    oldModificationTime = datetime.now() - timedelta(days=30)
    os.utime(pathFilenameCache, times=(oldModificationTime.timestamp(), oldModificationTime.timestamp()))
    OEISsequence = oeis._getOEISidValues(oeisIDrandom)
    assert OEISsequence is not None

def testInvalidCache(pathCacheTesting, oeisIDrandom):
    pathFilenameCache = pathCacheTesting / oeis._formatFilenameCache.format(oeisID=oeisIDrandom)
    pathFilenameCache.write_text("Invalid content")
    OEISsequence = oeis._getOEISidValues(oeisIDrandom)
    assert OEISsequence is not None

def testInvalidFileContent(pathCacheTesting, oeisIDrandom):
    pathFilenameCache = pathCacheTesting / oeis._formatFilenameCache.format(oeisID=oeisIDrandom)
    
    # Write invalid content to cache
    pathFilenameCache.write_text("# A999999\n1 1\n2 2\n")
    modificationTimeOriginal = pathFilenameCache.stat().st_mtime
    
    # Function should detect invalid content, fetch fresh data, and update cache
    OEISsequence = oeis._getOEISidValues(oeisIDrandom)
    
    # Verify the function succeeded
    assert OEISsequence is not None
    # Verify cache was updated (modification time changed)
    assert pathFilenameCache.stat().st_mtime > modificationTimeOriginal
    # Verify cache now contains correct sequence ID
    assert f"# {oeisIDrandom}" in pathFilenameCache.read_text()

def testNetworkError(monkeypatch, pathCacheTesting):
    def mockUrlopen(*args, **kwargs):
        raise urllib.error.URLError("Network error")
    
    monkeypatch.setattr(urllib.request, 'urlopen', mockUrlopen)
    with pytest.raises(urllib.error.URLError):
        oeis._getOEISidValues(next(iter(oeis.settingsOEISsequences)))

def testParseContentErrors():
    """Test invalid content parsing."""
    expectError(ValueError, oeis._parseBFileOEIS, "Invalid content\n1 2\n", 'A001415')

def testExtraComments(pathCacheTesting, oeisIDrandom):
    pathFilenameCache = pathCacheTesting / oeis._formatFilenameCache.format(oeisID=oeisIDrandom)
    
    # Write content with extra comment lines
    contentWithExtraComments = f"""# {oeisIDrandom}
# Extra comment line 1
# Extra comment line 2
1 2
2 4
3 6
# Another comment in the middle
4 8
5 10"""
    pathFilenameCache.write_text(contentWithExtraComments)
    
    OEISsequence = oeis._getOEISidValues(oeisIDrandom)
    # Verify sequence values are correct despite extra comments
    compareValues(2, lambda d: d[1], OEISsequence)  # First value
    compareValues(8, lambda d: d[4], OEISsequence)  # Value after mid-sequence comment
    compareValues(10, lambda d: d[5], OEISsequence)  # Last value

# ===== Command Line Interface Tests =====
def testGetOEISids():
    """Test that oeis.getOEISids prints all sequences with descriptions."""
    captureOutput = io.StringIO()
    with redirect_stdout(captureOutput):
        oeis.getOEISids()
    
    outputText = captureOutput.getvalue()
    from mapFolding.oeis import settingsOEISsequences
    # Check that all sequences are listed
    for oeisID in get_args(oeis.OEISsequenceID):
        assert oeisID in outputText
        assert settingsOEISsequences[oeisID]['description'] in outputText
    
    # Check that usage example is included
    assert "Usage example:" in outputText
    assert "oeisSequence_aOFn" in outputText

def testCommandLineInterface():
    """Test running as command line script."""
    captureOutput = io.StringIO()
    with redirect_stdout(captureOutput):
        oeis.getOEISids()
    outputText = captureOutput.getvalue()
    assert "Available OEIS sequences:" in outputText
