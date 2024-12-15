import io
import os
import random
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from typing import get_args
import pytest

from mapFolding.noCircularImportsIsAlie import getFoldingsTotalKnown
from mapFolding import getOEISids
from mapFolding import oeisSequence_aOFn, settingsOEISsequences, OEISsequenceID
from mapFolding.oeis import _formatFilenameCache, _getOEISsequence, _parseBFileOEIS
from mapFolding.oeis import _validateOEISid

from mapFolding import oeis

@pytest.fixture
def temporaryCache(tmp_path):
    """Temporarily replace the OEIS cache directory with a test directory."""
    pathCacheOriginal = oeis._pathCache
    oeis._pathCache = tmp_path
    yield tmp_path
    oeis._pathCache = pathCacheOriginal

@pytest.fixture(params=settingsOEISsequences.keys())
def oeisID(request):
    return request.param

@pytest.fixture
def sequenceIDsample() -> OEISsequenceID:
    """Get a consistent sample sequence ID for cache-related tests."""
    return next(iter(settingsOEISsequences))  # First sequence ID is fine for these tests

def test_calculate_sequence(oeisID):
    # Use known values directly from settings
    for n in settingsOEISsequences[oeisID]['testValuesValidation']:
        result = oeisSequence_aOFn(oeisID, n)
        expected = settingsOEISsequences[oeisID]['valuesKnown'][n]
        assert result == expected

def test_dimensions_lookup(oeisID):
    dictionaryValuesKnown = settingsOEISsequences[oeisID]['valuesKnown']
    # Get n values where n >= 2
    listCountTermsValid = [countTerm for countTerm in dictionaryValuesKnown.keys() if countTerm >= 2]
    countTerm = random.choice(listCountTermsValid)
    foldingsExpected = dictionaryValuesKnown[countTerm]
    
    # Get dimensions for this n
    listDimensions = sorted(settingsOEISsequences[oeisID]['dimensions'](countTerm))
    
    # Test the lookup
    assert getFoldingsTotalKnown(listDimensions) == foldingsExpected

def test_invalid_sequence():
    with pytest.raises(KeyError):
        oeisSequence_aOFn('A999999', 1) # type: ignore

def test_negative_n():
    with pytest.raises(ValueError):
        oeisSequence_aOFn('A001415', -1)

def testCacheMiss(temporaryCache, sequenceIDsample):
    pathFilenameCache = temporaryCache / _formatFilenameCache.format(oeisID=sequenceIDsample)
    
    assert not pathFilenameCache.exists()
    OEISsequence = _getOEISsequence(sequenceIDsample)
    assert OEISsequence is not None
    assert pathFilenameCache.exists()

def testCacheExpired(temporaryCache, sequenceIDsample):
    pathFilenameCache = temporaryCache / _formatFilenameCache.format(oeisID=sequenceIDsample)
    pathFilenameCache.write_text("# Old cache content")
    oldModificationTime = datetime.now() - timedelta(days=30)
    os.utime(pathFilenameCache, times=(oldModificationTime.timestamp(), oldModificationTime.timestamp()))
    OEISsequence = _getOEISsequence(sequenceIDsample)
    assert OEISsequence is not None

def testInvalidCache(temporaryCache, sequenceIDsample):
    pathFilenameCache = temporaryCache / _formatFilenameCache.format(oeisID=sequenceIDsample)
    pathFilenameCache.write_text("Invalid content")
    OEISsequence = _getOEISsequence(sequenceIDsample)
    assert OEISsequence is not None

def testInvalidFileContent(temporaryCache, sequenceIDsample):
    pathFilenameCache = temporaryCache / _formatFilenameCache.format(oeisID=sequenceIDsample)
    
    # Write invalid content to cache
    pathFilenameCache.write_text("# A999999\n1 1\n2 2\n")
    modificationTimeOriginal = pathFilenameCache.stat().st_mtime
    
    # Function should detect invalid content, fetch fresh data, and update cache
    OEISsequence = _getOEISsequence(sequenceIDsample)
    
    # Verify the function succeeded
    assert OEISsequence is not None
    # Verify cache was updated (modification time changed)
    assert pathFilenameCache.stat().st_mtime > modificationTimeOriginal
    # Verify cache now contains correct sequence ID
    assert f"# {sequenceIDsample}" in pathFilenameCache.read_text()

def testNetworkError(monkeypatch, temporaryCache):
    def mockUrlopen(*args, **kwargs):
        raise urllib.error.URLError("Network error")
    
    monkeypatch.setattr(urllib.request, 'urlopen', mockUrlopen)
    with pytest.raises(urllib.error.URLError):
        _getOEISsequence(next(iter(settingsOEISsequences)))

def testParseContentErrors():
    # Test invalid content parsing
    with pytest.raises(ValueError):
        _parseBFileOEIS("Invalid content\n1 2\n", 'A001415')

def testExtraComments(temporaryCache, sequenceIDsample):
    pathFilenameCache = temporaryCache / _formatFilenameCache.format(oeisID=sequenceIDsample)
    
    # Write content with extra comment lines
    contentWithExtraComments = f"""# {sequenceIDsample}
# Extra comment line 1
# Extra comment line 2
1 2
2 4
3 6
# Another comment in the middle
4 8
5 10"""
    pathFilenameCache.write_text(contentWithExtraComments)
    
    OEISsequence = _getOEISsequence(sequenceIDsample)
    # Verify sequence values are correct despite extra comments
    assert OEISsequence[1] == 2  # First value
    assert OEISsequence[4] == 8  # Value after mid-sequence comment
    assert OEISsequence[5] == 10  # Last value

def testGetOEISids():
    """Test that getOEISids prints all sequences with descriptions."""
    captureOutput = io.StringIO()
    with redirect_stdout(captureOutput):
        getOEISids()
    
    outputText = captureOutput.getvalue()
    
    # Check that all sequences are listed
    for oeisID in get_args(OEISsequenceID):
        assert oeisID in outputText
        assert settingsOEISsequences[oeisID]['description'] in outputText
    
    # Check that usage example is included
    assert "Usage example:" in outputText
    assert "oeisSequence_aOFn" in outputText

def testCommandLineInterface():
    """Test running as command line script."""
    captureOutput = io.StringIO()
    with redirect_stdout(captureOutput):
        from mapFolding.oeis import getOEISids
        getOEISids()
    outputText = captureOutput.getvalue()
    assert "Available OEIS sequences:" in outputText

def test__validateOEISid_valid_id(oeisID):
    assert _validateOEISid(oeisID) == oeisID

def test__validateOEISid_valid_id_case_insensitive(oeisID):
    assert _validateOEISid(oeisID.lower()) == oeisID.upper()
    assert _validateOEISid(oeisID.upper()) == oeisID.upper()
    assert _validateOEISid(oeisID.swapcase()) == oeisID.upper()

@pytest.mark.parametrize("invalidIDtext", [
    "A999999",
    "  A999999  ",
    "a999999",
    "A999999 "
])
def test__validateOEISid_invalid_id(invalidIDtext):
    with pytest.raises(KeyError):
        _validateOEISid(invalidIDtext)

def test__validateOEISid_partially_valid(sequenceIDsample):
    with pytest.raises(KeyError):
        _validateOEISid(f"{sequenceIDsample}extra")

