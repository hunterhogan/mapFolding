from datetime import datetime, timedelta
from mapFolding import oeis, settingsOEISsequences, oeisSequence_aOFn 
import os
import pytest
import urllib.error
import urllib.request
import random

@pytest.fixture(params=settingsOEISsequences.keys(), autouse=True)
def oeisID(request):
    return request.param

def test_calculate_sequence(oeisID):
    for n in settingsOEISsequences[oeisID]['testValuesValidation']:
        result = oeisSequence_aOFn(oeisID, n)
        expected = settingsOEISsequences[oeisID]['valuesKnown'][n]
        assert result == expected

def test_dimensions_lookup(oeisID):
    valuesKnown = settingsOEISsequences[oeisID]['valuesKnown']
    # Get random n and its value
    n = random.choice(list(valuesKnown.keys()))
    expectedValue = valuesKnown[n]
    
    # Get dimensions for this n
    dimensions = sorted(settingsOEISsequences[oeisID]['dimensions'](n))
    
    # Test the lookup
    assert oeis.dimensionsFoldingsTotalLookup[tuple(dimensions)] == expectedValue

def test_invalid_sequence():
    with pytest.raises(KeyError):
        oeisSequence_aOFn('A999999', 1) # type: ignore

def test_negative_n():
    with pytest.raises(ValueError):
        oeisSequence_aOFn('A001415', -1)

def testInvalidDimensions():
    # Test with invalid dimensions that don't match any known sequence
    dimensions = (999, 999)  # tuple of dimensions not in lookup
    assert dimensions not in oeis.dimensionsFoldingsTotalLookup

def testCacheMiss(tmp_path):
    # Simulate cache miss by using a temporary cache directory
    pathCacheOriginal = oeis.pathCache
    oeis.pathCache = tmp_path
    sequenceID = 'A001415'
    pathFilenameCache = oeis.pathCache / f"{sequenceID}.txt"
    assert not pathFilenameCache.exists()
    OEISsequence = oeis._getOEISsequence(sequenceID)
    assert OEISsequence is not None
    assert pathFilenameCache.exists()
    oeis.pathCache = pathCacheOriginal

def testCacheExpired(tmp_path):
    # Simulate expired cache by setting an old modification time
    pathCacheOriginal = oeis.pathCache
    oeis.pathCache = tmp_path
    sequenceID = 'A001415'
    pathFilenameCache = oeis.pathCache / f"{sequenceID}.txt"
    pathFilenameCache.write_text("# Old cache content")
    oldModificationTime = datetime.now() - timedelta(days=30)
    os.utime(pathFilenameCache, times=(oldModificationTime.timestamp(), oldModificationTime.timestamp()))
    OEISsequence = oeis._getOEISsequence(sequenceID)
    assert OEISsequence is not None
    oeis.pathCache = pathCacheOriginal

def testInvalidCache(tmp_path):
    # Simulate invalid cache content
    pathCacheOriginal = oeis.pathCache
    oeis.pathCache = tmp_path
    sequenceID = 'A001415'
    pathFilenameCache = oeis.pathCache / f"{sequenceID}.txt"
    pathFilenameCache.write_text("Invalid content")
    OEISsequence = oeis._getOEISsequence(sequenceID)
    assert OEISsequence is not None
    oeis.pathCache = pathCacheOriginal

def testInvalidFileContent(tmp_path):
    # Test recovery from invalid cache content
    pathCacheOriginal = oeis.pathCache
    oeis.pathCache = tmp_path
    sequenceID = 'A001415'
    pathFilenameCache = oeis.pathCache / f"{sequenceID}.txt"
    
    # Write invalid content to cache
    pathFilenameCache.write_text("# A999999\n1 1\n2 2\n")
    modificationTimeOriginal = pathFilenameCache.stat().st_mtime
    
    # Function should detect invalid content, fetch fresh data, and update cache
    OEISsequence = oeis._getOEISsequence(sequenceID)
    
    # Verify the function succeeded
    assert OEISsequence is not None
    # Verify cache was updated (modification time changed)
    assert pathFilenameCache.stat().st_mtime > modificationTimeOriginal
    # Verify cache now contains correct sequence ID
    assert f"# {sequenceID}" in pathFilenameCache.read_text()
    
    oeis.pathCache = pathCacheOriginal

def testNetworkError(monkeypatch, tmp_path):
    # Test network error when no valid cache exists
    pathCacheOriginal = oeis.pathCache
    oeis.pathCache = tmp_path  # Use empty temp directory to ensure no valid cache
    
    def mockUrlopen(*args, **kwargs):
        raise urllib.error.URLError("Network error")
    
    monkeypatch.setattr(urllib.request, 'urlopen', mockUrlopen)
    with pytest.raises(urllib.error.URLError):
        oeis._getOEISsequence('A001415')
    
    oeis.pathCache = pathCacheOriginal

def testParseContentErrors():
    # Test invalid content parsing
    with pytest.raises(ValueError):
        oeis._parseContent("Invalid content\n1 2\n", 'A001415')

def testExtraComments(tmp_path):
    # Test that extra comments don't affect sequence parsing
    pathCacheOriginal = oeis.pathCache
    oeis.pathCache = tmp_path
    sequenceID = 'A001415'
    pathFilenameCache = oeis.pathCache / f"{sequenceID}.txt"
    
    # Write content with extra comment lines
    contentWithExtraComments = f"""# {sequenceID}
# Extra comment line 1
# Extra comment line 2
1 2
2 4
3 6
# Another comment in the middle
4 8
5 10"""
    pathFilenameCache.write_text(contentWithExtraComments)
    
    OEISsequence = oeis._getOEISsequence(sequenceID)
    # Verify sequence values are correct despite extra comments
    assert OEISsequence[1] == 2  # First value
    assert OEISsequence[4] == 8  # Value after mid-sequence comment
    assert OEISsequence[5] == 10  # Last value
    
    oeis.pathCache = pathCacheOriginal
