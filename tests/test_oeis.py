import pytest
from mapFolding import settingsOEISsequences, oeisSequence_aOFn 
from datetime import datetime, timedelta
from mapFolding import oeis
import os

@pytest.mark.parametrize("seq_id", settingsOEISsequences.keys())
def test_calculate_sequence(seq_id):
    test_values = settingsOEISsequences[seq_id].get('testValuesValidation', [])
    expected_sequence = settingsOEISsequences[seq_id]['valuesKnown']
    
    for n in test_values:
        result = oeisSequence_aOFn(seq_id, n)
        expected = expected_sequence.get(n)
        assert result == expected

def test_invalid_sequence():
    with pytest.raises(KeyError):
        oeisSequence_aOFn('A999999', 1)

def test_negative_n():
    with pytest.raises(ValueError):
        oeisSequence_aOFn('A001415', -1)

def test_zero_n():
    for seq_id in settingsOEISsequences.keys():
        assert oeisSequence_aOFn(seq_id, 0) == 1

@pytest.mark.parametrize("seq_id,n", [
    ('A001415', 1),  # 2xn strip
    ('A001416', 1),  # 3xn strip
    ('A001417', 1),  # 2x2x...x2 n-dim
    ('A195646', 1),  # 3x3x...x3 n-dim
    ('A001418', 1),  # nxn sheet
])
def test_trivial_cases(seq_id, n):
    expected = settingsOEISsequences[seq_id]['valuesKnown'].get(n)
    assert oeisSequence_aOFn(seq_id, n) == expected

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
