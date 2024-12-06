import pytest
from mapFolding import (
    settingsOEISsequences, 
    oeisSequence_aOFn, 
    getOEISsequence,
)

@pytest.mark.parametrize("seq_id", settingsOEISsequences.keys())
def test_calculate_sequence(seq_id):
    test_values = settingsOEISsequences[seq_id].get('testValuesValidation', [])
    expected_sequence = getOEISsequence(seq_id)
    
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
    expected = getOEISsequence(seq_id).get(n)
    assert oeisSequence_aOFn(seq_id, n) == expected

