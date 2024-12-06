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
            
