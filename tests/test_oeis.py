import pytest
from mapFolding import (
    settingsOEISsequences, 
    oeisSequence_aOFn, 
    getOEISsequence,
    dimensionsFoldingsTotalLookup
)

@pytest.mark.parametrize("seq_id", settingsOEISsequences.keys())
def test_calculate_sequence(seq_id):
    test_values = settingsOEISsequences[seq_id].get('testValuesValidation', [])
    expected_sequence = getOEISsequence(seq_id)
    
    for n in test_values:
        result = oeisSequence_aOFn(seq_id, n)
        assert isinstance(result, int)
        
        # Get expected value using two methods
        expected = expected_sequence.get(n)
        if expected is not None:
            assert result == expected
            
        # Also verify against dimensionsFoldingsTotalLookup
        dimensions = settingsOEISsequences[seq_id]['dimensions'](n)
        dimensions.sort()
        expected_from_lookup = dimensionsFoldingsTotalLookup.get(tuple(dimensions))
        if expected_from_lookup is not None:
            assert result == expected_from_lookup