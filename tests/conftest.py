import random
import pytest
from mapFolding.oeis import settingsOEISsequences
from mapFolding import validateListDimensions

@pytest.fixture(params=settingsOEISsequences.keys())
def oeisID(request):
    return request.param

def hasTwoOrMorePositive(listDimensions):
    """Check if list has at least 2 positive dimensions."""
    return len([dim for dim in listDimensions if dim > 0]) >= 2

@pytest.fixture
def valid_dimensions_and_foldings(oeisID):
    """Pick a random countTerm with 2+ positive dimensions in the given seq."""
    dictionaryValuesKnown = settingsOEISsequences[oeisID]['valuesKnown']
    listCountTerms = [
        countTerm 
        for countTerm in dictionaryValuesKnown 
        if hasTwoOrMorePositive(settingsOEISsequences[oeisID]['dimensions'](countTerm))
    ]
    if not listCountTerms:
        pytest.skip(f"No valid dimensions for {oeisID}")
    chosenCountTerm = random.choice(listCountTerms)
    listValidated = validateListDimensions(
        settingsOEISsequences[oeisID]['dimensions'](chosenCountTerm)
    )
    return listValidated, dictionaryValuesKnown[chosenCountTerm]
