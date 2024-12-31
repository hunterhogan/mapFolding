import random
import pytest
from mapFolding.oeis import settingsOEISsequences
from mapFolding import validateListDimensions

@pytest.fixture(params=settingsOEISsequences.keys())
def oeisID(request):
    """Returns values from `settingsOEISsequences.keys()` not from `OEISsequenceID`."""
    return request.param

@pytest.fixture
def listDimensionsValidated(oeisID):
    """For each `oeisID` from the `pytest.fixture`, returns `listDimensions` if `validateListDimensions` approves."""
    while True:
        listOFn = list(settingsOEISsequences[oeisID]['valuesKnown'].keys())
        n = random.choice(listOFn)
        listDimensionsCandidate = settingsOEISsequences[oeisID]['getDimensions'](n)

        try:
            listDimensionsValidated = validateListDimensions(listDimensionsCandidate)
            return listDimensionsValidated
        except (ValueError, NotImplementedError):
            pass

@pytest.fixture
def listDimensionsForTests(oeisID):
    """For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestValidation`
    if `validateListDimensions` approves."""
    while True:
        n = random.choice(settingsOEISsequences[oeisID]['valuesTestValidation'])
        listDimensionsCandidate = settingsOEISsequences[oeisID]['getDimensions'](n)

        try:
            listDimensionsValidated = validateListDimensions(listDimensionsCandidate)
            return listDimensionsValidated
        except (ValueError, NotImplementedError):
            pass
