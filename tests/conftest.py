from mapFolding import oeis
from mapFolding import validateListDimensions
from mapFolding.oeis import settingsOEISsequences, OEISsequenceID
from typing import Any, Dict, List, Tuple
import pytest
import random
import sys

@pytest.fixture(params=settingsOEISsequences.keys())
def oeisID(request):
    """Returns values from `settingsOEISsequences.keys()` not from `OEISsequenceID`."""
    return request.param

@pytest.fixture
def oeisIDrandom() -> OEISsequenceID:
    """Return a random valid OEIS ID from settings."""
    return random.choice(list(settingsOEISsequences.keys()))

@pytest.fixture
def dictionaryDimensionsFoldingsTotal():
    """Returns a dictionary mapping dimension tuples to their known folding totals."""
    from tests import generateDictionaryDimensionsFoldingsTotal
    return generateDictionaryDimensionsFoldingsTotal()

@pytest.fixture
def listDimensions_validated(dictionaryDimensionsFoldingsTotal: Dict[Tuple[int], int]):
    """Returns one `listDimensions` approved by `validateListDimensions`. The average time to count foldings
    for a random `listDimensions` is multiple months, so this fixture is not suitable for testing counts."""
    while True:
        listDimensionsCandidate = list(random.choice(list(dictionaryDimensionsFoldingsTotal.keys())))

        try:
            listDimensionsValidated = validateListDimensions(listDimensionsCandidate)
            return listDimensionsValidated
        except (ValueError, NotImplementedError):
            pass

@pytest.fixture
def listDimensions_testCounts(oeisID):
    """For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestValidation`
    if `validateListDimensions` approves. Each `listDimensions` is suitable for testing counts."""
    while True:
        n = random.choice(settingsOEISsequences[oeisID]['valuesTestValidation'])
        listDimensionsCandidate = settingsOEISsequences[oeisID]['getDimensions'](n)

        try:
            return validateListDimensions(listDimensionsCandidate)
        except (ValueError, NotImplementedError):
            pass

@pytest.fixture
def listDimensions_valid() -> List[Tuple[List[int], int]]:
    """Provide comprehensive test cases for valid dimension inputs."""
    return [
        # ([2, 3], 45546), # test the test templates
        ([2, 3], 6),
        ([2, 3, 4], 24),
        ([0, 1, 2], 2),  # zeros ignored
        ([0], 0),  # edge case
        ([1] * 1000, 1),  # long list
        ([1, sys.maxsize], sys.maxsize),  # maxint
        ([2] * 10, 1024),  # power of 2
        ([3] * 3, 27),  # power of 3
        ([2, 2, 2, 2], 16),  # repeated dimensions
        ([1, 2, 3, 4, 5], 120),  # sequential
        ([sys.maxsize - 1, 1], sys.maxsize - 1),  # near maxint
    ]

@pytest.fixture
def listDimensions_invalid() -> List[Tuple[Any, type]]:
    """Provide comprehensive test cases for invalid dimension inputs."""
    return [
        # ([], TypeError),  # test the test templates
        # ([2, 3], ValueError), # test the test templates
        ([], ValueError),  # empty
        ([-1], ValueError),  # negative
        ([1.5], ValueError),  # float
        (['a'], ValueError),  # string
        ([None], TypeError),  # None
        ([[1, 2]], TypeError),  # nested
        (None, ValueError),  # None instead of list
        ([True], TypeError),  # bool
        ([float('inf')], ValueError),  # infinity
        ([float('nan')], ValueError),  # NaN
        ([sys.maxsize, sys.maxsize], OverflowError),  # overflow
        ([complex(1,1)], ValueError),  # complex number
    ]

@pytest.fixture
def pathCacheTesting(tmp_path):
    """Temporarily replace the OEIS cache directory with a test directory."""
    pathCacheOriginal = oeis._pathCache
    oeis._pathCache = tmp_path
    yield tmp_path
    oeis._pathCache = pathCacheOriginal
