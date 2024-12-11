import random
from typing import Dict, Tuple

import pytest

from mapFolding import foldings, getLeavesTotal, settingsOEISsequences


def buildTestPool():
    poolDimensionsToValue: Dict[Tuple, int] = {}
    for oeisID, settings in settingsOEISsequences.items():
        for n in settings['testValuesValidation']:
            dimensions = settings['dimensions'](n)
            dimensions.sort()
            poolDimensionsToValue[tuple(dimensions)] = settings['valuesKnown'][n]
    for dimensions in poolDimensionsToValue.copy():
        measure = [element for element in dimensions if element > 0]
        if len(measure) < 2:  # Skip 1D cases
            del poolDimensionsToValue[dimensions]
    return poolDimensionsToValue

@pytest.fixture(scope="module")
def poolTestCases():
    poolDimensionsToValue = buildTestPool()
    # Select 3 random test cases
    listDimensions = random.sample(list(poolDimensionsToValue.keys()), k=3)
    return [(list(dimensions), poolDimensionsToValue[dimensions]) for dimensions in listDimensions]

def test_foldings_computationDivisions(poolTestCases):
    for listDimensions, foldingsExpected in poolTestCases:
        leavesTotal = getLeavesTotal(listDimensions)
        leavesTotalMinimum = 2
        if leavesTotal <= leavesTotalMinimum:
            computationDivisions = leavesTotal
        else:
            computationDivisions = random.randint(leavesTotalMinimum, leavesTotal)
        
        foldingsTotal = sum(
            foldings(listDimensions, computationDivisions, index) 
            for index in range(computationDivisions)
        )
        assert foldingsTotal == foldingsExpected

def test_foldings_invalid_inputs():
    with pytest.raises(NotImplementedError):
        foldings([], 1, 0)  # Empty dimensions

    with pytest.raises(NotImplementedError):
        foldings([1], 1, 0)  # Only one dimension

    with pytest.raises(NotImplementedError):
        foldings([0, 0], 1, 0)  # No non-zero dimensions

    with pytest.raises(ValueError):
        foldings([1, -1], 1, 0) # Negative dimension

    with pytest.raises(ValueError):
        foldings([1,2], -1, 0) #Negative computationDivisions

    with pytest.raises(ValueError):
        foldings([1,2], 1, -1) #Negative computationIndex

    with pytest.raises(ValueError):
        foldings([1,2], 2, 2)  #computationIndex >= computationDivisions

    with pytest.raises(ValueError):
        foldings([1, 2], 10, 0) #computationDivisions > leavesTotal (for a 2x1 map)

    with pytest.raises(ValueError):
        foldings([1.5, 2], 1, 0) #Non-integer dimensions # type: ignore

    with pytest.raises(TypeError):
        foldings([1, 2], 'abc', 0) #Invalid type for computationDivisions # type: ignore

    with pytest.raises(TypeError):
        foldings([1, 2], 1, 'abc') #Invalid type for computationIndex # type: ignore

