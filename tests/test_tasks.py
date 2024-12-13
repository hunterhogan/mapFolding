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
    with pytest.raises(ValueError):
        foldings([], 1, 0)  # Empty dimensions (caught by intInnit)

    with pytest.raises(NotImplementedError):
        foldings([1], 1, 0)  # Only one dimension

    with pytest.raises(NotImplementedError):
        foldings([0, 0], 1, 0)  # No non-zero dimensions

    with pytest.raises(ValueError):
        foldings([1, -1], 1, 0)  # Negative dimension

    with pytest.raises(ValueError):
        foldings([1, 2], -1, 0)  # Negative computationDivisions

    with pytest.raises(ValueError):
        foldings([1, 2], 1, -1)  # Negative computationIndex

    with pytest.raises(ValueError):
        foldings([1, 2], 2, 2)  # computationIndex >= computationDivisions

    with pytest.raises(ValueError):
        foldings([1, 2], 10, 0)  # computationDivisions > leavesTotal

    with pytest.raises(ValueError):
        foldings([1.5, 2], 1, 0)  # Non-integer dimensions  # type: ignore

    with pytest.raises(TypeError):
        foldings([1, 2], 'abc', 0)  # Invalid type for computationDivisions  # type: ignore

    with pytest.raises(TypeError):
        foldings([1, 2], 1, 'abc')  # Invalid type for computationIndex  # type: ignore

def test_foldings_dimensions_parsing():
    with pytest.raises(ValueError):
        foldings([])  # Empty list
    
    with pytest.raises(ValueError):
        foldings([1, 2, 3.5])  # Non-integer dimensions # type: ignore

    with pytest.raises(ValueError):
        foldings([1, 2, -3])  # Negative dimensions

    # Test filtering of zero dimensions
    assert foldings([2, 0, 2], 1, 0) == foldings([2, 2], 1, 0)

def test_getLeavesTotal():
    # Test basic multiplication
    assert getLeavesTotal([2, 3]) == 6
    assert getLeavesTotal([2, 3, 4]) == 24

    # Test with zero dimensions
    assert getLeavesTotal([0, 0]) == 0
    assert getLeavesTotal([2, 0, 3]) == 6

    # Test error cases
    with pytest.raises(ValueError):
        getLeavesTotal([])  # Empty list

    with pytest.raises(ValueError):
        getLeavesTotal([-1, 2])  # Negative dimensions

    with pytest.raises(TypeError):
        getLeavesTotal(['1', '2'])  # Non-numeric values # type: ignore

def test_foldings_computation_divisions():
    with pytest.raises(ValueError):
        foldings([2, 2], 5, 0)  # computationDivisions > leavesTotal

    # Test computationIndex validation
    with pytest.raises(ValueError):
        foldings([2, 2], 2, -1)  # Negative computationIndex

    with pytest.raises(ValueError):
        foldings([2, 2], 2, 2)  # computationIndex >= computationDivisions

    # Test invalid dimensions
    with pytest.raises(NotImplementedError):
        foldings([1], 2, 0)  # Only one dimension

    # Test valid computation divisions
    assert foldings([2, 2], 2, 0) + foldings([2, 2], 2, 1) == foldings([2, 2])

