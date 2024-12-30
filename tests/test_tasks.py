import random
from typing import Dict, Tuple, List

import pytest

from mapFolding import foldings, getLeavesTotal, validateListDimensions
from mapFolding.oeis import settingsOEISsequences


def buildTestPool() -> Dict[Tuple[int, ...], int]:
    """Build a pool of valid test cases from OEIS sequences."""
    poolDimensionsToValue: Dict[Tuple[int, ...], int] = {}
    for oeisID, settings in settingsOEISsequences.items():
        for n in settings['testValuesValidation']:
            dimensions = settings['dimensions'](n)
            dimensions.sort()
            try:
                validateListDimensions(dimensions)
                poolDimensionsToValue[tuple(dimensions)] = settings['valuesKnown'][n]
            except Exception:
                pass
    return poolDimensionsToValue

@pytest.fixture(scope="module")
def poolTestCases() -> List[Tuple[List[int], int]]:
    """Provide a list of valid (dimensions, expected_value) test cases."""
    poolDimensionsToValue = buildTestPool()
    # Select 3 random test cases
    listDimensions = random.sample(list(poolDimensionsToValue.keys()), k=3)
    return [(list(dimensions), poolDimensionsToValue[dimensions]) for dimensions in listDimensions]

@pytest.mark.parametrize("task_parameters", [
    {'computationDivisions': 5000, 'computationIndex': 0, 'error': ".*"},  # computationDivisions > leavesTotal
    {'computationDivisions': 2, 'computationIndex': 2, 'error': ".*"},  # computationIndex >= computationDivisions
    {'computationDivisions': -1, 'computationIndex': 0, 'error': ".*"},  # Negative computationDivisions
    {'computationDivisions': 2, 'computationIndex': -1, 'error': ".*"},  # Negative computationIndex
    {'computationDivisions': 'abc', 'computationIndex': 0, 'error': ".*"},  # Invalid type computationDivisions
    {'computationDivisions': 2, 'computationIndex': 'abc', 'error': ".*"},  # Invalid type computationIndex
    {'computationDivisions': 2.5, 'computationIndex': 0, 'error': ".*"},  # Float computationDivisions
    {'computationDivisions': 2, 'computationIndex': 1.5, 'error': ".*"},  # Float computationIndex
    {'computationDivisions': None, 'computationIndex': 0, 'error': ".*"},  # None computationDivisions
    {'computationDivisions': 2, 'computationIndex': None, 'error': ".*"},  # None computationIndex
])
def test_task_validation(poolTestCases, task_parameters):
    """Test validation of computation task parameters."""
    # Get first test case dimensions
    listDimensions, DISCARDexpected = poolTestCases[0]
    
    with pytest.raises((ValueError, TypeError), match=task_parameters['error']):
        foldings(
            listDimensions, 
            task_parameters['computationDivisions'],  # type: ignore
            task_parameters['computationIndex']  # type: ignore
        )

# def test_foldings_computationDivisions(poolTestCases):
#     for listDimensions, foldingsExpected in poolTestCases:
#         leavesTotal = getLeavesTotal(listDimensions)
#         leavesTotalMinimum = 2
#         if leavesTotal <= leavesTotalMinimum:
#             computationDivisions = leavesTotal
#         else:
#             computationDivisions = random.randint(leavesTotalMinimum, leavesTotal)
        
#         foldingsTotal = sum(
#             foldings(listDimensions, computationDivisions, index) 
#             for index in range(computationDivisions)
#         )
#         assert foldingsTotal == foldingsExpected

