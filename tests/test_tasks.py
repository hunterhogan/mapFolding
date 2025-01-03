# import random

# import pytest

# from mapFolding import foldings, getLeavesTotal

# @pytest.mark.parametrize("taskDivisionParameters", [
#     {'computationDivisions': 5000, 'computationIndex': 0, 'error': ".*"},  # computationDivisions > leavesTotal
#     {'computationDivisions': 2, 'computationIndex': 2, 'error': ".*"},  # computationIndex >= computationDivisions
#     {'computationDivisions': -1, 'computationIndex': 0, 'error': ".*"},  # Negative computationDivisions
#     {'computationDivisions': 2, 'computationIndex': -1, 'error': ".*"},  # Negative computationIndex
#     {'computationDivisions': 'abc', 'computationIndex': 0, 'error': ".*"},  # Invalid type computationDivisions
#     {'computationDivisions': 2, 'computationIndex': 'abc', 'error': ".*"},  # Invalid type computationIndex
#     {'computationDivisions': 2.5, 'computationIndex': 0, 'error': ".*"},  # Float computationDivisions
#     {'computationDivisions': 2, 'computationIndex': 1.5, 'error': ".*"},  # Float computationIndex
#     {'computationDivisions': None, 'computationIndex': 0, 'error': ".*"},  # None computationDivisions
#     {'computationDivisions': 2, 'computationIndex': None, 'error': ".*"},  # None computationIndex
# ])
# def test_taskDivisionParameters(listDimensions_validated, taskDivisionParameters):
#     """Test validation of computation task parameters."""
#     with pytest.raises((ValueError, TypeError), match=taskDivisionParameters['error']):
#         foldings(
#             listDimensions_validated,
#             taskDivisionParameters['computationDivisions'],  # type: ignore
#             taskDivisionParameters['computationIndex']  # type: ignore
#         )

# def test_foldings_computationDivisions(listDimensions_testCounts, dictionaryDimensionsFoldingsTotal):
#     leavesTotal = getLeavesTotal(listDimensions_testCounts)
#     leavesTotalMinimum = 2
#     if leavesTotalMinimum >= leavesTotal:
#         computationDivisions = leavesTotal
#     else:
#         computationDivisions = random.randint(leavesTotalMinimum, leavesTotal)
    
#     foldingsTotal = sum(
#         foldings(listDimensions_testCounts, computationDivisions, index) 
#         for index in range(computationDivisions)
#     )
#     assert foldingsTotal == dictionaryDimensionsFoldingsTotal[tuple(listDimensions_testCounts)]

