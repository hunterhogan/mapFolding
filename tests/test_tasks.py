from .conftest import *
from mapFolding import countFolds

# NOTE The problems caused by the intersections of numba.jit, pytest coverage, and GitHub actions (pytest tests),
# are likely contributing to an intermittent test failure here: when listDimensions_testCounts, at `A001418`, supplies listDimensions = [1,1]
def test_foldings_computationDivisions(listDimensions_testCounts, dictionaryDimensionsFoldingsTotal):
    compareValues(dictionaryDimensionsFoldingsTotal[tuple(listDimensions_testCounts)], countFolds, listDimensions_testCounts, True)
