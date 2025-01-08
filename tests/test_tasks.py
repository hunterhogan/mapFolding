from .conftest import *
from mapFolding import countFolds

def test_foldings_computationDivisions(listDimensionsTest_countFolds, foldsTotalKnown):
    compareValues(foldsTotalKnown[tuple(listDimensionsTest_countFolds)], countFolds, listDimensionsTest_countFolds, True)
