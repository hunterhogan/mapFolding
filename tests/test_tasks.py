from .conftest import *
from mapFolding import countFolds
from mapFolding.babbage import defineConcurrencyLimit
from Z0Z_tools.pytest_parseParameters import makeTestSuiteConcurrencyLimit

def test_foldings_computationDivisions(listDimensionsTest_countFolds, foldsTotalKnown):
    compareValues(foldsTotalKnown[tuple(listDimensionsTest_countFolds)], countFolds, listDimensionsTest_countFolds, True)

def test_defineConcurrencyLimit():
    testSuite = makeTestSuiteConcurrencyLimit(defineConcurrencyLimit)
    for testName, testFunction in testSuite.items():
        testFunction()
