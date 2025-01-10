from .conftest import *
import pytest

def test_foldings_computationDivisions(listDimensionsTest_countFolds, foldsTotalKnown):
    standardComparison(foldsTotalKnown[tuple(listDimensionsTest_countFolds)], countFolds, listDimensionsTest_countFolds, True)

def test_defineConcurrencyLimit():
    testSuite = makeTestSuiteConcurrencyLimit(defineConcurrencyLimit)
    for testName, testFunction in testSuite.items():
        testFunction()

@pytest.mark.parametrize("cpuLimitValue", [{"invalid": True}, ["weird"]])
def test_countFolds_cpuLimitOopsie(cpuLimitValue):
    # This forces CPUlimit = oopsieKwargsie(cpuLimitValue).
    standardComparison(ValueError, countFolds, [2, 2], False, cpuLimitValue)
