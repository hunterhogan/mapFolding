from .conftest import *
import pytest

# TODO add a test. `C` = number of logical cores available. `n = C + 1`. Ensure that `[2,n]` is computed correctly.

def test_foldings_computationDivisions(listDimensionsTest_countFolds, foldsTotalKnown):
    standardComparison(foldsTotalKnown[tuple(listDimensionsTest_countFolds)], countFolds, listDimensionsTest_countFolds, True)

def test_defineConcurrencyLimit():
    testSuite = makeTestSuiteConcurrencyLimit(defineConcurrencyLimit)
    for testName, testFunction in testSuite.items():
        testFunction()

@pytest.mark.parametrize("cpuLimitValue", [{"invalid": True}, ["weird"]])
def test_countFolds_cpuLimitOopsie(cpuLimitValue):
    # This forces CPUlimit = oopsieKwargsie(cpuLimitValue).
    standardComparison(ValueError, countFolds, [2, 2], True, cpuLimitValue)
