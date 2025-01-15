from .conftest import *
import pytest
from typing import List, Dict, Tuple

# TODO add a test. `C` = number of logical cores available. `n = C + 1`. Ensure that `[2,n]` is computed correctly.

def test_foldings_computationDivisions(listDimensionsTest_countFolds: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int]) -> None:
    standardComparison(foldsTotalKnown[tuple(listDimensionsTest_countFolds)], countFolds, listDimensionsTest_countFolds, True)

def test_defineConcurrencyLimit() -> None:
    testSuite = makeTestSuiteConcurrencyLimit(defineConcurrencyLimit)
    for testName, testFunction in testSuite.items():
        testFunction()

@pytest.mark.parametrize("cpuLimitValue", [{"invalid": True}, ["weird"]])
def test_countFolds_cpuLimitOopsie(cpuLimitValue: Dict[str, bool] | List[str]) -> None:
    # This forces CPUlimit = oopsieKwargsie(cpuLimitValue).
    standardComparison(ValueError, countFolds, [2, 2], True, cpuLimitValue)
