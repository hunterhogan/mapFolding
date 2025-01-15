from .conftest import *
import pytest
from typing import List, Dict, Tuple

# TODO add a test. `C` = number of logical cores available. `n = C + 1`. Ensure that `[2,n]` is computed correctly.

def test_countFolds_computationDivisions(listDimensionsTest_countFolds: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int]) -> None:
    standardComparison(foldsTotalKnown[tuple(listDimensionsTest_countFolds)], countFolds, listDimensionsTest_countFolds, None, 'maximum')

def test_defineConcurrencyLimit() -> None:
    testSuite = makeTestSuiteConcurrencyLimit(defineConcurrencyLimit)
    for testName, testFunction in testSuite.items():
        testFunction()

@pytest.mark.parametrize("CPUlimitParameter", [{"invalid": True}, ["weird"]])
def test_countFolds_cpuLimitOopsie(listDimensionsTestFunctionality: List[int], CPUlimitParameter: Dict[str, bool] | List[str]) -> None:
    # This forces CPUlimit = oopsieKwargsie(cpuLimitValue).
    standardComparison(ValueError, countFolds, listDimensionsTestFunctionality, None, 'cpu', CPUlimitParameter)

def test_countFolds_invalid_computationDivisions(listDimensionsTestFunctionality: List[int]) -> None:
    standardComparison(ValueError, countFolds, listDimensionsTestFunctionality, None, {"wrong": "value"})
