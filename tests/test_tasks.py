from .conftest import *
import pytest
from typing import List, Dict, Tuple

# TODO add a test. `C` = number of logical cores available. `n = C + 1`. Ensure that `[2,n]` is computed correctly.
# Or, probably smarter: limit the number of cores, then run a test with C+1.

def test_countFolds_computationDivisions(listDimensionsTest_countFolds: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int]) -> None:
    standardComparison(foldsTotalKnown[tuple(listDimensionsTest_countFolds)], countFolds, listDimensionsTest_countFolds, None, 'maximum')

def test_defineConcurrencyLimit() -> None:
    testSuite = makeTestSuiteConcurrencyLimit(defineConcurrencyLimit)
    for testName, testFunction in testSuite.items():
        testFunction()

@pytest.mark.parametrize("CPUlimitParameter", [{"invalid": True}, ["weird"]])
def test_countFolds_cpuLimitOopsie(listDimensionsTestFunctionality: List[int], CPUlimitParameter: Dict[str, bool] | List[str]) -> None:
    standardComparison(ValueError, countFolds, listDimensionsTestFunctionality, None, 'cpu', CPUlimitParameter)

def test_countFolds_invalid_computationDivisions(listDimensionsTestFunctionality: List[int]) -> None:
    standardComparison(ValueError, countFolds, listDimensionsTestFunctionality, None, {"wrong": "value"})

@pytest.mark.parametrize("computationDivisions, concurrencyLimit, listDimensions, expectedTaskDivisions", [
    (None, 4, [9, 11], 0),
    ("maximum", 4, [7, 11], 77),
    ("cpu", 4, [3, 7], 4),
    (["invalid"], 4, [19, 23], ValueError),
    (20, 4, [3,5], ValueError)
])
def test_getTaskDivisions(computationDivisions, concurrencyLimit, listDimensions, expectedTaskDivisions) -> None:
    standardComparison(expectedTaskDivisions, getTaskDivisions, computationDivisions, concurrencyLimit, None, listDimensions)
