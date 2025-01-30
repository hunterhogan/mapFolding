from tests.conftest import *
import pytest
from typing import List, Dict, Tuple

# TODO add a test. `C` = number of logical cores available. `n = C + 1`. Ensure that `[2,n]` is computed correctly.
# Or, probably smarter: limit the number of cores, then run a test with C+1.

def test_algorithmSourceParallel(listDimensionsTestParallelization: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int], useAlgorithmDirectly) -> None:
    standardComparison(foldsTotalKnown[tuple(listDimensionsTestParallelization)], countFolds, listDimensionsTestParallelization, None, 'maximum')

def test_countFoldsComputationDivisionsInvalid(listDimensionsTestFunctionality: List[int]) -> None:
    standardComparison(ValueError, countFolds, listDimensionsTestFunctionality, None, {"wrong": "value"})

def test_countFoldsComputationDivisionsMaximum(listDimensionsTestParallelization: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int]) -> None:
    standardComparison(foldsTotalKnown[tuple(listDimensionsTestParallelization)], countFolds, listDimensionsTestParallelization, None, 'maximum')

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_defineConcurrencyLimit())
def test_defineConcurrencyLimit(nameOfTest, callablePytest):
    callablePytest()

# @pytest.mark.parametrize("CPUlimitParameter", [{"invalid": True}, ["weird"]])
# def test_countFolds_cpuLimitOopsie(listDimensionsTestFunctionality: List[int], CPUlimitParameter: Dict[str, bool] | List[str]) -> None:
#     standardComparison((AttributeError or ValueError), countFolds, listDimensionsTestFunctionality, None, 'cpu', CPUlimitParameter)

@pytest.mark.parametrize("computationDivisions, concurrencyLimit, listDimensions, expectedTaskDivisions", [
    (None, 4, [9, 11], 0),
    ("maximum", 4, [7, 11], 77),
    ("cpu", 4, [3, 7], 4),
    (["invalid"], 4, [19, 23], ValueError),
    (20, 4, [3,5], ValueError)
])
def test_getTaskDivisions(computationDivisions, concurrencyLimit, listDimensions, expectedTaskDivisions) -> None:
    standardComparison(expectedTaskDivisions, getTaskDivisions, computationDivisions, concurrencyLimit, None, listDimensions)
