from .conftest import *
from mapFolding import foldings, getLeavesTotal, validateListDimensions
from mapFolding.__idiotic_system__ import *
import pytest
import random
import sys
import unittest.mock

# ===== getLeavesTotal Tests =====
def test_getLeavesTotal_valid(listDimensions_valid):
    """Test getLeavesTotal with valid inputs."""
    for dimensions, expected in listDimensions_valid:
        compareValues(expected, getLeavesTotal, dimensions)

def test_getLeavesTotal_invalid(listDimensions_invalid):
    """Test getLeavesTotal with invalid inputs."""
    for dimensions, errorType in listDimensions_invalid:
        expectError(errorType, getLeavesTotal, dimensions)

@pytest.mark.parametrize("sequenceType", [list, tuple, range])
def test_getLeavesTotal_sequence_types(sequenceType):
    """Test getLeavesTotal with different sequence types."""
    if sequenceType is range:
        sequence = range(1, 4)
    else:
        sequence = sequenceType([1, 2, 3])
    compareValues(6, getLeavesTotal, sequence)

def test_getLeavesTotal_edge_cases():
    """Test edge cases for getLeavesTotal."""
    # Order independence
    compareValues(getLeavesTotal([2, 3, 4]), getLeavesTotal, [4, 2, 3])
    
    # Immutability
    listOriginal = [2, 3]
    compareValues(6, getLeavesTotal, listOriginal)
    compareValues([2, 3], lambda x: x, listOriginal)  # Check list wasn't modified
    
    # Overflow protection
    largeNumber = sys.maxsize // 2
    expectError(OverflowError, getLeavesTotal, [largeNumber, largeNumber, 2])

def test_getLeavesTotal_properties():
    """Test properties that should hold for getLeavesTotal."""
    def generateValidDimensions():
        return [random.randint(1, 5) for index in range(random.randint(2, 4))]
    
    def checkCommutative(inputValue, result):
        return getLeavesTotal(sorted(inputValue)) == result
    
    # templatePropertyTest(getLeavesTotal, "commutative", generateValidDimensions, checkCommutative)

# ===== Dimension Validation Tests =====
@pytest.mark.parametrize("dimensions,expected", [
    ([2, 2], True),
    ([3, 2], True),
    ([2, 0, 2], True),  # zeros handled
    ([1], False),  # single dimension
    ([0, 0], False),  # no positive dimensions
    ([], False),  # empty
    ([-1, 2], False),  # negative
])
def test_dimension_validation(dimensions, expected):
    """Test dimension validation logic."""
    if expected:
        validateListDimensions(dimensions)
    else:
        with pytest.raises((ValueError, NotImplementedError)):
            validateListDimensions(dimensions)

# ===== Parse Integers Tests =====
# def test_intInnit():
#     """Test integer parsing using the test suite generator."""
#     for testName, testFunction in makeTestSuiteIntInnit(parseListDimensions).items():
#         testFunction()

def test_foldings_cuda_enabled(mockedCUDA, listDimensions_testCounts, dictionaryDimensionsFoldingsTotal):
    """Verify CUDA path produces correct results when GPU is available."""
    with unittest.mock.patch('mapFolding.lego.useGPU', True):
        foldingsTotalActual = foldings(listDimensions_testCounts)
        expectedFoldingsTotal = dictionaryDimensionsFoldingsTotal[tuple(sorted(listDimensions_testCounts))]
        compareValues(expectedFoldingsTotal, lambda: foldingsTotalActual)
        mockedCUDA.to_device.assert_called()

def test_foldings_cuda_disabled(listDimensions_testCounts, dictionaryDimensionsFoldingsTotal):
    """Verify CPU fallback works when GPU is disabled."""
    with unittest.mock.patch('mapFolding.lego.useGPU', False):
        foldingsTotalActual = foldings(listDimensions_testCounts)
        expectedFoldingsTotal = dictionaryDimensionsFoldingsTotal[tuple(sorted(listDimensions_testCounts))]
        compareValues(expectedFoldingsTotal, lambda: foldingsTotalActual)

def test_foldings_results_consistent(mockedCUDA, listDimensions_testCounts, dictionaryDimensionsFoldingsTotal):
    """Test that GPU and CPU paths produce the same results."""
    expectedFoldingsTotal = dictionaryDimensionsFoldingsTotal[tuple(sorted(listDimensions_testCounts))]

    with unittest.mock.patch('mapFolding.lego.useGPU', True):
        foldingsTotalGPU = foldings(listDimensions_testCounts)

    with unittest.mock.patch('mapFolding.lego.useGPU', False):
        foldingsTotalCPU = foldings(listDimensions_testCounts)

    assert foldingsTotalGPU == foldingsTotalCPU == expectedFoldingsTotal
