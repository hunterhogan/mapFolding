from .conftest import *
from mapFolding.__fucking_idiotic_system__ import *
import pytest
import random
import sys

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
