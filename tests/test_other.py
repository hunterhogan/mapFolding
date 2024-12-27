import sys
from unittest.mock import patch, call
from typing import Any, List

import pytest

from mapFolding import (
    clearOEIScache, getLeavesTotal, parseListDimensions, 
    foldings, validateListDimensions
)
from mapFolding.oeis import settingsOEISsequences
from mapFolding.noCircularImportsIsAlie import getFoldingsTotalKnown

# ===== OEIS Cache Tests =====
@pytest.mark.parametrize("cacheExists", [True, False])
@patch('pathlib.Path.exists')
@patch('pathlib.Path.unlink')
def test_clear_OEIScache(mock_unlink, mock_exists, cacheExists):
    """Test OEIS cache clearing with both existing and non-existing cache."""
    mock_exists.return_value = cacheExists
    clearOEIScache()
    
    if cacheExists:
        assert mock_unlink.call_count == len(settingsOEISsequences)
        mock_unlink.assert_has_calls([call(missing_ok=True)] * len(settingsOEISsequences))
    else:
        mock_exists.assert_called_once()
        mock_unlink.assert_not_called()

# ===== getLeavesTotal Tests =====
@pytest.fixture
def validDimensionCases() -> List[tuple[List[int], int]]:
    """Provide test cases for valid dimension inputs."""
    return [
        ([2, 3], 6),
        ([2, 3, 4], 24),
        ([0, 1, 2], 2),  # zeros ignored
        ([0], 0),  # edge case
        ([1] * 1000, 1),  # long list
        ([1, sys.maxsize], sys.maxsize),  # maxint
    ]

@pytest.fixture
def invalidDimensionCases() -> List[tuple[Any, type]]:
    """Provide test cases for invalid dimension inputs."""
    return [
        ([], ValueError),  # empty
        ([-1], ValueError),  # negative
        ([1.5], ValueError),  # float
        (['a'], ValueError),  # string
        ([None], TypeError),  # None
        ([[1, 2]], TypeError),  # nested
        (None, ValueError),  # None instead of list
        ([True], TypeError),  # bool
    ]

def test_getLeavesTotal_valid(validDimensionCases):
    """Test getLeavesTotal with valid inputs."""
    for dimensions, expected in validDimensionCases:
        assert getLeavesTotal(dimensions) == expected

def test_getLeavesTotal_invalid(invalidDimensionCases):
    """Test getLeavesTotal with invalid inputs."""
    for dimensions, errorType in invalidDimensionCases:
        with pytest.raises(errorType):
            getLeavesTotal(dimensions)

@pytest.mark.parametrize("sequenceType", [list, tuple, range])
def test_getLeavesTotal_sequence_types(sequenceType):
    """Test getLeavesTotal with different sequence types."""
    if sequenceType is range:
        sequence = range(1, 4)
    else:
        sequence = sequenceType([1, 2, 3])
    assert getLeavesTotal(sequence) == 6 # type: ignore

def test_getLeavesTotal_edge_cases():
    """Test edge cases for getLeavesTotal."""
    # Order independence
    assert getLeavesTotal([2, 3, 4]) == getLeavesTotal([4, 2, 3])
    
    # Immutability
    listOriginal = [2, 3]
    getLeavesTotal(listOriginal)
    assert listOriginal == [2, 3]
    
    # Overflow protection
    largeNumber = sys.maxsize // 2
    with pytest.raises(OverflowError):
        getLeavesTotal([largeNumber, largeNumber, 2])

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

# ===== Parameter Validation Tests =====
@pytest.mark.parametrize("dimensions,divisions,index,errorType", [
    ([], 1, 0, ValueError),  # Empty dimensions
    ([1], 1, 0, NotImplementedError),  # Single dimension
    ([0, 0], 1, 0, NotImplementedError),  # No positive dimensions
    ([1, -1], 1, 0, ValueError),  # Negative dimension
    ([2, 2], -1, 0, ValueError),  # Negative divisions
    ([2, 2], 1, -1, ValueError),  # Negative index
    ([2, 2], 1.5, 0, ValueError),  # Float divisions
    ([2, 2], 1, 1.5, ValueError),  # Float index
])
def test_foldings_parameter_validation(dimensions, divisions, index, errorType):
    """Test parameter validation in foldings function."""
    with pytest.raises(errorType):
        foldings(dimensions, divisions, index)

