from unittest.mock import patch
import pytest
from mapFolding import clearOEIScache
from mapFolding.oeis import pathCache
from mapFolding import getLeavesTotal

@patch('shutil.rmtree')
@patch('pathlib.Path.exists')
@patch('pathlib.Path.mkdir')
def test_clear_existing_OEIScache(mock_mkdir, mock_exists, mock_rmtree):
    # Setup mocks
    mock_exists.return_value = True
    
    # Run the function
    clearOEIScache()
    
    # Verify the expected calls were made
    mock_rmtree.assert_called_once_with(pathCache)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

@patch('pathlib.Path.exists')
def test_clear_nonexistent_OEIScache(mock_exists):
    # Setup mock to simulate missing cache directory
    mock_exists.return_value = False
    
    # Run the function
    clearOEIScache()
    
    # Verify exists was called but rmtree wasn't needed
    mock_exists.assert_called_once_with()

@pytest.mark.parametrize("listDimensions, productExpected", [
    ([1], 1),
    ([2], 2),
    ([2, 3], 6),
    ([2, 3, 4], 24),
    ([1, 2, 3, 4], 24),
    ([10, 10], 100),
    ([2, 2, 2], 8),
    ([1, 1, 1], 1),
    ([0, 1, 2], 2),  # zeros should be ignored
    ([0, 0, 2], 2),  # multiple zeros
    ([0], 0),  # edge case: only zero
    ([1, 0], 1),  # zero at end
    ([0, 1], 1),  # zero at start
])
def test_getLeavesTotal_valid(listDimensions, productExpected):
    assert getLeavesTotal(listDimensions) == productExpected

@pytest.mark.parametrize("invalidInput, ERRORExpected", [
    ([], ValueError),  # empty list
    ([-1], ValueError),  # negative number
    ([1, -1], ValueError),  # negative number in middle
    ([1.5], ValueError),  # float
    ([1, 2.5], ValueError),  # float in list
    (['a'], ValueError),  # string
    ([1, 'b'], ValueError),  # string in list
    ([None], ValueError),  # None in list
    ([1, None], ValueError),  # None in list
    ([[1, 2]], ValueError),  # nested list
    ([{}], ValueError),  # dictionary
    (None, TypeError),  # None instead of list
    ((1, 2), TypeError),  # tuple instead of list
    ([float('inf')], ValueError),  # infinity
    ([float('nan')], ValueError),  # NaN
])
def test_getLeavesTotal_invalid(invalidInput, ERRORExpected):
    with pytest.raises(ERRORExpected):
        getLeavesTotal(invalidInput)

def test_getLeavesTotal_large_numbers():
    # Test with numbers near sys.maxsize to check for overflow handling
    import sys
    largeNumber = sys.maxsize // 1000  # Using smaller numbers to avoid overflow
    assert getLeavesTotal([largeNumber, 2]) == largeNumber * 2

def test_getLeavesTotal_mixed_order():
    # Test that order doesn't matter
    assert getLeavesTotal([2, 3, 4]) == getLeavesTotal([4, 2, 3])
    assert getLeavesTotal([0, 2, 0, 3]) == getLeavesTotal([2, 0, 3, 0])

def test_getLeavesTotal_immutable():
    # Test that the function doesn't modify the input list
    listOriginal = [0, 1, 2, 3]
    listTest = listOriginal.copy()
    getLeavesTotal(listTest)
    assert listTest == listOriginal

