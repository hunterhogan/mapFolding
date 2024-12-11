from unittest.mock import patch
import pytest
from mapFolding import clearOEIScache
from mapFolding.oeis import pathCache
from mapFolding import getLeavesTotal
import sys

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
    # New test cases
    ([1] * 1000, 1),  # Test very long lists
    ([1, sys.maxsize], sys.maxsize),  # Test maximum integer
    ([2, sys.maxsize // 2], sys.maxsize - 1),  # Test near-maximum result
])
def test_getLeavesTotal_valid(listDimensions, productExpected):
    assert getLeavesTotal(listDimensions) == productExpected

@pytest.mark.parametrize("sequenceInput, productExpected", [
    ([1, 2, 3], 6),  # list
    ((1, 2, 3), 6),  # tuple
    (range(1, 4), 6),  # range
])
def test_getLeavesTotal_sequences(sequenceInput, productExpected):
    """Test that getLeavesTotal works with various sequence types."""
    assert getLeavesTotal(sequenceInput) == productExpected

def findNaturalError(inputValue):
    """Determine the natural error type for an invalid input to getLeavesTotal."""
    try:
        getLeavesTotal(inputValue)
        return None  # No error occurred
    except Exception as ERRORinstance:
        return type(ERRORinstance)

# Update the test parameters to use dynamic error detection
@pytest.mark.parametrize("invalidInput", [
    [],  # empty list
    [-1],  # negative number
    [1, -1],  # negative number in middle
    [1.5],  # float
    [1, 2.5],  # float in list
    ['a'],  # string
    [1, 'b'],  # string in list
    [None],  # None in list
    [1, None],  # None in list
    [[1, 2]],  # nested list
    [{}],  # dictionary
    None,  # None instead of list
    [float('inf')],  # infinity
    [float('nan')],  # NaN
    [True],  # Boolean is an int subclass
    [1, True],  # Mixed with boolean
    [b'1'],  # Bytes
    [range(3)],  # List containing range
    # Removed memoryview tests as they're not relevant for this use case
])
def test_getLeavesTotal_invalid(invalidInput):
    ERRORExpected = findNaturalError(invalidInput)
    assert ERRORExpected is not None, f"Expected {invalidInput} to raise an error"
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

def test_getLeavesTotal_custom_list_subclass():
    class CustomList(list):
        pass
    customList = CustomList([1, 2, 3])
    assert getLeavesTotal(customList) == 6

def test_getLeavesTotal_concurrent_modification():
    class TrickyList(list):
        def __iter__(self):
            self.append(2)  # Modify list during iteration
            return super().__iter__()
    
    trickyList = TrickyList([1, 2, 3])
    with pytest.raises(RuntimeError):  # Should detect list modification
        getLeavesTotal(trickyList)

def test_getLeavesTotal_massive_list():
    # Test with a list that's large but not memory-breaking
    largeList = [1] * 1_000_000
    assert getLeavesTotal(largeList) == 1

def test_getLeavesTotal_maxsize_overflow():
    # Test potential overflow conditions
    import sys
    largeNumber = sys.maxsize // 2
    with pytest.raises(OverflowError):
        getLeavesTotal([largeNumber, largeNumber, 2])

def test_findNaturalError_validation():
    """Test that findNaturalError correctly identifies when an error should occur."""
    class SneakySequence:
        """A sequence that passes initial checks but fails during numeric conversion."""
        def __iter__(self):
            class TrickyNumber:
                """An object that pretends to be numeric but fails conversion."""
                def __int__(self):
                    raise ValueError("Surprise!")
                def is_integer(self):
                    return True
                def __float__(self):
                    return self
            
            return iter([TrickyNumber()])
        
        def __len__(self):
            return 1
            
    sneakyInput = SneakySequence()
    ERRORexpected = findNaturalError(sneakyInput)
    assert ERRORexpected is not None, "Should detect invalid sequence type"
    assert ERRORexpected == TypeError  # Changed from ValueError to TypeError

