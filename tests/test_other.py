from unittest.mock import patch, call
import pytest
from mapFolding import clearOEIScache, settingsOEISsequences, getLeavesTotal, parseListDimensions, getFoldingsTotalKnown
from mapFolding.oeis import _pathCache
import sys
from Z0Z_tools import makeTestSuiteIntInnit

@patch('pathlib.Path.exists')
@patch('pathlib.Path.unlink')
def test_clear_existing_OEIScache(mock_unlink, mock_exists):
    mock_exists.return_value = True
    
    clearOEIScache()
    
    assert mock_unlink.call_count == len(settingsOEISsequences)
    mock_unlink.assert_has_calls([
        call(missing_ok=True) for _ in range(len(settingsOEISsequences))
    ])

@patch('pathlib.Path.exists')
@patch('pathlib.Path.unlink')
def test_clear_nonexistent_OEIScache(mock_unlink, mock_exists):
    mock_exists.return_value = False
    
    clearOEIScache()
    
    # Verify exists was called but unlink wasn't needed
    mock_exists.assert_called_once()
    mock_unlink.assert_not_called()

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
])
def test_getLeavesTotal_invalid(invalidInput):
    ERRORExpected = findNaturalError(invalidInput)
    assert ERRORExpected is not None, f"Expected {invalidInput} to raise an error"
    with pytest.raises(ERRORExpected):
        getLeavesTotal(invalidInput)

def test_getLeavesTotal_large_numbers():
    # Test with numbers near sys.maxsize to check for overflow handling
    import sys
    largeNumber = sys.maxsize // 2
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
    assert ERRORexpected == TypeError 

def test_parseListInt():
    """Test the makeTestSuiteIntInnit function with various inputs."""
    testSuite = makeTestSuiteIntInnit(parseListDimensions)
    for smurfName, smurfFunction in testSuite.items():
        smurfFunction()    

def test_getFoldingsTotalKnown_valid_dimensions():
    # Test known dimensions from OEIS sequences
    assert getFoldingsTotalKnown([2, 2]) > 0
    assert getFoldingsTotalKnown([3, 2]) > 0
    assert getFoldingsTotalKnown([2, 0, 2]) > 0  # Should handle zero dimensions

def test_getFoldingsTotalKnown_invalid_dimensions():
    with pytest.raises(NotImplementedError):
        getFoldingsTotalKnown([1])  # Single dimension

    with pytest.raises(NotImplementedError):
        getFoldingsTotalKnown([0, 0])  # No positive dimensions

    with pytest.raises(ValueError):
        getFoldingsTotalKnown([])  # Empty list

    with pytest.raises(ValueError):
        getFoldingsTotalKnown([-1, 2])  # Negative dimension

    with pytest.raises(KeyError):
        getFoldingsTotalKnown([10, 10])  # Unknown dimensions
