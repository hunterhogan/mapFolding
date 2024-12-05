
import pytest
from mapFolding import foldings

test_cases = [
    ([2], 1),
    ([2, 2], 2),
    ([2, 3], 1),
    ([3, 3], 2),
    ([2, 2, 2], 8),
]

@pytest.mark.parametrize("dimensions,expected", test_cases)
def test_foldings(dimensions, expected):
    assert foldings(dimensions) == expected

def test_invalid_input():
    with pytest.raises(ValueError):
        foldings([])
    with pytest.raises(ValueError):
        foldings([1])
    with pytest.raises(ValueError):
        foldings([0, 2])