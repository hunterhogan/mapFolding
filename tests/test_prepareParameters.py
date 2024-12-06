
import pytest
from mapFolding.prepareParameters import countMinimumParsePoints

@pytest.mark.parametrize("dimensions,expected", [
    ([2, 2], 3),
    ([2, 3], 5),
    ([3, 3], 7),
    ([2, 2, 2], 7),
    ([], 0),
    ([1], 1),
    ([1, 1], 1),
])
def test_countMinimumParsePoints(dimensions, expected):
    assert countMinimumParsePoints(dimensions) == expected