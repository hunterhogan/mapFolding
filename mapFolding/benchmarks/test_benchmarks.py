from ...tests.conftest import *
from .benchmarking import recordBenchmarks, runBenchmarks
import numpy
import pathlib
import pytest
import unittest.mock
from typing import List

def test_recordBenchmarks_decorator(pathBenchmarksTesting: pathlib.Path,
                                    listDimensionsTestFunctionality: List[int],
                                    mockBenchmarkTimer: unittest.mock.MagicMock):
    """Test that the decorator correctly records benchmark data."""
    @recordBenchmarks()
    def functionTest(listDimensions: List[int]) -> int:
        return sum(listDimensions)

    with mockBenchmarkTimer:
        mockBenchmarkTimer.side_effect = [0, 1e9]
        result = functionTest(listDimensionsTestFunctionality)

    # Verify function still works normally
    assert result == sum(listDimensionsTestFunctionality)

    # Verify benchmark data was saved
    arrayBenchmarks = numpy.load(str(pathBenchmarksTesting), allow_pickle=True)
    assert len(arrayBenchmarks) == 1
    assert arrayBenchmarks[0]['time'] == 1.0
    assert tuple(arrayBenchmarks[0]['dimensions']) == tuple(listDimensionsTestFunctionality)

def test_recordBenchmarks_multiple_calls(pathBenchmarksTesting: pathlib.Path,
                                        listDimensionsTestFunctionality: List[int],
                                        mockBenchmarkTimer: unittest.mock.MagicMock):
    """Test that multiple function calls append to benchmark data."""
    @recordBenchmarks()
    def functionTest(listDimensions: List[int]) -> int:
        return sum(listDimensions)

    with mockBenchmarkTimer:
        mockBenchmarkTimer.side_effect = [0, 1e9, 2e9, 4e9]
        functionTest(listDimensionsTestFunctionality)
        functionTest(listDimensionsTestFunctionality)

    arrayBenchmarks = numpy.load(str(pathBenchmarksTesting), allow_pickle=True)
    assert len(arrayBenchmarks) == 2
    assert arrayBenchmarks[0]['time'] == 1.0
    assert arrayBenchmarks[1]['time'] == 2.0

# NOTE This test tries to collect benchmark data without ensuring that a function is decorated.
# def test_runBenchmarks_integration(pathBenchmarksTesting: pathlib.Path, listDimensionsTestFunctionality: List[int]):
#     """Test runBenchmarks creates valid benchmark data."""
#     countIterations = 2
#     runBenchmarks(countIterations)

#     arrayBenchmarks = numpy.load(str(pathBenchmarksTesting), allow_pickle=True)
#     assert len(arrayBenchmarks) > 0  # Should have recorded some benchmarks

#     # Verify data structure integrity
#     assert arrayBenchmarks.dtype.names == ('time', 'dimensions')
#     assert all(isinstance(record['time'], float) for record in arrayBenchmarks)
#     assert all(isinstance(record['dimensions'], tuple) for record in arrayBenchmarks)

#     # Verify at least one benchmark entry matches our test dimensions
#     assert any(tuple(listDimensionsTestFunctionality) == record['dimensions'] for record in arrayBenchmarks)

# NOTE This test tries to collect benchmark data without ensuring that a function is decorated.
# @pytest.mark.parametrize("countIterations", [1, 2])
# def test_runBenchmarks_iterations(countIterations: int, pathBenchmarksTesting: pathlib.Path, listDimensionsTestFunctionality: List[int]):
#     """Test runBenchmarks records data for each iteration."""
#     runBenchmarks(countIterations)
#     arrayBenchmarks = numpy.load(str(pathBenchmarksTesting), allow_pickle=True)

#     # Should have at least countIterations entries for our test dimensions
#     countMatches = sum(1 for record in arrayBenchmarks if tuple(listDimensionsTestFunctionality) == record['dimensions'])
#     assert countMatches >= countIterations
