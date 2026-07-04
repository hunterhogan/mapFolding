from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from numpy import dtype, ndarray, number
	from typing import Any

#================== Assert scalar and built-in containers ========================================================================

def assertEqualTo[个](actual: 个, expected: 个, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	"""Assert that two objects are equal, and if not, raise an AssertionError with a detailed message."""
	assert actual == expected, messageTestFailure(actual, expected, function, *arguments, **keywordArguments)

def assert_approx[个](actual: 个, expected: 个, pytest_rel: float, pytest_abs: float, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	assert actual == pytest.approx(expected, pytest_rel, pytest_abs, nan_ok=True), messageTestFailure(actual, expected, function, *arguments, **keywordArguments)

def messageTestFailure(actual: Any, expected: Any, function: str, *arguments: Any, **keywordArguments: Any) -> str:
	"""Format assertion message for any test comparison."""
	parameters: list[str] = [*map(repr, arguments), *starmap('{}={!r}'.format, keywordArguments.items())]
	return f'{function}({", ".join(parameters)}) = {actual!r}, but {expected = }.'

#================== Assert NumPy arrays ========================================================================

def assert_array_equal[形ndarray: ndarray[tuple[Any, ...], dtype[number]]](actual: 形ndarray, expected: 形ndarray, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	"""Assert that two arrays are equal, and if not, raise an AssertionError with a detailed message."""
	assert numpy.array_equal(actual, expected), messageTestFailure_ndarray(actual, expected, function, *arguments, **keywordArguments)

def assert_allclose[形ndarray: ndarray[tuple[Any, ...], dtype[number]]](actual: 形ndarray, expected: 形ndarray, rtol: float, atol: float, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	assert numpy.allclose(actual, expected, rtol, atol), messageTestFailure_ndarray(actual, expected, function, *arguments, **keywordArguments)

def messageTestFailure_ndarray[形ndarray: ndarray[tuple[Any, ...], dtype[number]]](actual: 形ndarray, expected: 形ndarray, function: str, *arguments: Any, **keywordArguments: Any) -> str:
	"""Printing arrays is absurd."""
	parameters: list[str] = [*map(repr, arguments), *starmap('{}={!r}'.format, keywordArguments.items())]
	return f'{function}({", ".join(parameters)}) = {actual.shape=},\t{actual.dtype=}, but {expected.shape=}, {expected.dtype=}.'
