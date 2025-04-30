from collections.abc import Callable
from mapFolding import getLeavesTotal, setProcessorLimit, validateListDimensions
from tests.conftest import standardizedEqualToCallableReturn
from typing import Any, Literal
from Z0Z_tools import intInnit
from Z0Z_tools.pytestForYourUse import PytestFor_intInnit, PytestFor_oopsieKwargsie
import multiprocessing
import numba
import pytest
import sys

@pytest.mark.parametrize("listDimensions,expected_intInnit,expected_validateListDimensions", [
	(None, ValueError, ValueError),  # None instead of list
	(['a'], ValueError, ValueError),  # string
	([-4, 2], [-4, 2], ValueError),  # negative
	([-3], [-3], ValueError),  # negative
	([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], (1, 2, 3, 4, 5)),  # sequential
	([1, sys.maxsize], [1, sys.maxsize], (1, sys.maxsize)),  # maxint
	([7.5], ValueError, ValueError),  # float
	([1] * 1000, [1] * 1000, (1,) * 1000),  # long list
	([11], [11], NotImplementedError),  # single dimension
	([2, 2, 2, 2], [2, 2, 2, 2], (2, 2, 2, 2)),  # repeated dimensions
	([2, 3, 4], [2, 3, 4], (2, 3, 4)),
	([2, 3], [2, 3], (2, 3)),
	([2] * 11, [2] * 11, (2,) * 11),  # power of 2
	([3] * 5, [3] * 5, (3,) * 5),  # power of 3
	([None], TypeError, TypeError),  # None
	([True], TypeError, TypeError),  # bool
	([[17, 39]], TypeError, TypeError),  # nested
	([], ValueError, ValueError),  # empty
	([complex(1,1)], ValueError, ValueError),  # complex number
	([float('inf')], ValueError, ValueError),  # infinity
	([float('nan')], ValueError, ValueError),  # NaN
	([sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], (sys.maxsize, sys.maxsize)),  # overflow protection
	(range(3, 7), [3, 4, 5, 6], (3, 4, 5, 6)),  # range sequence type
	(tuple([3, 5, 7]), [3, 5, 7], (3, 5, 7)),  # tuple sequence type
])
def test_listDimensionsAsParameter(listDimensions: None | list[str] | list[int] | list[float] | list[None] | list[bool] | list[list[int]] | list[complex] | range | tuple[int, ...],
							expected_intInnit: type[ValueError] | list[int] | type[TypeError],
							expected_validateListDimensions: type[ValueError] | type[NotImplementedError] | tuple[int, ...] | type[TypeError]) -> None:
	"""Test both validateListDimensions and getLeavesTotal with the same inputs."""
	standardizedEqualToCallableReturn(expected_intInnit, intInnit, listDimensions)
	standardizedEqualToCallableReturn(expected_validateListDimensions, validateListDimensions, listDimensions)

def test_getLeavesTotal_edge_cases() -> None:
	"""Test edge cases for getLeavesTotal."""
	# Order independence
	standardizedEqualToCallableReturn(getLeavesTotal((2, 3, 4)), getLeavesTotal, (4, 2, 3))

	# Input preservation
	mapShape = (2, 3)
	standardizedEqualToCallableReturn(6, getLeavesTotal, mapShape)
	assert mapShape == (2, 3), "Input tuple was modified"

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_intInnit())
def testIntInnit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_oopsieKwargsie())
def testOopsieKwargsie(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize("CPUlimit, expectedLimit", [
	(None, numba.get_num_threads()),
	(False, numba.get_num_threads()),
	(True, 1),
	(4, 4),
	(0.5, max(1, numba.get_num_threads() // 2)),
	(-0.5, max(1, numba.get_num_threads() // 2)),
	(-2, max(1, numba.get_num_threads() - 2)),
	(0, numba.get_num_threads()),
	(1, 1),
])
def test_setCPUlimitNumba(CPUlimit: None | float | bool | Literal[4] | Literal[-2] | Literal[0] | Literal[1], expectedLimit: Any | int) -> None:
	numba.set_num_threads(multiprocessing.cpu_count())
	standardizedEqualToCallableReturn(expectedLimit, setProcessorLimit, CPUlimit, 'numba')
