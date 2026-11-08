"""Resolve processor limits and divide computations into tasks.

`defineProcessorLimit` translates a `CPUlimit` preference into a worker count, and `getTaskDivisions`
translates a `computationDivisions` preference into a task count. The `cpuTotal` fixture fixes the
processor total so the expected worker counts do not depend on the machine.
"""
from __future__ import annotations

from mapFolding.beDRY import defineProcessorLimit, getTaskDivisions
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit

@pytest.mark.usefixtures('cpuTotal')
@pytest.mark.parametrize('concurrencyPackage', (None, 'multiprocessing', 'numba', 'unknown-package'))
@pytest.mark.parametrize('CPUlimit, expected', [(None, 8), (False, 8), (True, 1), (4, 4), (0.5, 4), (-0.5, 4), (-2, 6), (0, 8), (1, 1)])
def test_defineProcessorLimit(CPUlimit: ConcurrencyLimit, concurrencyPackage: str | None, expected: int) -> None:
	assertEqualTo(defineProcessorLimit(CPUlimit, concurrencyPackage), expected, defineProcessorLimit.__name__, CPUlimit, concurrencyPackage)

@pytest.mark.usefixtures('cpuTotal')
@pytest.mark.parametrize('concurrencyPackage', (None, 'multiprocessing', 'numba', 'unknown-package'))
@pytest.mark.parametrize('CPUlimit, expected', [([4], TypeError), ((2,), TypeError), ({2}, TypeError), ({'cores': 2}, TypeError)])
def test_defineProcessorLimitError(CPUlimit: list[int] | tuple[int, ...] | set[int] | dict[str, int], concurrencyPackage: str | None
	, expected: type[TypeError]) -> None:
	with pytest.raises(expected):
		#=SIN= Invalid argument type: the test verifies rejection of values outside `ConcurrencyLimit`.
		defineProcessorLimit(CPUlimit, concurrencyPackage)  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, totalLeaves, expected', [(None, 4, 99, 0), ('maximum', 4, 77, 77), ('cpu', 4, 21, 4)])
def test_getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, totalLeaves: int, expected: int) -> None:
	actual: int = getTaskDivisions(computationDivisions, concurrencyLimit, totalLeaves)
	assertEqualTo(actual, expected, getTaskDivisions.__name__, computationDivisions, concurrencyLimit, totalLeaves)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, totalLeaves, expected', [(['invalid'], 4, 437, ValueError), (20, 4, 15, ValueError)])
def test_getTaskDivisionsError(computationDivisions: int | str | None, concurrencyLimit: int, totalLeaves: int, expected: type[ValueError]) -> None:
	with pytest.raises(expected):
		getTaskDivisions(computationDivisions, concurrencyLimit, totalLeaves)
