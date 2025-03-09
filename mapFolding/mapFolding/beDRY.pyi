from collections.abc import Sequence
from mapFolding.theSSOT import Array1DElephino as Array1DElephino, Array1DFoldsTotal as Array1DFoldsTotal, Array1DLeavesTotal as Array1DLeavesTotal, Array3D as Array3D, ComputationState as ComputationState, getDatatypePackage as getDatatypePackage, getNumpyDtypeDefault as getNumpyDtypeDefault
from numpy import dtype, ndarray
from numpy.typing import DTypeLike as DTypeLike
from typing import Any

def validateListDimensions(listDimensions: Sequence[int]) -> tuple[int, ...]: ...
def getLeavesTotal(mapShape: tuple[int, ...]) -> int: ...
def makeConnectionGraph(mapShape: tuple[int, ...], leavesTotal: int, datatype: DTypeLike | None = None) -> Array3D: ...
def makeDataContainer(shape: int | tuple[int, ...], datatype: DTypeLike | None = None) -> Array1DLeavesTotal | Array1DElephino | Array1DFoldsTotal | ndarray[Any, dtype[Any]]: ...
def setCPUlimit(CPUlimit: Any | None) -> int:
    """Sets CPU limit for Numba concurrent operations. Note that it can only affect Numba-jitted functions that have not yet been imported.

\tParameters:
\t\tCPUlimit: whether and how to limit the CPU usage. See notes for details.
\tReturns:
\t\tconcurrencyLimit: The actual concurrency limit that was set
\tRaises:
\t\tTypeError: If CPUlimit is not of the expected types

\tLimits on CPU usage `CPUlimit`:
\t\t- `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
\t\t- `True`: Yes, limit the CPU usage; limits to 1 CPU.
\t\t- Integer `>= 1`: Limits usage to the specified number of CPUs.
\t\t- Decimal value (`float`) between 0 and 1: Fraction of total CPUs to use.
\t\t- Decimal value (`float`) between -1 and 0: Fraction of CPUs to *not* use.
\t\t- Integer `<= -1`: Subtract the absolute value from total CPUs.
\t"""
def getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, leavesTotal: int) -> int:
    """
\tDetermines whether to divide the computation into tasks and how many divisions.

\tParameters
\t----------
\tcomputationDivisions (None)
\t\tSpecifies how to divide computations:
\t\t- `None`: no division of the computation into tasks; sets task divisions to 0.
\t\t- int: direct set the number of task divisions; cannot exceed the map's total leaves.
\t\t- `'maximum'`: divides into `leavesTotal`-many `taskDivisions`.
\t\t- `'cpu'`: limits the divisions to the number of available CPUs, i.e. `concurrencyLimit`.
\tconcurrencyLimit
\t\tMaximum number of concurrent tasks allowed.
\tCPUlimit
\t\tfor error reporting.
\tlistDimensions
\t\tfor error reporting.

\tReturns
\t-------
\ttaskDivisions
\t\tHow many tasks must finish before the job can compute the total number of folds; `0` means no tasks, only job.

\tRaises
\t------
\tValueError
\t\tIf computationDivisions is an unsupported type or if resulting task divisions exceed total leaves.

\tNotes
\t-----
\tTask divisions should not exceed total leaves or the folds will be over-counted.
\t"""
def outfitCountFolds(mapShape: tuple[int, ...], computationDivisions: int | str | None = None, concurrencyLimit: int = 1) -> ComputationState: ...
