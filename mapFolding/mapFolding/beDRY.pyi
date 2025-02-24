import os
from collections.abc import Sequence
from mapFolding import computationState as computationState, getDatatypeModule as getDatatypeModule, getPathJobRootDEFAULT as getPathJobRootDEFAULT, hackSSOTdatatype as hackSSOTdatatype, hackSSOTdtype as hackSSOTdtype, indexMy as indexMy, indexTrack as indexTrack, setDatatypeElephino as setDatatypeElephino, setDatatypeFoldsTotal as setDatatypeFoldsTotal, setDatatypeLeavesTotal as setDatatypeLeavesTotal, setDatatypeModule as setDatatypeModule
from numpy import dtype, integer, ndarray
from numpy.typing import DTypeLike as DTypeLike, NDArray as NDArray
from pathlib import Path
from typing import Any

def getFilenameFoldsTotal(mapShape: Sequence[int] | ndarray[tuple[int], dtype[integer[Any]]]) -> str:
    """Imagine your computer has been counting folds for 70 hours, and when it tries to save your newly discovered value,
\tthe filename is invalid. I bet you think this function is more important after that thought experiment.

\tMake a standardized filename for the computed value `foldsTotal`.

\tThe filename takes into account
\t\t- the dimensions of the map, aka `mapShape`, aka `listDimensions`
\t\t- no spaces in the filename
\t\t- safe filesystem characters
\t\t- unique extension
\t\t- Python-safe strings:
\t\t\t- no starting with a number
\t\t\t- no reserved words
\t\t\t- no dashes or other special characters
\t\t\t- uh, I can't remember, but I found some other frustrating limitations
\t\t- if 'p' is still the first character of the filename, I picked that because it was the original identifier for the map shape in Lunnan's code

\tParameters:
\t\tmapShape: A sequence of integers representing the dimensions of the map.

\tReturns:
\t\tfilenameFoldsTotal: A filename string in format 'pMxN.foldsTotal' where M,N are sorted dimensions
\t"""
def getLeavesTotal(listDimensions: Sequence[int]) -> int:
    """
\tHow many leaves are in the map.

\tParameters:
\t\tlistDimensions: A list of integers representing dimensions.

\tReturns:
\t\tproductDimensions: The product of all positive integer dimensions.
\t"""
def getPathFilenameFoldsTotal(mapShape: Sequence[int] | ndarray[tuple[int], dtype[integer[Any]]], pathLikeWriteFoldsTotal: str | os.PathLike[str] | None = None) -> Path:
    """Get a standardized path and filename for the computed value `foldsTotal`.

\tIf you provide a directory, the function will append a standardized filename. If you provide a filename
\tor a relative path and filename, the function will prepend the default path.

\tParameters:
\t\tmapShape: List of dimensions for the map folding problem.
\t\tpathLikeWriteFoldsTotal (pathJobRootDEFAULT): Path, filename, or relative path and filename. If None, uses default path.
\t\t\tDefaults to None.

\tReturns:
\t\tpathFilenameFoldsTotal: Absolute path and filename.
\t"""
def getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, CPUlimit: bool | float | int | None, listDimensions: Sequence[int]) -> int:
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
def makeConnectionGraph(listDimensions: Sequence[int], **keywordArguments: str | None) -> ndarray[tuple[int, int, int], dtype[integer[Any]]]:
    """
\tConstructs a multi-dimensional connection graph representing the connections between the leaves of a map with the given dimensions.
\tAlso called a Cartesian product decomposition or dimensional product mapping.

\tParameters
\t\tlistDimensions: A sequence of integers representing the dimensions of the map.
\t\t**keywordArguments: Datatype management.

\tReturns
\t\tconnectionGraph: A 3D numpy array with shape of (dimensionsTotal, leavesTotal + 1, leavesTotal + 1).
\t"""
def makeDataContainer(shape: int | tuple[int, ...], datatype: DTypeLike | None = None) -> NDArray[integer[Any]]:
    '''Create a zeroed-out `ndarray` with the given shape and datatype.

\tParameters:
\t\tshape: The shape of the array. Can be an integer for 1D arrays
\t\t\tor a tuple of integers for multi-dimensional arrays.
\t\tdatatype (\'dtypeFoldsTotal\'): The desired data type for the array.
\t\t\tIf `None`, defaults to \'dtypeFoldsTotal\'. Defaults to None.

\tReturns:
\t\tdataContainer: A new array of given shape and type, filled with zeros.

\tNotes:
\t\tIf a version of the algorithm were to use something other than numpy, such as JAX or CUDA, because other
\t\tfunctions use this function, it would be much easier to change the datatype "ecosystem".
\t'''
def outfitCountFolds(listDimensions: Sequence[int], computationDivisions: int | str | None = None, CPUlimit: bool | float | int | None = None, **keywordArguments: str | bool | None) -> computationState:
    """
\tInitializes and configures the computation state for map folding computations.

\tParameters:
\t\tlistDimensions: The dimensions of the map to be folded
\t\tcomputationDivisions (None): see `getTaskDivisions`
\t\tCPUlimit (None): see `setCPUlimit`
\t\t**keywordArguments: Datatype management, it's complicated: see the code below.

\tReturns:
\t\tstateInitialized: The initialized computation state
\t"""
def parseDimensions(dimensions: Sequence[int], parameterName: str = 'listDimensions') -> list[int]:
    """
\tParse and validate the dimensions are non-negative integers.

\tParameters:
\t\tdimensions: Sequence of integers representing dimensions.
\t\tparameterName ('listDimensions'): Name of the parameter for error messages. Defaults to 'listDimensions'.
\tReturns:
\t\tlistNonNegative: List of validated non-negative integers.
\tRaises:
\t\tValueError: If any dimension is negative or if the list is empty.
\t\tTypeError: If any element cannot be converted to integer (raised by `intInnit`).
\t"""
def saveFoldsTotal(pathFilename: str | os.PathLike[str], foldsTotal: int) -> None:
    """
\tSave foldsTotal with multiple fallback mechanisms.

\tParameters:
\t\tpathFilename: Target save location
\t\tfoldsTotal: Critical computed value to save
\t"""
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
def validateListDimensions(listDimensions: Sequence[int]) -> list[int]:
    """
\tValidates and sorts a sequence of at least two positive dimensions.

\tParameters:
\t\tlistDimensions: A sequence of integer dimensions to be validated.

\tReturns:
\t\tdimensionsValidSorted: A list, with at least two elements, of only positive integers.

\tRaises:
\t\tValueError: If the input listDimensions is empty.
\t\tNotImplementedError: If the resulting list of positive dimensions has fewer than two elements.
\t"""
