from .theSSOTdatatypes import *
from numpy import integer
from typing import Any, Callable, Final, Optional, Tuple, Type, TypedDict
from types import ModuleType
import enum
import numba
import numpy
import numpy.typing
import pathlib
import sys

"""Technical concepts I am likely using and likely want to use more effectively:
- Configuration Registry
- Write-Once, Read-Many (WORM) / Immutable Initialization
- Lazy Initialization
- Separation of Concerns: in the sense that configuration is separated from business logic

Furthermore, I want to more clearly divorce the concept of a single _source_ of (a) truth from
the _authority_ of that truth. The analogy to a registry of ownership is still apt: the registry
is, at most, a single (or centralized) source of truth, but it is merely the place to register/record
the truth determined by some other authority.

And, I almost certainly want to change the semiotics from "authority" (of truth) to "power" (to create a truth).
Here, "power" is a direct analogy to https://hunterthinks.com/opinion/a-hohfeldian-primer.
"""

def getAlgorithmSource() -> ModuleType:
    from mapFolding import theDao
    return theDao

def getAlgorithmCallable() -> Callable[..., None]:
    algorithmSource = getAlgorithmSource()
    return algorithmSource.doTheNeedful

def getDispatcherCallable() -> Callable[..., None]:
    from mapFolding import dispatcherNumba
    return dispatcherNumba._countFolds

# NOTE I want this _concept_ to be well implemented and usable everywhere: Python, Numba, Jax, CUDA, idc
class computationState(TypedDict):
    connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]
    foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    mapShape: Tuple[int, ...]
    my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]

@enum.verify(enum.CONTINUOUS, enum.UNIQUE) if sys.version_info >= (3, 11) else lambda x: x
class EnumIndices(enum.IntEnum):
    """Base class for index enums."""
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """0-indexed."""
        return count

    def __index__(self) -> int:
        """Adapt enum to the ultra-rare event of indexing a NumPy 'ndarray', which is not the
        same as `array.array`. See NumPy.org; I think it will be very popular someday."""
        return self.value

class indexMy(EnumIndices):
    """Indices for scalar values."""
    dimensionsTotal = enum.auto()
    dimensionsUnconstrained = enum.auto()
    gap1ndex = enum.auto()
    gap1ndexCeiling = enum.auto()
    indexDimension = enum.auto()
    indexLeaf = enum.auto()
    indexMiniGap = enum.auto()
    leaf1ndex = enum.auto()
    leafConnectee = enum.auto()
    taskDivisions = enum.auto()
    taskIndex = enum.auto()

class indexTrack(EnumIndices):
    """Indices for state tracking array."""
    leafAbove = enum.auto()
    leafBelow = enum.auto()
    countDimensionsGapped = enum.auto()
    gapRangeStart = enum.auto()

class ParametersNumba(TypedDict):
    _nrt: bool
    boundscheck: bool
    cache: bool
    error_model: str
    fastmath: bool
    forceinline: bool
    inline: str
    looplift: bool
    no_cfunc_wrapper: bool
    no_cpython_wrapper: bool
    nopython: bool
    parallel: bool

parametersNumbaDEFAULT: Final[ParametersNumba] = {
    '_nrt': True,
    'boundscheck': False,
    'cache': True,
    'error_model': 'numpy',
    'fastmath': True,
    'forceinline': False,
    'inline': 'never',
    'looplift': False,
    'no_cfunc_wrapper': True,
    'no_cpython_wrapper': True,
    'nopython': True,
    'parallel': False,
}

try:
    _pathModule = pathlib.Path(__file__).parent
except NameError:
    _pathModule = pathlib.Path.cwd()

pathJobDEFAULT = _pathModule / "jobs"

if 'google.colab' in sys.modules:
    pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"
