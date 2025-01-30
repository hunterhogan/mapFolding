from numpy import integer
from numpy.typing import NDArray
from typing import Any, Callable, Final, Optional, Tuple, Type, TypedDict
from types import ModuleType
import enum
import numba
import numpy
import numpy.typing
import pathlib
import sys

def getAlgorithmSource() -> ModuleType:
    from mapFolding import theDao
    return theDao

def getAlgorithmCallable() -> Callable[..., None]:
    algorithmSource = getAlgorithmSource()
    return algorithmSource.doTheNeedful

def getDispatcherCallable() -> Callable[..., None]:
    from mapFolding import dispatcher
    return dispatcher._countFolds

# NOTE I want this _concept_ to be well implemented and usable everywhere: Python, Numba, Jax, CUDA, idc
class computationState(TypedDict):
    connectionGraph: NDArray[integer[Any]]
    foldGroups: NDArray[integer[Any]]
    gapsWhere: NDArray[integer[Any]]
    mapShape: Tuple[int, ...]
    my: NDArray[integer[Any]]
    track: NDArray[integer[Any]]

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

datatypeLargeDEFAULT: Final[str] = 'int64'
datatypeMediumDEFAULT: Final[str] = 'int16'
datatypeSmallDEFAULT: Final[str] = 'uint8'

# Design for the possibility that I will want to use module jabberPy and their datatypes, wockyPy
datatypeModuleDEFAULT: Final[str] = 'numpy'

def make_dtype(datatype: str, datatypeModule: Optional[str] = None) -> Type[Any]:
    if datatypeModule is None:
        datatypeModule = datatypeModuleDEFAULT
    return eval(f"{datatypeModule}.{datatype}")

dtypeLargeDEFAULT = make_dtype(datatypeLargeDEFAULT)
dtypeMediumDEFAULT = make_dtype(datatypeMediumDEFAULT)
dtypeSmallDEFAULT = make_dtype(datatypeSmallDEFAULT)

try:
    _pathModule = pathlib.Path(__file__).parent
except NameError:
    _pathModule = pathlib.Path.cwd()

pathJobDEFAULT = _pathModule / "jobs"

if 'google.colab' in sys.modules:
    pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"
