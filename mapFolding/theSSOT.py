from typing import Any, Tuple, Type, TypedDict
from numpy import integer
from numpy.typing import NDArray
import enum
import numpy
import numpy.typing
import pathlib
import sys

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
    dimensionsTotal = enum.auto() # connectionGraph.shape[0] or len(mapShape)
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

datatypeModule = 'numpy'

datatypeLarge = 'int64'
datatypeDefault = datatypeLarge
datatypeSmall = datatypeDefault

def make_dtype(_datatype: str) -> Type:
    return eval(f"{datatypeModule}.{_datatype}")

dtypeLarge = make_dtype(datatypeLarge)
dtypeDefault = make_dtype(datatypeDefault)
dtypeSmall = make_dtype(datatypeSmall)

try:
    _pathModule = pathlib.Path(__file__).parent
except NameError:
    _pathModule = pathlib.Path.cwd()

pathJobDEFAULT = _pathModule / "jobs"

if 'google.colab' in sys.modules:
    pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"
