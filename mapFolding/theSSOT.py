from typing import Any, Tuple, TypedDict
import enum
import numpy
import numpy.typing
import pathlib
import sys

datatypeModule = 'numpy'

datatypeLarge = 'int64'
datatypeDefault = datatypeLarge
datatypeSmall = datatypeDefault

make_dtype = lambda _datatype: eval(f"{datatypeModule}.{_datatype}")

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
    """Indices for dynamic values."""
    dimension1ndex = enum.auto()
    dimensionsUnconstrained = enum.auto()
    gap1ndex = enum.auto()
    gap1ndexCeiling = enum.auto()
    indexLeaf = enum.auto()
    indexMiniGap = enum.auto()
    leaf1ndex = enum.auto()
    leafConnectee = enum.auto()
    taskIndex = enum.auto()

class indexThe(EnumIndices):
    """Indices for static values."""
    dimensionsTotal = enum.auto()
    leavesTotal = enum.auto()
    taskDivisions = enum.auto()

class indexTrack(EnumIndices):
    """Indices for state tracking array."""
    leafAbove = enum.auto()
    leafBelow = enum.auto()
    countDimensionsGapped = enum.auto()
    gapRangeStart = enum.auto()

class computationState(TypedDict):
    connectionGraph: numpy.typing.NDArray[numpy.integer[Any]]
    foldsSubTotals: numpy.typing.NDArray[numpy.integer[Any]]
    gapsWhere: numpy.typing.NDArray[numpy.integer[Any]]
    mapShape: Tuple[int, ...]
    my: numpy.typing.NDArray[numpy.integer[Any]]
    the: numpy.typing.NDArray[numpy.integer[Any]]
    track: numpy.typing.NDArray[numpy.integer[Any]]
