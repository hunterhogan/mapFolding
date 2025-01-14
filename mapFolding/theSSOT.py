from typing import Any, Final, Literal, TypedDict, Tuple
import pathlib
import sys
import enum
import numba
import numpy
import numpy.typing

try:
    _pathModule = pathlib.Path(__file__).parent
except NameError:
    _pathModule = pathlib.Path.cwd()

pathJobDEFAULT = _pathModule / "jobs"
if 'google.colab' in sys.modules:
    pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"

@enum.verify(enum.CONTINUOUS, enum.UNIQUE)
class EnumIndices(enum.Enum):
    """Base class for index enums."""
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """0-indexed."""
        return count

    def __index__(self) -> int:
        """Make the enum work with array indexing."""
        return self.value

    # # Add integer-like behavior
    # def __int__(self) -> int:
    #     """Convert to integer."""
    #     return self.value

    # def __repr__(self) -> str:
    #     """Represent as enum member."""
    #     return f"{self.__class__.__name__}.{self.name}"

class indexMy(EnumIndices):
    """Indices for dynamic values."""
    dimension1ndex = enum.auto()
    dimensionsUnconstrained = enum.auto()
    doCountGaps = enum.auto()
    gap1ndex = enum.auto()
    gap1ndexLowerBound = enum.auto()
    indexLeaf = enum.auto()
    indexMiniGap = enum.auto()
    leaf1ndex = enum.auto()
    leafConnectee = enum.auto()
    lolaCondition = enum.auto()
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
    # TODO remove
    my = enum.auto()
"""
TODO improve semiotics: clarity, brevity
gapNotGap
"""
# TODO learn how to use TypedDict without numba freaking out
class Z0Z_computationState(TypedDict):
    connectionGraph: numpy.typing.NDArray[numpy.integer[Any]]
    foldsTotal: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]
    mapShape: Tuple[int, ...]
    my: numpy.typing.NDArray[numpy.integer[Any]]
    potentialGaps: numpy.typing.NDArray[numpy.integer[Any]]
    the: numpy.typing.NDArray[numpy.integer[Any]]
    track: numpy.typing.NDArray[numpy.integer[Any]]
class computationState(TypedDict):
    """Use this identifier when the class stabilizes"""
    reserved: None

class taskState(TypedDict):
    """Use this identifier when the class stabilizes"""
    reserved: None
