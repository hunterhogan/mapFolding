"""Prototype concept; especially as a complement to 'beDRY.py'."""
from typing import Final, Literal, TypedDict
import pathlib
import sys
import enum
import numba

try:
    _pathModule = pathlib.Path(__file__).parent
except NameError:
    _pathModule = pathlib.Path.cwd()

pathJobDEFAULT = _pathModule / "jobs"
if 'google.colab' in sys.modules:
    pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"

@enum.verify(enum.CONTINUOUS)
@enum.unique
class t(enum.Enum):
    """Indices for tracking array operations. Values must start at 0 for numpy array indexing."""
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count  # Start counting from 0

    leafAbove = enum.auto()
    leafBelow = enum.auto()
    countDimensionsGapped = enum.auto()
    gapRangeStart = enum.auto()
    my = enum.auto()

@enum.verify(enum.CONTINUOUS)
@enum.unique
class indexMy(enum.Enum):
    """Indices for tracking array operations. Values must start at 0 for numpy array indexing."""
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count  # Start counting from 0

    activeGap1ndex = enum.auto()
    activeLeaf1ndex = enum.auto()
    dimension1ndex = enum.auto()
    dimensionsUnconstrained = enum.auto()
    gap1ndexLowerBound = enum.auto()
    indexMiniGap = enum.auto()
    leaf1ndexConnectee = enum.auto()


# TODO learn how to use TypedDict without numba freaking out

class computationState(TypedDict):
    """Use this identifier when the class stabilizes"""
    reserved: None

class taskState(TypedDict):
    """Use this identifier when the class stabilizes"""
    reserved: None
