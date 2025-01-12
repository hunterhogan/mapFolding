"""Prototype concept; especially as a complement to 'beDRY.py'."""
from typing import Final, Literal, TypedDict
import pathlib
import sys

try:
    _pathModule = pathlib.Path(__file__).parent
except NameError:
    _pathModule = pathlib.Path.cwd()

pathJobDEFAULT = _pathModule / "jobs"
if 'google.colab' in sys.modules:
    pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"

leafAbove: Final[Literal[0]] = 0
leafBelow: Final[Literal[1]] = 1
countDimensionsGapped: Final[Literal[2]] = 2
gapRangeStart: Final[Literal[3]] = 3

# TODO learn how to use TypedDict without numba freaking out

class computationState(TypedDict):
    """Use this identifier when the class stabilizes"""
    reserved: None

class taskState(TypedDict):
    """Use this identifier when the class stabilizes"""
    reserved: None
