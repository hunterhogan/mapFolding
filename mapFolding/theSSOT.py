"""Prototype concept; especially as a complement to 'beDRY.py'."""
from typing import Final, Literal, TypedDict

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
