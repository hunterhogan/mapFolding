"""Prototype concept; especially as a complement to 'beDRY.py'."""
from typing import Final, Literal, List, TypedDict
import numpy

leafAbove: Final[Literal[0]] = 0
leafBelow: Final[Literal[1]] = 1
countDimensionsGapped: Final[Literal[2]] = 2
gapRangeStart: Final[Literal[3]] = 3
