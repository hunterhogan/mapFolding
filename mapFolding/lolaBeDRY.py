import numpy
from typing import List, TypedDict, Final, Literal

class countFoldsTask(TypedDict):
    activeGap1ndex: int
    activeLeaf1ndex: int
    connectionGraph: numpy.ndarray
    countDimensionsGapped: Literal[2]
    dimensionsTotal: int
    foldsTotal: int
    gapRangeStart: Literal[3]
    leafAbove: Literal[0]
    leafBelow: Literal[1]
    leavesTotal: int
    listDimensions: List[int]
    potentialGaps: numpy.ndarray
    taskIndex: int
    track: numpy.ndarray

leafAbove: Final[Literal[0]] = 0
leafBelow: Final[Literal[1]] = 1
countDimensionsGapped: Final[Literal[2]] = 2
gapRangeStart: Final[Literal[3]] = 3
