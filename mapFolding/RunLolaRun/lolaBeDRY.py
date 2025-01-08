import numpy
from typing import List, TypedDict, Final, Literal

class countFoldsTask(TypedDict):
    activeGap1ndex: numpy.uint8
    activeLeaf1ndex: numpy.uint8
    connectionGraph: numpy.ndarray
    countDimensionsGapped: Literal[2]
    dimensionsTotal: numpy.uint8
    foldsTotal: int
    gapRangeStart: Literal[3]
    leafAbove: Literal[0]
    leafBelow: Literal[1]
    leavesTotal: numpy.uint8
    listDimensions: List[int]
    potentialGaps: numpy.ndarray
    taskIndex: numpy.uint8
    track: numpy.ndarray

leafAbove: Final[Literal[0]] = 0
leafBelow: Final[Literal[1]] = 1
countDimensionsGapped: Final[Literal[2]] = 2
gapRangeStart: Final[Literal[3]] = 3
