"""Prototype concept: Import priority levels. Larger priority values should be imported before smaller priority values."""
"""
# TODO Across the entire package, restructure computationDivisions.

doWhile:
    taskDivisions: int
    taskIndex: int

At other places in the package, setCPUlimit and/or computationDivisions can work together to compute the integer values of taskDivisions and taskIndex.

# TODO learn how to profile the size of the compiled code and data. Use it when doing the package restructure, above.

# TODO to avoid dynamic memory allocation, use a preallocated array for at least:
dimensionsUnconstrained = numpy.uint8(0)
dimension1ndex = numpy.uint8(1)
gap1ndexLowerBound = numpy.uint8(0)
leaf1ndexConnectee = numpy.uint8(0)
indexMiniGap = numpy.uint8(0)
There may be advantages to putting the other "singleton" variables in a preallocated array, too. There do not seem to many
disadvantages to doing this.

"""
from .theSSOT import * # Priority 10,000
from .beDRY import getLeavesTotal, getTaskDivisions, makeConnectionGraph, outfitFoldings, setCPUlimit # Priority 1,000
from .beDRY import parseListDimensions, validateListDimensions
from .lola import countFolds # Priority 70. NOTE `countFolds` is the point of the package. Two things should be very stable: 1) the name of the function and 2) the first parameter will accept a `list` of integers representing the dimensions of a map.
from .oeis import oeisIDfor_n, getOEISids, clearOEIScache # Priority 30



__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'oeisIDfor_n',
]
