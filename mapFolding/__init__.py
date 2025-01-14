"""Prototype concept: Import priority levels. Larger priority values should be imported before smaller priority values."""

# TODO Across the entire package, restructure computationDivisions.

from .theSSOT import * # Priority 10,000
# TODO remove after restructuring
activeGap1ndex = indexMy.gap1ndex.value
activeLeaf1ndex = indexMy.leaf1ndex.value
dimension1ndex = indexMy.dimension1ndex.value
dimensionsUnconstrained = indexMy.dimensionsUnconstrained.value
gap1ndexLowerBound = indexMy.gap1ndexLowerBound.value
indexMiniGap = indexMy.indexMiniGap.value
leaf1ndexConnectee = indexMy.leafConnectee.value

from .beDRY import getLeavesTotal, getTaskDivisions, makeConnectionGraph, outfitFoldings, setCPUlimit, Z0Z_outfitFoldings # Priority 1,000
from .beDRY import parseDimensions, validateListDimensions
from .lola import countFolds # Priority 70. NOTE `countFolds` is the point of the package. Two things should be very stable: 1) the name of the function and 2) the first parameter will accept a `list` of integers representing the dimensions of a map.
# from .lolaTracing import countFolds # Priority 70. NOTE `countFolds` is the point of the package. Two things should be very stable: 1) the name of the function and 2) the first parameter will accept a `list` of integers representing the dimensions of a map.
from .oeis import oeisIDfor_n, getOEISids, clearOEIScache # Priority 30

__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'oeisIDfor_n',
]
