"""Test concept: Import priority levels. Larger priority values should be imported before smaller priority values.
This seems to be a little silly: no useful information is encoded in the priority value, so I don't know if a
new import should have a lower or higher priority.
Crazy concept: Python doesn't cram at least two import roles into one system, call it `import` and tell us how
awesome Python is. Alternatively, I learn about the secret system for mapping physical names to logical names."""

# TODO Across the entire package, restructure computationDivisions.
# test modules need updating still

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
# from .lola import countFolds # Priority 70. NOTE `countFolds` is the point of the package. Two things should be very stable: 1) the name of the function and 2) the first parameter will accept a `list` of integers representing the dimensions of a map.
from .lolaTracing import countFolds # Priority 70. NOTE `countFolds` is the point of the package. Two things should be very stable: 1) the name of the function and 2) the first parameter will accept a `list` of integers representing the dimensions of a map.
from .oeis import oeisIDfor_n, getOEISids, clearOEIScache # Priority 30

__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'oeisIDfor_n',
]
