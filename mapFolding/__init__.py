"""Test concept: Import priority levels. Larger priority values should be imported before smaller priority values.
This seems to be a little silly: no useful information is encoded in the priority value, so I don't know if a
new import should have a lower or higher priority.
Crazy concept: Python doesn't cram at least two import roles into one system, call it `import` and tell us how
awesome Python is. Alternatively, I learn about the secret system for mapping physical names to logical names."""

# TODO Across the entire package, restructure computationDivisions.
# test modules need updating still

from .theSSOT import *
from .beDRY import getTaskDivisions, makeConnectionGraph, outfitFoldings, setCPUlimit
from .beDRY import getLeavesTotal, parseDimensions, validateListDimensions
from .startHere import countFolds
from .oeis import oeisIDfor_n, getOEISids, clearOEIScache

__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'oeisIDfor_n',
]
