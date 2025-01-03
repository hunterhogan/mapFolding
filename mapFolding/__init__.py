"""
Map/stamp folding counter based on W. F. Lunnon, Multi-dimensional map-folding, "The Computer Journal", Volume 14, Issue 1, 1971, Pages 75-80, https://doi.org/10.1093/comjnl/14.1.75
"""

from .beDRY import getLeavesTotal, validateListDimensions, outfitFoldings, validateTaskDivisions
from .lego import foldings
from .oeis import oeisSequence_aOFn, getOEISids
from .clearOEIScache import clearOEIScache

__all__ = [
    'clearOEIScache',
    'foldings', 
    'getOEISids',
    'oeisSequence_aOFn', 
]
