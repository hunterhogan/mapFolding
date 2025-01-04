"""
Map/stamp folding counter based on W. F. Lunnon, Multi-dimensional map-folding, "The Computer Journal", Volume 14, Issue 1, 1971, Pages 75-80, https://doi.org/10.1093/comjnl/14.1.75
"""
# HEY! 
# The order of the imports affects the possibility of a partially initialized module error
# so-called circular imports are not actually a problem,
# the issue is that a partially initialized module may have
# a value that could change once the module is fully initialized
# So... does that mean the imports should be in roughly chronological/flow order?
# or reverse chronological/flow order?
# There MUST be an important concept I don't know about or this is a fucking idiotic system.
# I now suspect that both are true.

from .beDRY import getLeavesTotal, parseListDimensions, validateListDimensions
from .baseline import countFolds
from .oeis import oeisSequence_aOFn, getOEISids, clearOEIScache

__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'oeisSequence_aOFn',
]
