"""
Python implementation of 
W. F. Lunnon, Multi-dimensional map-folding, "The Computer Journal", Volume 14, Issue 1, 1971, Pages 75-80, https://doi.org/10.1093/comjnl/14.1.75 (BibTex)
"""
# HEY! 
# The order of the imports affects the possibility of a partially initialized module error
# so-called circular imports are not actually a problem,
# the issue is that a partially initialized module may have
# a value that could change once the module is fully initialized
# So... does that mean the imports should be in roughly chronological/flow order?
# or reverse chronological/flow order?
# There MUST be an import concept I don't know about or this is a fucking idiotic system.

# NOTE do not import modules with numba compiled function (e.g., `@numba.njit`) here
# because, for example, `numba.set_num_threads()` has no effect on compiled functions
from .beDRY import getLeavesTotal, parseListDimensions
from .types import OEISsequenceID
from .babbage import foldings
from .oeis import oeisSequence_aOFn, settingsOEISsequences, getOEISids
from .noCircularImportsIsAlie import getFoldingsTotalKnown
from .clearOEIScache import clearOEIScache

__all__ = [
    'clearOEIScache',
    'foldings',
    'getOEISids',
    'oeisSequence_aOFn',
    'settingsOEISsequences',
]