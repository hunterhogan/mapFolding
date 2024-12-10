"""
Python implementation of 
W. F. Lunnon, Multi-dimensional map-folding, "The Computer Journal", Volume 14, Issue 1, 1971, Pages 75-80, https://doi.org/10.1093/comjnl/14.1.75 (BibTex)
"""
# HEY! 
# The order of the imports affects the possibility of an 
# error due to importing from a partially initialized module
# so-called circular imports are not actually a problem,
# the issue is that a partially initialized module may have
# a value that could change once it is fully initialized

# NOTE do not import modules with numba compiled function (e.g., `@numba.njit`) here
# because, for example, `numba.set_num_threads()` has no effect on compiled functions
from .babbage import foldings
from .oeis import oeisSequence_aOFn, settingsOEISsequences, dimensionsFoldingsTotalLookup, OEISsequenceID
from .clearOEIScache import clearOEIScache

__all__ = [
    'clearOEIScache',
    'dimensionsFoldingsTotalLookup',
    'foldings',
    'oeisSequence_aOFn',
    'settingsOEISsequences',
]