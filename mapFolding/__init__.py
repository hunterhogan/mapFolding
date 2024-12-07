"""
Python implementation of 
W. F. Lunnon, Multi-dimensional map-folding, "The Computer Journal", Volume 14, Issue 1, 1971, Pages 75-80, https://doi.org/10.1093/comjnl/14.1.75 (BibTex)
"""
import importlib.metadata

metadata = importlib.metadata.metadata("mapFolding")
__version__ = metadata["Version"]
__author__ = metadata["Author"]

# HEY! 
# The order of the imports affects the possibility of an 
# error due to importing from a partially initialized module

from .lovelace import foldings
from .oeis import oeisSequence_aOFn, settingsOEISsequences, dimensionsFoldingsTotalLookup
from .clearOEIScache import clearOEIScache

__all__ = [
    '__author__',
    '__version__',
    'clearOEIScache',
    'dimensionsFoldingsTotalLookup',
    'foldings',
    'oeisSequence_aOFn',
    'settingsOEISsequences',
]