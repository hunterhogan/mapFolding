"""
Map folding enumeration and counting algorithms with optimization capabilities.

This package implements algorithms to count and enumerate the various ways
a rectangular map can be folded, based on the mathematical problem described
in Lunnon's 1971 paper. It provides multiple layers of functionality, from
high-level user interfaces to low-level algorithmic optimizations and code
transformation tools.

Core modules:
- basecamp: Public API with simplified interfaces for end users
- theDao: Core computational algorithm using a functional state-transformation approach
- beDRY: Utility functions for common operations and parameter management
- theSSOT: Single Source of Truth for configuration, types, and state management
- oeis: Interface to the Online Encyclopedia of Integer Sequences for known results

Extended functionality:
- someAssemblyRequired: Code transformation framework that optimizes the core algorithm
  through AST manipulation, dataclass transformation, and compilation techniques

Special directories:
- .cache/: Stores cached data from external sources like OEIS to improve performance
- syntheticModules/: Contains dynamically generated, optimized implementations of the
  core algorithm created by the code transformation framework

This package strives to balance algorithm readability and understandability with
high-performance computation capabilities, allowing users to compute map folding
totals for larger dimensions than previously feasible.
"""
from mapFolding.basecamp import countFolds as countFolds
from mapFolding.oeis import clearOEIScache, getOEISids, OEIS_for_n

__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'OEIS_for_n',
]
