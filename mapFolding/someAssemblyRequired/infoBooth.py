"""
Configuration constants and computational complexity estimates for map folding operations.

Provides default identifiers for code generation, module organization, and computational
resource planning. The module serves as a central registry for configuration values
used throughout the map folding system, particularly for synthetic module generation
and optimization decision-making.

The complexity estimates enable informed choices about computational strategies based
on empirical measurements and theoretical analysis of map folding algorithms for
specific dimensional configurations.
"""

from hunterMakesPy import identifierDotAttribute
from typing import Final

dictionaryEstimatesMapFolding: Final[dict[tuple[int, ...], int]] = {
	(2,2,2,2,2,2,2,2): 798148657152000,
	(2,21): 776374224866624,
	(3,15): 824761667826225,
	(3,3,3,3): 85109616000000000000000000000000,
	(8,8): 791274195985524900,
}
"""Estimates of multidimensional map folding `foldsTotal`."""

identifierCallableSourceDEFAULT: Final[str] = 'count'
"""Default identifier for the core counting function in algorithms."""

identifierCallableSourceDispatcherDEFAULT: Final[str] = 'doTheNeedful'
"""Default identifier for dispatcher functions that route computational tasks."""

identifierCountingDEFAULT: Final[str] = 'groupsOfFolds'
"""Default identifier for the primary counting variable in map folding computations."""

identifierDataclassInstanceDEFAULT: Final[str] = 'state'
"""Default variable name for dataclass instances in generated code."""

identifierModuleDataPackingDEFAULT: Final[str] = 'dataPacking'
"""Default identifier for modules containing data packing and unpacking functions."""

identifierModuleSourceAlgorithmDEFAULT: Final[str] = 'daoOfMapFolding'
"""Default identifier for the algorithm source module containing the base implementation."""

logicalPathInfixAlgorithmDEFAULT: Final[identifierDotAttribute] = 'algorithms'
"""Default logical path component for handmade algorithms."""

logicalPathInfixDEFAULT: Final[identifierDotAttribute] = 'syntheticModules'
"""Default logical path component for organizing synthetic generated modules."""

