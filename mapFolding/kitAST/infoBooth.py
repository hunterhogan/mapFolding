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

from __future__ import annotations

from copy import deepcopy
from mapFolding.kitFilesystem import getPathRootJobDEFAULT
from mapFolding.theSSOT import settingsPackage
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from hunterMakesPy import identifierDotAttribute
	from typing import Final

dictionaryEstimatesMapFolding: Final[dict[tuple[int, ...], int]] = {
	(2, 2, 2, 2, 2, 2, 2, 2): 798148657152000,  # Probably less than 12 days with my T4 discovery.
	(2, 21): 776374224866624,
	(3, 15): 824761667826225,
	(3, 3, 3, 3): 85109616000000000000000000000000,
	(8, 8): 791274195985524900,  # Two tests, months apart, estimated 300,000 hours to compute.
}
"""Estimates of multidimensional map folding `totalFolds`."""

class Default(TypedDict):
	"""Default values."""

	filesystem: dict[str, PurePosixPath]
	function: dict[str, str]
	logicalPath: dict[str, identifierDotAttribute]
	module: dict[str, str]
	variable: dict[str, str]

default = Default(
	filesystem={
		'jobModule': PurePosixPath(getPathRootJobDEFAULT())
		, 'sourcePackage': PurePosixPath(settingsPackage.pathPackage)
	}
	, function={
		'counting': 'count'
		, 'dispatcher': 'doTheNeedful'
		, 'initializeState': 'transitionOnGroupsOfFolds'
	}
	, logicalPath={
		'algorithm': 'algorithms'
		, 'synthetic': 'syntheticModules'
	}
	, module={
		'algorithm': 'daoOfMapFolding'
		, 'dataBasket': 'dataBaskets'
		, 'initializeState': 'initializeState'
	}
	, variable={
		'counting': 'groupsOfFolds'
		, 'stateDataclass': 'MapFoldingState'
		, 'stateInstance': 'state'
	}
)

# TODO Figure out how to centralize more variables. Example: I renamed
# mapFolding\algorithms\symmetricFolds.py to mapFolding\algorithms\foldsSymmetric.py.
defaultFoldsSymmetric: Default = deepcopy(default)
defaultFoldsSymmetric['function']['_processCompletedFutures'] = '_processCompletedFutures'
defaultFoldsSymmetric['function']['filterAsymmetricFolds'] = 'filterAsymmetricFolds'
defaultFoldsSymmetric['function']['getSymmetricTotalFolds'] = 'getSymmetricTotalFolds'
defaultFoldsSymmetric['function']['initializeConcurrencyManager'] = 'initializeConcurrencyManager'
defaultFoldsSymmetric['logicalPath']['assembly'] = 'kitAST.foldsSymmetric'
defaultFoldsSymmetric['logicalPath']['synthetic'] += '.foldsSymmetric'
defaultFoldsSymmetric['module']['algorithm'] = 'algorithm'
defaultFoldsSymmetric['module']['asynchronous'] = 'asynchronous'
# TODO Ambitious: can I dynamically extract this value from the hand-made algorithm?
defaultFoldsSymmetric['variable']['counting'] = 'symmetricFolds'
# TODO Didn't I make a clever function to dynamically extract this value? `findDataclass`
defaultFoldsSymmetric['variable']['stateDataclass'] = 'SymmetricFoldsState'
