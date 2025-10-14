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

from copy import deepcopy
from hunterMakesPy import identifierDotAttribute
from typing import Final, TypedDict

dictionaryEstimatesMapFolding: Final[dict[tuple[int, ...], int]] = {
	(2,2,2,2,2,2,2,2): 798148657152000,
	(2,21): 776374224866624,
	(3,15): 824761667826225,
	(3,3,3,3): 85109616000000000000000000000000,
	(8,8): 791274195985524900, # A test estimated 300,000 hours to compute.
}
"""Estimates of multidimensional map folding `foldsTotal`."""

class Default(TypedDict):
	"""Default identifiers."""

	function: dict[str, str]
	logicalPath: dict[str, identifierDotAttribute]
	module: dict[str, str]
	variable: dict[str, str]

default = Default(
	function = {
		'counting': 'count',
		'dispatcher': 'doTheNeedful',
		'initializeState': 'transitionOnGroupsOfFolds',
	},
	logicalPath = {
		'algorithm': 'algorithms',
		'synthetic': 'syntheticModules',
	},
	module = {
		'algorithm': 'daoOfMapFolding',
		'dataBasket': 'dataBaskets',
		'initializeState': 'initializeState',
	},
	variable = {
		'counting': 'groupsOfFolds',
		'stateDataclass': 'MapFoldingState',
		'stateInstance': 'state',
	},
)

defaultA007822: Default = deepcopy(default)
defaultA007822['function']['_processCompletedFutures'] = '_processCompletedFutures'
defaultA007822['function']['filterAsymmetricFolds'] = 'filterAsymmetricFolds'
defaultA007822['function']['getSymmetricFoldsTotal'] = 'getSymmetricFoldsTotal'
defaultA007822['function']['initializeConcurrencyManager'] = 'initializeConcurrencyManager'
defaultA007822['logicalPath']['assembly'] = 'someAssemblyRequired.A007822'
defaultA007822['logicalPath']['synthetic'] += '.A007822'
defaultA007822['module']['algorithm'] = 'algorithm'
defaultA007822['module']['asynchronous'] = 'asynchronous'
defaultA007822['variable']['counting'] = 'symmetricFolds'
defaultA007822['variable']['stateDataclass'] = 'SymmetricFoldsState'
