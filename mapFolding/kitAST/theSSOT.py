# ruff: ignore[undocumented-public-module]
# DOCUMENT
from __future__ import annotations

from copy import deepcopy
from mapFolding.kitFilesystem import getPathRootJobDEFAULT
from mapFolding.theSSOT import settingsPackage
from mapFolding.theTypes import Default
from pathlib import PurePosixPath
from typing import Final

default = Default(
	filesystem={
		'jobModule': PurePosixPath(getPathRootJobDEFAULT())
		, 'sourcePackage': PurePosixPath(settingsPackage.pathPackage)
	}
	, function={
		'counting': 'count'
		, 'dispatcher': 'doTheNeedful'
	}
	, logicalPath={
		'algorithm': 'algorithms'
		, 'synthetic': 'synthesized'
	}
	, module={
		'dataBasket': 'dataBaskets'
		, 'initializeState': 'initializeState'
	}
	, variable={
		'stateInstance': 'state'
	}
)
# TODO Figure out how to centralize more variables. Example: I renamed
# mapFolding\algorithms\symmetricFolds.py to mapFolding\algorithms\foldsSymmetric.py.
defaultMapFolding: Default = deepcopy(default)
defaultMapFolding['function'].update({
	'initializeState': 'transitionOnGroupsOfFolds'
})
defaultMapFolding['module'].update({
	'algorithm': 'daoOfMapFolding'
})
defaultMapFolding['variable'].update({
	'counting': 'groupsOfFolds'
# TODO Didn't I make a clever function to dynamically extract this value? `findDataclass`
	, 'stateDataclass': 'MapFoldingState'
})

defaultFoldsSymmetric: Default = deepcopy(default)
defaultFoldsSymmetric['function'].update({
	'_processCompletedFutures': '_processCompletedFutures'
	, 'filterAsymmetricFolds': 'filterAsymmetricFolds'
	, 'getSymmetricTotalFolds': 'getSymmetricTotalFolds'
	, 'initializeConcurrencyManager': 'initializeConcurrencyManager'
    , 'initializeState': defaultMapFolding['function']['initializeState']
})
defaultFoldsSymmetric['logicalPath']['synthetic'] += '.foldsSymmetric'
defaultFoldsSymmetric['logicalPath'].update({'assembly': 'kitAST.foldsSymmetric'})
defaultFoldsSymmetric['module'].update({
	'algorithm': 'algorithm'
	, 'asynchronous': 'asynchronous'
})
defaultFoldsSymmetric['variable'].update({
# TODO Ambitious: can I dynamically extract this value from the hand-made algorithm?
	'counting': 'symmetricFolds'
# TODO Didn't I make a clever function to dynamically extract this value? `findDataclass`
	, 'stateDataclass': 'SymmetricFoldsState'
})

dictionaryEstimatesMapFolding: Final[dict[tuple[int, ...], int]] = {
	(2, 2, 2, 2, 2, 2, 2, 2): 798148657152000,  # Probably less than 12 days with my T4 discovery.
	(2, 21): 776374224866624,
	(3, 15): 824761667826225,
	(3, 3, 3, 3): 85109616000000000000000000000000,
	(8, 8): 791274195985524900,  # Two tests, months apart, estimated 300,000 hours to compute.
}
"""Estimates of multidimensional map folding `totalFolds`."""
