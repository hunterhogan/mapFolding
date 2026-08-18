"""Store shared naming defaults for AST-generated map-folding modules.

(AI generated docstring)

You can use this module to retrieve canonical identifier mappings, logical paths,
filesystem roots, and selected estimate data for AST-driven code generation. The module
centralizes the default naming schemes for generic algorithms, map-folding algorithms,
and symmetry-aware map-folding algorithms.

Contents
--------
Variables
	default
		Baseline identifier and path mapping for generated algorithm modules.
	defaultFoldsSymmetric
		Identifier and path mapping specialized for symmetric-fold counting modules.
	defaultMapFolding
		Identifier and path mapping specialized for multidimensional map-folding modules.
	dictionaryEstimatesMapFolding
		Selected estimated `totalFolds` values for expensive multidimensional shapes.
"""
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
		, 'pathRoot': PurePosixPath(settingsPackage.pathPackage)
		, 'sourcePackage': PurePosixPath(settingsPackage.pathPackage)
	}
	, function={
		'counting': 'count'
		, 'dispatcher': 'doTheNeedful'
	}
	, logicalPath={
		'algorithm': 'algorithms'
		, 'default': ''
		, 'synthetic': 'synthesized'
	}
	, module={
		'dataBasket': 'dataBaskets'
		, 'default': ''
		, 'package': settingsPackage.identifierPackage
		, 'initializeState': 'initializeState'
	}
	, variable={
		'counting': 'groupsOfFolds'
		, 'stateInstance': 'state'
	}
)

defaultMapFolding: Default = deepcopy(default)
defaultMapFolding['function'].update({
	'initializeState': 'transitionOnGroupsOfFolds'
})
defaultMapFolding['module'].update({
	'algorithm': 'daoOfMapFolding'
	, 'algorithmNumba': 'daoOfMapFoldingNumba'
	, 'countParallelNumba': 'countParallelNumba'
	, 'inlineNumba': 'inlineNumba'
	, 'theorem2': 'theorem2'
	, 'theorem2Numba': 'theorem2Numba'
	, 'theorem2Trimmed': 'theorem2Trimmed'
})
defaultMapFolding['variable'].update({
	'counting': 'groupsOfFolds'
	, 'stateDataclass': 'StateMapFolding'
})
defaultMapFolding['logicalPath']['default'] = defaultMapFolding['logicalPath']['algorithm']
defaultMapFolding['module']['default'] = defaultMapFolding['module']['algorithm']

defaultFoldsSymmetric: Default = deepcopy(defaultMapFolding)
defaultFoldsSymmetric['function'].update({
	'_processCompletedFutures': '_processCompletedFutures'
	, 'activeLeafGreaterThan0': 'activeLeafGreaterThan0'
	, 'filterAsymmetricFolds': 'filterAsymmetricFolds'
	, 'getSymmetricTotalFolds': 'getSymmetricTotalFolds'
	, 'initializeConcurrencyManager': 'initializeConcurrencyManager'
	, 'initializeState': defaultMapFolding['function']['initializeState']
})
defaultFoldsSymmetric['logicalPath']['synthetic'] += '.foldsSymmetric'
defaultFoldsSymmetric['logicalPath'].update({'assembly': 'kitAST.foldsSymmetric'})
defaultFoldsSymmetric['module'].update({
	'algorithm': 'algorithm'
	, 'algorithmNumba': 'algorithmNumba'
	, 'algorithmSource': 'foldsSymmetric'
	, 'asynchronous': 'asynchronous'
	, 'asynchronousAnnex': '_asynchronousAnnex'
})
defaultFoldsSymmetric['variable'].update({
	'counting': 'symmetricFolds'
	, 'indices': 'indices'
	, 'maxWorkers': 'maxWorkers'
	, 'stateDataclass': 'StateMapFoldingSymmetric'
})
defaultFoldsSymmetric['logicalPath']['default'] = defaultFoldsSymmetric['logicalPath']['synthetic']
defaultFoldsSymmetric['module']['default'] = defaultFoldsSymmetric['module']['algorithm']

dictionaryEstimatesMapFolding: Final[dict[tuple[int, ...], int]] = {
	(2, 2, 2, 2, 2, 2, 2, 2): 798148657152000
	, (2, 21): 776374224866624
	, (3, 15): 824761667826225
	, (3, 3, 3, 3): 85109616000000000000000000000000
	, (8, 8): 791274195985524900  # Two tests, months apart, estimated 300,000 hours to compute.
}
"""Estimates of multidimensional map folding `totalFolds`."""
