"""Access shared identifier mappings and estimated counts for source generation.

(AI generated docstring)

You can use this module to inspect package roots, source and generated module identifiers,
state names, and selected progress estimates. The specialized mappings are independent
copies of the base mapping, while each mapping remains mutable.

Contents
--------
Variables
	default
		Access the base identifier and filesystem defaults for module generation.
	defaultMapFolding
		Access identifier and output defaults for multidimensional folding modules.
	defaultMapFoldingSymmetric
		Access identifier and output defaults for symmetric folding modules.
	defaultMatrixMeanders
		Access identifier and output defaults for matrix-meander modules.
	lookupMapFoldingEstimates
		Look up selected fold-count estimates for expensive map shapes.
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
		, 'numbaDecorator': 'jit'
		, 'numbaModule': 'jit_module'
		, 'numbaVectorize': 'vectorize'
	}
	, logicalPath={
		'algorithm': 'algorithms'
		, 'default': ''
		, 'synthetic': 'synthesized'
		, 'numba': 'numba'
	}
	, module={
		'dataBasket': 'dataBaskets'
		, 'algorithmNumba': 'algorithmNumba'
		, 'default': ''
		, 'package': settingsPackage.identifierPackage
		, 'initializeState': 'initializeState'
	}
	, variable={
		'counting': 'groupsOfFolds'
		, 'numbaProgressBar': 'ProgressBarGroupsOfFolds'
		, 'stateInstance': 'state'
	}
)
"""Access the base identifier and filesystem defaults for module generation.

(AI generated docstring)

This mapping supplies package roots, source and generated logical paths, state names,
and base counting and dispatcher names. The specialized defaults copy this mapping.
"""

defaultMapFolding: Default = deepcopy(default)
"""Access identifier and output defaults for multidimensional folding modules.

(AI generated docstring)

This mapping extends the base defaults with state initialization, Theorem 2 variants,
compilation variants, and the source algorithm identifiers.
"""
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
	, 'stateDataclassParallel': 'StateMapFoldingParallel'
})
defaultMapFolding['logicalPath']['default'] = defaultMapFolding['logicalPath']['algorithm']
defaultMapFolding['module']['default'] = defaultMapFolding['module']['algorithm']

defaultMapFoldingSymmetric: Default = deepcopy(defaultMapFolding)
"""Access identifier and output defaults for symmetric folding modules.

(AI generated docstring)

This mapping extends the map-folding defaults with symmetric state, filtering, worker
management, and synchronous and asynchronous generation identifiers.
"""
defaultMapFoldingSymmetric['function'].update({
	'_processCompletedFutures': '_processCompletedFutures'
	, 'activeLeafGreaterThan0': 'activeLeafGreaterThan0'
	, 'filterAsymmetricFolds': 'filterAsymmetricFolds'
	, 'getSymmetricTotalFolds': 'getSymmetricTotalFolds'
	, 'initializeConcurrencyManager': 'initializeConcurrencyManager'
	, 'initializeState': defaultMapFolding['function']['initializeState']
})
defaultMapFoldingSymmetric['logicalPath']['synthetic'] += '.mapFoldingSymmetric'
defaultMapFoldingSymmetric['logicalPath'].update({'assembly': 'kitAST.mapFoldingSymmetric'})
defaultMapFoldingSymmetric['module'].update({
	'algorithm': 'algorithm'
	, 'algorithmNumba': 'algorithmNumba'
	, 'algorithmSource': 'foldsSymmetric'
	, 'asynchronous': 'asynchronous'
	, 'asynchronousAnnex': '_asynchronousAnnex'
})
defaultMapFoldingSymmetric['variable'].update({
	'counting': 'symmetricFolds'
	, 'indices': 'indices'
	, 'maxWorkers': 'maxWorkers'
	, 'stateDataclass': 'StateMapFoldingSymmetric'
})
defaultMapFoldingSymmetric['logicalPath']['default'] = defaultMapFoldingSymmetric['logicalPath']['synthetic']
defaultMapFoldingSymmetric['module']['default'] = defaultMapFoldingSymmetric['module']['algorithm']

defaultMatrixMeanders: Default = deepcopy(default)
"""Access identifier and output defaults for matrix-meander modules.

(AI generated docstring)

This mapping extends the base defaults with shared counting, width tests, pruning
variants, data containers, and generated matrix-meander paths.
"""
defaultMatrixMeanders['function'].update({
	'bigInt': 'countBigInt'
	, 'bigIntTest': 'integersWide吗'
	, 'bigIntTestNumPy': 'integersWideNumPy吗'
	, 'bigIntTestPandas': 'integersWidePandas吗'
	, 'Dyck': 'walkDyckPath'
	, 'flipDyckEndpoint': 'flipDyckEndpoint'
	, 'aggregateArcCodes': 'aggregateArcCodes'
	, 'prune': 'prune'
	, 'pruneArray': 'pruneArray'
	, 'pruneDataFrame': 'pruneDataFrame'
})
defaultMatrixMeanders['logicalPath']['synthetic'] += '.matrixMeanders'
defaultMatrixMeanders['module'].update({
	'algorithm': 'matrixMeanders'
	, 'bigInt': 'matrixMeandersShare'
	, 'chop': 'chop'
	, 'prune': 'prune'
	, 'pruneNumPy': 'pruneNumPy'
	, 'prunePandas': 'prunePandas'
	, 'numpy': 'matrixMeandersNumPy'
	, 'pandas': 'matrixMeandersPandas'
	, 'common': 'matrixMeandersShare'
	, 'share': 'matrixMeandersShare'
})
defaultMatrixMeanders['variable'].update({
	'arrayMeanders': 'arrayMeanders'
	, 'dataframeAnalyzed': 'dataframeAnalyzed'
	, 'tqdmBoundary': 'tqdmBoundary'
})
defaultMatrixMeanders['logicalPath']['default'] = defaultMatrixMeanders['logicalPath']['algorithm']
defaultMatrixMeanders['module']['default'] = defaultMatrixMeanders['module']['algorithm']

lookupMapFoldingEstimates: Final[dict[tuple[int, ...], int]] = {
	(2, 2, 2, 2, 2, 2, 2, 2): 798148657152000
	, (2, 21): 776374224866624
	, (2, 22): 4037141814510720
	, (3, 15): 824761667826225
	, (3, 3, 3, 3): 85109616000000000000000000000000
	, (8, 8): 791274195985524900  # Two tests, months apart, estimated 300,000 hours to compute.
}
"""Estimates of multidimensional map folding `totalFolds`."""
