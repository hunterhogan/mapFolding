# fundamentals
from mapFolding.theSSOT import (
	computationState,
	EnumIndices,
	getDispatcherCallable,
	getPathPackage,
	indexMy,
	indexTrack,
	myPackageNameIs,
)

# Datatype management
from mapFolding.theSSOT import (
	getDatatypeModule,
	hackSSOTdatatype,
	hackSSOTdtype,
	setDatatypeElephino,
	setDatatypeFoldsTotal,
	setDatatypeLeavesTotal,
	setDatatypeModule,
)

# Synthesize modules
from mapFolding.theSSOT import (
	formatFilenameModuleDEFAULT,
	getAlgorithmDispatcher,
	getAlgorithmSource,
	getPathJobRootDEFAULT,
	getPathSyntheticModules,
	moduleOfSyntheticModules,
	Z0Z_getDatatypeModuleScalar,
	Z0Z_getDecoratorCallable,
	Z0Z_setDatatypeModuleScalar,
	Z0Z_setDecoratorCallable,
	Z0Z_identifierCountFolds,
)

# Parameters for the prima donna
from mapFolding.theSSOT import (
	ParametersNumba,
	parametersNumbaDEFAULT,
	parametersNumbaFailEarly,
	parametersNumbaMinimum,
	parametersNumbaParallelDEFAULT,
	parametersNumbaSuperJit,
	parametersNumbaSuperJitParallel,
)

# Coping
from mapFolding.theSSOT import (
	FREAKOUT,
)

from mapFolding.beDRY import (
	getFilenameFoldsTotal,
	getPathFilenameFoldsTotal,
	outfitCountFolds,
	saveFoldsTotal,
)

from mapFolding.basecamp import countFolds
from mapFolding.oeis import clearOEIScache, getOEISids, oeisIDfor_n

__all__: list[str] = [
	'clearOEIScache',
	'countFolds',
	'getOEISids',
	'oeisIDfor_n',
]
