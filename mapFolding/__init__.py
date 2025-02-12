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
    hackSSOTdatatype,
    hackSSOTdtype,
    setDatatypeElephino,
    setDatatypeFoldsTotal,
    setDatatypeLeavesTotal,
    setDatatypeModule,
)

# Synthesize modules
from mapFolding.theSSOT import (
    getAlgorithmCallable,
    getAlgorithmSource,
    getPathJobRootDEFAULT,
    getPathSyntheticModules,
    moduleOfSyntheticModules,
    Z0Z_getDatatypeModuleScalar,
    Z0Z_getDecoratorCallable,
    Z0Z_setDatatypeModuleScalar,
    Z0Z_setDecoratorCallable,
)

# Parameters for the prima donna
from mapFolding.theSSOT import (
    ParametersNumba,
    parametersNumbaDEFAULT,
    parametersNumbaFailEarly,
    parametersNumbaParallelDEFAULT,
    parametersNumbaSuperJit,
    parametersNumbaSuperJitParallel,
)

from mapFolding.beDRY import (
    getFilenameFoldsTotal,
    getPathFilenameFoldsTotal,
    outfitCountFolds,
    saveFoldsTotal,
)

from mapFolding.basecamp import countFolds
from mapFolding.oeis import clearOEIScache, getOEISids, oeisIDfor_n

__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'oeisIDfor_n',
]
