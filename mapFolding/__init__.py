from mapFolding.theSSOT import (
    computationState,
    EnumIndices,
    getAlgorithmCallable,
    getAlgorithmSource,
    getDispatcherCallable,
    hackSSOTdtype,
    hackSSOTdatatype,
    indexMy,
    indexTrack,
    relativePathSyntheticModules,
    ParametersNumba,
    parametersNumbaDEFAULT,
    pathJobDEFAULT,
    setDatatypeElephino,
    setDatatypeFoldsTotal,
    setDatatypeLeavesTotal,
    setDatatypeModule,
)
from mapFolding.beDRY import getFilenameFoldsTotal, getPathFilenameFoldsTotal, outfitCountFolds, saveFoldsTotal
from mapFolding.basecamp import countFolds
from mapFolding.oeis import clearOEIScache, getOEISids, oeisIDfor_n

__all__ = [
    'clearOEIScache',
    'countFolds',
    'getOEISids',
    'oeisIDfor_n',
]
