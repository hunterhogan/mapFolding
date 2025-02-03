from mapFolding.theSSOT import (
    computationState,
    # datatypeLargeDEFAULT,
    # datatypeMediumDEFAULT,
    # datatypeModuleDEFAULT,
    # datatypeSmallDEFAULT,
    # dtypeLargeDEFAULT,
    # dtypeMediumDEFAULT,
    # dtypeSmallDEFAULT,
    EnumIndices,
    getAlgorithmCallable,
    getAlgorithmSource,
    getDispatcherCallable,
    hackSSOTdtype,
    indexMy,
    indexTrack,
    # make_dtype,
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
