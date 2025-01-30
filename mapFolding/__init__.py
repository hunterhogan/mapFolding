from mapFolding.theSSOT import (
    computationState,
    datatypeLargeDEFAULT,
    datatypeMediumDEFAULT,
    datatypeModuleDEFAULT,
    datatypeSmallDEFAULT,
    dtypeLargeDEFAULT,
    dtypeMediumDEFAULT,
    dtypeSmallDEFAULT,
    getAlgorithmSource,
    getAlgorithmCallable,
    getDispatcherCallable,
    indexMy,
    indexTrack,
    make_dtype,
    pathJobDEFAULT,
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
