from mapFolding import getPathFilenameFoldsTotal, computationState
from mapFolding import outfitCountFolds
from typing import Any, Literal, Optional, Sequence, Type, overload
import pathlib
import pickle

@overload
def makeStateJob(listDimensions: Sequence[int], writeJob: Literal[True] = True
                 , **keywordArguments: Optional[Type[Any]]) -> pathlib.Path:
    ...

@overload
def makeStateJob(listDimensions: Sequence[int], writeJob: Literal[False] = False
                 , **keywordArguments: Optional[Type[Any]]) -> computationState:
    ...

def makeStateJob(listDimensions: Sequence[int], writeJob: bool = True, **keywordArguments: Optional[Type[Any]]) -> computationState | pathlib.Path:

    stateUniversal: computationState = outfitCountFolds(listDimensions, computationDivisions=None, CPUlimit=None, **keywordArguments)

    from mapFolding.syntheticModules import countInitialize
    countInitialize(stateUniversal['connectionGraph'], stateUniversal['gapsWhere'], stateUniversal['my'], stateUniversal['track'])

    if not writeJob:
        return stateUniversal

    pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal['mapShape'])
    suffix = pathFilenameChopChop.suffix
    pathJob = pathlib.Path(str(pathFilenameChopChop)[0:-len(suffix)])
    pathJob.mkdir(parents=True, exist_ok=True)
    pathFilenameJob = pathJob / 'stateJob.pkl'

    pathFilenameJob.write_bytes(pickle.dumps(stateUniversal))
    return pathFilenameJob
