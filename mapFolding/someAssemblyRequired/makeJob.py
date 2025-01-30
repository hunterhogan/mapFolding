from mapFolding import getPathFilenameFoldsTotal
from mapFolding import outfitCountFolds
from typing import Any, Optional, Sequence, Type
import pathlib
import pickle

def makeStateJob(listDimensions: Sequence[int], **keywordArguments: Optional[Type[Any]]) -> pathlib.Path:

    stateUniversal = outfitCountFolds(listDimensions, computationDivisions=None, CPUlimit=None, **keywordArguments)

    from syntheticModules import countInitialize
    countInitialize(stateUniversal['connectionGraph'], stateUniversal['gapsWhere'], stateUniversal['my'], stateUniversal['track'])

    pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal['mapShape'])
    suffix = pathFilenameChopChop.suffix
    pathJob = pathlib.Path(str(pathFilenameChopChop)[0:-len(suffix)])
    pathJob.mkdir(parents=True, exist_ok=True)
    pathFilenameJob = pathJob / 'stateJob.pkl'

    pathFilenameJob.write_bytes(pickle.dumps(stateUniversal))
    return pathFilenameJob
