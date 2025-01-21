from typing import Any, Optional, Sequence, Type, Union

def Z0Z_makeJob(listDimensions: Sequence[int], **keywordArguments: Optional[Type[Any]]):
    from mapFolding import outfitCountFolds
    stateUniversal = outfitCountFolds(listDimensions, computationDivisions=None, CPUlimit=None, **keywordArguments)
    from mapFolding.countInitialize import countInitialize
    countInitialize(stateUniversal['connectionGraph'], stateUniversal['gapsWhere'], stateUniversal['my'], stateUniversal['the'], stateUniversal['track'])
    from mapFolding import getPathFilenameFoldsTotal
    pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal['mapShape'])
    import pathlib
    suffix = pathFilenameChopChop.suffix
    pathJob = pathlib.Path(str(pathFilenameChopChop)[0:-len(suffix)])
    pathJob.mkdir(parents=True, exist_ok=True)
    pathFilenameJob = pathJob / 'stateJob.pkl'

    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateUniversal['mapShape'], pathFilenameJob.parent)
    stateJob = {**stateUniversal, 'pathFilenameFoldsTotal': pathFilenameFoldsTotal}

    del stateJob['mapShape']

    import pickle
    pathFilenameJob.write_bytes(pickle.dumps(stateJob))
    return pathFilenameJob

def runJob(pathFilename):
    from typing import Final
    import numpy
    from pathlib import Path
    pathFilenameJob = Path(pathFilename)
    from pickle import loads
    stateJob = loads(pathFilenameJob.read_bytes())

    connectionGraph: numpy.ndarray = stateJob['connectionGraph']
    foldsSubTotals: numpy.ndarray = stateJob['foldsSubTotals']
    gapsWhere: numpy.ndarray = stateJob['gapsWhere']
    my: numpy.ndarray = stateJob['my']
    pathFilenameFoldsTotal: Final[Path] = stateJob['pathFilenameFoldsTotal']
    the: Final[numpy.ndarray] = stateJob['the']
    track: numpy.ndarray = stateJob['track']

    from mapFolding.countSequentialNoNumba import countSequential
    countSequential(connectionGraph, foldsSubTotals, gapsWhere, my, the, track)

    print(foldsSubTotals.sum().item())
    Path(pathFilenameFoldsTotal).parent.mkdir(parents=True, exist_ok=True)
    Path(pathFilenameFoldsTotal).write_text(str(foldsSubTotals.sum().item()))
    print(pathFilenameFoldsTotal)
