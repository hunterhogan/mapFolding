from pathlib import Path
from mapFolding import outfitCountFolds, getPathFilenameFoldsTotal, saveFoldsTotal
from typing import Any, Optional, Sequence, Type, Union
import os

def countFolds(listDimensions: Sequence[int], pathishWriteFoldsTotal: Optional[Union[str, os.PathLike[str]]] = None, computationDivisions: Optional[Union[int, str]] = None, CPUlimit: Optional[Union[int, float, bool]] = None, **keywordArguments: Optional[Type[Any]]) -> int:
    """Count the total number of possible foldings for a given map dimensions.

    Parameters:
        listDimensions: List of integers representing the dimensions of the map to be folded.
        pathishWriteFoldsTotal (None): Path, filename, or pathFilename to write the total fold count to.
            If a directory is provided, creates a file with a default name based on map dimensions.
        computationDivisions (None):
            Whether and how to divide the computational work. See notes for details.
        CPUlimit (None): This is only relevant if there are `computationDivisions`: whether and how to limit the CPU usage. See notes for details.
        **keywordArguments: Additional arguments including `dtypeDefault` and `dtypeLarge` for data type specifications.
    Returns:
        foldsSubTotals: Total number of distinct ways to fold a map of the given dimensions.

    Computation divisions:
        - None: no division of the computation into tasks; sets task divisions to 0
        - int: direct set the number of task divisions; cannot exceed the map's total leaves
        - "maximum": divides into `leavesTotal`-many `taskDivisions`
        - "cpu": limits the divisions to the number of available CPUs, i.e. `concurrencyLimit`

    Limits on CPU usage `CPUlimit`:
        - `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
        - `True`: Yes, limit the CPU usage; limits to 1 CPU.
        - Integer `>= 1`: Limits usage to the specified number of CPUs.
        - Decimal value (`float`) between 0 and 1: Fraction of total CPUs to use.
        - Decimal value (`float`) between -1 and 0: Fraction of CPUs to *not* use.
        - Integer `<= -1`: Subtract the absolute value from total CPUs.

    N.B.: You probably don't want to divide the computation into tasks.
        If you want to compute a large `foldsTotal`, dividing the computation into tasks is usually a bad idea. Dividing the algorithm into tasks is inherently inefficient: efficient division into tasks means there would be no overlap in the work performed by each task. When dividing this algorithm, the amount of overlap is between 50% and 90% by all tasks: at least 50% of the work done by every task must be done by _all_ tasks. If you improve the computation time, it will only change by -10 to -50% depending on (at the very least) the ratio of the map dimensions and the number of leaves. If an undivided computation would take 10 hours on your computer, for example, the computation will still take at least 5 hours but you might reduce the time to 9 hours. Most of the time, however, you will increase the computation time. If logicalCores >= leavesTotal, it will probably be faster. If logicalCores <= 2 * leavesTotal, it will almost certainly be slower for all map dimensions.
    """
    stateUniversal = outfitCountFolds(listDimensions, computationDivisions=computationDivisions, CPUlimit=CPUlimit, **keywordArguments)

    pathFilenameFoldsTotal = None
    if pathishWriteFoldsTotal is not None:
        pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateUniversal['mapShape'], pathishWriteFoldsTotal)

    from mapFolding.babbage import _countFolds
    _countFolds(**stateUniversal)

    foldsTotal = stateUniversal['foldsSubTotals'].sum().item()

    if pathFilenameFoldsTotal is not None:
        saveFoldsTotal(pathFilenameFoldsTotal, foldsTotal)

    return foldsTotal

def Z0Z_makeJob(listDimensions: Sequence[int], **keywordArguments: Optional[Type[Any]]) -> Path:
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
    import pickle
    pathFilenameJob.write_bytes(pickle.dumps(dict(stateUniversal)))
    return pathFilenameJob

def runJob(pathFilename: str):
    from ctypes import c_ulonglong
    from mapFolding import getPathFilenameFoldsTotal
    from mapFolding import saveFoldsTotal
    from mapFolding.countSequentialNoNumba import countSequential
    from pathlib import Path
    from pickle import loads
    from typing import Final
    from typing import Tuple
    import numpy

    pathFilenameJob: Final = Path(pathFilename)

    stateJob = loads(pathFilenameJob.read_bytes())

    connectionGraph: Final[numpy.ndarray] = stateJob['connectionGraph']
    foldsSubTotals: numpy.ndarray = stateJob['foldsSubTotals']
    gapsWhere: numpy.ndarray = stateJob['gapsWhere']
    mapShape: Final[Tuple[int, ...]] = stateJob['mapShape']
    my: numpy.ndarray = stateJob['my']
    the: Final[numpy.ndarray] = stateJob['the']
    track: numpy.ndarray = stateJob['track']
    del stateJob

    pathFilenameFoldsTotal: Final[Path] = getPathFilenameFoldsTotal(mapShape, pathFilenameJob.parent)
    foldsTotal = c_ulonglong(0)

    def compileThis():
        nonlocal connectionGraph, foldsTotal, foldsSubTotals, gapsWhere, my, pathFilenameFoldsTotal, the, track
        countSequential(connectionGraph, foldsSubTotals, gapsWhere, my, the, track)
        foldsTotal = foldsSubTotals.sum().item()
        print(foldsTotal)
        saveFoldsTotal(pathFilenameFoldsTotal, foldsTotal)
        print(pathFilenameFoldsTotal)

    compileThis()
