from mapFolding import outfitFoldings
from typing import Optional, Sequence, Type, Union
import os
import pathlib

def countFolds(listDimensions: Sequence[int], writeFoldsTotal: Optional[Union[str, os.PathLike[str]]] = None, computationDivisions: Optional[Union[int, str]] = None, CPUlimit: Optional[Union[int, float, bool]] = None, **keywordArguments: Optional[Type]) -> int:
    """Count the total number of possible foldings for a given map dimensions.

    Parameters:
        listDimensions: List of integers representing the dimensions of the map to be folded.
        writeFoldsTotal (None): Path or filename to write the total fold count.
            If a directory is provided, creates a file with default name based on map dimensions.
        computationDivisions (None):
            Whether and how to divide the computational work. See notes for details.
        CPUlimit (None): This is only relevant if there are `computationDivisions`: whether and how to limit the CPU usage. See notes for details.
        **keywordArguments: Additional arguments including `dtypeDefault` and `dtypeLarge` for data type specifications.
    Returns:
        foldsTotal: Total number of distinct ways to fold a map of the given dimensions.

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
    stateUniversal = outfitFoldings(listDimensions, computationDivisions=computationDivisions, CPUlimit=CPUlimit, **keywordArguments)

    pathFilenameFoldsTotal = None
    if writeFoldsTotal is not None:
        pathFilenameFoldsTotal = pathlib.Path(writeFoldsTotal)
        if pathFilenameFoldsTotal.is_dir():
            filenameFoldsTotalDEFAULT = str(sorted(stateUniversal['mapShape'])).replace(' ', '') + '.foldsTotal'
            pathFilenameFoldsTotal = pathFilenameFoldsTotal / filenameFoldsTotalDEFAULT
        pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)

    # NOTE Don't import a module with a numba.jit function until you want the function to compile and to freeze all settings for that function.
    from mapFolding.babbage import _countFolds
    foldsTotal = _countFolds(**stateUniversal)
    # foldsTotal = benchmarkSherpa(**stateUniversal)

    if pathFilenameFoldsTotal is not None:
        try:
            pathFilenameFoldsTotal.write_text(str(foldsTotal))
        except Exception as ERRORmessage:
            print(ERRORmessage)
            print(f"\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n\n{foldsTotal=}\n\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal")

    return foldsTotal

# from numpy import integer
# from numpy.typing import NDArray
# from typing import Any, Tuple
# from mapFolding.benchmarks.benchmarking import recordBenchmarks
# @recordBenchmarks()
# def benchmarkSherpa(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], mapShape: Tuple[int, ...], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
#     from mapFolding.babbage import _countFolds
#     return _countFolds(connectionGraph, foldsTotal, mapShape, my, gapsWhere, the, track)
