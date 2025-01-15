from numpy import integer
from numpy.typing import NDArray
from typing import Any, Tuple
import numba
import numpy
from mapFolding import outfitFoldings
# from mapFolding.benchmarks.benchmarking import recordBenchmarks
from typing import Optional, Union, Sequence, Type
import os
import pathlib

# TODO the current tests expect positional `listDimensions, computationDivisions`, so after restructuring you can arrange the parameters however you want.
def countFolds(
    listDimensions: Sequence[int],
    computationDivisions: Optional[Union[int, str]] = None,
    CPUlimit: Optional[Union[int, float, bool]] = None,
    writeFoldsTotal: Optional[Union[str, os.PathLike[str]]] = None,
    **keywordArguments: Optional[Type]
    ):
    """keywordArguments:
    dtypeDefault: Optional[Type]
    dtypeLarge: Optional[Type]

    writeFoldsTotal: path, filename, or pathFilename
    """
    stateUniversal = outfitFoldings(listDimensions, computationDivisions=computationDivisions, CPUlimit=CPUlimit, **keywordArguments)

    pathFilenameFoldsTotal = None
    if writeFoldsTotal is not None:
        pathFilenameFoldsTotal = pathlib.Path(writeFoldsTotal)
        if pathFilenameFoldsTotal.is_dir():
            filenameFoldsTotalDEFAULT = str(sorted(stateUniversal['mapShape'])).replace(' ', '') + '.foldsTotal'
            pathFilenameFoldsTotal = pathFilenameFoldsTotal / filenameFoldsTotalDEFAULT
        pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)

    from mapFolding.babbage import _countFolds
    foldsTotal = _countFolds(**stateUniversal)
    # foldsTotal = benchmarkSherpa(**stateUniversal)

    if pathFilenameFoldsTotal is not None:
        try:
            pathFilenameFoldsTotal.write_text(str(foldsTotal))
        except Exception as ERRORmessage:
            print(ERRORmessage)
            print("\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal")
            print(f"{foldsTotal=}")
            print("\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal")

    return foldsTotal

# @recordBenchmarks()
# def benchmarkSherpa(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], mapShape: Tuple[int, ...], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
#     from mapFolding.babbage import _countFolds
#     return _countFolds(connectionGraph, foldsTotal, mapShape, my, gapsWhere, the, track)
