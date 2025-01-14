from typing import Tuple
import numpy
import numba
from mapFolding.lovelace import countFoldsCompiled

@numba.jit(cache=True)
def _countFolds(connectionGraph: numpy.ndarray, foldsTotal: numpy.ndarray, mapShape: Tuple[int, ...], my: numpy.ndarray, potentialGaps: numpy.ndarray, the: numpy.ndarray, track: numpy.ndarray):
    return countFoldsCompiled(connectionGraph, foldsTotal, mapShape, my, potentialGaps, the, track)
