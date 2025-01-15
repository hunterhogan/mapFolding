from mapFolding.lovelace import countFoldsCompiled
from numpy import integer
from numpy.typing import NDArray
from typing import Any, Tuple
import numba
import numpy

@numba.jit(cache=True)
def _countFolds(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], mapShape: Tuple[int, ...], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    # TODO learn if I really must change this jitted function to get the super jit to recompile
    # print('babbage')
    return countFoldsCompiled(connectionGraph, foldsTotal, my, gapsWhere, the, track)
