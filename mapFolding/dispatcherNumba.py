from mapFolding import indexMy
from numpy import integer
from numpy.typing import NDArray
from syntheticModules import countInitialize, countParallel, countSequential
from typing import Any, Tuple
import numba

# TODO synthesize this module
# source: theDao.doTheNeedful

# very simple jit
@numba.jit(cache=False)
# the call is `dispatcher(**stateUniversal)`, so the signature must accept all keys in `stateUniversal`, which is `StateUniversal`
def _countFolds(connectionGraph: NDArray[integer[Any]]
                , foldGroups: NDArray[integer[Any]]
                , gapsWhere: NDArray[integer[Any]]
                , mapShape: Tuple[int, ...]
                , my: NDArray[integer[Any]]
                , track: NDArray[integer[Any]]
                ) -> None:

    # print("Numba, you need to ignore the cached files and recompile the functions called below.")

    # build three calls to the three functions, and we know exactly what those are because we are making them, too.
    countInitialize(connectionGraph=connectionGraph, gapsWhere=gapsWhere, my=my, track=track)

    if my[indexMy.taskDivisions.value] > 0:
        countParallel(connectionGraph=connectionGraph, foldGroups=foldGroups, gapsWhere=gapsWhere, my=my, track=track)
    else:
        countSequential(connectionGraph=connectionGraph, foldGroups=foldGroups, gapsWhere=gapsWhere, my=my, track=track)
