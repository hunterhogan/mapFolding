from mapFolding import indexMy, indexTrack
from numpy import integer
from numpy.typing import NDArray
from typing import Any, Tuple
import numba
import numpy
from syntheticModules import countInitialize
from syntheticModules import countParallel
from syntheticModules import countSequential

@numba.jit((numba.uint8[:, :, ::1], numba.int64[::1], numba.uint8[::1], numba.uint8[::1], numba.uint8[::1], numba.uint8[:, ::1]))
def doTheNeedful(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]
                , foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                , gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                , mapShape: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                , my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                , track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]
                ) -> None:
    countInitialize(connectionGraph, gapsWhere, my, track)
    if my[indexMy.taskDivisions.value] > 0:
        countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
    else:
        countSequential(connectionGraph, foldGroups, gapsWhere, my, track)
