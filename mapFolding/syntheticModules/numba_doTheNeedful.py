from mapFolding import indexMy, indexTrack
from numpy import integer
from numpy.typing import NDArray
from typing import Any, Tuple
import numba
import numpy
from mapFolding.syntheticModules.numba_countInitialize import countInitialize
from mapFolding.syntheticModules.numba_countParallel import countParallel
from mapFolding.syntheticModules.numba_countSequential import countSequential

@numba.jit((numba.uint8[:, :, ::1], numba.int64[::1], numba.uint8[::1], numba.uint8[::1], numba.uint8[::1], numba.uint8[:, ::1]))
def doTheNeedful(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], mapShape: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    """
        What in tarnation is this stupid module and function?

        - This function is not in the same module as `countFolds` so that we can delay Numba just-in-time (jit) compilation of this function and the finalization of its settings until we are ready.
        - This function is not in the same module as the next function, which does the hard work, so that we can delay `numba.jit` compilation of the next function.
        - This function is "jitted" but the next function is super jitted, which makes it too arrogant to talk to plebian Python functions. It will, however, reluctantly talk to basic jitted functions.
        - So this module can talk to the next function, and because this module isn't as arrogant, it will talk to the low-class `countFolds` that called this function. Well, with a few restrictions, of course:
            - No `TypedDict`
            - The plebs must clean up their own memory problems
            - No oversized integers
            - No global variables, only global constants
            - It won't accept pleb nonlocal variables either
            - Python "class": they are all inferior to the jit class
            - No `**kwargs`
            - and just a few dozen-jillion other things.
        """
    countInitialize(connectionGraph, gapsWhere, my, track)
    if my[indexMy.taskDivisions.value] > 0:
        countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
    else:
        countSequential(connectionGraph, foldGroups, gapsWhere, my, track)