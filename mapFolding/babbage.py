from mapFolding.importSelector import countSequential, countParallel, countInitialize
from mapFolding import indexThe
from numpy import integer
from numpy.typing import NDArray
from typing import Any, Tuple
import numba
import numpy

@numba.jit(cache=True)
def _countFolds(connectionGraph: NDArray[integer[Any]], foldsSubTotals: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], mapShape: Tuple[int, ...], my: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    """
    What in tarnation is this stupid module and function?

    - This function is not in the same module as `countFolds` so that we can delay Numba just-in-time (jit) compilation of this function and the finalization of its settings until we are ready.
    - This function is not in the same module as `countFoldsCompiled`, which is the function that does the hard, so that we can delay `numba.jit` compilation of `countFoldsCompiled`.
    - `countFoldsCompiled` is not merely "jitted", it is super jitted, which makes it too arrogant to talk to plebian Python functions. It will, however, reluctantly talk to basic jitted functions.
    - The function in this module is jitted, so it can talk to `countFoldsCompiled`, and because it isn't so arrogant, it will talk to the low-class `countFolds` with only a few restrictions, such as:
        - No `TypedDict`
        - No Python v 3.13
        - The plebs must clean up their own memory problems
        - No oversized integers
        - No global variables, only global constants
        - They don't except pleb nonlocal variables either
        - Python "class": they are all inferior to a jit
        - No `**kwargs`
        - and just a few dozen-jillion other things.

    """
    # print("babbage")
    countInitialize(connectionGraph=connectionGraph, gapsWhere=gapsWhere, my=my, the=the, track=track)

    if the[indexThe.taskDivisions.value] > 0:
        countParallel(connectionGraph=connectionGraph, foldsSubTotals=foldsSubTotals, gapsWherePARALLEL=gapsWhere, myPARALLEL=my, the=the, trackPARALLEL=track)
    else:
        countSequential(connectionGraph=connectionGraph, foldsSubTotals=foldsSubTotals, gapsWhere=gapsWhere, my=my, the=the, track=track)
