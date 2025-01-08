from Z0Z_tools import oopsieKwargsie, defineConcurrencyLimit
from mapFolding import outfitFoldings
from typing import Final, List, Optional, Union
import numba
import numpy

def countFolds(listDimensions: List[int], computationDivisions: bool = False, CPUlimit: Optional[Union[int, float, bool]] = None)-> int:
    """
    Count the distinct ways to fold a multi-dimensional map.
    Parameters:
        listDimensions: list of integers, the dimensions of the map.
        computationDivisions: whether to divide the computation into tasks. Dividing into tasks is NEVER* faster and is often many times slower. (*: that I have seen)
        CPUlimit: whether and how to limit the CPU usage. See notes for details.

    Limits on CPU usage `CPUlimit`
        - `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
        - `True`: Yes, limit the CPU usage; limits to 1 CPU.
        - Integer `>= 1`: Limits usage to the specified number of CPUs.
        - Decimal value (`float`) between 0 and 1: Fraction of total CPUs to use.
        - Decimal value (`float`) between -1 and 0: Fraction of CPUs to *not* use.
        - Integer `<= -1`: Subtract the absolute value from total CPUs.
    """

    taskDivisions = computationDivisions if isinstance(computationDivisions, bool) else oopsieKwargsie(computationDivisions)
    if not isinstance(taskDivisions, bool):
        raise ValueError(f"I received {computationDivisions} for the parameter, `computationDivisions`, but I need 'True' or 'False'.")

    if not (CPUlimit is None or isinstance(CPUlimit, (bool, int, float))):
        CPUlimit = oopsieKwargsie(CPUlimit)

    numba.set_num_threads(defineConcurrencyLimit(CPUlimit))

    dtypeDefault: Final = numpy.uint8
    dtypeMaximum: Final = numpy.uint16

    validatedDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions, dtypeDefault, dtypeMaximum)

    dimensionsTotal: Final[int] = len(validatedDimensions)

    from .lovelace import _countFolds
    return _countFolds(leavesTotal, dimensionsTotal, connectionGraph, taskDivisions, track, potentialGaps)
