from collections.abc import Sequence
from mapFolding.beDRY import outfitCountFolds as outfitCountFolds, setCPUlimit as setCPUlimit, validateListDimensions as validateListDimensions
from mapFolding.filesystem import getPathFilenameFoldsTotal as getPathFilenameFoldsTotal, saveFoldsTotal as saveFoldsTotal
from mapFolding.theSSOT import ComputationState as ComputationState, getPackageDispatcher as getPackageDispatcher
from os import PathLike

def countFolds(listDimensions: Sequence[int], pathLikeWriteFoldsTotal: str | PathLike[str] | None = None, computationDivisions: int | str | None = None, CPUlimit: int | float | bool | None = None) -> int:
    '''Count the total number of possible foldings for a given map dimensions.

\tParameters:
\t\tlistDimensions: List of integers representing the dimensions of the map to be folded.
\t\tpathLikeWriteFoldsTotal (None): Path, filename, or pathFilename to write the total fold count to.
\t\t\tIf a directory is provided, creates a file with a default name based on map dimensions.
\t\tcomputationDivisions (None):
\t\t\tWhether and how to divide the computational work. See notes for details.
\t\tCPUlimit (None): This is only relevant if there are `computationDivisions`: whether and how to limit the CPU usage. See notes for details.
\tReturns:
\t\tfoldsTotal: Total number of distinct ways to fold a map of the given dimensions.

\tComputation divisions:
\t\t- None: no division of the computation into tasks; sets task divisions to 0
\t\t- int: direct set the number of task divisions; cannot exceed the map\'s total leaves
\t\t- "maximum": divides into `leavesTotal`-many `taskDivisions`
\t\t- "cpu": limits the divisions to the number of available CPUs, i.e. `concurrencyLimit`

\tLimits on CPU usage `CPUlimit`:
\t\t- `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
\t\t- `True`: Yes, limit the CPU usage; limits to 1 CPU.
\t\t- Integer `>= 1`: Limits usage to the specified number of CPUs.
\t\t- Decimal value (`float`) between 0 and 1: Fraction of total CPUs to use.
\t\t- Decimal value (`float`) between -1 and 0: Fraction of CPUs to *not* use.
\t\t- Integer `<= -1`: Subtract the absolute value from total CPUs.

\tN.B.: You probably don\'t want to divide the computation into tasks.
\t\tIf you want to compute a large `foldsTotal`, dividing the computation into tasks is usually a bad idea. Dividing the algorithm into tasks is inherently inefficient: efficient division into tasks means there would be no overlap in the work performed by each task. When dividing this algorithm, the amount of overlap is between 50% and 90% by all tasks: at least 50% of the work done by every task must be done by _all_ tasks. If you improve the computation time, it will only change by -10 to -50% depending on (at the very least) the ratio of the map dimensions and the number of leaves. If an undivided computation would take 10 hours on your computer, for example, the computation will still take at least 5 hours but you might reduce the time to 9 hours. Most of the time, however, you will increase the computation time. If logicalCores >= leavesTotal, it will probably be faster. If logicalCores <= 2 * leavesTotal, it will almost certainly be slower for all map dimensions.
\t'''
