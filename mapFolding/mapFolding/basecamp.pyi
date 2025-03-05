from collections.abc import Sequence
from mapFolding.beDRY import ComputationState as ComputationState, outfitCountFolds as outfitCountFolds, setCPUlimit as setCPUlimit, validateListDimensions as validateListDimensions
from mapFolding.filesystem import getPathFilenameFoldsTotal as getPathFilenameFoldsTotal, saveFoldsTotal as saveFoldsTotal
from mapFolding.theSSOT import getPackageDispatcher as getPackageDispatcher
from os import PathLike

def countFolds(listDimensions: Sequence[int], pathLikeWriteFoldsTotal: str | PathLike[str] | None = None, computationDivisions: int | str | None = None, CPUlimit: int | float | bool | None = None) -> int: ...
