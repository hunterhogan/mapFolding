from collections.abc import Sequence
from mapFolding import ComputationState as ComputationState, getPackageDispatcher as getPackageDispatcher, getPathFilenameFoldsTotal as getPathFilenameFoldsTotal, outfitCountFolds as outfitCountFolds, saveFoldsTotal as saveFoldsTotal, setCPUlimit as setCPUlimit, validateListDimensions as validateListDimensions
from os import PathLike

def countFolds(listDimensions: Sequence[int], pathLikeWriteFoldsTotal: str | PathLike[str] | None = None, computationDivisions: int | str | None = None, CPUlimit: int | float | bool | None = None) -> int: ...
