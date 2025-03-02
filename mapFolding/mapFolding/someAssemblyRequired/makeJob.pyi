from collections.abc import Sequence
from mapFolding import ComputationState as ComputationState, getAlgorithmSource as getAlgorithmSource, getPathFilenameFoldsTotal as getPathFilenameFoldsTotal, outfitCountFolds as outfitCountFolds, validateListDimensions as validateListDimensions
from typing import Any

def makeStateJob(listDimensions: Sequence[int], *, writeJob: bool = True, **keywordArguments: Any): ...
