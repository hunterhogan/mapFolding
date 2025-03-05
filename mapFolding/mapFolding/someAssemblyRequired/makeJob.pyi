from collections.abc import Sequence
from mapFolding.beDRY import ComputationState as ComputationState, outfitCountFolds as outfitCountFolds, validateListDimensions as validateListDimensions
from mapFolding.filesystem import getPathFilenameFoldsTotal as getPathFilenameFoldsTotal
from mapFolding.theSSOT import getAlgorithmSource as getAlgorithmSource
from typing import Any

def makeStateJob(listDimensions: Sequence[int], *, writeJob: bool = True, **keywordArguments: Any): ...
