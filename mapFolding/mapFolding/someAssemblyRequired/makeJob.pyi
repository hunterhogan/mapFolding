from collections.abc import Sequence
from mapFolding.beDRY import ComputationState as ComputationState, outfitCountFolds as outfitCountFolds, validateListDimensions as validateListDimensions
from mapFolding.filesystem import getPathFilenameFoldsTotal as getPathFilenameFoldsTotal
from mapFolding.theSSOT import getAlgorithmSource as getAlgorithmSource
from pathlib import Path
from typing import Any, Literal, overload

@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[True], **keywordArguments: Any) -> Path: ...
@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[False], **keywordArguments: Any) -> ComputationState: ...
