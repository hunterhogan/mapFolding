from collections.abc import Sequence
from mapFolding import computationState as computationState, getAlgorithmSource as getAlgorithmSource, getPathFilenameFoldsTotal as getPathFilenameFoldsTotal, outfitCountFolds as outfitCountFolds
from pathlib import Path
from typing import Literal, overload

@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[True], **keywordArguments: str | None) -> Path: ...
@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[False], **keywordArguments: str | None) -> computationState: ...
