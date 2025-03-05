from mapFolding.syntheticModules.numbaCount import countInitialize as countInitialize, countParallel as countParallel, countSequential as countSequential
from mapFolding.theSSOT import indexMy as indexMy
from numpy import dtype, integer, ndarray
from typing import Any

def doTheNeedful(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], foldGroups: ndarray[tuple[int], dtype[integer[Any]]], gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], mapShape: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None: ...
