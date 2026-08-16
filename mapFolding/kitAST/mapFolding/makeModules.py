"""makeMapFoldingModules."""
from __future__ import annotations

from astToolkit import parsePathFilename2astModule
from mapFolding.kitAST.kitMakeModules import getModule
from mapFolding.kitAST.mapFolding._count import (
	makeInlineNumba, makeInlineParallelNumba, makeTheorem2, numbaOnTheorem2, trimTheorem2)
from mapFolding.kitAST.mapFolding._doTheNeedful import makeInitializeState
from mapFolding.kitAST.numba.kitNumba import make_jit_module
from mapFolding.kitAST.theSSOT import defaultMapFolding
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import PurePath

def makeModulesMapFolding() -> None:
	"""Make multidimensional map folding modules."""
	make_jit_module(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	makeInlineNumba(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	makeInlineParallelNumba(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	makeInitializeState(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	pathFilename: PurePath = makeTheorem2(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	pathFilename = trimTheorem2(parsePathFilename2astModule(pathFilename), defaultMapFolding)
	numbaOnTheorem2(parsePathFilename2astModule(pathFilename), defaultMapFolding)

if __name__ == '__main__':
	makeModulesMapFolding()
