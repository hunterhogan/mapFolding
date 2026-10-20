"""Regenerate multidimensional map-folding modules from configured source.

You can use this module to coordinate dependent source transformations and overwrite the
configured sequential, parallel, initialization, and Theorem 2 output modules.

Contents
--------
Functions
	makeModulesMapFolding
		Regenerate the configured multidimensional counting modules.
"""
from __future__ import annotations

from astToolkit.filesystem import parsePathFilename2astModule
from mapFolding.kitAST.mapFolding._count import makeInlineNumba, makeInlineParallelNumba, makeTheorem2, makeTheorem2Numba, trimTheorem2
from mapFolding.kitAST.mapFolding._doTheNeedful import makeInitializeState
from mapFolding.kitAST.numba.kitNumba import make_jit_module
from mapFolding.kitAST.paths import getModule
from mapFolding.kitAST.theSSOT import defaultMapFolding
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import PurePath

def makeModulesMapFolding() -> None:
	"""Regenerate the configured multidimensional counting modules.

	You can use this function to write module-level compilation, sequential and parallel counting,
	state initialization, and the Theorem 2 variants. The function uses
	`mapFolding.kitAST.theSSOT.defaultMapFolding` [1] for source and output identifiers.
	Each dependent Theorem 2 transformation loads the preceding generated file.
	The function writes the configured output modules.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding.makeModules` uses this call [2].

		```python
		makeModulesMapFolding()
		```

	References
	----------
	[1] `mapFolding.kitAST.theSSOT.defaultMapFolding`

	[2] `mapFolding.kitAST.mapFolding.makeModules`
	"""
	make_jit_module(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	makeInlineNumba(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	makeInlineParallelNumba(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	makeInitializeState(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	pathFilename: PurePath = makeTheorem2(getModule(identifiers=defaultMapFolding), defaultMapFolding)
	pathFilename = trimTheorem2(parsePathFilename2astModule(pathFilename), defaultMapFolding)
	makeTheorem2Numba(parsePathFilename2astModule(pathFilename), defaultMapFolding)

if __name__ == '__main__':
	makeModulesMapFolding()
