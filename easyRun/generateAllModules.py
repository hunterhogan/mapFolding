"""Generate all modules that require some assembly."""
from __future__ import annotations

from mapFolding.someAssemblyRequired.foldsSymmetric.makeModulesFoldsSymmetric import makeFoldsSymmetricModules
from mapFolding.someAssemblyRequired.foldsSymmetric.makeModulesFoldsSymmetricAsynchronous import makeFoldsSymmetricAsynchronousModules
from mapFolding.someAssemblyRequired.mapFoldingModules.makeModulesMapFolding import makeMapFoldingModules
from mapFolding.someAssemblyRequired.meanders.makeModulesMeanders import makeMeandersModules
from mapFolding.zCuzDocStoopid.makeDocstrings import do

makeMapFoldingModules()

makeFoldsSymmetricModules()
makeFoldsSymmetricAsynchronousModules()

makeMeandersModules()

do()
