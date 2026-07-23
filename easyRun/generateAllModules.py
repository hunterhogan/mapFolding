"""Generate all modules that require some assembly."""
from __future__ import annotations

from mapFolding.someAssemblyRequired.A007822.makeModulesA007822 import makeA007822Modules
from mapFolding.someAssemblyRequired.A007822.makeModulesA007822Asynchronous import makeA007822AsynchronousModules
from mapFolding.someAssemblyRequired.mapFoldingModules.makeModulesMapFolding import makeMapFoldingModules
from mapFolding.someAssemblyRequired.meanders.makeModulesMeanders import makeMeandersModules
from mapFolding.zCuzDocStoopid.makeDocstrings import do

makeMapFoldingModules()

makeA007822Modules()
makeA007822AsynchronousModules()

makeMeandersModules()

do()
