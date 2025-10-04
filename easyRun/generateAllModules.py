"""Generate all modules that require some assembly."""
from mapFolding.someAssemblyRequired.A007822.makeA007822AsynchronousModules import makeA007822AsynchronousModules
from mapFolding.someAssemblyRequired.A007822.makeA007822Modules import makeA007822Modules
from mapFolding.someAssemblyRequired.mapFoldingModules.makeMapFoldingModules import makeMapFoldingModules
from mapFolding.someAssemblyRequired.meanders.makeMeandersModules import makeMeandersModules

# TODO from mapFolding.zCuzDocStoopid import makeDocstrings

makeMapFoldingModules()

makeA007822Modules()
makeA007822AsynchronousModules()

makeMeandersModules()
