"""Overwrite all synthetic modules."""
from __future__ import annotations

from mapFolding.kitAST.mapFolding.makeModules import makeModulesMapFolding
from mapFolding.kitAST.mapFoldingSymmetric.makeModules import makeModulesFoldsSymmetric
from mapFolding.kitAST.mapFoldingSymmetric.makeModulesAsynchronous import makeModulesFoldsSymmetricAsynchronous
from mapFolding.kitAST.matrixMeanders.makeModules import makeModulesMeanders
from mapFolding.kitAST.matrixMeanders.makeModulesAxis import makeModulesAxis
from mapFolding.kitAST.oeis.oeisModules import makeModulesOEIS
from mapFolding.oeis import update_oeisdata

update_oeisdata()
makeModulesMapFolding()
makeModulesFoldsSymmetric()
makeModulesFoldsSymmetricAsynchronous()
makeModulesMeanders()
makeModulesAxis()

makeModulesOEIS()
