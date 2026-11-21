"""Overwrite all synthetic modules."""
from __future__ import annotations

from mapFolding.kitAST.mapFolding.makeModules import makeModulesMapFolding
from mapFolding.kitAST.mapFoldingSymmetric.makeModules import makeModulesFoldsSymmetric
from mapFolding.kitAST.mapFoldingSymmetric.makeModulesAsynchronous import makeModulesFoldsSymmetricAsynchronous
from mapFolding.kitAST.matrixMeanders.makeModules import makeModulesMeanders
from mapFolding.kitAST.oeisModules import makeModulesOEIS
from mapFolding.oeis import settingsOEIS
from mapFolding.oeis._oeisdata import removeDirectoriesEmpty
from mapFolding.oeis._oeisdataClone import updateSnapshot

updateSnapshot()
removeDirectoriesEmpty(settingsOEIS.pathTerms)
makeModulesMapFolding()
makeModulesFoldsSymmetric()
makeModulesFoldsSymmetricAsynchronous()
makeModulesMeanders()

makeModulesOEIS()
