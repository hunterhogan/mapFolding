"""Overwrite all synthetic modules."""
from __future__ import annotations

from mapFolding.kitAST.mapFolding.makeModules import makeModulesMapFolding
from mapFolding.kitAST.mapFoldingSymmetric.makeModules import makeModulesFoldsSymmetric
from mapFolding.kitAST.mapFoldingSymmetric.makeModulesAsynchronous import makeModulesFoldsSymmetricAsynchronous
from mapFolding.kitAST.matrixMeanders.makeModules import makeModulesMeanders
from mapFolding.kitAST.oeisIDByFormula import makeOEISidByFormula

makeModulesMapFolding()
makeModulesFoldsSymmetric()
makeModulesFoldsSymmetricAsynchronous()
makeModulesMeanders()

makeOEISidByFormula()
