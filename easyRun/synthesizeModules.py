"""Overwrite all synthetic modules."""
from __future__ import annotations

from mapFolding.kitAST.mapFolding.makeModules import makeModulesMapFolding
# TODO Make the A007822 folding relation explicit in this symmetry terminology.
from mapFolding.kitAST.mapFoldingSymmetric.makeModules import makeModulesFoldsSymmetric
# TODO Make the A007822 folding relation explicit in this symmetry terminology.
from mapFolding.kitAST.mapFoldingSymmetric.makeModulesAsynchronous import makeModulesFoldsSymmetricAsynchronous
from mapFolding.kitAST.matrixMeanders.makeModules import makeModulesMeanders
from mapFolding.kitAST.oeisIDByFormula import makeOEISidByFormula
from mapFolding.kitAST.oeisTests import makeTestsByFormula
from mapFolding.oeis._oeisdataClone import updateSnapshot

updateSnapshot()
makeModulesMapFolding()
# TODO Make the A007822 folding relation explicit in this symmetry terminology.
makeModulesFoldsSymmetric()
# TODO Make the A007822 folding relation explicit in this symmetry terminology.
makeModulesFoldsSymmetricAsynchronous()
makeModulesMeanders()

# TODO These must not be separate processes.
makeOEISidByFormula()
makeTestsByFormula()
