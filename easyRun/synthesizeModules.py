"""Overwrite all synthetic modules."""
from __future__ import annotations

from mapFolding.kitAST.foldsSymmetric.makeModules import makeModulesFoldsSymmetric
from mapFolding.kitAST.foldsSymmetric.makeModulesAsynchronous import makeModulesFoldsSymmetricAsynchronous
from mapFolding.kitAST.makeProofOEISidByFormula import makeOEISidByFormulaLookup
from mapFolding.kitAST.mapFolding.makeModules import makeModulesMapFolding
from mapFolding.kitAST.meanders.makeModulesMeanders import makeModulesMeanders
from mapFolding.theSSOT import settingsPackage
from mapFolding.zCuzDocStoopid.makeDocstrings import sourcePrefix, transformOEISidByFormula
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

makeModulesMapFolding()

makeModulesFoldsSymmetric()
makeModulesFoldsSymmetricAsynchronous()

makeModulesMeanders()

pathRoot: Path = settingsPackage.pathPackage / "oeis"
pathFilenameSource: Path = next(iter(pathRoot.glob(f"{sourcePrefix}*.py"))).absolute()
pathFilenameSource = transformOEISidByFormula(pathFilenameSource)

makeOEISidByFormulaLookup(pathFilenameSource)
