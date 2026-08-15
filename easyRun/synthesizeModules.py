"""Overwrite all synthetic modules."""
from __future__ import annotations

from mapFolding.someAssemblyRequired.foldsSymmetric.makeModules import makeFoldsSymmetricModules
from mapFolding.someAssemblyRequired.foldsSymmetric.makeModulesAsynchronous import makeModulesFoldsSymmetricAsynchronous
from mapFolding.someAssemblyRequired.makeProofOEISidByFormula import makeOEISidByFormulaLookup
from mapFolding.someAssemblyRequired.mapFolding.makeModules import makeMapFoldingModules
from mapFolding.someAssemblyRequired.meanders.makeModulesMeanders import makeMeandersModules
from mapFolding.theSSOT import settingsPackage
from mapFolding.zCuzDocStoopid.makeDocstrings import sourcePrefix, transformOEISidByFormula
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

makeMapFoldingModules()

makeFoldsSymmetricModules()
makeModulesFoldsSymmetricAsynchronous()

makeMeandersModules()

pathRoot: Path = settingsPackage.pathPackage / "oeis"
pathFilenameSource: Path = next(iter(pathRoot.glob(f"{sourcePrefix}*.py"))).absolute()
pathFilenameSource = transformOEISidByFormula(pathFilenameSource)

makeOEISidByFormulaLookup(pathFilenameSource)
