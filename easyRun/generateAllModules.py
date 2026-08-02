"""Generate all modules that require some assembly."""
from __future__ import annotations

from mapFolding.someAssemblyRequired.foldsSymmetric.makeModulesFoldsSymmetric import makeFoldsSymmetricModules
from mapFolding.someAssemblyRequired.foldsSymmetric.makeModulesFoldsSymmetricAsynchronous import makeFoldsSymmetricAsynchronousModules
from mapFolding.someAssemblyRequired.makeProofOEISidByFormula import makeOEISidByFormulaLookup
from mapFolding.someAssemblyRequired.mapFoldingModules.makeModulesMapFolding import makeMapFoldingModules
from mapFolding.someAssemblyRequired.meanders.makeModulesMeanders import makeMeandersModules
from mapFolding.theSSOT import settingsPackage
from mapFolding.zCuzDocStoopid.makeDocstrings import sourcePrefix, transformOEISidByFormula
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

makeMapFoldingModules()

makeFoldsSymmetricModules()
makeFoldsSymmetricAsynchronousModules()

makeMeandersModules()

pathRoot: Path = settingsPackage.pathPackage / "oeis"
pathFilenameSource: Path = next(iter(pathRoot.glob(f"{sourcePrefix}*.py"))).absolute()
pathFilenameSource = transformOEISidByFormula(pathFilenameSource)

makeOEISidByFormulaLookup(pathFilenameSource)
