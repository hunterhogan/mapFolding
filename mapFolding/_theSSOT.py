from importlib import import_module as importlib_import_module
from inspect import getfile as inspect_getfile
from pathlib import Path
from tomli import load as tomli_load
import dataclasses

packageNamePACKAGING_HARDCODED = "mapFolding"
concurrencyPackageHARDCODED = 'multiprocessing'

# Evaluate When Packaging
# https://github.com/hunterhogan/mapFolding/issues/18
try:
	packageNamePACKAGING: str = tomli_load(Path("../pyproject.toml").open('rb'))["project"]["name"]
except Exception:
	packageNamePACKAGING = packageNamePACKAGING_HARDCODED

# Evaluate When Installing
# https://github.com/hunterhogan/mapFolding/issues/18
def getPathPackageINSTALLING() -> Path:
	pathPackage: Path = Path(inspect_getfile(importlib_import_module(packageNamePACKAGING)))
	if pathPackage.is_file():
		pathPackage = pathPackage.parent
	return pathPackage

@dataclasses.dataclass
class PackageSettings:
	fileExtension: str = dataclasses.field(default='.py', metadata={'evaluateWhen': 'installing'})
	packageName: str = dataclasses.field(default = packageNamePACKAGING, metadata={'evaluateWhen': 'packaging'})
	pathPackage: Path = dataclasses.field(default_factory=getPathPackageINSTALLING, metadata={'evaluateWhen': 'installing'})
	concurrencyPackage: str | None = None
	"""Package to use for concurrent execution (e.g., 'multiprocessing', 'numba')."""

concurrencyPackage = concurrencyPackageHARDCODED
packageSettings = PackageSettings(concurrencyPackage=concurrencyPackage)
