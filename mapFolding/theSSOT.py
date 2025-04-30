from importlib import import_module as importlib_import_module
from inspect import getfile as inspect_getfile
from pathlib import Path
from tomli import load as tomli_load
import dataclasses

# Evaluate When Packaging https://github.com/hunterhogan/mapFolding/issues/18
try:
	packageNamePACKAGING: str = tomli_load(Path("../pyproject.toml").open('rb'))["project"]["name"]
except Exception:
	packageNamePACKAGING = "mapFolding"

# Evaluate When Installing https://github.com/hunterhogan/mapFolding/issues/18
def getPathPackageINSTALLING() -> Path:
	pathPackage: Path = Path(inspect_getfile(importlib_import_module(packageNamePACKAGING)))
	if pathPackage.is_file():
		pathPackage = pathPackage.parent
	return pathPackage

# PackageSettings in theSSOT.py and immutability https://github.com/hunterhogan/mapFolding/issues/11
@dataclasses.dataclass
class PackageSettings:
	"""
	In _all_ of the root directory and "syntheticModules", the _only_ `PackageSettings` that are used are:
		The.concurrencyPackage
		The.dispatcher
		The.packageName
		The.pathPackage

	From `astToolkit`:
	class PackageSettings:
		fileExtension: str = dataclasses.field(default='.py', metadata={'evaluateWhen': 'installing'})
		packageName: str = dataclasses.field(default = packageNamePACKAGING, metadata={'evaluateWhen': 'packaging'})
		pathPackage: Path = dataclasses.field(default_factory=getPathPackageINSTALLING, metadata={'evaluateWhen': 'installing'})

	"""

	fileExtension: str = dataclasses.field(default='.py', metadata={'evaluateWhen': 'installing'})
	packageName: str = dataclasses.field(default = packageNamePACKAGING, metadata={'evaluateWhen': 'packaging'})
	pathPackage: Path = dataclasses.field(default_factory=getPathPackageINSTALLING, metadata={'evaluateWhen': 'installing'})
