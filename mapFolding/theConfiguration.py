from types import ModuleType
from mapFolding.theWrongWay import *
from sys import modules as sysModules
from pathlib import Path
from importlib import import_module as importlib_import_module
from inspect import getfile as inspect_getfile
from typing import Final
"""
evaluateWhenPACKAGING
evaluateWhenINSTALLING
"""
try:
	import tomli
	TRYmyPackageNameIs: str = tomli.load(Path("../pyproject.toml").open('rb'))["project"]["name"]
except Exception:
	TRYmyPackageNameIs: str = myPackageNameIsPACKAGING

myPackageNameIs: Final[str] = TRYmyPackageNameIs

def getPathPackageINSTALLING() -> Path:
	pathPackage = Path(inspect_getfile(importlib_import_module(myPackageNameIs)))
	if pathPackage.is_file():
		pathPackage: Path = pathPackage.parent
	return pathPackage

pathPackage: Path = getPathPackageINSTALLING()

moduleOfSyntheticModules: Final[str] = "syntheticModules"
formatNameModule = "numba_{callableTarget}"
formatFilenameModule = formatNameModule + ".py"
dispatcherCallableName = "doTheNeedful"
nameModuleDispatcher: str = formatNameModule.format(callableTarget=dispatcherCallableName)
Z0Z_filenameModuleWrite = 'numbaCount.py'
Z0Z_filenameWriteElseCallableTarget: str = 'count'

def getDispatcherCallable():
    logicalPathModule: str = f"{myPackageNameIs}.{moduleOfSyntheticModules}.{nameModuleDispatcher}"
    moduleImported: ModuleType = importlib_import_module(logicalPathModule)
    return getattr(moduleImported, dispatcherCallableName)

def getAlgorithmSource() -> ModuleType:
	logicalPathModule: str = f"{myPackageNameIs}.{algorithmSourcePACKAGING}"
	moduleImported: ModuleType = importlib_import_module(logicalPathModule)
	return moduleImported
	# from mapFolding import theDao
	# return theDao

# TODO learn how to see this from the user's perspective
def getPathJobRootDEFAULT() -> Path:
	if 'google.colab' in sysModules:
		pathJobDEFAULT: Path = Path("/content/drive/MyDrive") / "jobs"
	else:
		pathJobDEFAULT = pathPackage / "jobs"
	return pathJobDEFAULT

listCallablesDispatchees: list[str] = listCallablesDispatcheesHARDCODED

additional_importsHARDCODED.append(myPackageNameIs)
