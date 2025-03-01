from ctypes import c_uint8, c_int16
from numpy import int64 as numpy_int64, int16 as numpy_int16, uint8 as numpy_uint8
from numpy import dtype, ndarray
from importlib import import_module as importlib_import_module
from inspect import getfile as inspect_getfile
from mapFolding.theWrongWay import *
from pathlib import Path
from sys import modules as sysModules
from types import ModuleType
from typing import Final, TypeAlias
"""
evaluateWhenPACKAGING
evaluateWhenINSTALLING
"""
try:
	import tomli
	TRYmyPackageNameIs: str = tomli.load(Path("../pyproject.toml").open('rb'))["project"]["name"]
except Exception:
	TRYmyPackageNameIs = myPackageNameIsPACKAGING

myPackageNameIs: Final[str] = TRYmyPackageNameIs

def getPathPackageINSTALLING() -> Path:
	pathPackage: Path = Path(inspect_getfile(importlib_import_module(myPackageNameIs)))
	if pathPackage.is_file():
		pathPackage = pathPackage.parent
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

# TODO learn how to see this from the user's perspective
def getPathJobRootDEFAULT() -> Path:
	if 'google.colab' in sysModules:
		pathJobDEFAULT: Path = Path("/content/drive/MyDrive") / "jobs"
	else:
		pathJobDEFAULT = pathPackage / "jobs"
	return pathJobDEFAULT

listCallablesDispatchees: list[str] = listCallablesDispatcheesHARDCODED

additional_importsHARDCODED.append(myPackageNameIs)

concurrencyPackage: Final[str] = 'numba'

# TODO this is stupid: the values will get out of line, but I can't figure out how tyo keep them inline or check they are inline without so much code that the verification code is likely to introduce problems
# DatatypeLeavesTotal: TypeAlias = c_uint8
# numpyLeavesTotal: TypeAlias = numpy_uint8
# DatatypeElephino: TypeAlias = c_int16
# numpyElephino: TypeAlias = numpy_int16
DatatypeLeavesTotal: TypeAlias = int
numpyLeavesTotal: TypeAlias = numpy_int64
DatatypeElephino: TypeAlias = int
numpyElephino: TypeAlias = numpy_int64
DatatypeFoldsTotal: TypeAlias = int
numpyFoldsTotal: TypeAlias = numpy_int64

Array3D: TypeAlias = ndarray[tuple[int, int, int], dtype[numpyLeavesTotal]]
Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[numpyLeavesTotal]]
Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[numpy_int16]]
Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[numpy_int64]]
