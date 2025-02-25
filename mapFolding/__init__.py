from collections import defaultdict
from types import ModuleType
import importlib

_dictionaryListsImportFrom: dict[str, list[str]] = defaultdict(list)

def __getattr__(name: str):
	if name not in _mapSymbolToModule:
		raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

	try:
		moduleAsStr: str = _mapSymbolToModule[name]
		module: ModuleType =  importlib.import_module(moduleAsStr)
		blankSymbol = getattr(module, name)
	except (ImportError, ModuleNotFoundError, AttributeError):
		raise

	# The need to inject into globals tells us that the symbol has not actually been imported
	globals()[name] = blankSymbol
	return blankSymbol

_dictionaryListsImportFrom['mapFolding.basecamp'].extend([
	'countFolds',
])

_dictionaryListsImportFrom['mapFolding.beDRY'].extend([
	'getFilenameFoldsTotal',
	'getPathFilenameFoldsTotal',
	'outfitCountFolds',
	'saveFoldsTotal',
])

_dictionaryListsImportFrom['mapFolding.oeis'].extend([
	'clearOEIScache',
	'getOEISids',
	'oeisIDfor_n',
])

# fundamentals
_dictionaryListsImportFrom['mapFolding.theSSOT'].extend([
	'computationState',
	'EnumIndices',
	'getDispatcherCallable',
	'indexMy',
	'indexTrack',
	'myPackageNameIs',
	'pathPackage',
])

# Datatype management
_dictionaryListsImportFrom['mapFolding.theSSOT'].extend([
	'getDatatypeModule',
	'hackSSOTdatatype',
	'hackSSOTdtype',
	'setDatatypeElephino',
	'setDatatypeFoldsTotal',
	'setDatatypeLeavesTotal',
	'setDatatypeModule',
])

# Synthesize modules
_dictionaryListsImportFrom['mapFolding.theSSOT'].extend([
	'additional_importsHARDCODED',
	'formatFilenameModule',
	'getAlgorithmDispatcher',
	'getAlgorithmSource',
	'getPathJobRootDEFAULT',
	'getPathSyntheticModules',
	'listCallablesDispatchees',
	'moduleOfSyntheticModules',
	'Z0Z_filenameModuleWrite',
	'Z0Z_filenameWriteElseCallableTarget',
	'Z0Z_getDatatypeModuleScalar',
	'Z0Z_getDecoratorCallable',
	'Z0Z_identifierCountFolds',
	'Z0Z_setDatatypeModuleScalar',
	'Z0Z_setDecoratorCallable',
])

# Parameters for the prima donna
_dictionaryListsImportFrom['mapFolding.theSSOT'].extend([
	'ParametersNumba',
	'parametersNumbaDEFAULT',
	'parametersNumbaFailEarly',
	'parametersNumbaMinimum',
	'parametersNumbaParallelDEFAULT',
	'parametersNumbaSuperJit',
	'parametersNumbaSuperJitParallel',
])

# Coping
_dictionaryListsImportFrom['mapFolding.theSSOT'].extend([
	'FREAKOUT',
])

_mapSymbolToModule: dict[str, str] = {}
for moduleAsStr, listSymbolsAsStr in _dictionaryListsImportFrom.items():
	for symbolAsStr in listSymbolsAsStr:
		_mapSymbolToModule[symbolAsStr] = moduleAsStr

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from basecamp import *
	from beDRY import *
	from oeis import *
	from theDao import *
	from theSSOT import *
