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

# fundamentals
_dictionaryListsImportFrom['mapFolding.theSSOT'].extend([
	'computationState',
	'concurrencyPackage',
	'getPackageDispatcher',
	'myPackageNameIs',
	'pathPackage',
])

# Datatype management
_dictionaryListsImportFrom['mapFolding.theSSOT'].extend([
	'Array1DElephino',
	'Array1DFoldsTotal',
	'Array1DLeavesTotal',
	'Array3D',
	'DatatypeElephino',
	'DatatypeFoldsTotal',
	'DatatypeLeavesTotal',
	'getDatatypeModule',
	'getNumpyDtypeDefault',
	'numpyElephino',
	'numpyFoldsTotal',
	'numpyLeavesTotal',
])

# Synthesize modules
_dictionaryListsImportFrom['mapFolding.theSSOT'].extend([
	'additional_importsHARDCODED',
	'getAlgorithmDispatcher',
	'getAlgorithmSource',
	'getPathJobRootDEFAULT',
	'getPathSyntheticModules',
	'listNumbaCallableDispatchees',
	'moduleOfSyntheticModulesPACKAGING',
	'ParametersSynthesizeNumbaCallable',
	'Z0Z_dispatcherOfDataCallable',
	'Z0Z_dispatcherOfDataFilename',
	'Z0Z_filenameModuleWrite',
	'Z0Z_filenameWriteElseCallableTarget',
	'Z0Z_formatFilenameModuleSynthetic',
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

_dictionaryListsImportFrom['mapFolding.basecamp'].extend([
	'countFolds',
])

_dictionaryListsImportFrom['mapFolding.beDRY'].extend([
	'ComputationState',
	'getFilenameFoldsTotal',
	'getLeavesTotal',
	'getPathFilenameFoldsTotal',
	'getTaskDivisions',
	'makeConnectionGraph',
	'makeDataContainer',
	'outfitCountFolds',
	'saveFoldsTotal',
	'setCPUlimit',
	'validateListDimensions',
])

_dictionaryListsImportFrom['mapFolding.oeis'].extend([
	'clearOEIScache',
	'getOEISids',
	'getOEISidValues',
	'OEIS_for_n',
	'oeisIDfor_n',
	'oeisIDsImplemented',
	'settingsOEIS',
	'validateOEISid',
])

_mapSymbolToModule: dict[str, str] = {}
for moduleAsStr, listSymbolsAsStr in _dictionaryListsImportFrom.items():
	for symbolAsStr in listSymbolsAsStr:
		_mapSymbolToModule[symbolAsStr] = moduleAsStr

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from theDao import *
	from theSSOT import *
	from beDRY import *
	from basecamp import *
	from oeis import *
