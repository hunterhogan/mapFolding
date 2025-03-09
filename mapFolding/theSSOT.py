from collections.abc import Callable
from importlib import import_module as importlib_import_module
from inspect import getfile as inspect_getfile
from numba.core.compiler import CompilerBase as numbaCompilerBase
from numpy import dtype, ndarray, int64 as numpy_int64, int16 as numpy_int16
from numpy.typing import DTypeLike
from pathlib import Path
from sys import modules as sysModules
from types import ModuleType
from typing import Any, cast, Final, NamedTuple, TYPE_CHECKING, TypeAlias
import dataclasses
import tomli

try:
	from typing import NotRequired
except Exception:
	from typing_extensions import NotRequired # type: ignore

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict

Z0Z_packageFlow = 'algorithm'
# Z0Z_packageFlow = 'numba'

# =============================================================================
# The Wrong Way The Wrong Way The Wrong Way The Wrong Way The Wrong Way
# Evaluate When Packaging Evaluate When Packaging Evaluate When Packaging

sourceAlgorithmPACKAGING: str = 'theDao'
datatypeModulePACKAGING: Final[str] = 'numpy'
dispatcherCallableNamePACKAGING: str = 'doTheNeedful'
moduleOfSyntheticModulesPACKAGING: Final[str] = 'syntheticModules'

dataclassIdentifierAsStrPACKAGING: str = 'ComputationState'
dataclassInstanceAsStrPACKAGING: str = 'state'

try:
	myPackageNameIsPACKAGING: str = tomli.load(Path("../pyproject.toml").open('rb'))["project"]["name"]
except Exception:
	myPackageNameIsPACKAGING: str = "mapFolding"

# =============================================================================
# The Wrong Way The Wrong Way The Wrong Way The Wrong Way The Wrong Way
# Evaluate When Installing Evaluate When Installing Evaluate When Installing

fileExtensionINSTALLING: str = '.py'

def getPathPackageINSTALLING() -> Path:
	pathPackage: Path = Path(inspect_getfile(importlib_import_module(myPackageNameIsPACKAGING)))
	if pathPackage.is_file():
		pathPackage = pathPackage.parent
	return pathPackage

# =============================================================================
# The Wrong Way The Wrong Way The Wrong Way The Wrong Way The Wrong Way
# Hardcoding Hardcoding Hardcoding Hardcoding Hardcoding Hardcoding Hardcoding

# NOTE see also `ParametersSynthesizeNumbaCallable` below

# =============================================================================
# Temporary or transient or something; probably still the wrong way

# the data converter and the dispatcher could be in the same module.
Z0Z_DataConverterFilename = 'dataNamespaceFlattened' + fileExtensionINSTALLING
Z0Z_DataConverterCallable = 'flattenData'

# =============================================================================
# The right way, perhaps.

# =====================
# Create enduring identifiers from the hopefully transient identifiers above.
myPackageNameIs: Final[str] = myPackageNameIsPACKAGING
pathPackage: Path = getPathPackageINSTALLING()
theSourceAlgorithm: str = sourceAlgorithmPACKAGING
theDatatypeModule: Final[str] = datatypeModulePACKAGING
theDispatcherCallableName: str = dispatcherCallableNamePACKAGING
theModuleOfSyntheticModules: Final[str] = moduleOfSyntheticModulesPACKAGING
theDataclassIdentifierAsStr: str = dataclassIdentifierAsStrPACKAGING
theDataclassInstanceAsStr: str = dataclassInstanceAsStrPACKAGING
theFileExtension: str = fileExtensionINSTALLING

# =============================================================================
# The right way.
autoflake_additional_imports: list[str] = []
autoflake_additional_imports.append(myPackageNameIs)
concurrencyPackage: str = 'not implemented'

# =============================================================================
# The relatively flexible type system needs a different paradigm, but I don't
# know what it should be. The system needs to 1) help optimize computation, 2)
# make it possible to change the basic type of the package (e.g., from numpy
# to superTypePy), 3) make it possible to synthesize the optimized modules used
# by the package (numba, at the moment), and 4) make it possible to synthesize
# arbitrary modules with different type systems.

DatatypeLeavesTotal: TypeAlias = int
# this would be uint8, but mapShape (2,2,2,2, 2,2,2,2) has 256 leaves, so generic containers accommodate
numpyLeavesTotal: TypeAlias = numpy_int16

DatatypeElephino: TypeAlias = int
numpyElephino: TypeAlias = numpy_int16

DatatypeFoldsTotal: TypeAlias = int
numpyFoldsTotal: TypeAlias = numpy_int64
numpyDtypeDefault = numpyFoldsTotal

Array3D: TypeAlias = ndarray[tuple[int, int, int], dtype[numpyLeavesTotal]]
Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[numpyLeavesTotal]]
Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[numpyElephino]]
Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[numpyFoldsTotal]]

# =============================================================================
# The right way.
# (The dataclass, not the typing of the dataclass.)
# (Also, my noobplementation of the dataclass certainly needs improvement.)

@dataclasses.dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
class ComputationState:
	mapShape: tuple[DatatypeLeavesTotal, ...]
	leavesTotal: DatatypeLeavesTotal
	taskDivisions: DatatypeLeavesTotal

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'description': 'A 3D array representing the connection graph of the map.'})
	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)

	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None) # pyright: ignore[reportAssignmentType]
	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None) # pyright: ignore[reportAssignmentType]
	gapRangeStart: Array1DElephino = dataclasses.field(default=None) # pyright: ignore[reportAssignmentType]
	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None) # pyright: ignore[reportAssignmentType]
	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None) # pyright: ignore[reportAssignmentType]
	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None) # pyright: ignore[reportAssignmentType]
	foldGroups: Array1DFoldsTotal = dataclasses.field(default=None) # pyright: ignore[reportAssignmentType]

	foldsTotal: DatatypeFoldsTotal = DatatypeFoldsTotal(0)
	gap1ndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	gap1ndexCeiling: DatatypeElephino = DatatypeElephino(0)
	groupsOfFolds: DatatypeFoldsTotal = DatatypeFoldsTotal(0)
	indexDimension: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexLeaf: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexMiniGap: DatatypeElephino = DatatypeElephino(0)
	leaf1ndex: DatatypeElephino = DatatypeElephino(1)
	leafConnectee: DatatypeElephino = DatatypeElephino(0)
	taskIndex: DatatypeLeavesTotal = dataclasses.field(default=DatatypeLeavesTotal(0), metadata={'myType': DatatypeLeavesTotal})
	# taskIndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)

	def __post_init__(self):
		from mapFolding.beDRY import makeConnectionGraph, makeDataContainer
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		self.connectionGraph = makeConnectionGraph(self.mapShape, self.leavesTotal, numpyLeavesTotal)

		if self.dimensionsUnconstrained is None: # pyright: ignore[reportUnnecessaryComparison]
			self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal))

		if self.foldGroups is None:
			self.foldGroups = makeDataContainer(max(2, int(self.taskDivisions) + 1), numpyFoldsTotal)
			self.foldGroups[-1] = self.leavesTotal

		leavesTotalAsInt = int(self.leavesTotal)

		if self.countDimensionsGapped is None:
			self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, numpyElephino)
		if self.gapRangeStart is None:
			self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.gapsWhere is None:
			self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.leafAbove is None:
			self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.leafBelow is None:
			self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)

	def getFoldsTotal(self):
		self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)

	# factory? constructor?
	# state.taskIndex = state.taskIndex.type(indexSherpa)
	# self.fieldName = self.fieldName.fieldType(indexSherpa)
	# state.taskIndex.toMyType(indexSherpa)

# =============================================================================
# The most right way I know how to implement.

logicalPathModuleSourceAlgorithm: str = f"{myPackageNameIs}.{theSourceAlgorithm}"
logicalPathModuleDispatcher: str = logicalPathModuleSourceAlgorithm

def getSourceAlgorithm() -> ModuleType:
	moduleImported: ModuleType = importlib_import_module(logicalPathModuleSourceAlgorithm)
	return moduleImported

def getAlgorithmDispatcher():
	moduleImported: ModuleType = getSourceAlgorithm()
	# TODO I think I need to use `inspect` to type the return value
	dispatcherCallable = getattr(moduleImported, theDispatcherCallableName)
	return dispatcherCallable

def getPathSyntheticModules() -> Path:
	return pathPackage / theModuleOfSyntheticModules

# TODO learn how to see this from the user's perspective
def getPathJobRootDEFAULT() -> Path:
	if 'google.colab' in sysModules:
		pathJobDEFAULT: Path = Path("/content/drive/MyDrive") / "jobs"
	else:
		pathJobDEFAULT = pathPackage / "jobs"
	return pathJobDEFAULT

_datatypeModule: str = ''
def getDatatypeModule() -> str:
	global _datatypeModule
	if not _datatypeModule:
		_datatypeModule = theDatatypeModule
	return _datatypeModule

def getNumpyDtypeDefault() -> DTypeLike:
	return numpyDtypeDefault

# =============================================================================
# The coping way.

class FREAKOUT(Exception): pass

# =============================================================================
# Temporary or transient or something; probably still the wrong way

Z0Z_formatNameModuleSynthetic = "numba_{callableTarget}"
Z0Z_formatFilenameModuleSynthetic = Z0Z_formatNameModuleSynthetic + fileExtensionINSTALLING
Z0Z_nameModuleDispatcherSynthetic: str = Z0Z_formatNameModuleSynthetic.format(callableTarget=theDispatcherCallableName)
Z0Z_filenameModuleWrite = 'numbaCount' + fileExtensionINSTALLING
Z0Z_logicalPathDispatcherSynthetic: str = '.'.join([myPackageNameIs, theModuleOfSyntheticModules, Z0Z_nameModuleDispatcherSynthetic])
Z0Z_filenameWriteElseCallableTarget: str = 'count'

_datatypeModuleScalar = ''
_decoratorCallable = ''

# =============================================================================
# The most right way I know how to implement.

if Z0Z_packageFlow == 'numba': # pyright: ignore [reportUnnecessaryComparison]
	autoflake_additional_imports.append('numba')
	concurrencyPackage: str = 'numba'
	logicalPathModuleDispatcher = Z0Z_logicalPathDispatcherSynthetic
	_datatypeModuleScalar = 'numba'
	_decoratorCallable = 'jit'

def getPackageDispatcher():
	moduleImported: ModuleType = importlib_import_module(logicalPathModuleDispatcher)
	dispatcherCallable = getattr(moduleImported, theDispatcherCallableName)
	return dispatcherCallable

def Z0Z_getDatatypeModuleScalar() -> str:
	return _datatypeModuleScalar

def Z0Z_setDatatypeModuleScalar(moduleName: str) -> str:
	global _datatypeModuleScalar
	_datatypeModuleScalar = moduleName
	return _datatypeModuleScalar

def Z0Z_getDecoratorCallable() -> str:
	return _decoratorCallable

def Z0Z_setDecoratorCallable(decoratorName: str) -> str:
	global _decoratorCallable
	_decoratorCallable = decoratorName
	return _decoratorCallable

class ParametersNumba(TypedDict):
	_dbg_extend_lifetimes: NotRequired[bool]
	_dbg_optnone: NotRequired[bool]
	_nrt: NotRequired[bool]
	boundscheck: NotRequired[bool]
	cache: bool
	debug: NotRequired[bool]
	error_model: str
	fastmath: bool
	forceinline: bool
	forceobj: NotRequired[bool]
	inline: str
	locals: NotRequired[dict[str, Any]]
	looplift: bool
	no_cfunc_wrapper: bool
	no_cpython_wrapper: bool
	no_rewrites: NotRequired[bool]
	nogil: NotRequired[bool]
	nopython: bool
	parallel: bool
	pipeline_class: NotRequired[type[numbaCompilerBase]]
	signature_or_function: NotRequired[Any | Callable[..., Any] | str | tuple[Any, ...]]
	target: NotRequired[str]

parametersNumbaFailEarly: Final[ParametersNumba] = { '_nrt': True, 'boundscheck': True, 'cache': True, 'error_model': 'python', 'fastmath': False, 'forceinline': True, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': True, 'parallel': False, }
"""For a production function: speed is irrelevant, error discovery is paramount, must be compatible with anything downstream."""

parametersNumbaDEFAULT: Final[ParametersNumba] = { '_nrt': True, 'boundscheck': False, 'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': True, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': True, 'parallel': False, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""

parametersNumbaParallelDEFAULT: Final[ParametersNumba] = { **parametersNumbaDEFAULT, '_nrt': True, 'parallel': True, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""

parametersNumbaSuperJit: Final[ParametersNumba] = { **parametersNumbaDEFAULT, 'no_cfunc_wrapper': True, 'no_cpython_wrapper': True, }
"""Speed, no helmet, no talking to non-jitted functions."""

parametersNumbaSuperJitParallel: Final[ParametersNumba] = { **parametersNumbaSuperJit, '_nrt': True, 'parallel': True, }
"""Speed, no helmet, concurrency, no talking to non-jitted functions."""

parametersNumbaMinimum: Final[ParametersNumba] = { '_nrt': True, 'boundscheck': True, 'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': False, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': False, 'forceobj': True, 'parallel': False, }

class ParametersSynthesizeNumbaCallable(NamedTuple):
	callableTarget: str
	parametersNumba: ParametersNumba | None = None
	inlineCallables: bool = False

listNumbaCallableDispatchees: list[ParametersSynthesizeNumbaCallable] = [
	ParametersSynthesizeNumbaCallable('countParallel', parametersNumbaSuperJitParallel, True),
	ParametersSynthesizeNumbaCallable('countSequential', parametersNumbaSuperJit, True),
	ParametersSynthesizeNumbaCallable('countInitialize', parametersNumbaDEFAULT, True),
]

"""Technical concepts I am likely using and likely want to use more effectively:
- Configuration Registry
- Write-Once, Read-Many (WORM) / Immutable Initialization
- Lazy Initialization
- Separate configuration from business logic

theSSOT and yourSSOT

delay realization/instantiation until a concrete value is desired
moment of truth: when the value is needed, not when the value is defined
"""
