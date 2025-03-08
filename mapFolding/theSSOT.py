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

# =============================================================================
# The Wrong Way The Wrong Way The Wrong Way The Wrong Way The Wrong Way
# Evaluate When Packaging Evaluate When Packaging Evaluate When Packaging

algorithmSourcePACKAGING: str = 'theDao'
datatypeModulePACKAGING: Final[str] = 'numpy'
dispatcherCallableNamePACKAGING: str = 'doTheNeedful'
moduleOfSyntheticModulesPACKAGING: Final[str] = 'syntheticModules'

dataclassIdentifierPACKAGING: str = 'ComputationState'
dataclassInstancePACKAGING: str = 'state'

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

additional_importsHARDCODED: list[str] = ['numba']

# NOTE see also `ParametersSynthesizeNumbaCallable` below

# =============================================================================
# Temporary or transient or something; probably still the wrong way

Z0Z_formatNameModuleSynthetic = "numba_{callableTarget}"
Z0Z_formatFilenameModuleSynthetic = Z0Z_formatNameModuleSynthetic + fileExtensionINSTALLING
Z0Z_dispatcherCallableName = dispatcherCallableNamePACKAGING
Z0Z_nameModuleDispatcherSynthetic: str = Z0Z_formatNameModuleSynthetic.format(callableTarget=Z0Z_dispatcherCallableName)
Z0Z_filenameModuleWrite = 'numbaCount' + fileExtensionINSTALLING
Z0Z_filenameWriteElseCallableTarget: str = 'count'
# the data converter and the dispatcher could be in the same module.
Z0Z_DataConverterFilename = 'dataNamespaceFlattened' + fileExtensionINSTALLING
Z0Z_DataConverterCallable = 'flattenData'
concurrencyPackage: str = 'numba'

# =============================================================================
# The right way, perhaps.

myPackageNameIs: Final[str] = myPackageNameIsPACKAGING
pathPackage: Path = getPathPackageINSTALLING()

theAlgorithmSource: str = algorithmSourcePACKAGING
theDatatypeModule: Final[str] = datatypeModulePACKAGING
theDispatcherCallableName: str = dispatcherCallableNamePACKAGING
theModuleOfSyntheticModules: Final[str] = moduleOfSyntheticModulesPACKAGING
theDataclassIdentifier: str = dataclassIdentifierPACKAGING
theDataclassInstance: str = dataclassInstancePACKAGING
theFileExtension: str = fileExtensionINSTALLING

additional_importsHARDCODED.append(myPackageNameIs)

# TODO this is stupid: the values will get out of line, but I can't figure out how to keep them inline or check they are inline without so much code that the verification code is likely to introduce problems
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
Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[numpyElephino]]
Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[numpyFoldsTotal]]

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

def getAlgorithmSource() -> ModuleType:
	logicalPathModule: str = f"{myPackageNameIs}.{theAlgorithmSource}"
	moduleImported: ModuleType = importlib_import_module(logicalPathModule)
	return moduleImported

def getAlgorithmDispatcher():
	moduleImported: ModuleType = getAlgorithmSource()
	dispatcherCallable = getattr(moduleImported, theDispatcherCallableName)
	return cast(Callable[[ComputationState], ComputationState], dispatcherCallable)

# I DON'T KNOW!
def getPackageDispatcher():
	moduleImported: ModuleType = getAlgorithmSource()
	dispatcherCallable = getattr(moduleImported, theDispatcherCallableName)
	return cast(Callable[[ComputationState], ComputationState], dispatcherCallable)

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
	return numpyFoldsTotal

# =============================================================================
# More truth

def getPathSyntheticModules() -> Path:
	return pathPackage / theModuleOfSyntheticModules

_datatypeModuleScalar = 'numba'
_decoratorCallable = 'jit'
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

class FREAKOUT(Exception): pass

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
"""
refactor theDao, use dataclass or something that makes sense
refactor the synthesize modules to transform the into data structures that work for numba
"""

"""Technical concepts I am likely using and likely want to use more effectively:
- Configuration Registry
- Write-Once, Read-Many (WORM) / Immutable Initialization
- Lazy Initialization
- Separate configuration from business logic

theSSOT and yourSSOT

delay realization/instantiation until a concrete value is desired
moment of truth: when the value is needed, not when the value is defined
"""

class ParametersSynthesizeNumbaCallable(NamedTuple):
	callableTarget: str
	parametersNumba: ParametersNumba | None = None
	inlineCallables: bool = False

listNumbaCallableDispatchees: list[ParametersSynthesizeNumbaCallable] = [
	ParametersSynthesizeNumbaCallable('countParallel', parametersNumbaSuperJitParallel, True),
	ParametersSynthesizeNumbaCallable('countSequential', parametersNumbaSuperJit, True),
	ParametersSynthesizeNumbaCallable('countInitialize', parametersNumbaDEFAULT, True),
]
