from _typeshed import Incomplete
from collections.abc import Callable
from numba.core.compiler import CompilerBase as numbaCompilerBase
from numpy.typing import DTypeLike as DTypeLike
from pathlib import Path
from types import ModuleType
from typing import Any, Final, NamedTuple, TypeAlias, TypedDict
from typing_extensions import NotRequired

algorithmSourcePACKAGING: str
datatypeModulePACKAGING: Final[str]
dispatcherCallableNamePACKAGING: str
moduleOfSyntheticModulesPACKAGING: Final[str]
dataclassIdentifierPACKAGING: str
myPackageNameIsPACKAGING: str

def getPathPackageINSTALLING() -> Path: ...

additional_importsHARDCODED: list[str]
myPackageNameIs: Final[str]
pathPackage: Path
Z0Z_formatNameModuleSynthetic: str
Z0Z_formatFilenameModuleSynthetic: Incomplete
Z0Z_nameModuleDispatcherSynthetic: str
Z0Z_filenameModuleWrite: str
Z0Z_filenameWriteElseCallableTarget: str
Z0Z_dispatcherOfDataFilename: str
Z0Z_dispatcherOfDataCallable: str
concurrencyPackage: str

def getAlgorithmSource() -> ModuleType: ...
def getAlgorithmDispatcher(): ...
def getPackageDispatcher(): ...
def getPathJobRootDEFAULT() -> Path: ...

DatatypeLeavesTotal: TypeAlias
numpyLeavesTotal: TypeAlias
DatatypeElephino: TypeAlias
numpyElephino: TypeAlias
DatatypeFoldsTotal: TypeAlias
numpyFoldsTotal: TypeAlias
Array3D: TypeAlias
Array1DLeavesTotal: TypeAlias
Array1DElephino: TypeAlias
Array1DFoldsTotal: TypeAlias
_datatypeModule: str

def getDatatypeModule() -> str: ...
def getNumpyDtypeDefault() -> DTypeLike: ...
def getPathSyntheticModules() -> Path: ...

_datatypeModuleScalar: str
_decoratorCallable: str

def Z0Z_getDatatypeModuleScalar() -> str: ...
def Z0Z_setDatatypeModuleScalar(moduleName: str) -> str: ...
def Z0Z_getDecoratorCallable() -> str: ...
def Z0Z_setDecoratorCallable(decoratorName: str) -> str: ...

class FREAKOUT(Exception): ...

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

parametersNumbaFailEarly: Final[ParametersNumba]
parametersNumbaDEFAULT: Final[ParametersNumba]
parametersNumbaParallelDEFAULT: Final[ParametersNumba]
parametersNumbaSuperJit: Final[ParametersNumba]
parametersNumbaSuperJitParallel: Final[ParametersNumba]
parametersNumbaMinimum: Final[ParametersNumba]

class ParametersSynthesizeNumbaCallable(NamedTuple):
    callableTarget: str
    parametersNumba: ParametersNumba | None = ...
    inlineCallables: bool = ...

listNumbaCallableDispatchees: list[ParametersSynthesizeNumbaCallable]
