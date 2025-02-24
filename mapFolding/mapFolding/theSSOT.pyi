from mapFolding.theSSOTdatatypes import *
from collections.abc import Callable
from numba.core.compiler import CompilerBase as numbaCompilerBase
from numpy import dtype, integer, ndarray
from pathlib import Path
from types import ModuleType
from typing import Any, Final, TypedDict
from typing_extensions import NotRequired

myPackageNameIs: str
moduleOfSyntheticModules: str
formatFilenameModuleDEFAULT: str
dispatcherCallableNameDEFAULT: str

def getPathPackage() -> Path: ...
def getPathJobRootDEFAULT() -> Path: ...
def getPathSyntheticModules() -> Path: ...
def getAlgorithmSource() -> ModuleType: ...
def getAlgorithmDispatcher() -> Callable[..., None]: ...
def getDispatcherCallable() -> Callable[..., None]: ...

class computationState(TypedDict):
    connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]]
    foldGroups: ndarray[tuple[int], dtype[integer[Any]]]
    gapsWhere: ndarray[tuple[int], dtype[integer[Any]]]
    mapShape: ndarray[tuple[int], dtype[integer[Any]]]
    my: ndarray[tuple[int], dtype[integer[Any]]]
    track: ndarray[tuple[int, int], dtype[integer[Any]]]

_datatypeModuleScalar: str
_decoratorCallable: str

def Z0Z_getDatatypeModuleScalar() -> str: ...
def Z0Z_setDatatypeModuleScalar(moduleName: str) -> str: ...
def Z0Z_getDecoratorCallable() -> str: ...
def Z0Z_setDecoratorCallable(decoratorName: str) -> str: ...

class FREAKOUT(Exception): ...

Z0Z_identifierCountFolds: str

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
