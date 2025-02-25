from collections.abc import Callable
from mapFolding.theSSOTdatatypes import *
from numba.core.compiler import CompilerBase as numbaCompilerBase
from numpy import dtype, integer, ndarray
from pathlib import Path
from types import ModuleType
from typing import Any, Final, TYPE_CHECKING, cast

try:
	from typing import NotRequired
except Exception:
	from typing_extensions import NotRequired # type: ignore

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict

"""Technical concepts I am likely using and likely want to use more effectively:
- Configuration Registry
- Write-Once, Read-Many (WORM) / Immutable Initialization
- Lazy Initialization
- Separate configuration from business logic

theSSOT and yourSSOT

delay realization/instantiation until a concrete value is desired
moment of truth: when the value is needed, not when the value is defined
"""

"""
listDimensions: list[int]
mapShape
tupleDimensions: tuple[int, ...]
dimensionsTuple
dimensionTuple
"""

def getPathSyntheticModules() -> Path:
	return pathPackage / moduleOfSyntheticModules

def getAlgorithmDispatcher() -> Callable[..., None]:
	algorithmSource: ModuleType = getAlgorithmSource()
	return cast(Callable[..., None], algorithmSource.doTheNeedful) # 'doTheNeedful' is duplicated and there is not a SSOT for it

# NOTE I want this _concept_, not necessarily this method, to be well implemented and usable everywhere: Python, Numba, Jax, CUDA, idc
class computationState(TypedDict):
	connectionGraph:	ndarray[tuple[int, int, int], dtype[integer[Any]]]
	foldGroups:			ndarray[tuple[int]			, dtype[integer[Any]]]
	gapsWhere:			ndarray[tuple[int]			, dtype[integer[Any]]]
	mapShape:			ndarray[tuple[int]			, dtype[integer[Any]]]
	my:					ndarray[tuple[int]			, dtype[integer[Any]]]
	track:				ndarray[tuple[int, int]		, dtype[integer[Any]]]

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

class FREAKOUT(Exception):
	pass

# The following identifier is declared in theDao.py.
# TODO Learn how to assign theDao.py the power to set this truth
# while using theSSOT.py as the SSOT.
Z0Z_identifierCountFolds = 'groupsOfFolds'

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
