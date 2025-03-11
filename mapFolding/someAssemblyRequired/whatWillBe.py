"""
- Settings for synthesizing the modules used by the package (i.e., the flow for numba)
- Settings for synthesizing modules that could be used by the package (e.g., the flow for JAX)
- Therefore, an abstracted system for creating settings for the package
- And with only a little more effort, an abstracted system for creating settings to synthesize arbitrary subsets of modules for arbitrary packages
"""
from collections.abc import Callable
from mapFolding.theSSOT import formatStrModuleForCallableSynthetic, formatStrModuleSynthetic, packageFlowSynthetic, theFileExtension, thePackageName
from numba.core.compiler import CompilerBase as numbaCompilerBase
from typing import Any, TYPE_CHECKING, Final, NamedTuple

try:
	from typing import NotRequired
except Exception:
	from typing_extensions import NotRequired

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict

# =============================================================================
# Temporary or transient or something; probably still the wrong way

# the data converter and the dispatcher could be in the same module.
Z0Z_DataConverterFilename = 'dataNamespaceFlattened' + theFileExtension
Z0Z_DataConverterCallable = 'flattenData'

# =============================================================================
# The right way.
Z0Z_autoflake_additional_imports: list[str] = []
Z0Z_autoflake_additional_imports.append(thePackageName)

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

_datatypeModuleScalar = ''
_decoratorCallable = ''

# if numba
_datatypeModuleScalar = 'numba'
_decoratorCallable = 'jit'
Z0Z_autoflake_additional_imports.append('numba')

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


fileExtension = theFileExtension
formatStrFilenameSynthetic = formatStrModuleSynthetic + fileExtension
formatStrFilenameForCallableSynthetic = formatStrModuleForCallableSynthetic + fileExtension
filenameModuleSyntheticWrite = formatStrFilenameSynthetic.format(packageFlow=packageFlowSynthetic)
filenameWriteCallableTargetDEFAULT: str = 'count'
