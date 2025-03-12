from collections.abc import Callable
from numba.core.compiler import CompilerBase as numbaCompilerBase
from typing import Any, TYPE_CHECKING, Final

try:
	from typing import NotRequired
except Exception:
	from typing_extensions import NotRequired

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict

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

parametersNumbaFailEarly: Final[ParametersNumba] = {
		'_nrt': True,
		'boundscheck': True,
		'cache': True,
		'error_model': 'python',
		'fastmath': False,
		'forceinline': True,
		'inline': 'always',
		'looplift': False,
		'no_cfunc_wrapper': False,
		'no_cpython_wrapper': False,
		'nopython': True,
		'parallel': False, }
"""For a production function: speed is irrelevant, error discovery is paramount, must be compatible with anything downstream."""

parametersNumbaDEFAULT: Final[ParametersNumba] = {
		'_nrt': True,
		'boundscheck': False,
		'cache': True,
		'error_model': 'numpy',
		'fastmath': True,
		'forceinline': True,
		'inline': 'always',
		'looplift': False,
		'no_cfunc_wrapper': False,
		'no_cpython_wrapper': False,
		'nopython': True,
		'parallel': False, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""

parametersNumbaParallelDEFAULT: Final[ParametersNumba] = {
		**parametersNumbaDEFAULT,
		'_nrt': True,
		'parallel': True, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""

parametersNumbaSuperJit: Final[ParametersNumba] = {
		**parametersNumbaDEFAULT,
		'no_cfunc_wrapper': True,
		'no_cpython_wrapper': True, }
"""Speed, no helmet, no talking to non-jitted functions."""

parametersNumbaSuperJitParallel: Final[ParametersNumba] = {
		**parametersNumbaSuperJit,
		'_nrt': True,
		'parallel': True, }
"""Speed, no helmet, concurrency, no talking to non-jitted functions."""

parametersNumbaMinimum: Final[ParametersNumba] = {
		'_nrt': True,
		'boundscheck': True,
		'cache': True,
		'error_model': 'numpy',
		'fastmath': True,
		'forceinline': False,
		'inline': 'always',
		'looplift': False,
		'no_cfunc_wrapper': False,
		'no_cpython_wrapper': False,
		'nopython': False,
		'forceobj': True,
		'parallel': False, }
