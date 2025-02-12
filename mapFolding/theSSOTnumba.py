"""I have so much truth, I need two files to contain it all!"""
"""TODO learn how to use this efficiently and effectively to solve problems, be DRY, and have SSOT."""
from typing import Final, TYPE_CHECKING, Dict, Any, Union, Callable, Tuple, Any
import numba
import numba.core.compiler
try:
	from typing import NotRequired
except ImportError:
	from typing_extensions import NotRequired

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict

"""
Old notes that are not entirely accurate.

| **Option**			  | **Description**									 | **Why**			   | **Size**		| **But**				  |
| ----------------------- | --------------------------------------------------- | --------------------- | --------------- | ------------------------ |
| `_dbg_extend_lifetimes` | Debug option to extend object lifetimes			 | Debugging			 |				 |						  |
| `_dbg_optnone`		  | Disable optimization for debugging				  | Debugging			 |				 |						  |
| `debug`				 | Enable debug mode with additional checks			| Debugging			 |				 |						  |
| `no_rewrites`		   | Disable AST rewrites optimization				   | Debugging			 |				 |						  |
| `boundscheck`		   | Enable array bounds checking (slows execution)	  | Error checking		| Larger		  | Slower				   |
| `error_model`		   | Divide by zero: kill or chill?					  | Error checking		| ?			   |						  |
| `_nrt`				  | Enable No Runtime type checking					 | Startup speed		 | Smaller		 | No type protection	   |
| `fastmath`			  | Reduce float potential precision					| Float speed		   | Smaller		 | Discriminatory, untested |
| `forceinline`		   | Force function inlining							 | Reduce function calls | Likely larger   |						  |
| `forceobj`			  | Force object mode compilation					   | Inclusiveness		 | Larger		  | Slower execution		 |
| `inline`				| Algorithmically choose inlining					 | Speed				 | Slightly larger |						  |
| `looplift`			  | Enable loop lifting optimization					| Speed (if applicable) | Larger		  | Exclusionary			 |
| `no_cfunc_wrapper`	  | Disable C function wrapper generation			   | Size				  | Smaller		 | Exclusionary			 |
| `no_cpython_wrapper`	| Disable Python C-API wrapper generation			 | Size				  | Smallest		| Exclusionary			 |

"""
# NOTE Deepseek removed forceinline=True, inline='always'
# TODO try to implement all possible parameters, but use `NotRequired` for the more esoteric ones
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
	locals: NotRequired[Dict[str, Any]]
	looplift: bool
	no_cfunc_wrapper: bool
	no_cpython_wrapper: bool
	no_rewrites: NotRequired[bool]
	nogil: NotRequired[bool]
	nopython: bool
	parallel: bool
	pipeline_class: NotRequired[numba.core.compiler.CompilerBase]
	signature_or_function: NotRequired[Union[Any, Callable, str, Tuple]]
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
	'parallel': False,
}
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
	'parallel': False,
}
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""

parametersNumbaParallelDEFAULT: Final[ParametersNumba] = {
	**parametersNumbaDEFAULT,
	'_nrt': True,
	'parallel': True,
}
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""

parametersNumbaSuperJit: Final[ParametersNumba] = {
	**parametersNumbaDEFAULT,
	'no_cfunc_wrapper': True,
	'no_cpython_wrapper': True,
}
"""Speed, no helmet, no talking to non-jitted functions."""

parametersNumbaSuperJitParallel: Final[ParametersNumba] = {
	**parametersNumbaSuperJit,
	'_nrt': True,
	'parallel': True,
}
"""Speed, no helmet, concurrency, no talking to non-jitted functions.
Claude says, "The NRT is Numba's memory management system that handles memory allocation and deallocation for array operations. Because of array copying, you need to have NRT enabled." IDK which AI assistant autocompleted this, but, "The NRT is a bit slower than the default memory management system, but it's necessary for certain operations."
"""
