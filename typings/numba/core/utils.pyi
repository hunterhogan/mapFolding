import functools
import json
import typing as _tp
from _typeshed import Incomplete
from collections.abc import Generator, Mapping, MutableMapping, MutableSet
from numba.core import config as config, types as types
from numba.core.config import DEVELOPER_MODE as DEVELOPER_MODE, MACHINE_BITS as MACHINE_BITS, PYVERSION as PYVERSION
from types import ModuleType

def erase_traceback(exc_value):
    """
    Erase the traceback and hanging locals from the given exception instance.
    """
def safe_relpath(path, start=...):
    '''
    Produces a "safe" relative path, on windows relpath doesn\'t work across
    drives as technically they don\'t share the same root.
    See: https://bugs.python.org/issue7195 for details.
    '''

BINOPS_TO_OPERATORS: Incomplete
INPLACE_BINOPS_TO_OPERATORS: Incomplete
ALL_BINOPS_TO_OPERATORS: Incomplete
UNARY_BUITINS_TO_OPERATORS: Incomplete
OPERATORS_TO_BUILTINS: Incomplete
_shutting_down: bool

def _at_shutdown() -> None: ...
def shutting_down(globals=...):
    """
    Whether the interpreter is currently shutting down.
    For use in finalizers, __del__ methods, and similar; it is advised
    to early bind this function rather than look it up when calling it,
    since at shutdown module globals may be cleared.
    """

class ThreadLocalStack:
    """A TLS stack container.

    Uses the BORG pattern and stores states in threadlocal storage.
    """
    _tls: Incomplete
    stack_name: str
    _registered: Incomplete
    def __init_subclass__(cls, *, stack_name, **kwargs) -> None: ...
    _stack: Incomplete
    def __init__(self) -> None: ...
    def push(self, state) -> None:
        """Push to the stack
        """
    def pop(self):
        """Pop from the stack
        """
    def top(self):
        """Get the top item on the stack.

        Raises IndexError if the stack is empty. Users should check the size
        of the stack beforehand.
        """
    def __len__(self) -> int: ...
    def enter(self, state) -> Generator[None]:
        """A contextmanager that pushes ``state`` for the duration of the
        context.
        """

class ConfigOptions:
    OPTIONS: Incomplete
    _values: Incomplete
    def __init__(self) -> None: ...
    def set(self, name, value: bool = True) -> None: ...
    def unset(self, name) -> None: ...
    def _check_attr(self, name) -> None: ...
    def __getattr__(self, name): ...
    def __setattr__(self, name, value) -> None: ...
    def __repr__(self) -> str: ...
    def copy(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __hash__(self): ...

def order_by_target_specificity(target, templates, fnkey: str = ''):
    '''This orders the given templates from most to least specific against the
    current "target". "fnkey" is an indicative typing key for use in the
    exception message in the case that there\'s no usable templates for the
    current "target".
    '''
T = _tp.TypeVar('T')

class OrderedSet(MutableSet[T]):
    _data: Incomplete
    def __init__(self, iterable: _tp.Iterable[T] = ()) -> None: ...
    def __contains__(self, key) -> bool: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def add(self, item) -> None: ...
    def discard(self, item) -> None: ...

class MutableSortedSet(MutableSet[T], _tp.Generic[T]):
    """Mutable Sorted Set
    """
    _values: Incomplete
    def __init__(self, values: _tp.Iterable[T] = ()) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, x: T) -> bool: ...
    def add(self, x: T): ...
    def discard(self, value: T): ...
    def update(self, values) -> None: ...
Tk = _tp.TypeVar('Tk')
Tv = _tp.TypeVar('Tv')

class SortedMap(Mapping[Tk, Tv], _tp.Generic[Tk, Tv]):
    """Immutable
    """
    _values: Incomplete
    _index: Incomplete
    def __init__(self, seq) -> None: ...
    def __getitem__(self, k): ...
    def __len__(self) -> int: ...
    def __iter__(self): ...

class MutableSortedMap(MutableMapping[Tk, Tv], _tp.Generic[Tk, Tv]):
    _dct: dict[Tk, Tv]
    def __init__(self, dct: Incomplete | None = None) -> None: ...
    def __getitem__(self, k: Tk) -> Tv: ...
    def __setitem__(self, k: Tk, v: Tv): ...
    def __delitem__(self, k: Tk): ...
    def __len__(self) -> int: ...
    def __iter__(self) -> int: ...

class UniqueDict(dict):
    def __setitem__(self, key, value) -> None: ...

def runonce(fn): ...
def bit_length(intval):
    """
    Return the number of bits necessary to represent integer `intval`.
    """
def stream_list(lst) -> Generator[Incomplete, None, Incomplete]:
    """
    Given a list, return an infinite iterator of iterators.
    Each iterator iterates over the list from the last seen point up to
    the current end-of-list.

    In effect, each iterator will give the newly appended elements from the
    previous iterator instantiation time.
    """

class BenchmarkResult:
    func: Incomplete
    loop: Incomplete
    records: Incomplete
    best: Incomplete
    def __init__(self, func, records, loop) -> None: ...
    def __repr__(self) -> str: ...

def format_time(tm): ...
def benchmark(func, maxsec: int = 1): ...

_dynamic_modname: str
_dynamic_module: Incomplete

def chain_exception(new_exc, old_exc):
    """Set the __cause__ attribute on *new_exc* for explicit exception
    chaining.  Returns the inplace modified *new_exc*.
    """
def get_nargs_range(pyfunc):
    """Return the minimal and maximal number of Python function
    positional arguments.
    """
def unify_function_types(numba_types):
    """Return a normalized tuple of Numba function types so that

        Tuple(numba_types)

    becomes

        UniTuple(dtype=<unified function type>, count=len(numba_types))

    If the above transformation would be incorrect, return the
    original input as given. For instance, if the input tuple contains
    types that are not function or dispatcher type, the transformation
    is considered incorrect.
    """
def unified_function_type(numba_types, require_precise: bool = True):
    """Returns a unified Numba function type if possible.

    Parameters
    ----------
    numba_types : Sequence of numba Type instances.
    require_precise : bool
      If True, the returned Numba function type must be precise.

    Returns
    -------
    typ : {numba.core.types.Type, None}
      A unified Numba function type. Or ``None`` when the Numba types
      cannot be unified, e.g. when the ``numba_types`` contains at
      least two different Numba function type instances.

    If ``numba_types`` contains a Numba dispatcher type, the unified
    Numba function type will be an imprecise ``UndefinedFunctionType``
    instance, or None when ``require_precise=True`` is specified.

    Specifying ``require_precise=False`` enables unifying imprecise
    Numba dispatcher instances when used in tuples or if-then branches
    when the precise Numba function cannot be determined on the first
    occurrence that is not a call expression.
    """

class _RedirectSubpackage(ModuleType):
    """Redirect a subpackage to a subpackage.

    This allows all references like:

    >>> from numba.old_subpackage import module
    >>> module.item

    >>> import numba.old_subpackage.module
    >>> numba.old_subpackage.module.item

    >>> from numba.old_subpackage.module import item
    """
    __old_module_states: Incomplete
    __new_module: Incomplete
    def __init__(self, old_module_locals, new_module) -> None: ...
    def __reduce__(self): ...

def get_hashable_key(value):
    """
        Given a value, returns a key that can be used
        as a hash. If the value is hashable, we return
        the value, otherwise we return id(value).

        See discussion in gh #6957
    """

class threadsafe_cached_property(functools.cached_property):
    _lock: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def __get__(self, *args, **kwargs): ...

def dump_llvm(fndesc, module) -> None: ...

class _lazy_pformat:
    args: Incomplete
    kwargs: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def __str__(self) -> str: ...

class _LazyJSONEncoder(json.JSONEncoder):
    def default(self, obj): ...
