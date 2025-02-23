from .abstract import Callable as Callable, DTypeSpec as DTypeSpec, Dummy as Dummy, Literal as Literal, Type as Type, weakref as weakref
from .common import Opaque as Opaque
from .misc import unliteral as unliteral
from _typeshed import Incomplete
from numba.core import config as config, errors as errors, types as types, utils as utils
from numba.core.typeconv import Conversion as Conversion
from typing import NamedTuple

_logger: Incomplete
_termcolor: Incomplete

class _FAILURE(NamedTuple):
    template: Incomplete
    matched: Incomplete
    error: Incomplete
    literal: Incomplete

_termwidth: Incomplete
_header_lead: str
_header_template: Incomplete
_reason_template: str

def _wrapper(tmp, indent: int = 0): ...

_overload_template: str
_err_reasons: Incomplete

def _bt_as_lines(bt):
    """
    Converts a backtrace into a list of lines, squashes it a bit on the way.
    """
def argsnkwargs_to_str(args, kwargs): ...

class _ResolutionFailures:
    """Collect and format function resolution failures.
    """
    _context: Incomplete
    _function_type: Incomplete
    _args: Incomplete
    _kwargs: Incomplete
    _failures: Incomplete
    _depth: Incomplete
    _max_depth: int
    _scale: int
    def __init__(self, context, function_type, args, kwargs, depth: int = 0) -> None: ...
    def __len__(self) -> int: ...
    def add_error(self, calltemplate, matched, error, literal) -> None:
        """
        Args
        ----
        calltemplate : CallTemplate
        error : Exception or str
            Error message
        """
    def format(self):
        """Return a formatted error message from all the gathered errors.
        """
    def format_error(self, error):
        """Format error message or exception
        """
    def get_loc(self, classtemplate, error):
        """Get source location information from the error message.
        """
    def raise_error(self) -> None: ...

def _unlit_non_poison(ty):
    """Apply unliteral(ty) and raise a TypingError if type is Poison.
    """

class BaseFunction(Callable):
    """
    Base type class for some function types.
    """
    templates: Incomplete
    typing_key: Incomplete
    _impl_keys: Incomplete
    _depth: int
    def __init__(self, template) -> None: ...
    @property
    def key(self): ...
    def augment(self, other):
        """
        Augment this function type with the other function types' templates,
        so as to support more input types.
        """
    def get_impl_key(self, sig):
        """
        Get the implementation key (used by the target context) for the
        given signature.
        """
    def get_call_type(self, context, args, kws): ...
    def get_call_signatures(self): ...

class Function(BaseFunction, Opaque):
    """
    Type class for builtin functions implemented by Numba.
    """

class BoundFunction(Callable, Opaque):
    """
    A function with an implicit first argument (denoted as *this* below).
    """
    template: Incomplete
    typing_key: Incomplete
    this: Incomplete
    def __init__(self, template, this) -> None: ...
    def unify(self, typingctx, other): ...
    def copy(self, this): ...
    @property
    def key(self): ...
    def get_impl_key(self, sig):
        """
        Get the implementation key (used by the target context) for the
        given signature.
        """
    def get_call_type(self, context, args, kws): ...
    def get_call_signatures(self): ...

class MakeFunctionLiteral(Literal, Opaque): ...

class _PickleableWeakRef(weakref.ref):
    """
    Allow a weakref to be pickled.

    Note that if the object referred to is not kept alive elsewhere in the
    pickle, the weakref will immediately expire after being constructed.
    """
    def __getnewargs__(self): ...

class WeakType(Type):
    """
    Base class for types parametered by a mortal object, to which only
    a weak reference is kept.
    """
    _wr: Incomplete
    def _store_object(self, obj) -> None: ...
    def _get_object(self): ...
    @property
    def key(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class Dispatcher(WeakType, Callable, Dummy):
    """
    Type class for @jit-compiled functions.
    """
    def __init__(self, dispatcher) -> None: ...
    def dump(self, tab: str = '') -> None: ...
    def get_call_type(self, context, args, kws):
        """
        Resolve a call to this dispatcher using the given argument types.
        A signature returned and it is ensured that a compiled specialization
        is available for it.
        """
    def get_call_signatures(self): ...
    @property
    def dispatcher(self):
        """
        A strong reference to the underlying numba.dispatcher.Dispatcher
        instance.
        """
    def get_overload(self, sig):
        """
        Get the compiled overload for the given signature.
        """
    def get_impl_key(self, sig):
        """
        Get the implementation key for the given signature.
        """
    def unify(self, context, other): ...
    def can_convert_to(self, typingctx, other): ...

class ObjModeDispatcher(Dispatcher):
    """Dispatcher subclass that enters objectmode function.
    """

class ExternalFunctionPointer(BaseFunction):
    """
    A pointer to a native function (e.g. exported via ctypes or cffi).
    *get_pointer* is a Python function taking an object
    and returning the raw pointer value as an int.
    """
    sig: Incomplete
    requires_gil: Incomplete
    get_pointer: Incomplete
    cconv: Incomplete
    def __init__(self, sig, get_pointer, cconv: Incomplete | None = None) -> None: ...
    @property
    def key(self): ...

class ExternalFunction(Function):
    """
    A named native function (resolvable by LLVM) accepting an explicit
    signature. For internal use only.
    """
    symbol: Incomplete
    sig: Incomplete
    def __init__(self, symbol, sig) -> None: ...
    @property
    def key(self): ...

class NamedTupleClass(Callable, Opaque):
    """
    Type class for namedtuple classes.
    """
    instance_class: Incomplete
    def __init__(self, instance_class) -> None: ...
    def get_call_type(self, context, args, kws) -> None: ...
    def get_call_signatures(self): ...
    def get_impl_key(self, sig): ...
    @property
    def key(self): ...

class NumberClass(Callable, DTypeSpec, Opaque):
    '''
    Type class for number classes (e.g. "np.float64").
    '''
    instance_type: Incomplete
    def __init__(self, instance_type) -> None: ...
    def get_call_type(self, context, args, kws) -> None: ...
    def get_call_signatures(self): ...
    def get_impl_key(self, sig): ...
    @property
    def key(self): ...
    @property
    def dtype(self): ...

class _RecursiveCallOverloads(NamedTuple):
    qualname: Incomplete
    uid: Incomplete

class RecursiveCall(Opaque):
    """
    Recursive call to a Dispatcher.
    """
    _overloads: Incomplete
    dispatcher_type: Incomplete
    def __init__(self, dispatcher_type) -> None: ...
    def add_overloads(self, args, qualname, uid) -> None:
        """Add an overload of the function.

        Parameters
        ----------
        args :
            argument types
        qualname :
            function qualifying name
        uid :
            unique id
        """
    def get_overloads(self, args):
        """Get the qualifying name and unique id for the overload given the
        argument types.
        """
    @property
    def key(self): ...
