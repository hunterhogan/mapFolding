from _typeshed import Incomplete
from numba.core import cgutils as cgutils, errors as errors, types as types
from numba.core.ccallback import CFunc as CFunc
from numba.core.dispatcher import Dispatcher as Dispatcher
from numba.core.imputils import lower_cast as lower_cast, lower_constant as lower_constant
from numba.core.types import FunctionPrototype as FunctionPrototype, FunctionType as FunctionType, UndefinedFunctionType as UndefinedFunctionType, WrapperAddressProtocol as WrapperAddressProtocol
from numba.extending import NativeValue as NativeValue, box as box, models as models, register_model as register_model, typeof_impl as typeof_impl, unbox as unbox

def typeof_function_type(val, c): ...

class FunctionProtoModel(models.PrimitiveModel):
    """FunctionProtoModel describes the signatures of first-class functions
    """
    def __init__(self, dmm, fe_type) -> None: ...

class FunctionModel(models.StructModel):
    """FunctionModel holds addresses of function implementations
    """
    def __init__(self, dmm, fe_type) -> None: ...

def lower_constant_dispatcher(context, builder, typ, pyval): ...
def lower_constant_function_type(context, builder, typ, pyval): ...
def _get_wrapper_address(func, sig):
    """Return the address of a compiled cfunc wrapper function of `func`.

    Warning: The compiled function must be compatible with the given
    signature `sig`. If it is not, then result of calling the compiled
    function is undefined. The compatibility is ensured when passing
    in a first-class function to a Numba njit compiled function either
    as an argument or via namespace scoping.

    Parameters
    ----------
    func : object
      A Numba cfunc or jit decoreated function or an object that
      implements the wrapper address protocol (see note below).
    sig : Signature
      The expected function signature.

    Returns
    -------
    addr : int
      An address in memory (pointer value) of the compiled function
      corresponding to the specified signature.

    Note: wrapper address protocol
    ------------------------------

    An object implements the wrapper address protocol iff the object
    provides a callable attribute named __wrapper_address__ that takes
    a Signature instance as the argument, and returns an integer
    representing the address or pointer value of a compiled function
    for the given signature.

    """
def _get_jit_address(func, sig):
    """Similar to ``_get_wrapper_address()`` but get the `.jit_addr` instead.
    """
def _lower_get_address(context, builder, func, sig, failure_mode, *, function_name):
    """Low-level call to <function_name>(func, sig).

    When calling this function, GIL must be acquired.
    """

lower_get_wrapper_address: Incomplete
lower_get_jit_address: Incomplete

def unbox_function_type(typ, obj, c): ...
def box_function_type(typ, val, c): ...
def lower_cast_function_type_to_function_type(context, builder, fromty, toty, val): ...
def lower_cast_dispatcher_to_function_type(context, builder, fromty, toty, val): ...
