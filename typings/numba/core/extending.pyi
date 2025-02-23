from _typeshed import Incomplete
from numba._helperlib import _import_cython_function as _import_cython_function
from numba.core import config as config, errors as errors, types as types, utils as utils
from numba.core.datamodel import models as models
from numba.core.imputils import lower_builtin as lower_builtin, lower_cast as lower_cast, lower_getattr as lower_getattr, lower_getattr_generic as lower_getattr_generic, lower_setattr as lower_setattr, lower_setattr_generic as lower_setattr_generic
from numba.core.pythonapi import NativeValue as NativeValue, box as box, reflect as reflect, unbox as unbox
from numba.core.serialize import ReduceMixin as ReduceMixin
from numba.core.typing.asnumbatype import as_numba_type as as_numba_type
from numba.core.typing.templates import infer as infer, infer_getattr as infer_getattr
from numba.core.typing.typeof import typeof_impl as typeof_impl
from typing import NamedTuple

def type_callable(func):
    """
    Decorate a function as implementing typing for the callable *func*.
    *func* can be a callable object (probably a global) or a string
    denoting a built-in operation (such 'getitem' or '__array_wrap__')
    """

_overload_default_jit_options: Incomplete

def overload(func, jit_options={}, strict: bool = True, inline: str = 'never', prefer_literal: bool = False, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    *func* in nopython mode.

    The decorated function will have the same formal parameters as *func*
    and be passed the Numba types of those parameters.  It should return
    a function implementing *func* for the given types.

    Here is an example implementing len() for tuple types::

        @overload(len)
        def tuple_len(seq):
            if isinstance(seq, types.BaseTuple):
                n = len(seq)
                def len_impl(seq):
                    return n
                return len_impl

    Compiler options can be passed as an dictionary using the **jit_options**
    argument.

    Overloading strictness (that the typing and implementing signatures match)
    is enforced by the **strict** keyword argument, it is recommended that this
    is set to True (default).

    To handle a function that accepts imprecise types, an overload
    definition can return 2-tuple of ``(signature, impl_function)``, where
    the ``signature`` is a ``typing.Signature`` specifying the precise
    signature to be used; and ``impl_function`` is the same implementation
    function as in the simple case.

    If the kwarg inline determines whether the overload is inlined in the
    calling function and can be one of three values:
    * 'never' (default) - the overload is never inlined.
    * 'always' - the overload is always inlined.
    * a function that takes two arguments, both of which are instances of a
      namedtuple with fields:
        * func_ir
        * typemap
        * calltypes
        * signature
      The first argument holds the information from the caller, the second
      holds the information from the callee. The function should return Truthy
      to determine whether to inline, this essentially permitting custom
      inlining rules (typical use might be cost models).

    The *prefer_literal* option allows users to control if literal types should
    be tried first or last. The default (`False`) is to use non-literal types.
    Implementations that can specialize based on literal values should set the
    option to `True`. Note, this option maybe expanded in the near future to
    allow for more control (e.g. disabling non-literal types).

    **kwargs prescribes additional arguments passed through to the overload
    template. The only accepted key at present is 'target' which is a string
    corresponding to the target that this overload should be bound against.
    """
def register_jitable(*args, **kwargs):
    """
    Register a regular python function that can be executed by the python
    interpreter and can be compiled into a nopython function when referenced
    by other jit'ed functions.  Can be used as::

        @register_jitable
        def foo(x, y):
            return x + y

    Or, with compiler options::

        @register_jitable(_nrt=False) # disable runtime allocation
        def foo(x, y):
            return x + y

    """
def overload_attribute(typ, attr, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    attribute *attr* for the given Numba type in nopython mode.

    *kwargs* are passed to the underlying `@overload` call.

    Here is an example implementing .nbytes for array types::

        @overload_attribute(types.Array, 'nbytes')
        def array_nbytes(arr):
            def get(arr):
                return arr.size * arr.itemsize
            return get
    """
def _overload_method_common(typ, attr, **kwargs):
    """Common code for overload_method and overload_classmethod
    """
def overload_method(typ, attr, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    method *attr* for the given Numba type in nopython mode.

    *kwargs* are passed to the underlying `@overload` call.

    Here is an example implementing .take() for array types::

        @overload_method(types.Array, 'take')
        def array_take(arr, indices):
            if isinstance(indices, types.Array):
                def take_impl(arr, indices):
                    n = indices.shape[0]
                    res = np.empty(n, arr.dtype)
                    for i in range(n):
                        res[i] = arr[indices[i]]
                    return res
                return take_impl
    """
def overload_classmethod(typ, attr, **kwargs):
    '''
    A decorator marking the decorated function as typing and implementing
    classmethod *attr* for the given Numba type in nopython mode.


    Similar to ``overload_method``.


    Here is an example implementing a classmethod on the Array type to call
    ``np.arange()``::

        @overload_classmethod(types.Array, "make")
        def ov_make(cls, nitems):
            def impl(cls, nitems):
                return np.arange(nitems)
            return impl

    The above code will allow the following to work in jit-compiled code::

        @njit
        def foo(n):
            return types.Array.make(n)
    '''
def make_attribute_wrapper(typeclass, struct_attr, python_attr):
    """
    Make an automatic attribute wrapper exposing member named *struct_attr*
    as a read-only attribute named *python_attr*.
    The given *typeclass*'s model must be a StructModel subclass.
    """

class _Intrinsic(ReduceMixin):
    """
    Dummy callable for intrinsic
    """
    _memo: Incomplete
    _recent: Incomplete
    __uuid: Incomplete
    _ctor_kwargs: Incomplete
    _name: Incomplete
    _defn: Incomplete
    _prefer_literal: Incomplete
    def __init__(self, name, defn, prefer_literal: bool = False, **kwargs) -> None: ...
    @property
    def _uuid(self):
        """
        An instance-specific UUID, to avoid multiple deserializations of
        a given instance.

        Note this is lazily-generated, for performance reasons.
        """
    def _set_uuid(self, u) -> None: ...
    def _register(self) -> None: ...
    def __call__(self, *args, **kwargs) -> None:
        """
        This is only defined to pretend to be a callable from CPython.
        """
    def __repr__(self) -> str: ...
    def __deepcopy__(self, memo): ...
    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
    @classmethod
    def _rebuild(cls, uuid, name, defn):
        """
        NOTE: part of ReduceMixin protocol
        """

def intrinsic(*args, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    *func* in nopython mode using the llvmlite IRBuilder API.  This is an escape
    hatch for expert users to build custom LLVM IR that will be inlined to
    the caller.

    The first argument to *func* is the typing context.  The rest of the
    arguments corresponds to the type of arguments of the decorated function.
    These arguments are also used as the formal argument of the decorated
    function.  If *func* has the signature ``foo(typing_context, arg0, arg1)``,
    the decorated function will have the signature ``foo(arg0, arg1)``.

    The return values of *func* should be a 2-tuple of expected type signature,
    and a code-generation function that will passed to ``lower_builtin``.
    For unsupported operation, return None.

    Here is an example implementing a ``cast_int_to_byte_ptr`` that cast
    any integer to a byte pointer::

        @intrinsic
        def cast_int_to_byte_ptr(typingctx, src):
            # check for accepted types
            if isinstance(src, types.Integer):
                # create the expected type signature
                result_type = types.CPointer(types.uint8)
                sig = result_type(types.uintp)
                # defines the custom code generation
                def codegen(context, builder, signature, args):
                    # llvm IRBuilder code here
                    [src] = args
                    rtype = signature.return_type
                    llrtype = context.get_value_type(rtype)
                    return builder.inttoptr(src, llrtype)
                return sig, codegen
    """
def get_cython_function_address(module_name, function_name):
    """
    Get the address of a Cython function.

    Args
    ----
    module_name:
        Name of the Cython module
    function_name:
        Name of the Cython function

    Returns
    -------
    A Python int containing the address of the function

    """
def include_path():
    """Returns the C include directory path.
    """
def sentry_literal_args(pysig, literal_args, args, kwargs):
    """Ensures that the given argument types (in *args* and *kwargs*) are
    literally typed for a function with the python signature *pysig* and the
    list of literal argument names in *literal_args*.

    Alternatively, this is the same as::

        SentryLiteralArgs(literal_args).for_pysig(pysig).bind(*args, **kwargs)
    """

class SentryLiteralArgs(NamedTuple('_SentryLiteralArgs', [('literal_args', Incomplete)])):
    """
    Parameters
    ----------
    literal_args : Sequence[str]
        A sequence of names for literal arguments

    Examples
    --------

    The following line:

    >>> SentryLiteralArgs(literal_args).for_pysig(pysig).bind(*args, **kwargs)

    is equivalent to:

    >>> sentry_literal_args(pysig, literal_args, args, kwargs)
    """
    def for_function(self, func):
        """Bind the sentry to the signature of *func*.

        Parameters
        ----------
        func : Function
            A python function.

        Returns
        -------
        obj : BoundLiteralArgs
        """
    def for_pysig(self, pysig):
        """Bind the sentry to the given signature *pysig*.

        Parameters
        ----------
        pysig : inspect.Signature


        Returns
        -------
        obj : BoundLiteralArgs
        """

class BoundLiteralArgs(NamedTuple('BoundLiteralArgs', [('pysig', Incomplete), ('literal_args', Incomplete)])):
    """
    This class is usually created by SentryLiteralArgs.
    """
    def bind(self, *args, **kwargs):
        """Bind to argument types.
        """

def is_jitted(function):
    """Returns True if a function is wrapped by one of the Numba @jit
    decorators, for example: numba.jit, numba.njit

    The purpose of this function is to provide a means to check if a function is
    already JIT decorated.
    """
