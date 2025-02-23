from _typeshed import Incomplete
from collections.abc import Generator
from enum import Enum
from numba.core import cgutils as cgutils, types as types, typing as typing, utils as utils
from numba.core.typing.templates import BaseRegistryLoader as BaseRegistryLoader
from typing import NamedTuple

class Registry:
    """
    A registry of function and attribute implementations.
    """
    name: Incomplete
    functions: Incomplete
    getattrs: Incomplete
    setattrs: Incomplete
    casts: Incomplete
    constants: Incomplete
    def __init__(self, name: str = 'unspecified') -> None: ...
    def lower(self, func, *argtys):
        '''
        Decorate an implementation of *func* for the given argument types.
        *func* may be an actual global function object, or any
        pseudo-function supported by Numba, such as "getitem".

        The decorated implementation has the signature
        (context, builder, sig, args).
        '''
    def _decorate_attr(self, impl, ty, attr, impl_list, decorator): ...
    def lower_getattr(self, ty, attr):
        """
        Decorate an implementation of __getattr__ for type *ty* and
        the attribute *attr*.

        The decorated implementation will have the signature
        (context, builder, typ, val).
        """
    def lower_getattr_generic(self, ty):
        """
        Decorate the fallback implementation of __getattr__ for type *ty*.

        The decorated implementation will have the signature
        (context, builder, typ, val, attr).  The implementation is
        called for attributes which haven't been explicitly registered
        with lower_getattr().
        """
    def lower_setattr(self, ty, attr):
        """
        Decorate an implementation of __setattr__ for type *ty* and
        the attribute *attr*.

        The decorated implementation will have the signature
        (context, builder, sig, args).
        """
    def lower_setattr_generic(self, ty):
        """
        Decorate the fallback implementation of __setattr__ for type *ty*.

        The decorated implementation will have the signature
        (context, builder, sig, args, attr).  The implementation is
        called for attributes which haven't been explicitly registered
        with lower_setattr().
        """
    def lower_cast(self, fromty, toty):
        """
        Decorate the implementation of implicit conversion between
        *fromty* and *toty*.

        The decorated implementation will have the signature
        (context, builder, fromty, toty, val).
        """
    def lower_constant(self, ty):
        """
        Decorate the implementation for creating a constant of type *ty*.

        The decorated implementation will have the signature
        (context, builder, ty, pyval).
        """
    def __repr__(self) -> str: ...

class RegistryLoader(BaseRegistryLoader):
    """
    An incremental loader for a target registry.
    """
    registry_items: Incomplete

builtin_registry: Incomplete
lower_builtin: Incomplete
lower_getattr: Incomplete
lower_getattr_generic: Incomplete
lower_setattr: Incomplete
lower_setattr_generic: Incomplete
lower_cast: Incomplete
lower_constant: Incomplete

def _decorate_getattr(impl, ty, attr): ...
def _decorate_setattr(impl, ty, attr): ...
def fix_returning_optional(context, builder, sig, status, retval): ...
def user_function(fndesc, libs):
    """
    A wrapper inserting code calling Numba-compiled *fndesc*.
    """
def user_generator(gendesc, libs):
    """
    A wrapper inserting code calling Numba-compiled *gendesc*.
    """
def iterator_impl(iterable_type, iterator_type):
    """
    Decorator a given class as implementing *iterator_type*
    (by providing an `iternext()` method).
    """

class _IternextResult:
    """
    A result wrapper for iteration, passed by iternext_impl() into the
    wrapped function.
    """
    __slots__: Incomplete
    _context: Incomplete
    _builder: Incomplete
    _pairobj: Incomplete
    def __init__(self, context, builder, pairobj) -> None: ...
    def set_exhausted(self) -> None:
        """
        Mark the iterator as exhausted.
        """
    def set_valid(self, is_valid: bool = True) -> None:
        """
        Mark the iterator as valid according to *is_valid* (which must
        be either a Python boolean or a LLVM inst).
        """
    def yield_(self, value) -> None:
        """
        Mark the iterator as yielding the given *value* (a LLVM inst).
        """
    def is_valid(self):
        """
        Return whether the iterator is marked valid.
        """
    def yielded_value(self):
        """
        Return the iterator's yielded value, if any.
        """

class RefType(Enum):
    """
    Enumerate the reference type
    """
    NEW = 1
    BORROWED = 2
    UNTRACKED = 3

def iternext_impl(ref_type: Incomplete | None = None):
    """
    Wrap the given iternext() implementation so that it gets passed
    an _IternextResult() object easing the returning of the iternext()
    result pair.

    ref_type: a numba.targets.imputils.RefType value, the reference type used is
    that specified through the RefType enum.

    The wrapped function will be called with the following signature:
        (context, builder, sig, args, iternext_result)
    """
def call_getiter(context, builder, iterable_type, val):
    """
    Call the `getiter()` implementation for the given *iterable_type*
    of value *val*, and return the corresponding LLVM inst.
    """
def call_iternext(context, builder, iterator_type, val):
    """
    Call the `iternext()` implementation for the given *iterator_type*
    of value *val*, and return a convenience _IternextResult() object
    reflecting the results.
    """
def call_len(context, builder, ty, val):
    """
    Call len() on the given value.  Return None if len() isn't defined on
    this type.
    """

class _ForIterLoop(NamedTuple):
    value: Incomplete
    do_break: Incomplete

def for_iter(context, builder, iterable_type, val) -> Generator[Incomplete]:
    """
    Simulate a for loop on the given iterable.  Yields a namedtuple with
    the given members:
    - `value` is the value being yielded
    - `do_break` is a callable to early out of the loop
    """
def impl_ret_new_ref(ctx, builder, retty, ret):
    """
    The implementation returns a new reference.
    """
def impl_ret_borrowed(ctx, builder, retty, ret):
    """
    The implementation returns a borrowed reference.
    This function automatically incref so that the implementation is
    returning a new reference.
    """
def impl_ret_untracked(ctx, builder, retty, ret):
    """
    The return type is not a NRT object.
    """
def force_error_model(context, model_name: str = 'numpy') -> Generator[None]:
    """
    Temporarily change the context's error model.
    """
def numba_typeref_ctor(*args, **kwargs) -> None:
    """A stub for use internally by Numba when a call is emitted
    on a TypeRef.
    """
