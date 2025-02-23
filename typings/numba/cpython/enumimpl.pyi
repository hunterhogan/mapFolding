from numba.core import types as types
from numba.core.extending import overload_method as overload_method
from numba.core.imputils import impl_ret_untracked as impl_ret_untracked, lower_builtin as lower_builtin, lower_cast as lower_cast, lower_constant as lower_constant, lower_getattr as lower_getattr, lower_getattr_generic as lower_getattr_generic

def enum_eq(context, builder, sig, args): ...
def enum_is(context, builder, sig, args): ...
def enum_ne(context, builder, sig, args): ...
def enum_value(context, builder, ty, val): ...
def int_enum_to_int(context, builder, fromty, toty, val):
    """
    Convert an IntEnum member to its raw integer value.
    """
def enum_constant(context, builder, ty, pyval):
    """
    Return a LLVM constant representing enum member *pyval*.
    """
def enum_class_getattr(context, builder, ty, val, attr):
    """
    Return an enum member by attribute name.
    """
def enum_class_getitem(context, builder, sig, args):
    """
    Return an enum member by index name.
    """
def intenum_hash(val): ...
