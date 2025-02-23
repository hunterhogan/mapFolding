from numba.core import cgutils as cgutils, types as types, typing as typing
from numba.core.imputils import impl_ret_untracked as impl_ret_untracked, lower_builtin as lower_builtin, lower_cast as lower_cast, lower_getattr_generic as lower_getattr_generic, lower_setattr_generic as lower_setattr_generic

def always_return_true_impl(context, builder, sig, args): ...
def always_return_false_impl(context, builder, sig, args): ...
def optional_is_none(context, builder, sig, args):
    """
    Check if an Optional value is invalid
    """
def optional_getattr(context, builder, typ, value, attr):
    """
    Optional.__getattr__ => redirect to the wrapped type.
    """
def optional_setattr(context, builder, sig, args, attr):
    """
    Optional.__setattr__ => redirect to the wrapped type.
    """
def optional_to_optional(context, builder, fromty, toty, val):
    """
    The handling of optional->optional cast must be special cased for
    correct propagation of None value.  Given type T and U. casting of
    T? to U? (? denotes optional) should always succeed.   If the from-value
    is None, the None value the casted value (U?) should be None; otherwise,
    the from-value is casted to U. This is different from casting T? to U,
    which requires the from-value must not be None.
    """
def any_to_optional(context, builder, fromty, toty, val): ...
def optional_to_any(context, builder, fromty, toty, val): ...
