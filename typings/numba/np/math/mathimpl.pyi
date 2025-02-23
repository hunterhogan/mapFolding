from _typeshed import Incomplete
from numba.core import cgutils as cgutils, config as config, types as types
from numba.core.extending import overload as overload
from numba.core.imputils import impl_ret_untracked as impl_ret_untracked
from numba.core.typing import signature as signature
from numba.cpython.unsafe.numbers import trailing_zeros as trailing_zeros

_NP_FLT_FINFO: Incomplete
FLT_MAX: Incomplete
FLT_MIN: Incomplete
_NP_DBL_FINFO: Incomplete
DBL_MAX: Incomplete
DBL_MIN: Incomplete
FLOAT_ABS_MASK: int
FLOAT_SIGN_MASK: int
DOUBLE_ABS_MASK: int
DOUBLE_SIGN_MASK: int

def is_nan(builder, val):
    """
    Return a condition testing whether *val* is a NaN.
    """
def is_inf(builder, val):
    """
    Return a condition testing whether *val* is an infinite.
    """
def is_finite(builder, val):
    """
    Return a condition testing whether *val* is a finite.
    """
def f64_as_int64(builder, val):
    """
    Bitcast a double into a 64-bit integer.
    """
def int64_as_f64(builder, val):
    """
    Bitcast a 64-bit integer into a double.
    """
def f32_as_int32(builder, val):
    """
    Bitcast a float into a 32-bit integer.
    """
def int32_as_f32(builder, val):
    """
    Bitcast a 32-bit integer into a float.
    """
def negate_real(builder, val):
    """
    Negate real number *val*, with proper handling of zeros.
    """
def call_fp_intrinsic(builder, name, args):
    """
    Call a LLVM intrinsic floating-point operation.
    """
def _unary_int_input_wrapper_impl(wrapped_impl):
    """
    Return an implementation factory to convert the single integral input
    argument to a float64, then defer to the *wrapped_impl*.
    """
def unary_math_int_impl(fn, float_impl) -> None: ...
def unary_math_intr(fn, intrcode):
    """
    Implement the math function *fn* using the LLVM intrinsic *intrcode*.
    """
def unary_math_extern(fn, f32extern, f64extern, int_restype: bool = False):
    """
    Register implementations of Python function *fn* using the
    external function named *f32extern* and *f64extern* (for float32
    and float64 inputs, respectively).
    If *int_restype* is true, then the function's return value should be
    integral, otherwise floating-point.
    """

exp_impl: Incomplete
log_impl: Incomplete
log10_impl: Incomplete
sin_impl: Incomplete
cos_impl: Incomplete
log1p_impl: Incomplete
expm1_impl: Incomplete
erf_impl: Incomplete
erfc_impl: Incomplete
tan_impl: Incomplete
asin_impl: Incomplete
acos_impl: Incomplete
atan_impl: Incomplete
asinh_impl: Incomplete
acosh_impl: Incomplete
atanh_impl: Incomplete
sinh_impl: Incomplete
cosh_impl: Incomplete
tanh_impl: Incomplete
log2_impl: Incomplete
ceil_impl: Incomplete
floor_impl: Incomplete
gamma_impl: Incomplete
sqrt_impl: Incomplete
trunc_impl: Incomplete
lgamma_impl: Incomplete

def isnan_float_impl(context, builder, sig, args): ...
def isnan_int_impl(context, builder, sig, args): ...
def isinf_float_impl(context, builder, sig, args): ...
def isinf_int_impl(context, builder, sig, args): ...
def isfinite_float_impl(context, builder, sig, args): ...
def isfinite_int_impl(context, builder, sig, args): ...
def copysign_float_impl(context, builder, sig, args): ...
def frexp_impl(context, builder, sig, args): ...
def ldexp_impl(context, builder, sig, args): ...
def atan2_s64_impl(context, builder, sig, args): ...
def atan2_u64_impl(context, builder, sig, args): ...
def atan2_float_impl(context, builder, sig, args): ...
def hypot_s64_impl(context, builder, sig, args): ...
def hypot_u64_impl(context, builder, sig, args): ...
def hypot_float_impl(context, builder, sig, args): ...
def radians_float_impl(context, builder, sig, args): ...
def degrees_float_impl(context, builder, sig, args): ...
def pow_impl(context, builder, sig, args): ...
def _unsigned(T) -> None:
    """Convert integer to unsigned integer of equivalent width."""
def _unsigned_impl(T): ...
def gcd_impl(context, builder, sig, args): ...
