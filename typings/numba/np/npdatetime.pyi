from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types
from numba.core.cgutils import create_constant_array as create_constant_array
from numba.core.config import IS_32BITS as IS_32BITS
from numba.core.errors import LoweringError as LoweringError
from numba.core.imputils import impl_ret_untracked as impl_ret_untracked, lower_builtin as lower_builtin, lower_cast as lower_cast, lower_constant as lower_constant
from numba.extending import overload_method as overload_method
from numba.np import npdatetime_helpers as npdatetime_helpers, npyfuncs as npyfuncs, numpy_support as numpy_support

DATETIME64: Incomplete
TIMEDELTA64: Incomplete
NAT: Incomplete
TIMEDELTA_BINOP_SIG: Incomplete

def scale_by_constant(builder, val, factor):
    """
    Multiply *val* by the constant *factor*.
    """
def unscale_by_constant(builder, val, factor):
    """
    Divide *val* by the constant *factor*.
    """
def add_constant(builder, val, const):
    """
    Add constant *const* to *val*.
    """
def scale_timedelta(context, builder, val, srcty, destty):
    """
    Scale the timedelta64 *val* from *srcty* to *destty*
    (both numba.types.NPTimedelta instances)
    """
def normalize_timedeltas(context, builder, left, right, leftty, rightty):
    """
    Scale either *left* or *right* to the other's unit, in order to have
    homogeneous units.
    """
def alloc_timedelta_result(builder, name: str = 'ret'):
    """
    Allocate a NaT-initialized datetime64 (or timedelta64) result slot.
    """
def alloc_boolean_result(builder, name: str = 'ret'):
    """
    Allocate an uninitialized boolean result slot.
    """
def is_not_nat(builder, val):
    """
    Return a predicate which is true if *val* is not NaT.
    """
def are_not_nat(builder, vals):
    """
    Return a predicate which is true if all of *vals* are not NaT.
    """

normal_year_months: Incomplete
leap_year_months: Incomplete
normal_year_months_acc: Incomplete
leap_year_months_acc: Incomplete

def datetime_constant(context, builder, ty, pyval): ...
def timedelta_pos_impl(context, builder, sig, args): ...
def timedelta_neg_impl(context, builder, sig, args): ...
def timedelta_abs_impl(context, builder, sig, args): ...
def timedelta_sign_impl(context, builder, sig, args):
    """
    np.sign(timedelta64)
    """
def timedelta_add_impl(context, builder, sig, args): ...
def timedelta_sub_impl(context, builder, sig, args): ...
def _timedelta_times_number(context, builder, td_arg, td_type, number_arg, number_type, return_type): ...
def timedelta_times_number(context, builder, sig, args): ...
def number_times_timedelta(context, builder, sig, args): ...
def timedelta_over_number(context, builder, sig, args): ...
def timedelta_over_timedelta(context, builder, sig, args): ...
def timedelta_floor_div_timedelta(context, builder, sig, args): ...
def timedelta_mod_timedelta(context, builder, sig, args): ...
def _create_timedelta_comparison_impl(ll_op, default_value): ...
def _create_timedelta_ordering_impl(ll_op): ...

timedelta_eq_timedelta_impl: Incomplete
timedelta_ne_timedelta_impl: Incomplete
timedelta_lt_timedelta_impl: Incomplete
timedelta_le_timedelta_impl: Incomplete
timedelta_gt_timedelta_impl: Incomplete
timedelta_ge_timedelta_impl: Incomplete

def is_leap_year(builder, year_val):
    """
    Return a predicate indicating whether *year_val* (offset by 1970) is a
    leap year.
    """
def year_to_days(builder, year_val):
    """
    Given a year *year_val* (offset to 1970), return the number of days
    since the 1970 epoch.
    """
def reduce_datetime_for_unit(builder, dt_val, src_unit, dest_unit): ...
def convert_datetime_for_arith(builder, dt_val, src_unit, dest_unit):
    """
    Convert datetime *dt_val* from *src_unit* to *dest_unit*.
    """
def _datetime_timedelta_arith(ll_op_name): ...

_datetime_plus_timedelta: Incomplete
_datetime_minus_timedelta: Incomplete

def datetime_plus_timedelta(context, builder, sig, args): ...
def timedelta_plus_datetime(context, builder, sig, args): ...
def datetime_minus_timedelta(context, builder, sig, args): ...
def datetime_minus_datetime(context, builder, sig, args): ...
def _create_datetime_comparison_impl(ll_op): ...

datetime_eq_datetime_impl: Incomplete
datetime_ne_datetime_impl: Incomplete
datetime_lt_datetime_impl: Incomplete
datetime_le_datetime_impl: Incomplete
datetime_gt_datetime_impl: Incomplete
datetime_ge_datetime_impl: Incomplete

def _gen_datetime_max_impl(NAT_DOMINATES): ...

datetime_maximum_impl: Incomplete
datetime_fmax_impl: Incomplete

def _gen_datetime_min_impl(NAT_DOMINATES): ...

datetime_minimum_impl: Incomplete
datetime_fmin_impl: Incomplete

def _gen_timedelta_max_impl(NAT_DOMINATES): ...

timedelta_maximum_impl: Incomplete
timedelta_fmax_impl: Incomplete

def _gen_timedelta_min_impl(NAT_DOMINATES): ...

timedelta_minimum_impl: Incomplete
timedelta_fmin_impl: Incomplete

def _cast_to_timedelta(context, builder, val): ...
def _np_isnat_impl(context, builder, sig, args): ...
def _cast_npdatetime_int64(context, builder, fromty, toty, val): ...
def ol_hash_npdatetime(x): ...
