from .arrayobj import _empty_nd_impl as _empty_nd_impl, array_copy as array_copy, make_array as make_array
from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import cgutils as cgutils, config as config, types as types
from numba.core.errors import NumbaPerformanceWarning as NumbaPerformanceWarning, NumbaTypeError as NumbaTypeError, TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic, overload as overload, register_jitable as register_jitable
from numba.core.imputils import impl_ret_borrowed as impl_ret_borrowed, impl_ret_new_ref as impl_ret_new_ref, impl_ret_untracked as impl_ret_untracked, lower_builtin as lower_builtin
from numba.core.typing import signature as signature

ll_char: Incomplete
ll_char_p: Incomplete
ll_void_p = ll_char_p
ll_intc: Incomplete
ll_intc_p: Incomplete
intp_t: Incomplete
ll_intp_p: Incomplete
F_INT_nptype: Incomplete
F_INT_nbtype: Incomplete
_blas_kinds: Incomplete

def get_blas_kind(dtype, func_name: str = '<BLAS function>'): ...
def ensure_blas() -> None: ...
def ensure_lapack() -> None: ...
def make_constant_slot(context, builder, ty, val): ...

class _BLAS:
    """
    Functions to return type signatures for wrapped
    BLAS functions.
    """
    def __init__(self) -> None: ...
    @classmethod
    def numba_xxnrm2(cls, dtype): ...
    @classmethod
    def numba_xxgemm(cls, dtype): ...

class _LAPACK:
    """
    Functions to return type signatures for wrapped
    LAPACK functions.
    """
    def __init__(self) -> None: ...
    @classmethod
    def numba_xxgetrf(cls, dtype): ...
    @classmethod
    def numba_ez_xxgetri(cls, dtype): ...
    @classmethod
    def numba_ez_rgeev(cls, dtype): ...
    @classmethod
    def numba_ez_cgeev(cls, dtype): ...
    @classmethod
    def numba_ez_xxxevd(cls, dtype): ...
    @classmethod
    def numba_xxpotrf(cls, dtype): ...
    @classmethod
    def numba_ez_gesdd(cls, dtype): ...
    @classmethod
    def numba_ez_geqrf(cls, dtype): ...
    @classmethod
    def numba_ez_xxgqr(cls, dtype): ...
    @classmethod
    def numba_ez_gelsd(cls, dtype): ...
    @classmethod
    def numba_xgesv(cls, dtype): ...

def make_contiguous(context, builder, sig, args) -> Generator[Incomplete]:
    """
    Ensure that all array arguments are contiguous, if necessary by
    copying them.
    A new (sig, args) tuple is yielded.
    """
def check_c_int(context, builder, n) -> None:
    """
    Check whether *n* fits in a C `int`.
    """
def check_blas_return(context, builder, res) -> None:
    """
    Check the integer error return from one of the BLAS wrappers in
    _helperlib.c.
    """
def check_lapack_return(context, builder, res) -> None:
    """
    Check the integer error return from one of the LAPACK wrappers in
    _helperlib.c.
    """
def call_xxdot(context, builder, conjugate, dtype, n, a_data, b_data, out_data) -> None:
    """
    Call the BLAS vector * vector product function for the given arguments.
    """
def call_xxgemv(context, builder, do_trans, m_type, m_shapes, m_data, v_data, out_data) -> None:
    """
    Call the BLAS matrix * vector product function for the given arguments.
    """
def call_xxgemm(context, builder, x_type, x_shapes, x_data, y_type, y_shapes, y_data, out_type, out_shapes, out_data):
    """
    Call the BLAS matrix * matrix product function for the given arguments.
    """
def dot_2_mm(context, builder, sig, args):
    """
    np.dot(matrix, matrix)
    """
def dot_2_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix)
    """
def dot_2_mv(context, builder, sig, args):
    """
    np.dot(matrix, vector)
    """
def dot_2_vv(context, builder, sig, args, conjugate: bool = False):
    """
    np.dot(vector, vector)
    np.vdot(vector, vector)
    """
def dot_2(left, right):
    """
    np.dot(a, b)
    """
def matmul_2(left, right):
    """
    a @ b
    """
def dot_2_impl(name, left, right): ...
def vdot(left, right):
    """
    np.vdot(a, b)
    """
def dot_3_vm_check_args(a, b, out) -> None: ...
def dot_3_mv_check_args(a, b, out) -> None: ...
def dot_3_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix, out)
    np.dot(matrix, vector, out)
    """
def dot_3_mm(context, builder, sig, args):
    """
    np.dot(matrix, matrix, out)
    """
def dot_3(left, right, out):
    """
    np.dot(a, b, out)
    """

fatal_error_func: Incomplete

def _check_finite_matrix(a) -> None: ...
def _check_linalg_matrix(a, func_name, la_prefix: bool = True) -> None: ...
def _check_homogeneous_types(func_name, *types) -> None: ...
def _copy_to_fortran_order() -> None: ...
def ol_copy_to_fortran_order(a): ...
def _inv_err_handler(r) -> None: ...
def _dummy_liveness_func(a):
    """pass a list of variables to be preserved through dead code elimination"""
def inv_impl(a): ...
def _handle_err_maybe_convergence_problem(r) -> None: ...
def _check_linalg_1_or_2d_matrix(a, func_name, la_prefix: bool = True) -> None: ...
def cho_impl(a): ...
def eig_impl(a): ...
def eigvals_impl(a): ...
def eigh_impl(a): ...
def eigvalsh_impl(a): ...
def svd_impl(a, full_matrices: int = 1): ...
def qr_impl(a): ...
def _system_copy_in_b(bcpy, b, nrhs) -> None:
    """
    Correctly copy 'b' into the 'bcpy' scratch space.
    """
def _system_copy_in_b_impl(bcpy, b, nrhs): ...
def _system_compute_nrhs(b) -> None:
    """
    Compute the number of right hand sides in the system of equations
    """
def _system_compute_nrhs_impl(b): ...
def _system_check_dimensionally_valid(a, b) -> None:
    """
    Check that AX=B style system input is dimensionally valid.
    """
def _system_check_dimensionally_valid_impl(a, b): ...
def _system_check_non_empty(a, b) -> None:
    """
    Check that AX=B style system input is not empty.
    """
def _system_check_non_empty_impl(a, b): ...
def _lstsq_residual(b, n, nrhs) -> None:
    """
    Compute the residual from the 'b' scratch space.
    """
def _lstsq_residual_impl(b, n, nrhs): ...
def _lstsq_solution(b, bcpy, n) -> None:
    """
    Extract 'x' (the lstsq solution) from the 'bcpy' scratch space.
    Note 'b' is only used to check the system input dimension...
    """
def _lstsq_solution_impl(b, bcpy, n): ...
def lstsq_impl(a, b, rcond: float = -1.0): ...
def _solve_compute_return(b, bcpy) -> None:
    """
    Extract 'x' (the solution) from the 'bcpy' scratch space.
    Note 'b' is only used to check the system input dimension...
    """
def _solve_compute_return_impl(b, bcpy): ...
def solve_impl(a, b): ...
def pinv_impl(a, rcond: float = 1e-15): ...
def _get_slogdet_diag_walker(a):
    """
    Walks the diag of a LUP decomposed matrix
    uses that det(A) = prod(diag(lup(A)))
    and also that log(a)+log(b) = log(a*b)
    The return sign is adjusted based on the values found
    such that the log(value) stays in the real domain.
    """
def slogdet_impl(a): ...
def det_impl(a): ...
def _compute_singular_values(a) -> None:
    """
    Compute singular values of *a*.
    """
def _compute_singular_values_impl(a):
    """
    Returns a function to compute singular values of `a`
    """
def _oneD_norm_2(a) -> None:
    """
    Compute the L2-norm of 1D-array *a*.
    """
def _oneD_norm_2_impl(a): ...
def _get_norm_impl(x, ord_flag): ...
def norm_impl(x, ord: Incomplete | None = None): ...
def cond_impl(x, p: Incomplete | None = None): ...
def _get_rank_from_singular_values(sv, t):
    """
    Gets rank from singular values with cut-off at a given tolerance
    """
def matrix_rank_impl(A, tol: Incomplete | None = None):
    """
    Computes rank for matrices and vectors.
    The only issue that may arise is that because numpy uses double
    precision lapack calls whereas numba uses type specific lapack
    calls, some singular values may differ and therefore counting the
    number of them above a tolerance may lead to different counts,
    and therefore rank, in some cases.
    """
def matrix_power_impl(a, n):
    """
    Computes matrix power. Only integer powers are supported in numpy.
    """
def matrix_trace_impl(a, offset: int = 0):
    """
    Computes the trace of an array.
    """
def _check_scalar_or_lt_2d_mat(a, func_name, la_prefix: bool = True) -> None: ...
def outer_impl_none(a, b, out): ...
def outer_impl_arr(a, b, out): ...
def _get_outer_impl(a, b, out): ...
def outer_impl(a, b, out: Incomplete | None = None): ...
def _kron_normaliser_impl(x): ...
def _kron_return(a, b): ...
def kron_impl(a, b): ...
