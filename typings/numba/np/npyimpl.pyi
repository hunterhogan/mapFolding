from _typeshed import Incomplete
from numba.core import callconv as callconv, cgutils as cgutils, config as config, errors as errors, types as types, typing as typing, utils as utils
from numba.core.extending import intrinsic as intrinsic, overload as overload
from numba.core.imputils import Registry as Registry, force_error_model as force_error_model, impl_ret_borrowed as impl_ret_borrowed, impl_ret_new_ref as impl_ret_new_ref
from numba.core.typing import npydecl as npydecl
from numba.np import arrayobj as arrayobj, numpy_support as numpy_support, ufunc_db as ufunc_db
from numba.np.arrayobj import _getitem_array_generic as _getitem_array_generic
from numba.np.numpy_support import _ufunc_loop_sig as _ufunc_loop_sig, from_dtype as from_dtype, select_array_wrapper as select_array_wrapper, ufunc_find_matching_loop as ufunc_find_matching_loop
from numba.np.ufunc.sigparse import parse_signature as parse_signature
from typing import NamedTuple

registry: Incomplete

class _ScalarIndexingHelper:
    def update_indices(self, loop_indices, name) -> None: ...
    def as_values(self) -> None: ...

class _ScalarHelper:
    '''Helper class to handle scalar arguments (and result).
    Note that store_data is only used when generating code for
    a scalar ufunc and to write the output value.

    For loading, the value is directly used without having any
    kind of indexing nor memory backing it up. This is the use
    for input arguments.

    For storing, a variable is created in the stack where the
    value will be written.

    Note that it is not supported (as it is unneeded for our
    current use-cases) reading back a stored value. This class
    will always "load" the original value it got at its creation.
    '''
    context: Incomplete
    builder: Incomplete
    val: Incomplete
    base_type: Incomplete
    shape: Incomplete
    _ptr: Incomplete
    def __init__(self, ctxt, bld, val, ty) -> None: ...
    def create_iter_indices(self): ...
    def load_data(self, indices): ...
    def store_data(self, indices, val) -> None: ...
    @property
    def return_val(self): ...

class _ArrayIndexingHelper(NamedTuple('_ArrayIndexingHelper', [('array', Incomplete), ('indices', Incomplete)])):
    def update_indices(self, loop_indices, name) -> None: ...
    def as_values(self):
        """
        The indexing helper is built using alloca for each value, so it
        actually contains pointers to the actual indices to load. Note
        that update_indices assumes the same. This method returns the
        indices as values
        """

class _ArrayHelper(NamedTuple('_ArrayHelper', [('context', Incomplete), ('builder', Incomplete), ('shape', Incomplete), ('strides', Incomplete), ('data', Incomplete), ('layout', Incomplete), ('base_type', Incomplete), ('ndim', Incomplete), ('return_val', Incomplete)])):
    """Helper class to handle array arguments/result.
    It provides methods to generate code loading/storing specific
    items as well as support code for handling indices.
    """
    def create_iter_indices(self): ...
    def _load_effective_address(self, indices): ...
    def load_data(self, indices): ...
    def store_data(self, indices, value) -> None: ...

class _ArrayGUHelper(NamedTuple('_ArrayHelper', [('context', Incomplete), ('builder', Incomplete), ('shape', Incomplete), ('strides', Incomplete), ('data', Incomplete), ('layout', Incomplete), ('base_type', Incomplete), ('ndim', Incomplete), ('inner_arr_ty', Incomplete), ('is_input_arg', Incomplete)])):
    """Helper class to handle array arguments/result.
    It provides methods to generate code loading/storing specific
    items as well as support code for handling indices.

    Contrary to _ArrayHelper, this class can create a view to a subarray
    """
    def create_iter_indices(self): ...
    def _load_effective_address(self, indices): ...
    def load_data(self, indices): ...
    def guard_shape(self, loopshape) -> None: ...
    def guard_match_core_dims(self, other: _ArrayGUHelper, ndims: int): ...

def _prepare_argument(ctxt, bld, inp, tyinp, where: str = 'input operand'):
    """returns an instance of the appropriate Helper (either
    _ScalarHelper or _ArrayHelper) class to handle the argument.
    using the polymorphic interface of the Helper classes, scalar
    and array cases can be handled with the same code"""

_broadcast_onto_sig: Incomplete

def _broadcast_onto(src_ndim, src_shape, dest_ndim, dest_shape):
    """Low-level utility function used in calculating a shape for
    an implicit output array.  This function assumes that the
    destination shape is an LLVM pointer to a C-style array that was
    already initialized to a size of one along all axes.

    Returns an integer value:
    >= 1  :  Succeeded.  Return value should equal the number of dimensions in
             the destination shape.
    0     :  Failed to broadcast because source shape is larger than the
             destination shape (this case should be weeded out at type
             checking).
    < 0   :  Failed to broadcast onto destination axis, at axis number ==
             -(return_value + 1).
    """
def _build_array(context, builder, array_ty, input_types, inputs):
    """Utility function to handle allocation of an implicit output array
    given the target context, builder, output array type, and a list of
    _ArrayHelper instances.
    """
def _unpack_output_types(ufunc, sig): ...
def _unpack_output_values(ufunc, builder, values): ...
def _pack_output_values(ufunc, context, builder, typ, values): ...
def numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel_class): ...
def numpy_gufunc_kernel(context, builder, sig, args, ufunc, kernel_class) -> None: ...

class _Kernel:
    context: Incomplete
    builder: Incomplete
    outer_sig: Incomplete
    def __init__(self, context, builder, outer_sig) -> None: ...
    def cast(self, val, fromty, toty):
        """Numpy uses cast semantics that are different from standard Python
        (for example, it does allow casting from complex to float).

        This method acts as a patch to context.cast so that it allows
        complex to real/int casts.

        """
    def generate(self, *args): ...

def _ufunc_db_function(ufunc):
    """Use the ufunc loop type information to select the code generation
    function from the table provided by the dict_of_kernels. The dict
    of kernels maps the loop identifier to a function with the
    following signature: (context, builder, signature, args).

    The loop type information has the form 'AB->C'. The letters to the
    left of '->' are the input types (specified as NumPy letter
    types).  The letters to the right of '->' are the output
    types. There must be 'ufunc.nin' letters to the left of '->', and
    'ufunc.nout' letters to the right.

    For example, a binary float loop resulting in a float, will have
    the following signature: 'ff->f'.

    A given ufunc implements many loops. The list of loops implemented
    for a given ufunc can be accessed using the 'types' attribute in
    the ufunc object. The NumPy machinery selects the first loop that
    fits a given calling signature (in our case, what we call the
    outer_sig). This logic is mimicked by 'ufunc_find_matching_loop'.
    """
def register_ufunc_kernel(ufunc, kernel, lower): ...
def register_unary_operator_kernel(operator, ufunc, kernel, lower, inplace: bool = False): ...
def register_binary_operator_kernel(op, ufunc, kernel, lower, inplace: bool = False): ...
def array_positive_impl(context, builder, sig, args):
    """Lowering function for +(array) expressions.  Defined here
    (numba.targets.npyimpl) since the remaining array-operator
    lowering functions are also registered in this module.
    """
def register_ufuncs(ufuncs, lower) -> None: ...
def _make_dtype_object(typingctx, desc):
    """Given a string or NumberClass description *desc*, returns the dtype object.
    """
def numpy_dtype(desc):
    """Provide an implementation so that numpy.dtype function can be lowered.
    """
