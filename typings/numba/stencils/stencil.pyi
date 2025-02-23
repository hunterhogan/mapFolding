from _typeshed import Incomplete
from numba.core import config as config, ir as ir, ir_utils as ir_utils, registry as registry, types as types, typing as typing, utils as utils
from numba.core.errors import NumbaValueError as NumbaValueError
from numba.core.extending import register_jitable as register_jitable
from numba.core.imputils import lower_builtin as lower_builtin
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, CallableTemplate as CallableTemplate, infer_global as infer_global, signature as signature
from numba.misc.special import literal_unroll as literal_unroll
from numba.np import numpy_support as numpy_support

class StencilFuncLowerer:
    """Callable class responsible for lowering calls to a specific StencilFunc.
    """
    stencilFunc: Incomplete
    def __init__(self, sf) -> None: ...
    def __call__(self, context, builder, sig, args): ...

def raise_if_incompatible_array_sizes(a, *args) -> None: ...
def slice_addition(the_slice, addend):
    """ Called by stencil in Python mode to add the loop index to a
        user-specified slice.
    """

class StencilFunc:
    """
    A special type to hold stencil information for the IR.
    """
    id_counter: int
    id: Incomplete
    kernel_ir: Incomplete
    mode: Incomplete
    options: Incomplete
    kws: Incomplete
    _typingctx: Incomplete
    _targetctx: Incomplete
    neighborhood: Incomplete
    _type_cache: Incomplete
    _lower_me: Incomplete
    def __init__(self, kernel_ir, mode, options) -> None: ...
    def replace_return_with_setitem(self, blocks, index_vars, out_name):
        '''
        Find return statements in the IR and replace them with a SetItem
        call of the value "returned" by the kernel into the result array.
        Returns the block labels that contained return statements.
        '''
    def add_indices_to_kernel(self, kernel, index_names, ndim, neighborhood, standard_indexed, typemap, calltypes):
        """
        Transforms the stencil kernel as specified by the user into one
        that includes each dimension's index variable as part of the getitem
        calls.  So, in effect array[-1] becomes array[index0-1].
        """
    def get_return_type(self, argtys): ...
    def _install_type(self, typingctx) -> None:
        """Constructs and installs a typing class for a StencilFunc object in
        the input typing context.
        """
    def compile_for_argtys(self, argtys, kwtys, return_type, sigret): ...
    def _type_me(self, argtys, kwtys):
        """
        Implement AbstractTemplate.generic() for the typing class
        built by StencilFunc._install_type().
        Return the call-site signature.
        """
    def copy_ir_with_calltypes(self, ir, calltypes):
        """
        Create a copy of a given IR along with its calltype information.
        We need a copy of the calltypes because copy propagation applied
        to the copied IR will change the calltypes and make subsequent
        uses of the original IR invalid.
        """
    def _stencil_wrapper(self, result, sigret, return_type, typemap, calltypes, *args): ...
    def __call__(self, *args, **kwargs): ...

def stencil(func_or_mode: str = 'constant', **options): ...
def _stencil(mode, options): ...
def stencil_dummy_lower(context, builder, sig, args):
    """lowering for dummy stencil calls"""
