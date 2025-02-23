from _typeshed import Incomplete
from numba.core import config as config, ir as ir, ir_utils as ir_utils, rewrites as rewrites, types as types, typing as typing, utils as utils
from numba.core.errors import NumbaValueError as NumbaValueError
from numba.core.ir_utils import GuardException as GuardException, compile_to_numba_ir as compile_to_numba_ir, find_callname as find_callname, find_const as find_const, get_call_table as get_call_table, guard as guard, mk_unique_var as mk_unique_var, replace_arg_nodes as replace_arg_nodes, require as require
from numba.core.typing import signature as signature
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, infer_global as infer_global
from numba.core.utils import OPERATORS_TO_BUILTINS as OPERATORS_TO_BUILTINS
from numba.np import numpy_support as numpy_support

def _compute_last_ind(dim_size, index_const): ...

class StencilPass:
    func_ir: Incomplete
    typemap: Incomplete
    calltypes: Incomplete
    array_analysis: Incomplete
    typingctx: Incomplete
    targetctx: Incomplete
    flags: Incomplete
    def __init__(self, func_ir, typemap, calltypes, array_analysis, typingctx, targetctx, flags) -> None: ...
    def run(self) -> None:
        """ Finds all calls to StencilFuncs in the IR and converts them to parfor.
        """
    def replace_return_with_setitem(self, blocks, exit_value_var, parfor_body_exit_label) -> None:
        '''
        Find return statements in the IR and replace them with a SetItem
        call of the value "returned" by the kernel into the result array.
        Returns the block labels that contained return statements.
        '''
    def _mk_stencil_parfor(self, label, in_args, out_arr, stencil_ir, index_offsets, target, return_type, stencil_func, arg_to_arr_dict):
        """ Converts a set of stencil kernel blocks to a parfor.
        """
    def _get_stencil_last_ind(self, dim_size, end_length, gen_nodes, scope, loc): ...
    def _get_stencil_start_ind(self, start_length, gen_nodes, scope, loc): ...
    def _replace_stencil_accesses(self, stencil_ir, parfor_vars, in_args, index_offsets, stencil_func, arg_to_arr_dict):
        """ Convert relative indexing in the stencil kernel to standard indexing
            by adding the loop index variables to the corresponding dimensions
            of the array index tuples.
        """
    def _add_index_offsets(self, index_list, index_offsets, new_body, scope, loc):
        """ Does the actual work of adding loop index variables to the
            relative index constants or variables.
        """
    def _add_offset_to_slice(self, slice_var, offset_var, out_nodes, scope, loc): ...

def get_stencil_ir(sf, typingctx, args, scope, loc, input_dict, typemap, calltypes):
    """get typed IR from stencil bytecode
    """

class DummyPipeline:
    state: Incomplete
    def __init__(self, typingctx, targetctx, args, f_ir) -> None: ...

def _get_const_index_expr(stencil_ir, func_ir, index_var):
    """
    infer index_var as constant if it is of a expression form like c-1 where c
    is a constant in the outer function.
    index_var is assumed to be inside stencil kernel
    """
def _get_const_index_expr_inner(stencil_ir, func_ir, index_var):
    """inner constant inference function that calls constant, unary and binary
    cases.
    """
def _get_const_two_irs(ir1, ir2, var):
    """get constant in either of two IRs if available
    otherwise, throw GuardException
    """
def _get_const_unary_expr(stencil_ir, func_ir, index_def):
    """evaluate constant unary expr if possible
    otherwise, raise GuardException
    """
def _get_const_binary_expr(stencil_ir, func_ir, index_def):
    """evaluate constant binary expr if possible
    otherwise, raise GuardException
    """
