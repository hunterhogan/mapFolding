from _typeshed import Incomplete
from collections import OrderedDict as OrderedDict
from collections.abc import Generator
from numba import pndindex as pndindex, prange as prange
from numba.core import analysis as analysis, config as config, errors as errors, ir as ir, ir_utils as ir_utils, postproc as postproc, rewrites as rewrites, typeinfer as typeinfer, types as types, typing as typing, utils as utils
from numba.core.analysis import compute_cfg_from_blocks as compute_cfg_from_blocks, compute_dead_maps as compute_dead_maps, compute_live_map as compute_live_map, compute_use_defs as compute_use_defs
from numba.core.controlflow import CFGraph as CFGraph
from numba.core.extending import lower_builtin as lower_builtin, overload as overload, register_jitable as register_jitable
from numba.core.imputils import impl_ret_untracked as impl_ret_untracked
from numba.core.ir_utils import GuardException as GuardException, add_offset_to_labels as add_offset_to_labels, apply_copy_propagate as apply_copy_propagate, build_definitions as build_definitions, canonicalize_array_math as canonicalize_array_math, compile_to_numba_ir as compile_to_numba_ir, copy_propagate as copy_propagate, dprint_func_ir as dprint_func_ir, find_build_sequence as find_build_sequence, find_callname as find_callname, find_potential_aliases as find_potential_aliases, find_topo_order as find_topo_order, get_block_copies as get_block_copies, get_call_table as get_call_table, get_definition as get_definition, get_name_var_table as get_name_var_table, get_np_ufunc_typ as get_np_ufunc_typ, get_stmt_writes as get_stmt_writes, guard as guard, has_no_side_effect as has_no_side_effect, index_var_of_get_setitem as index_var_of_get_setitem, is_get_setitem as is_get_setitem, is_getitem as is_getitem, is_setitem as is_setitem, mk_alloc as mk_alloc, mk_loop_header as mk_loop_header, mk_range_block as mk_range_block, mk_unique_var as mk_unique_var, next_label as next_label, remove_dead as remove_dead, rename_labels as rename_labels, replace_arg_nodes as replace_arg_nodes, replace_returns as replace_returns, replace_var_names as replace_var_names, replace_vars as replace_vars, replace_vars_inner as replace_vars_inner, require as require, set_index_var_of_get_setitem as set_index_var_of_get_setitem, simplify as simplify, simplify_CFG as simplify_CFG, transfer_scope as transfer_scope, visit_vars as visit_vars, visit_vars_inner as visit_vars_inner
from numba.core.types.functions import Function as Function
from numba.core.typing import npydecl as npydecl, signature as signature
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, infer_global as infer_global
from numba.np.npdatetime_helpers import datetime_maximum as datetime_maximum, datetime_minimum as datetime_minimum
from numba.np.numpy_support import as_dtype as as_dtype, numpy_version as numpy_version
from numba.parfors import array_analysis as array_analysis
from numba.parfors.array_analysis import assert_equiv as assert_equiv, random_1arg_size as random_1arg_size, random_2arg_sizelast as random_2arg_sizelast, random_3arg_sizelast as random_3arg_sizelast, random_calls as random_calls, random_int_args as random_int_args
from numba.stencils import stencilparfor as stencilparfor
from numba.stencils.stencilparfor import StencilPass as StencilPass
from typing import NamedTuple

_termwidth: int
_txtwrapper: Incomplete

def print_wrapped(x) -> None: ...

sequential_parfor_lowering: bool

def init_prange() -> None: ...
def init_prange_overload(): ...

class internal_prange:
    def __new__(cls, *args): ...

def min_parallel_impl(return_type, arg): ...
def max_parallel_impl(return_type, arg): ...
def argmin_parallel_impl(in_arr): ...
def argmax_parallel_impl(in_arr): ...
def dotvv_parallel_impl(a, b): ...
def dotvm_parallel_impl(a, b): ...
def dotmv_parallel_impl(a, b): ...
def dot_parallel_impl(return_type, atyp, btyp): ...
def sum_parallel_impl(return_type, arg): ...
def prod_parallel_impl(return_type, arg): ...
def mean_parallel_impl(return_type, arg): ...
def var_parallel_impl(return_type, arg): ...
def std_parallel_impl(return_type, arg): ...
def arange_parallel_impl(return_type, *args, dtype: Incomplete | None = None): ...
def linspace_parallel_impl(return_type, *args): ...

swap_functions_map: Incomplete

def fill_parallel_impl(return_type, arr, val):
    """Parallel implementation of ndarray.fill.  The array on
       which to operate is retrieved from get_call_name and
       is passed along with the value to fill.
    """

replace_functions_ndarray: Incomplete

def max_checker(arr_size) -> None: ...
def min_checker(arr_size) -> None: ...
def argmin_checker(arr_size) -> None: ...
def argmax_checker(arr_size) -> None: ...

class checker_impl(NamedTuple):
    name: Incomplete
    func: Incomplete

replace_functions_checkers_map: Incomplete

class LoopNest:
    """The LoopNest class holds information of a single loop including
    the index variable (of a non-negative integer value), and the
    range variable, e.g. range(r) is 0 to r-1 with step size 1.
    """
    index_variable: Incomplete
    start: Incomplete
    stop: Incomplete
    step: Incomplete
    def __init__(self, index_variable, start, stop, step) -> None: ...
    def __repr__(self) -> str: ...
    def list_vars(self): ...

class Parfor(ir.Expr, ir.Stmt):
    id_counter: int
    id: Incomplete
    loop_nests: Incomplete
    init_block: Incomplete
    loop_body: Incomplete
    index_var: Incomplete
    params: Incomplete
    equiv_set: Incomplete
    patterns: Incomplete
    flags: Incomplete
    no_sequential_lowering: Incomplete
    races: Incomplete
    redvars: Incomplete
    reddict: Incomplete
    lowerer: Incomplete
    def __init__(self, loop_nests, init_block, loop_body, loc, index_var, equiv_set, pattern, flags, *, no_sequential_lowering: bool = False, races=...) -> None: ...
    def __repr__(self) -> str: ...
    def get_loop_nest_vars(self): ...
    def list_vars(self):
        """list variables used (read/written) in this parfor by
        traversing the body and combining block uses.
        """
    def get_shape_classes(self, var, typemap: Incomplete | None = None):
        """get the shape classes for a given variable.
        If a typemap is specified then use it for type resolution
        """
    def dump(self, file: Incomplete | None = None) -> None: ...
    def validate_params(self, typemap) -> None:
        """
        Check that Parfors params are of valid types.
        """

def _analyze_parfor(parfor, equiv_set, typemap, array_analysis):
    """Recursive array analysis for parfor nodes.
    """

class ParforDiagnostics:
    """Holds parfor diagnostic info, this is accumulated throughout the
    PreParforPass and ParforPass, also in the closure inlining!
    """
    func: Incomplete
    replaced_fns: Incomplete
    internal_name: str
    fusion_info: Incomplete
    nested_fusion_info: Incomplete
    fusion_reports: Incomplete
    hoist_info: Incomplete
    def __init__(self) -> None: ...
    func_ir: Incomplete
    name: Incomplete
    line: Incomplete
    fusion_enabled: Incomplete
    purpose: str
    initial_parfors: Incomplete
    def setup(self, func_ir, fusion_enabled) -> None: ...
    @property
    def has_setup(self): ...
    _has_setup: Incomplete
    @has_setup.setter
    def has_setup(self, state) -> None: ...
    def count_parfors(self, blocks: Incomplete | None = None): ...
    def _get_nested_parfors(self, parfor, parfors_list) -> None: ...
    def _get_parfors(self, blocks, parfors_list) -> None: ...
    def get_parfors(self): ...
    def hoisted_allocations(self): ...
    def compute_graph_info(self, _a):
        """
        compute adjacency list of the fused loops
        and find the roots in of the lists
        """
    def get_stats(self, fadj, nadj, root):
        """
        Computes the number of fused and serialized loops
        based on a fusion adjacency list `fadj` and a nested
        parfors adjacency list `nadj` for the root, `root`
        """
    def reachable_nodes(self, adj, root):
        """
        returns a list of nodes reachable in an adjacency list from a
        specified root
        """
    def sort_pf_by_line(self, pf_id, parfors_simple):
        """
        pd_id - the parfors id
        parfors_simple - the simple parfors map
        """
    def get_parfors_simple(self, print_loop_search): ...
    def get_all_lines(self, parfors_simple): ...
    def source_listing(self, parfors_simple, purpose_str) -> None: ...
    def print_unoptimised(self, lines): ...
    def print_optimised(self, lines): ...
    def allocation_hoist(self) -> None: ...
    def instruction_hoist(self) -> None: ...
    def dump(self, level: int = 1) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class PreParforPass:
    """Preprocessing for the Parfor pass. It mostly inlines parallel
    implementations of numpy functions if available.
    """
    func_ir: Incomplete
    typemap: Incomplete
    calltypes: Incomplete
    typingctx: Incomplete
    targetctx: Incomplete
    options: Incomplete
    swapped: Incomplete
    replace_functions_map: Incomplete
    stats: Incomplete
    def __init__(self, func_ir, typemap, calltypes, typingctx, targetctx, options, swapped={}, replace_functions_map: Incomplete | None = None) -> None: ...
    def run(self) -> None:
        """Run pre-parfor processing pass.
        """
    def _replace_parallel_functions(self, blocks):
        """
        Replace functions with their parallel implementation in
        replace_functions_map if available.
        The implementation code is inlined to enable more optimization.
        """

def find_template(op): ...

class ParforPassStates:
    """This class encapsulates all internal states of the ParforPass.
    """
    func_ir: Incomplete
    typemap: Incomplete
    calltypes: Incomplete
    typingctx: Incomplete
    targetctx: Incomplete
    return_type: Incomplete
    options: Incomplete
    diagnostics: Incomplete
    swapped_fns: Incomplete
    fusion_info: Incomplete
    nested_fusion_info: Incomplete
    array_analysis: Incomplete
    flags: Incomplete
    metadata: Incomplete
    def __init__(self, func_ir, typemap, calltypes, return_type, typingctx, targetctx, options, flags, metadata, diagnostics=...) -> None: ...

class ConvertInplaceBinop:
    """Parfor subpass to convert setitem on Arrays
    """
    pass_states: Incomplete
    rewritten: Incomplete
    def __init__(self, pass_states) -> None:
        """
        Parameters
        ----------
        pass_states : ParforPassStates
        """
    def run(self, blocks) -> None: ...
    def _inplace_binop_to_parfor(self, equiv_set, loc, op, target, value):
        """generate parfor from setitem node with a boolean or slice array indices.
        The value can be either a scalar or an array variable, and if a boolean index
        is used for the latter case, the same index must be used for the value too.
        """
    def _type_getitem(self, args): ...

def get_index_var(x): ...

class ConvertSetItemPass:
    """Parfor subpass to convert setitem on Arrays
    """
    pass_states: Incomplete
    rewritten: Incomplete
    def __init__(self, pass_states) -> None:
        """
        Parameters
        ----------
        pass_states : ParforPassStates
        """
    def run(self, blocks): ...
    def _setitem_to_parfor(self, equiv_set, loc, target, index, value, shape: Incomplete | None = None):
        """generate parfor from setitem node with a boolean or slice array indices.
        The value can be either a scalar or an array variable, and if a boolean index
        is used for the latter case, the same index must be used for the value too.
        """
    def _type_getitem(self, args): ...

def _make_index_var(typemap, scope, index_vars, body_block, force_tuple: bool = False):
    """ When generating a SetItem call to an array in a parfor, the general
    strategy is to generate a tuple if the array is more than 1 dimension.
    If it is 1 dimensional then you can use a simple variable.  This routine
    is also used when converting pndindex to parfor but pndindex requires a
    tuple even if the iteration space is 1 dimensional.  The pndindex use of
    this function will use force_tuple to make the output index a tuple even
    if it is one dimensional.
    """
def _mk_parfor_loops(typemap, size_vars, scope, loc):
    """
    Create loop index variables and build LoopNest objects for a parfor.
    """

class ConvertNumpyPass:
    """
    Convert supported Numpy functions, as well as arrayexpr nodes, to
    parfor nodes.
    """
    pass_states: Incomplete
    rewritten: Incomplete
    def __init__(self, pass_states) -> None: ...
    def run(self, blocks) -> None: ...
    def _is_C_order(self, arr_name): ...
    def _is_C_or_F_order(self, arr_name): ...
    def _arrayexpr_to_parfor(self, equiv_set, lhs, arrayexpr, avail_vars):
        """generate parfor from arrayexpr node, which is essentially a
        map with recursive tree.
        """
    def _is_supported_npycall(self, expr):
        """check if we support parfor translation for
        this Numpy call.
        """
    def _numpy_to_parfor(self, equiv_set, lhs, expr): ...
    def _numpy_map_to_parfor(self, equiv_set, call_name, lhs, args, kws, expr):
        """generate parfor from Numpy calls that are maps.
        """

class ConvertReducePass:
    """
    Find reduce() calls and convert them to parfors.
    """
    pass_states: Incomplete
    rewritten: Incomplete
    def __init__(self, pass_states) -> None: ...
    def run(self, blocks) -> None: ...
    def _reduce_to_parfor(self, equiv_set, lhs, args, loc):
        """
        Convert a reduce call to a parfor.
        The call arguments should be (call_name, array, init_value).
        """
    def _mk_reduction_body(self, call_name, scope, loc, index_vars, in_arr, acc_var):
        """
        Produce the body blocks for a reduction function indicated by call_name.
        """

class ConvertLoopPass:
    """Build Parfor nodes from prange loops.
    """
    pass_states: Incomplete
    rewritten: Incomplete
    def __init__(self, pass_states) -> None: ...
    def run(self, blocks): ...
    def _is_parallel_loop(self, func_var, call_table): ...
    def _get_loop_kind(self, func_var, call_table):
        """see if prange is user prange or internal"""
    def _get_prange_init_block(self, entry_block, call_table, prange_args):
        """
        If there is init_prange, find the code between init_prange and prange
        calls. Remove the code from entry_block and return it.
        """
    def _is_prange_init(self, func_var, call_table): ...
    def _replace_loop_access_indices(self, loop_body, index_set, new_index):
        """
        Replace array access indices in a loop body with a new index.
        index_set has all the variables that are equivalent to loop index.
        """
    def _replace_multi_dim_ind(self, ind_var, index_set, new_index) -> None:
        """
        replace individual indices in multi-dimensional access variable, which
        is a build_tuple
        """

def _find_mask(typemap, func_ir, arr_def):
    """check if an array is of B[...M...], where M is a
    boolean array, and other indices (if available) are ints.
    If found, return B, M, M's type, and a tuple representing mask indices.
    Otherwise, raise GuardException.
    """

class ParforPass(ParforPassStates):
    """ParforPass class is responsible for converting NumPy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    """
    def _pre_run(self) -> None: ...
    def run(self) -> None:
        """run parfor conversion pass: replace Numpy calls
        with Parfors when possible and optimize the IR."""
    def _find_mask(self, arr_def):
        """check if an array is of B[...M...], where M is a
        boolean array, and other indices (if available) are ints.
        If found, return B, M, M's type, and a tuple representing mask indices.
        Otherwise, raise GuardException.
        """
    def _mk_parfor_loops(self, size_vars, scope, loc):
        """
        Create loop index variables and build LoopNest objects for a parfor.
        """

class ParforFusionPass(ParforPassStates):
    """ParforFusionPass class is responsible for fusing parfors
    """
    def run(self) -> None:
        """run parfor fusion pass"""
    def fuse_parfors(self, array_analysis, blocks, func_ir, typemap) -> None: ...
    def fuse_recursive_parfor(self, parfor, equiv_set, func_ir, typemap) -> None: ...

class ParforPreLoweringPass(ParforPassStates):
    """ParforPreLoweringPass class is responsible for preparing parfors for lowering.
    """
    def run(self) -> None:
        """run parfor prelowering pass"""

def _remove_size_arg(call_name, expr) -> None:
    """remove size argument from args or kws"""
def _get_call_arg_types(expr, typemap): ...
def _arrayexpr_tree_to_ir(func_ir, typingctx, typemap, calltypes, equiv_set, init_block, expr_out_var, expr, parfor_index_tuple_var, all_parfor_indices, avail_vars):
    """generate IR from array_expr's expr tree recursively. Assign output to
    expr_out_var and returns the whole IR as a list of Assign nodes.
    """
def _gen_np_divide(arg1, arg2, out_ir, typemap):
    """generate np.divide() instead of / for array_expr to get numpy error model
    like inf for division by zero (test_division_by_zero).
    """
def _gen_arrayexpr_getitem(equiv_set, var, parfor_index_tuple_var, all_parfor_indices, el_typ, calltypes, typingctx, typemap, init_block, out_ir):
    """if there is implicit dimension broadcast, generate proper access variable
    for getitem. For example, if indices are (i1,i2,i3) but shape is (c1,0,c3),
    generate a tuple with (i1,0,i3) for access.  Another example: for (i1,i2,i3)
    and (c1,c2) generate (i2,i3).
    """
def _find_func_var(typemap, func, avail_vars, loc):
    """find variable in typemap which represents the function func.
    """
def lower_parfor_sequential(typingctx, func_ir, typemap, calltypes, metadata) -> None: ...
def _lower_parfor_sequential_block(block_label, block, new_blocks, typemap, calltypes, parfor_found, scope): ...
def _find_first_parfor(body): ...
def get_parfor_params(blocks, options_fusion, fusion_info):
    """find variables used in body of parfors from outside and save them.
    computed as live variables at entry of first block.
    """
def _combine_params_races_for_ssa_names(scope, params, races):
    """Returns `(params|races1, races1)`, where `races1` contains all variables
    in `races` are NOT referring to the same unversioned (SSA) variables in
    `params`.
    """
def get_parfor_params_inner(parfor, pre_defs, options_fusion, fusion_info): ...
def _find_parfors(body) -> Generator[Incomplete]: ...
def _is_indirect_index(func_ir, index, nest_indices): ...
def get_array_indexed_with_parfor_index_internal(loop_body, index, ret_indexed, ret_not_indexed, nest_indices, func_ir) -> None: ...
def get_array_indexed_with_parfor_index(loop_body, index, nest_indices, func_ir): ...
def get_parfor_outputs(parfor, parfor_params):
    """get arrays that are written to inside the parfor and need to be passed
    as parameters to gufunc.
    """

_RedVarInfo: Incomplete

def get_parfor_reductions(func_ir, parfor, parfor_params, calltypes, reductions: Incomplete | None = None, reduce_varnames: Incomplete | None = None, param_uses: Incomplete | None = None, param_nodes: Incomplete | None = None, var_to_param: Incomplete | None = None):
    """find variables that are updated using their previous values and an array
    item accessed with parfor index, e.g. s = s+A[i]
    """
def check_conflicting_reduction_operators(param, nodes) -> None:
    """In prange, a user could theoretically specify conflicting
       reduction operators.  For example, in one spot it is += and
       another spot *=.  Here, we raise an exception if multiple
       different reduction operators are used in one prange.
    """
def get_reduction_init(nodes):
    """
    Get initial value for known reductions.
    Currently, only += and *= are supported.
    """
def supported_reduction(x, func_ir): ...
def get_reduce_nodes(reduction_node, nodes, func_ir):
    """
    Get nodes that combine the reduction variable with a sentinel variable.
    Recognizes the first node that combines the reduction variable with another
    variable.
    """
def get_expr_args(expr):
    """
    Get arguments of an expression node
    """
def visit_parfor_pattern_vars(parfor, callback, cbdata) -> None: ...
def visit_vars_parfor(parfor, callback, cbdata) -> None: ...
def parfor_defs(parfor, use_set: Incomplete | None = None, def_set: Incomplete | None = None):
    """list variables written in this parfor by recursively
    calling compute_use_defs() on body and combining block defs.
    """
def _parfor_use_alloca(parfor, alloca_set) -> None:
    """
    Reduction variables for parfors and the reduction variables within
    nested parfors must be stack allocated.
    """
def parfor_insert_dels(parfor, curr_dead_set):
    """insert dels in parfor. input: dead variable set right after parfor.
    returns the variables for which del was inserted.
    """
def maximize_fusion(func_ir, blocks, typemap, up_direction: bool = True) -> None:
    """
    Reorder statements to maximize parfor fusion. Push all parfors up or down
    so they are adjacent.
    """
def maximize_fusion_inner(func_ir, block, call_table, alias_map, arg_aliases, up_direction: bool = True): ...
def expand_aliases(the_set, alias_map, arg_aliases): ...
def _can_reorder_stmts(stmt, next_stmt, func_ir, call_table, alias_map, arg_aliases):
    """
    Check dependencies to determine if a parfor can be reordered in the IR block
    with a non-parfor statement.
    """
def is_assert_equiv(func_ir, expr): ...
def get_parfor_writes(parfor): ...

class FusionReport(NamedTuple):
    first: Incomplete
    second: Incomplete
    message: Incomplete

def try_fuse(equiv_set, parfor1, parfor2, metadata, func_ir, typemap):
    """try to fuse parfors and return a fused parfor, otherwise return None
    """
def fuse_parfors_inner(parfor1, parfor2): ...
def remove_duplicate_definitions(blocks, nameset) -> None:
    """Remove duplicated definition for variables in the given nameset, which
    is often a result of parfor fusion.
    """
def has_cross_iter_dep(parfor, func_ir, typemap, index_positions: Incomplete | None = None, indexed_arrays: Incomplete | None = None, non_indexed_arrays: Incomplete | None = None): ...
def dprint(*s) -> None: ...
def get_parfor_pattern_vars(parfor):
    """ get the variables used in parfor pattern information
    """
def remove_dead_parfor(parfor, lives, lives_n_aliases, arg_aliases, alias_map, func_ir, typemap):
    """ remove dead code inside parfor including get/sets
    """
def _update_parfor_get_setitems(block_body, index_var, alias_map, saved_values, lives) -> None:
    """
    replace getitems of a previously set array in a block of parfor loop body
    """
def remove_dead_parfor_recursive(parfor, lives, arg_aliases, alias_map, func_ir, typemap) -> None:
    """create a dummy function from parfor and call remove dead recursively
    """
def _add_liveness_return_block(blocks, lives, typemap): ...
def find_potential_aliases_parfor(parfor, args, typemap, func_ir, alias_map, arg_aliases) -> None: ...
def simplify_parfor_body_CFG(blocks):
    """simplify CFG of body loops in parfors"""
def wrap_parfor_blocks(parfor, entry_label: Incomplete | None = None):
    """wrap parfor blocks for analysis/optimization like CFG"""
def unwrap_parfor_blocks(parfor, blocks: Incomplete | None = None) -> None:
    """
    unwrap parfor blocks after analysis/optimization.
    Allows changes to the parfor loop.
    """
def get_copies_parfor(parfor, typemap):
    """find copies generated/killed by parfor"""
def apply_copies_parfor(parfor, var_dict, name_var_table, typemap, calltypes, save_copies) -> None:
    """apply copy propagate recursively in parfor"""
def push_call_vars(blocks, saved_globals, saved_getattrs, typemap, nested: bool = False) -> None:
    """push call variables to right before their call site.
    assuming one global/getattr is created for each call site and control flow
    doesn't change it.
    """
def _get_saved_call_nodes(fname, saved_globals, saved_getattrs, block_defs, rename_dict):
    """ Implement the copying of globals or getattrs for the purposes noted in
        push_call_vars.  We make a new var and assign to it a copy of the
        global or getattr.  We remember this new assignment node and add an
        entry in the renaming dictionary so that for this block the original
        var name is replaced by the new var name we created.
    """
def repr_arrayexpr(arrayexpr):
    """Extract operators from arrayexpr to represent it abstractly as a string.
    """
def fix_generator_types(generator_info, return_type, typemap) -> None:
    """postproc updates generator_info with live variables after transformations
    but generator variables have types in return_type that are updated here.
    """
def get_parfor_call_table(parfor, call_table: Incomplete | None = None, reverse_call_table: Incomplete | None = None): ...
def get_parfor_tuple_table(parfor, tuple_table: Incomplete | None = None): ...
def get_parfor_array_accesses(parfor, accesses: Incomplete | None = None): ...
def parfor_add_offset_to_labels(parfor, offset) -> None: ...
def parfor_find_max_label(parfor): ...
def parfor_typeinfer(parfor, typeinferer) -> None: ...
def build_parfor_definitions(parfor, definitions: Incomplete | None = None):
    """get variable definition table for parfors"""
def dummy_return_in_loop_body(loop_body) -> Generator[None]:
    """adds dummy return to last block of parfor loop body for CFG computation
    """

class ReduceInfer(AbstractTemplate):
    def generic(self, args, kws): ...

def ensure_parallel_support() -> None:
    """Check if the platform supports parallel=True and raise if it does not.
    """
