from _typeshed import Incomplete
from numba.core import cgutils as cgutils, compiler as compiler, config as config, ir as ir, lowering as lowering, sigutils as sigutils, types as types
from numba.core.errors import CompilerError as CompilerError, InternalError as InternalError, NotDefinedError as NotDefinedError, NumbaParallelSafetyWarning as NumbaParallelSafetyWarning
from numba.core.ir_utils import add_offset_to_labels as add_offset_to_labels, find_max_label as find_max_label, find_topo_order as find_topo_order, fixup_var_define_in_scope as fixup_var_define_in_scope, get_call_table as get_call_table, get_definition as get_definition, get_global_func_typ as get_global_func_typ, get_name_var_table as get_name_var_table, get_np_ufunc_typ as get_np_ufunc_typ, get_unused_var_name as get_unused_var_name, guard as guard, is_const_call as is_const_call, is_pure as is_pure, legalize_names as legalize_names, remove_dels as remove_dels, rename_labels as rename_labels, replace_var_names as replace_var_names, transfer_scope as transfer_scope, visit_vars_inner as visit_vars_inner
from numba.core.typing import signature as signature
from numba.parfors import parfor as parfor
from numba.parfors.parfor import ensure_parallel_support as ensure_parallel_support
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder as ParforLoweringBuilder

class ParforLower(lowering.Lower):
    """This is a custom lowering class that extends standard lowering so as
    to accommodate parfor.Parfor nodes."""
    def lower_inst(self, inst) -> None: ...
    @property
    def _disable_sroa_like_opt(self):
        """
        Force disable this because Parfor use-defs is incompatible---it only
        considers use-defs in blocks that must be executing.
        See https://github.com/numba/numba/commit/017e2ff9db87fc34149b49dd5367ecbf0bb45268
        """

def _lower_parfor_parallel(lowerer, parfor): ...
def _lower_parfor_parallel_std(lowerer, parfor) -> None:
    """Lowerer that handles LLVM code generation for parfor.
    This function lowers a parfor IR node to LLVM.
    The general approach is as follows:
    1) The code from the parfor's init block is lowered normally
       in the context of the current function.
    2) The body of the parfor is transformed into a gufunc function.
    3) Code is inserted into the main function that calls do_scheduling
       to divide the iteration space for each thread, allocates
       reduction arrays, calls the gufunc function, and then invokes
       the reduction function across the reduction arrays to produce
       the final reduction values.
    """

_ReductionInfo: Incomplete

def _parfor_lowering_finalize_reduction(parfor, redarrs, lowerer, parfor_reddict, thread_count_var) -> None:
    """Emit code to finalize the reduction from the intermediate values of
    each thread.
    """

class ParforsUnexpectedReduceNodeError(InternalError):
    def __init__(self, inst) -> None: ...

def _lower_trivial_inplace_binops(parfor, lowerer, thread_count_var, reduce_info) -> None:
    """Lower trivial inplace-binop reduction.
    """
def _lower_non_trivial_reduce(parfor, lowerer, thread_count_var, reduce_info) -> None:
    """Lower non-trivial reduction such as call to `functools.reduce()`.
    """
def _lower_var_to_var_assign(lowerer, inst):
    """Lower Var->Var assignment.

    Returns True if-and-only-if `inst` is a Var->Var assignment.
    """
def _emit_getitem_call(idx, lowerer, reduce_info):
    """Emit call to ``redarr_var[idx]``
    """
def _emit_binop_reduce_call(binop, lowerer, thread_count_var, reduce_info):
    """Emit call to the ``binop`` for the reduction variable.
    """
def _is_right_op_and_rhs_is_init(inst, redvar_name, op):
    """Is ``inst`` an inplace-binop and the RHS is the reduction init?
    """
def _fix_redvar_name_ssa_mismatch(parfor, lowerer, inst, redvar_name):
    """Fix reduction variable name mismatch due to SSA.
    """
def _create_shape_signature(get_shape_classes, num_inputs, num_reductions, args, func_sig, races, typemap):
    """Create shape signature for GUFunc
    """
def _print_block(block) -> None: ...
def _print_body(body_dict) -> None:
    """Pretty-print a set of IR blocks.
    """
def wrap_loop_body(loop_body): ...
def unwrap_loop_body(loop_body) -> None: ...
def add_to_def_once_sets(a_def, def_once, def_more) -> None:
    """If the variable is already defined more than once, do nothing.
       Else if defined exactly once previously then transition this
       variable to the defined more than once set (remove it from
       def_once set and add to def_more set).
       Else this must be the first time we've seen this variable defined
       so add to def_once set.
    """
def compute_def_once_block(block, def_once, def_more, getattr_taken, typemap, module_assigns) -> None:
    """Effect changes to the set of variables defined once or more than once
       for a single block.
       block - the block to process
       def_once - set of variable names known to be defined exactly once
       def_more - set of variable names known to be defined more than once
       getattr_taken - dict mapping variable name to tuple of object and attribute taken
       module_assigns - dict mapping variable name to the Global that they came from
    """
def wrap_find_topo(loop_body): ...
def compute_def_once_internal(loop_body, def_once, def_more, getattr_taken, typemap, module_assigns) -> None:
    """Compute the set of variables defined exactly once in the given set of blocks
       and use the given sets for storing which variables are defined once, more than
       once and which have had a getattr call on them.
    """
def compute_def_once(loop_body, typemap):
    """Compute the set of variables defined exactly once in the given set of blocks.
    """
def find_vars(var, varset): ...
def _hoist_internal(inst, dep_on_param, call_table, hoisted, not_hoisted, typemap, stored_arrays): ...
def find_setitems_block(setitems, itemsset, block, typemap) -> None: ...
def find_setitems_body(setitems, itemsset, loop_body, typemap) -> None:
    """
      Find the arrays that are written into (goes into setitems) and the
      mutable objects (mostly arrays) that are written into other arrays
      (goes into itemsset).
    """
def empty_container_allocator_hoist(inst, dep_on_param, call_table, hoisted, not_hoisted, typemap, stored_arrays): ...
def hoist(parfor_params, loop_body, typemap, wrapped_blocks): ...
def redtyp_is_scalar(redtype): ...
def redtyp_to_redarraytype(redtyp):
    """Go from a reducation variable type to a reduction array type used to hold
       per-worker results.
    """
def redarraytype_to_sig(redarraytyp):
    """Given a reduction array type, find the type of the reduction argument to the gufunc.
    """
def legalize_names_with_typemap(names, typemap):
    """ We use ir_utils.legalize_names to replace internal IR variable names
        containing illegal characters (e.g. period) with a legal character
        (underscore) so as to create legal variable names.
        The original variable names are in the typemap so we also
        need to add the legalized name to the typemap as well.
    """
def to_scalar_from_0d(x): ...
def _create_gufunc_for_parfor_body(lowerer, parfor, typemap, typingctx, targetctx, flags, locals, has_aliases, index_var_typ, races):
    """
    Takes a parfor and creates a gufunc function for its body.
    There are two parts to this function.
    1) Code to iterate across the iteration space as defined by the schedule.
    2) The parfor body that does the work for a single point in the iteration space.
    Part 1 is created as Python text for simplicity with a sentinel assignment to mark the point
    in the IR where the parfor body should be added.
    This Python text is 'exec'ed into existence and its IR retrieved with run_frontend.
    The IR is scanned for the sentinel assignment where that basic block is split and the IR
    for the parfor body inserted.
    """
def replace_var_with_array_in_block(vars, block, typemap, calltypes): ...
def replace_var_with_array_internal(vars, loop_body, typemap, calltypes) -> None: ...
def replace_var_with_array(vars, loop_body, typemap, calltypes) -> None: ...
def call_parallel_gufunc(lowerer, cres, gu_signature, outer_sig, expr_args, expr_arg_types, loop_ranges, redvars, reddict, redarrdict, init_block, index_var_typ, races, exp_name_to_tuple_var):
    """
    Adds the call to the gufunc function from the main function.
    """
