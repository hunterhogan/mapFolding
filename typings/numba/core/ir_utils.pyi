import numba
from _typeshed import Incomplete
from numba.core import analysis as analysis, config as config, ir as ir, postproc as postproc, rewrites as rewrites, types as types, typing as typing
from numba.core.analysis import compute_cfg_from_blocks as compute_cfg_from_blocks, compute_live_map as compute_live_map, compute_use_defs as compute_use_defs
from numba.core.errors import CompilerError as CompilerError, NumbaPendingDeprecationWarning as NumbaPendingDeprecationWarning, TypingError as TypingError, UnsupportedError as UnsupportedError
from numba.core.extending import _Intrinsic as _Intrinsic
from numba.core.typing.templates import signature as signature

_unique_var_count: int

def mk_unique_var(prefix): ...

class _MaxLabel:
    _value: Incomplete
    def __init__(self, value: int = 0) -> None: ...
    def next(self): ...
    def update(self, newval) -> None: ...

_the_max_label: Incomplete

def get_unused_var_name(prefix, var_table):
    """ Get a new var name with a given prefix and
        make sure it is unused in the given variable table.
    """
def next_label(): ...
def mk_alloc(typingctx, typemap, calltypes, lhs, size_var, dtype, scope, loc, lhs_typ):
    """generate an array allocation with np.empty() and return list of nodes.
    size_var can be an int variable or tuple of int variables.
    lhs_typ is the type of the array being allocated.
    """
def convert_size_to_var(size_var, typemap, scope, loc, nodes): ...
def get_np_ufunc_typ(func):
    """get type of the incoming function from builtin registry"""
def mk_range_block(typemap, start, stop, step, calltypes, scope, loc):
    """make a block that initializes loop range and iteration variables.
    target label in jump needs to be set.
    """
def _mk_range_args(typemap, start, stop, step, scope, loc): ...
def get_global_func_typ(func):
    """get type variable for func() from builtin registry"""
def mk_loop_header(typemap, phi_var, calltypes, scope, loc):
    """make a block that is a loop header updating iteration variables.
    target labels in branch need to be set.
    """
def legalize_names(varnames):
    """returns a dictionary for conversion of variable names to legal
    parameter names.
    """
def get_name_var_table(blocks):
    """create a mapping from variable names to their ir.Var objects"""
def replace_var_names(blocks, namedict):
    """replace variables (ir.Var to ir.Var) from dictionary (name -> name)"""
def replace_var_callback(var, vardict): ...
def replace_vars(blocks, vardict) -> None:
    """replace variables (ir.Var to ir.Var) from dictionary (name -> ir.Var)"""
def replace_vars_stmt(stmt, vardict) -> None: ...
def replace_vars_inner(node, vardict): ...

visit_vars_extensions: Incomplete

def visit_vars(blocks, callback, cbdata) -> None:
    """go over statements of block bodies and replace variable names with
    dictionary.
    """
def visit_vars_stmt(stmt, callback, cbdata) -> None: ...
def visit_vars_inner(node, callback, cbdata): ...

add_offset_to_labels_extensions: Incomplete

def add_offset_to_labels(blocks, offset):
    """add an offset to all block labels and jump/branch targets
    """

find_max_label_extensions: Incomplete

def find_max_label(blocks): ...
def flatten_labels(blocks):
    """makes the labels in range(0, len(blocks)), useful to compare CFGs
    """
def remove_dels(blocks) -> None:
    """remove ir.Del nodes"""
def remove_args(blocks) -> None:
    """remove ir.Arg nodes"""
def dead_code_elimination(func_ir, typemap: Incomplete | None = None, alias_map: Incomplete | None = None, arg_aliases: Incomplete | None = None) -> None:
    """ Performs dead code elimination and leaves the IR in a valid state on
    exit
    """
def remove_dead(blocks, args, func_ir, typemap: Incomplete | None = None, alias_map: Incomplete | None = None, arg_aliases: Incomplete | None = None):
    """dead code elimination using liveness and CFG info.
    Returns True if something has been removed, or False if nothing is removed.
    """

remove_dead_extensions: Incomplete

def remove_dead_block(block, lives, call_table, arg_aliases, alias_map, alias_set, func_ir, typemap):
    """remove dead code using liveness info.
    Mutable arguments (e.g. arrays) that are not definitely assigned are live
    after return of function.
    """

remove_call_handlers: Incomplete

def remove_dead_random_call(rhs, lives, call_list): ...
def has_no_side_effect(rhs, lives, call_table):
    """ Returns True if this expression has no side effects that
        would prevent re-ordering.
    """

is_pure_extensions: Incomplete

def is_pure(rhs, lives, call_table):
    """ Returns True if every time this expression is evaluated it
        returns the same result.  This is not the case for things
        like calls to numpy.random.
    """
def is_const_call(module_name, func_name): ...

alias_analysis_extensions: Incomplete
alias_func_extensions: Incomplete

def get_canonical_alias(v, alias_map): ...
def find_potential_aliases(blocks, args, typemap, func_ir, alias_map: Incomplete | None = None, arg_aliases: Incomplete | None = None):
    """find all array aliases and argument aliases to avoid remove as dead"""
def _add_alias(lhs, rhs, alias_map, arg_aliases) -> None: ...
def is_immutable_type(var, typemap): ...
def copy_propagate(blocks, typemap):
    """compute copy propagation information for each block using fixed-point
     iteration on data flow equations:
     in_b = intersect(predec(B))
     out_b = gen_b | (in_b - kill_b)
    """
def init_copy_propagate_data(blocks, entry, typemap):
    """get initial condition of copy propagation data flow for each block.
    """

copy_propagate_extensions: Incomplete

def get_block_copies(blocks, typemap):
    """get copies generated and killed by each block
    """

apply_copy_propagate_extensions: Incomplete

def apply_copy_propagate(blocks, in_copies, name_var_table, typemap, calltypes, save_copies: Incomplete | None = None):
    """apply copy propagation to IR: replace variables when copies available"""
def fix_setitem_type(stmt, typemap, calltypes) -> None:
    """Copy propagation can replace setitem target variable, which can be array
    with 'A' layout. The replaced variable can be 'C' or 'F', so we update
    setitem call type reflect this (from matrix power test)
    """
def dprint_func_ir(func_ir, title, blocks: Incomplete | None = None) -> None:
    """Debug print function IR, with an optional blocks argument
    that may differ from the IR's original blocks.
    """
def find_topo_order(blocks, cfg: Incomplete | None = None):
    """find topological order of blocks such that true branches are visited
    first (e.g. for_break test in test_dataflow). This is written as an iterative
    implementation of post order traversal to avoid recursion limit issues.
    """

call_table_extensions: Incomplete

def get_call_table(blocks, call_table: Incomplete | None = None, reverse_call_table: Incomplete | None = None, topological_ordering: bool = True):
    """returns a dictionary of call variables and their references.
    """

tuple_table_extensions: Incomplete

def get_tuple_table(blocks, tuple_table: Incomplete | None = None):
    """returns a dictionary of tuple variables and their values.
    """
def get_stmt_writes(stmt): ...
def rename_labels(blocks):
    """rename labels of function body blocks according to topological sort.
    The set of labels of these blocks will remain unchanged.
    """
def simplify_CFG(blocks):
    """transform chains of blocks that have no loop into a single block"""

arr_math: Incomplete

def canonicalize_array_math(func_ir, typemap, calltypes, typingctx) -> None: ...

array_accesses_extensions: Incomplete

def get_array_accesses(blocks, accesses: Incomplete | None = None):
    """returns a set of arrays accessed and their indices.
    """
def is_slice_index(index):
    """see if index is a slice index or has slice in it"""
def merge_adjacent_blocks(blocks) -> None: ...
def restore_copy_var_names(blocks, save_copies, typemap):
    """
    restores variable names of user variables after applying copy propagation
    """
def simplify(func_ir, typemap, calltypes, metadata) -> None: ...

class GuardException(Exception): ...

def require(cond) -> None:
    """
    Raise GuardException if the given condition is False.
    """
def guard(func, *args, **kwargs):
    """
    Run a function with given set of arguments, and guard against
    any GuardException raised by the function by returning None,
    or the expected return results if no such exception was raised.
    """
def get_definition(func_ir, name, **kwargs):
    """
    Same as func_ir.get_definition(name), but raise GuardException if
    exception KeyError is caught.
    """
def build_definitions(blocks, definitions: Incomplete | None = None):
    """Build the definitions table of the given blocks by scanning
    through all blocks and instructions, useful when the definitions
    table is out-of-sync.
    Will return a new definition table if one is not passed.
    """

build_defs_extensions: Incomplete

def find_callname(func_ir, expr, typemap: Incomplete | None = None, definition_finder=...):
    """Try to find a call expression's function and module names and return
    them as strings for unbounded calls. If the call is a bounded call, return
    the self object instead of module name. Raise GuardException if failed.

    Providing typemap can make the call matching more accurate in corner cases
    such as bounded call on an object which is inside another object.
    """
def find_build_sequence(func_ir, var):
    """Check if a variable is constructed via build_tuple or
    build_list or build_set, and return the sequence and the
    operator, or raise GuardException otherwise.
    Note: only build_tuple is immutable, so use with care.
    """
def find_const(func_ir, var):
    """Check if a variable is defined as constant, and return
    the constant value, or raise GuardException otherwise.
    """
def compile_to_numba_ir(mk_func, glbls, typingctx: Incomplete | None = None, targetctx: Incomplete | None = None, arg_typs: Incomplete | None = None, typemap: Incomplete | None = None, calltypes: Incomplete | None = None):
    """
    Compile a function or a make_function node to Numba IR.

    Rename variables and
    labels to avoid conflict if inlined somewhere else. Perform type inference
    if typingctx and other typing inputs are available and update typemap and
    calltypes.
    """
def _create_function_from_code_obj(fcode, func_env, func_arg, func_clo, glbls):
    '''
    Creates a function from a code object. Args:
    * fcode - the code object
    * func_env - string for the freevar placeholders
    * func_arg - string for the function args (e.g. "a, b, c, d=None")
    * func_clo - string for the closure args
    * glbls - the function globals
    '''
def get_ir_of_code(glbls, fcode):
    """
    Compile a code object to get its IR, ir.Del nodes are emitted
    """
def replace_arg_nodes(block, args) -> None:
    """
    Replace ir.Arg(...) with variables
    """
def replace_returns(blocks, target, return_label) -> None:
    """
    Return return statement by assigning directly to target, and a jump.
    """
def gen_np_call(func_as_str, func, lhs, args, typingctx, typemap, calltypes): ...
def dump_block(label, block) -> None: ...
def dump_blocks(blocks) -> None: ...
def is_operator_or_getitem(expr):
    """true if expr is unary or binary operator or getitem"""
def is_get_setitem(stmt):
    """stmt is getitem assignment or setitem (and static cases)"""
def is_getitem(stmt):
    """true if stmt is a getitem or static_getitem assignment"""
def is_setitem(stmt):
    """true if stmt is a SetItem or StaticSetItem node"""
def index_var_of_get_setitem(stmt):
    """get index variable for getitem/setitem nodes (and static cases)"""
def set_index_var_of_get_setitem(stmt, new_index) -> None: ...
def is_namedtuple_class(c):
    """check if c is a namedtuple class"""
def fill_block_with_call(newblock, callee, label_next, inputs, outputs):
    """Fill *newblock* to call *callee* with arguments listed in *inputs*.
    The returned values are unwrapped into variables in *outputs*.
    The block would then jump to *label_next*.
    """
def fill_callee_prologue(block, inputs, label_next):
    """
    Fill a new block *block* that unwraps arguments using names in *inputs* and
    then jumps to *label_next*.

    Expected to use with *fill_block_with_call()*
    """
def fill_callee_epilogue(block, outputs):
    """
    Fill a new block *block* to prepare the return values.
    This block is the last block of the function.

    Expected to use with *fill_block_with_call()*
    """
def find_outer_value(func_ir, var):
    """Check if a variable is a global value, and return the value,
    or raise GuardException otherwise.
    """
def raise_on_unsupported_feature(func_ir, typemap) -> None:
    """
    Helper function to walk IR and raise if it finds op codes
    that are unsupported. Could be extended to cover IR sequences
    as well as op codes. Intended use is to call it as a pipeline
    stage just prior to lowering to prevent LoweringErrors for known
    unsupported features.
    """
def warn_deprecated(func_ir, typemap) -> None: ...
def resolve_func_from_module(func_ir, node):
    """
    This returns the python function that is being getattr'd from a module in
    some IR, it resolves import chains/submodules recursively. Should it not be
    possible to find the python function being called None will be returned.

    func_ir - the FunctionIR object
    node - the IR node from which to start resolving (should be a `getattr`).
    """
def enforce_no_dels(func_ir) -> None:
    """
    Enforce there being no ir.Del nodes in the IR.
    """
def enforce_no_phis(func_ir) -> None:
    """
    Enforce there being no ir.Expr.phi nodes in the IR.
    """
def legalize_single_scope(blocks):
    """Check the given mapping of ir.Block for containing a single scope.
    """
def check_and_legalize_ir(func_ir, flags: numba.core.compiler.Flags):
    """
    This checks that the IR presented is legal
    """
def convert_code_obj_to_function(code_obj, caller_ir):
    """
    Converts a code object from a `make_function.code` attr in the IR into a
    python function, caller_ir is the FunctionIR of the caller and is used for
    the resolution of freevars.
    """
def fixup_var_define_in_scope(blocks) -> None:
    """Fixes the mapping of ir.Block to ensure all referenced ir.Var are
    defined in every scope used by the function. Such that looking up a variable
    from any scope in this function will not fail.

    Note: This is a workaround. Ideally, all the blocks should refer to the
    same ir.Scope, but that property is not maintained by all the passes.
    """
def transfer_scope(block, scope):
    """Transfer the ir.Block to use the given ir.Scope.
    """
def is_setup_with(stmt): ...
def is_terminator(stmt): ...
def is_raise(stmt): ...
def is_return(stmt): ...
def is_pop_block(stmt): ...
