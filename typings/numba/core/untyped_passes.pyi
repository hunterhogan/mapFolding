from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import bytecode as bytecode, config as config, consts as consts, errors as errors, interpreter as interpreter, ir as ir, postproc as postproc, rewrites as rewrites, transforms as transforms, types as types
from numba.core.analysis import compute_cfg_from_blocks as compute_cfg_from_blocks, compute_use_defs as compute_use_defs, dead_branch_prune as dead_branch_prune, find_literally_calls as find_literally_calls, rewrite_semantic_constants as rewrite_semantic_constants
from numba.core.compiler_machinery import AnalysisPass as AnalysisPass, FunctionPass as FunctionPass, SSACompliantMixin as SSACompliantMixin, register_pass as register_pass
from numba.core.ir_utils import GuardException as GuardException, build_definitions as build_definitions, compile_to_numba_ir as compile_to_numba_ir, convert_code_obj_to_function as convert_code_obj_to_function, find_max_label as find_max_label, fixup_var_define_in_scope as fixup_var_define_in_scope, get_definition as get_definition, get_name_var_table as get_name_var_table, guard as guard, rename_labels as rename_labels, replace_var_names as replace_var_names, resolve_func_from_module as resolve_func_from_module, simplify_CFG as simplify_CFG, transfer_scope as transfer_scope
from numba.core.ssa import reconstruct_ssa as reconstruct_ssa
from numba.misc.special import literal_unroll as literal_unroll

def fallback_context(state, msg) -> Generator[None]:
    """
    Wraps code that would signal a fallback to object mode
    """

class ExtractByteCode(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Extract bytecode from function
        """

class TranslateByteCode(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Analyze bytecode and translating to Numba IR
        """

class FixupArgs(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class IRProcessing(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class RewriteSemanticConstants(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """

class DeadBranchPrune(SSACompliantMixin, FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """
    def get_analysis_usage(self, AU) -> None: ...

class InlineClosureLikes(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class GenericRewrites(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Perform any intermediate representation rewrites before type
        inference.
        """

class WithLifting(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Extract with-contexts
        """

class InlineInlinables(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.jit decorator directly
    into the site of its call depending on the value set in the 'inline' kwarg
    to the decorator.

    This is an untyped pass. CFG simplification is performed at the end of the
    pass but no block level clean up is performed on the mutated IR (typing
    information is not available to do so).
    """
    _name: str
    _DEBUG: bool
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """Run inlining of inlinables
        """
    def _do_work(self, state, work_list, block, i, expr, inline_worker): ...

class PreserveIR(AnalysisPass):
    """
    Preserves the IR in the metadata
    """
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class FindLiterallyCalls(FunctionPass):
    """Find calls to `numba.literally()` and signal if its requirement is not
    satisfied.
    """
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class CanonicalizeLoopExit(FunctionPass):
    """A pass to canonicalize loop exit by splitting it from function exit.
    """
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...
    def _split_exit_block(self, fir, cfg, exit_label) -> None: ...

class CanonicalizeLoopEntry(FunctionPass):
    """A pass to canonicalize loop header by splitting it from function entry.

    This is needed for loop-lifting; esp in py3.8
    """
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...
    def _split_entry_block(self, fir, cfg, loop, entry_label) -> None: ...

class PrintIRCFG(FunctionPass):
    _name: str
    _ver: int
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class MakeFunctionToJitFunction(FunctionPass):
    '''
    This swaps an ir.Expr.op == "make_function" i.e. a closure, for a compiled
    function containing the closure body and puts it in ir.Global. It\'s a 1:1
    statement value swap. `make_function` is already untyped
    '''
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class TransformLiteralUnrollConstListToTuple(FunctionPass):
    """ This pass spots a `literal_unroll([<constant values>])` and rewrites it
    as a `literal_unroll(tuple(<constant values>))`.
    """
    _name: str
    _accepted_types: Incomplete
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class MixedContainerUnroller(FunctionPass):
    _name: str
    _DEBUG: bool
    _accepted_types: Incomplete
    def __init__(self) -> None: ...
    def analyse_tuple(self, tup):
        """
        Returns a map of type->list(indexes) for a typed tuple
        """
    def add_offset_to_labels_w_ignore(self, blocks, offset, ignore: Incomplete | None = None):
        """add an offset to all block labels and jump/branch targets
        don't add an offset to anything in the ignore list
        """
    def inject_loop_body(self, switch_ir, loop_ir, caller_max_label, dont_replace, switch_data):
        '''
        Injects the "loop body" held in `loop_ir` into `switch_ir` where ever
        there is a statement of the form `SENTINEL.<int> = RHS`. It also:
        * Finds and then deliberately does not relabel non-local jumps so as to
          make the switch table suitable for injection into the IR from which
          the loop body was derived.
        * Looks for `typed_getitem` and wires them up to loop body version
          specific variables or, if possible, directly writes in their constant
          value at their use site.

        Args:
        - switch_ir, the switch table with SENTINELS as generated by
          self.gen_switch
        - loop_ir, the IR of the loop blocks (derived from the original func_ir)
        - caller_max_label, the maximum label in the func_ir caller
        - dont_replace, variables that should not be renamed (to handle
          references to variables that are incoming at the loop head/escaping at
          the loop exit.
        - switch_data, the switch table data used to generated the switch_ir,
          can be generated by self.analyse_tuple.

        Returns:
        - A type specific switch table with each case containing a versioned
          loop body suitable for injection as a replacement for the loop_ir.
        '''
    def gen_switch(self, data, index):
        """
        Generates a function with a switch table like
        def foo():
            if PLACEHOLDER_INDEX in (<integers>):
                SENTINEL = None
            elif PLACEHOLDER_INDEX in (<integers>):
                SENTINEL = None
            ...
            else:
                raise RuntimeError

        The data is a map of (type : indexes) for example:
        (int64, int64, float64)
        might give:
        {int64: [0, 1], float64: [2]}

        The index is the index variable for the driving range loop over the
        mixed tuple.
        """
    def apply_transform(self, state): ...
    def unroll_loop(self, state, loop_info) -> None: ...
    def run_pass(self, state): ...

class IterLoopCanonicalization(FunctionPass):
    """ Transforms loops that are induced by `getiter` into range() driven loops
    If the typemap is available this will only impact Tuple and UniTuple, if it
    is not available it will impact all matching loops.
    """
    _name: str
    _DEBUG: bool
    _accepted_types: Incomplete
    _accepted_calls: Incomplete
    def __init__(self) -> None: ...
    def assess_loop(self, loop, func_ir, partial_typemap: Incomplete | None = None): ...
    def transform(self, loop, func_ir, cfg): ...
    def run_pass(self, state): ...

class PropagateLiterals(FunctionPass):
    """Implement literal propagation based on partial type inference"""
    _name: str
    def __init__(self) -> None: ...
    def get_analysis_usage(self, AU) -> None: ...
    def run_pass(self, state): ...

class LiteralPropagationSubPipelinePass(FunctionPass):
    """Implement literal propagation based on partial type inference"""
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...
    def get_analysis_usage(self, AU) -> None: ...

class LiteralUnroll(FunctionPass):
    """Implement the literal_unroll semantics"""
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class SimplifyCFG(FunctionPass):
    """Perform CFG simplification"""
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class ReconstructSSA(FunctionPass):
    """Perform SSA-reconstruction

    Produces minimal SSA.
    """
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...
    def _patch_locals(self, state) -> None: ...

class RewriteDynamicRaises(FunctionPass):
    """Replace existing raise statements by dynamic raises in Numba IR.
    """
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...
