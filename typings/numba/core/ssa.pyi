from _typeshed import Incomplete
from collections.abc import Generator
from numba import config as config
from numba.core import errors as errors, ir as ir, ir_utils as ir_utils
from numba.core.analysis import compute_cfg_from_blocks as compute_cfg_from_blocks
from numba.core.utils import OrderedSet as OrderedSet, _lazy_pformat as _lazy_pformat

_logger: Incomplete

def reconstruct_ssa(func_ir):
    """Apply SSA reconstruction algorithm on the given IR.

    Produces minimal SSA using Choi et al algorithm.
    """

class _CacheListVars:
    _saved: Incomplete
    def __init__(self) -> None: ...
    def get(self, inst): ...

def _run_ssa(blocks):
    """Run SSA reconstruction on IR blocks of a function.
    """
def _fix_ssa_vars(blocks, varname, defmap, cfg, df_plus, cache_list_vars):
    """Rewrite all uses to ``varname`` given the definition map
    """
def _iterated_domfronts(cfg):
    """Compute the iterated dominance frontiers (DF+ in literatures).

    Returns a dictionary which maps block label to the set of labels of its
    iterated dominance frontiers.
    """
def _compute_phi_locations(iterated_df, defmap): ...
def _fresh_vars(blocks, varname):
    """Rewrite to put fresh variable names
    """
def _get_scope(blocks): ...
def _find_defs_violators(blocks, cfg):
    """
    Returns
    -------
    res : Set[str]
        The SSA violators in a dictionary of variable names.
    """
def _run_block_analysis(blocks, states, handler) -> None: ...
def _run_block_rewrite(blocks, states, handler): ...
def _make_states(blocks): ...
def _run_ssa_block_pass(states, blk, handler) -> Generator[Incomplete]: ...

class _BaseHandler:
    """A base handler for all the passes used here for the SSA algorithm.
    """
    def on_assign(self, states, assign) -> None:
        """
        Called when the pass sees an ``ir.Assign``.

        Subclasses should override this for custom behavior

        Parameters
        -----------
        states : dict
        assign : numba.ir.Assign

        Returns
        -------
        stmt : numba.ir.Assign or None
            For rewrite passes, the return value is used as the replacement
            for the given statement.
        """
    def on_other(self, states, stmt) -> None:
        """
        Called when the pass sees an ``ir.Stmt`` that's not an assignment.

        Subclasses should override this for custom behavior

        Parameters
        -----------
        states : dict
        assign : numba.ir.Stmt

        Returns
        -------
        stmt : numba.ir.Stmt or None
            For rewrite passes, the return value is used as the replacement
            for the given statement.
        """

class _GatherDefsHandler(_BaseHandler):
    '''Find all defs and uses of variable in each block

    ``states["label"]`` is a int; label of the current block
    ``states["defs"]`` is a Mapping[str, List[Tuple[ir.Assign, int]]]:
        - a mapping of the name of the assignee variable to the assignment
          IR node and the block label.
    ``states["uses"]`` is a Mapping[Set[int]]
    '''
    def on_assign(self, states, assign) -> None: ...
    def on_other(self, states, stmt) -> None: ...

class UndefinedVariable:
    def __init__(self) -> None: ...
    target: Incomplete

class _FreshVarHandler(_BaseHandler):
    """Replaces assignment target with new fresh variables.
    """
    def on_assign(self, states, assign): ...
    def on_other(self, states, stmt): ...

class _FixSSAVars(_BaseHandler):
    """Replace variable uses in IR nodes to the correct reaching variable
    and introduce Phi nodes if necessary. This class contains the core of
    the SSA reconstruction algorithm.

    See Ch 5 of the Inria SSA book for reference. The method names used here
    are similar to the names used in the pseudocode in the book.
    """
    _cache_list_vars: Incomplete
    def __init__(self, cache_list_vars) -> None: ...
    def on_assign(self, states, assign): ...
    def on_other(self, states, stmt): ...
    def _fix_var(self, states, stmt, used_vars):
        """Fix all variable uses in ``used_vars``.
        """
    def _find_def(self, states, stmt):
        """Find definition of ``stmt`` for the statement ``stmt``
        """
    def _find_def_from_top(self, states, label, loc):
        """Find definition reaching block of ``label``.

        This method would look at all dominance frontiers.
        Insert phi node if necessary.
        """
    def _find_def_from_bottom(self, states, label, loc):
        """Find definition from within the block at ``label``.
        """
    def _stmt_index(self, defstmt, block, stop: int = -1):
        """Find the positional index of the statement at ``block``.

        Assumptions:
        - no two statements can point to the same object.
        """

def _warn_about_uninitialized_variable(varname, loc) -> None: ...
