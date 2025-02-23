from _typeshed import Incomplete
from numba.core import errors as errors, ir as ir, ir_utils as ir_utils
from numba.core.analysis import compute_cfg_from_blocks as compute_cfg_from_blocks, compute_use_defs as compute_use_defs, find_top_level_loops as find_top_level_loops
from numba.core.utils import PYVERSION as PYVERSION
from typing import NamedTuple

_logger: Incomplete

def _extract_loop_lifting_candidates(cfg, blocks):
    """
    Returns a list of loops that are candidate for loop lifting
    """
def find_region_inout_vars(blocks, livemap, callfrom, returnto, body_block_ids):
    """Find input and output variables to a block region.
    """

class _loop_lift_info(NamedTuple):
    loop: Incomplete
    inputs: Incomplete
    outputs: Incomplete
    callfrom: Incomplete
    returnto: Incomplete

def _loop_lift_get_candidate_infos(cfg, blocks, livemap):
    """
    Returns information on looplifting candidates.
    """
def _loop_lift_modify_call_block(liftedloop, block, inputs, outputs, returnto):
    """
    Transform calling block from top-level function to call the lifted loop.
    """
def _loop_lift_prepare_loop_func(loopinfo, blocks) -> None:
    """
    Inplace transform loop blocks for use as lifted loop.
    """
def _loop_lift_modify_blocks(func_ir, loopinfo, blocks, typingctx, targetctx, flags, locals):
    """
    Modify the block inplace to call to the lifted-loop.
    Returns a dictionary of blocks of the lifted-loop.
    """
def _has_multiple_loop_exits(cfg, lpinfo):
    '''Returns True if there is more than one exit in the loop.

    NOTE: "common exits" refers to the situation where a loop exit has another
    loop exit as its successor. In that case, we do not need to alter it.
    '''
def _pre_looplift_transform(func_ir):
    """Canonicalize loops for looplifting.
    """
def loop_lifting(func_ir, typingctx, targetctx, flags, locals):
    """
    Loop lifting transformation.

    Given a interpreter `func_ir` returns a 2 tuple of
    `(toplevel_interp, [loop0_interp, loop1_interp, ....])`
    """
def canonicalize_cfg_single_backedge(blocks):
    """
    Rewrite loops that have multiple backedges.
    """
def canonicalize_cfg(blocks):
    """
    Rewrite the given blocks to canonicalize the CFG.
    Returns a new dictionary of blocks.
    """
def with_lifting(func_ir, typingctx, targetctx, flags, locals):
    """With-lifting transformation

    Rewrite the IR to extract all withs.
    Only the top-level withs are extracted.
    Returns the (the_new_ir, the_lifted_with_ir)
    """
def _get_with_contextmanager(func_ir, blocks, blk_start):
    """Get the global object used for the context manager
    """
def _legalize_with_head(blk) -> None:
    """Given *blk*, the head block of the with-context, check that it doesn't
    do anything else.
    """
def _cfg_nodes_in_region(cfg, region_begin, region_end):
    """Find the set of CFG nodes that are in the given region
    """
def find_setupwiths(func_ir):
    """Find all top-level with.

    Returns a list of ranges for the with-regions.
    """
def _rewrite_return(func_ir, target_block_label):
    """Rewrite a return block inside a with statement.

    Arguments
    ---------

    func_ir: Function IR
      the CFG to transform
    target_block_label: int
      the block index/label of the block containing the POP_BLOCK statement


    This implements a CFG transformation to insert a block between two other
    blocks.

    The input situation is:

    ┌───────────────┐
    │   top         │
    │   POP_BLOCK   │
    │   bottom      │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │               │
    │    RETURN     │
    │               │
    └───────────────┘

    If such a pattern is detected in IR, it means there is a `return` statement
    within a `with` context. The basic idea is to rewrite the CFG as follows:

    ┌───────────────┐
    │   top         │
    │   POP_BLOCK   │
    │               │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │               │
    │     bottom    │
    │               │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │               │
    │    RETURN     │
    │               │
    └───────────────┘

    We split the block that contains the `POP_BLOCK` statement into two blocks.
    Everything from the beginning of the block up to and including the
    `POP_BLOCK` statement is considered the 'top' and everything below is
    considered 'bottom'. Finally the jump statements are re-wired to make sure
    the CFG remains valid.

    """
def _eliminate_nested_withs(with_ranges): ...
def consolidate_multi_exit_withs(withs: dict, blocks, func_ir):
    """Modify the FunctionIR to merge the exit blocks of with constructs.
    """
def _fix_multi_exit_blocks(func_ir, exit_nodes, *, split_condition: Incomplete | None = None):
    """Modify the FunctionIR to create a single common exit node given the
    original exit nodes.

    Parameters
    ----------
    func_ir :
        The FunctionIR. Mutated inplace.
    exit_nodes :
        The original exit nodes. A sequence of block keys.
    split_condition : callable or None
        If not None, it is a callable with the signature
        `split_condition(statement)` that determines if the `statement` is the
        splitting point (e.g. `POP_BLOCK`) in an exit node.
        If it's None, the exit node is not split.
    """
