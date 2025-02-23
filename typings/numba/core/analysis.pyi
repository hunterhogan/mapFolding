from .controlflow import CFGraph as CFGraph
from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import consts as consts, errors as errors, ir as ir, types as types
from numba.misc import special as special
from typing import NamedTuple

class _use_defs_result(NamedTuple):
    usemap: Incomplete
    defmap: Incomplete

ir_extension_usedefs: Incomplete

def compute_use_defs(blocks):
    """
    Find variable use/def per block.
    """
def compute_live_map(cfg, blocks, var_use_map, var_def_map):
    """
    Find variables that must be alive at the ENTRY of each block.
    We use a simple fix-point algorithm that iterates until the set of
    live variables is unchanged for each block.
    """

class _dead_maps_result(NamedTuple):
    internal: Incomplete
    escaping: Incomplete
    combined: Incomplete

def compute_dead_maps(cfg, blocks, live_map, var_def_map):
    """
    Compute the end-of-live information for variables.
    `live_map` contains a mapping of block offset to all the living
    variables at the ENTRY of the block.
    """
def compute_live_variables(cfg, blocks, var_def_map, var_dead_map):
    """
    Compute the live variables at the beginning of each block
    and at each yield point.
    The ``var_def_map`` and ``var_dead_map`` indicates the variable defined
    and deleted at each block, respectively.
    """
def compute_cfg_from_blocks(blocks): ...
def find_top_level_loops(cfg) -> Generator[Incomplete]:
    """
    A generator that yields toplevel loops given a control-flow-graph
    """
def _fix_loop_exit(cfg, loop):
    """
    Fixes loop.exits for Py3.8+ bytecode CFG changes.
    This is to handle `break` inside loops.
    """

class nullified(NamedTuple):
    condition: Incomplete
    taken_br: Incomplete
    rewrite_stmt: Incomplete

def dead_branch_prune(func_ir, called_args):
    """
    Removes dead branches based on constant inference from function args.
    This directly mutates the IR.

    func_ir is the IR
    called_args are the actual arguments with which the function is called
    """
def rewrite_semantic_constants(func_ir, called_args) -> None:
    """
    This rewrites values known to be constant by their semantics as ir.Const
    nodes, this is to give branch pruning the best chance possible of killing
    branches. An example might be rewriting len(tuple) as the literal length.

    func_ir is the IR
    called_args are the actual arguments with which the function is called
    """
def find_literally_calls(func_ir, argtypes) -> None:
    """An analysis to find `numba.literally` call inside the given IR.
    When an unsatisfied literal typing request is found, a `ForceLiteralArg`
    exception is raised.

    Parameters
    ----------

    func_ir : numba.ir.FunctionIR

    argtypes : Sequence[numba.types.Type]
        The argument types.
    """

ir_extension_use_alloca: Incomplete

def must_use_alloca(blocks):
    """
    Analyzes a dictionary of blocks to find variables that must be
    stack allocated with alloca.  For each statement in the blocks,
    determine if that statement requires certain variables to be
    stack allocated.  This function uses the extension point
    ir_extension_use_alloca to allow other IR node types like parfors
    to register to be processed by this analysis function.  At the
    moment, parfors are the only IR node types that may require
    something to be stack allocated.
    """
