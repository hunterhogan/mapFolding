from _typeshed import Incomplete
from functools import cached_property as cached_property
from numba.core import analysis as analysis, ir as ir, ir_utils as ir_utils, transforms as transforms

class YieldPoint:
    block: Incomplete
    inst: Incomplete
    live_vars: Incomplete
    weak_live_vars: Incomplete
    def __init__(self, block, inst) -> None: ...

class GeneratorInfo:
    yield_points: Incomplete
    state_vars: Incomplete
    def __init__(self) -> None: ...
    def get_yield_points(self):
        """
        Return an iterable of YieldPoint instances.
        """

class VariableLifetime:
    """
    For lazily building information of variable lifetime
    """
    _blocks: Incomplete
    def __init__(self, blocks) -> None: ...
    @cached_property
    def cfg(self): ...
    @cached_property
    def usedefs(self): ...
    @cached_property
    def livemap(self): ...
    @cached_property
    def deadmaps(self): ...

ir_extension_insert_dels: Incomplete

class PostProcessor:
    """
    A post-processor for Numba IR.
    """
    func_ir: Incomplete
    def __init__(self, func_ir) -> None: ...
    def run(self, emit_dels: bool = False, extend_lifetimes: bool = False):
        """
        Run the following passes over Numba IR:
        - canonicalize the CFG
        - emit explicit `del` instructions for variables
        - compute lifetime of variables
        - compute generator info (if function is a generator function)
        """
    def _populate_generator_info(self) -> None:
        """
        Fill `index` for the Yield instruction and create YieldPoints.
        """
    def _compute_generator_info(self) -> None:
        """
        Compute the generator's state variables as the union of live variables
        at all yield points.
        """
    def _insert_var_dels(self, extend_lifetimes: bool = False) -> None:
        """
        Insert del statements for each variable.
        Returns a 2-tuple of (variable definition map, variable deletion map)
        which indicates variables defined and deleted in each block.

        The algorithm avoids relying on explicit knowledge on loops and
        distinguish between variables that are defined locally vs variables that
        come from incoming blocks.
        We start with simple usage (variable reference) and definition (variable
        creation) maps on each block. Propagate the liveness info to predecessor
        blocks until it stabilize, at which point we know which variables must
        exist before entering each block. Then, we compute the end of variable
        lives and insert del statements accordingly. Variables are deleted after
        the last use. Variable referenced by terminators (e.g. conditional
        branch and return) are deleted by the successors or the caller.
        """
    def _patch_var_dels(self, internal_dead_map, escaping_dead_map, extend_lifetimes: bool = False) -> None:
        """
        Insert delete in each block
        """
    def remove_dels(self) -> None:
        """
        Strips the IR of Del nodes
        """
