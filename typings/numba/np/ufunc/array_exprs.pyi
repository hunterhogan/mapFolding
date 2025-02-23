import ast
from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import compiler as compiler, ir as ir, rewrites as rewrites, targetconfig as targetconfig, types as types
from numba.core.typing import npydecl as npydecl
from numba.np.ufunc.dufunc import DUFunc as DUFunc

def _is_ufunc(func): ...

class RewriteArrayExprs(rewrites.Rewrite):
    """The RewriteArrayExprs class is responsible for finding array
    expressions in Numba intermediate representation code, and
    rewriting those expressions to a single operation that will expand
    into something similar to a ufunc call.
    """
    def __init__(self, state, *args, **kws) -> None: ...
    crnt_block: Incomplete
    typemap: Incomplete
    array_assigns: Incomplete
    const_assigns: Incomplete
    def match(self, func_ir, block, typemap, calltypes):
        """
        Using typing and a basic block, search the basic block for array
        expressions.
        Return True when one or more matches were found, False otherwise.
        """
    def _match_array_expr(self, instr, expr, target_name) -> None:
        """
        Find whether the given assignment (*instr*) of an expression (*expr*)
        to variable *target_name* is an array expression.
        """
    def _has_explicit_output(self, expr, func):
        """
        Return whether the *expr* call to *func* (a ufunc) features an
        explicit output argument.
        """
    def _get_array_operator(self, ir_expr): ...
    def _get_operands(self, ir_expr):
        """Given a Numba IR expression, return the operands to the expression
        in order they appear in the expression.
        """
    def _translate_expr(self, ir_expr):
        """Translate the given expression from Numba IR to an array expression
        tree.
        """
    def _handle_matches(self):
        """Iterate over the matches, trying to find which instructions should
        be rewritten, deleted, or moved.
        """
    def _get_final_replacement(self, replacement_map, instr):
        """Find the final replacement instruction for a given initial
        instruction by chasing instructions in a map from instructions
        to replacement instructions.
        """
    def apply(self):
        """When we've found array expressions in a basic block, rewrite that
        block, returning a new, transformed block.
        """

_unaryops: Incomplete
_binops: Incomplete
_cmpops: Incomplete

def _arr_expr_to_ast(expr):
    """Build a Python expression AST from an array expression built by
    RewriteArrayExprs.
    """
def _legalize_parameter_names(var_list) -> Generator[Incomplete]:
    """
    Legalize names in the variable list for use as a Python function's
    parameter names.
    """

class _EraseInvalidLineRanges(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST) -> ast.AST: ...

def _fix_invalid_lineno_ranges(astree: ast.AST):
    """Inplace fixes invalid lineno ranges.
    """
def _lower_array_expr(lowerer, expr):
    """Lower an array expression built by RewriteArrayExprs.
    """
