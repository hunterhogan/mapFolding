from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import config as config, consts as consts, errors as errors
from numba.core.errors import ConstantInferenceError as ConstantInferenceError, NotDefinedError as NotDefinedError, RedefinedError as RedefinedError, VerificationError as VerificationError
from numba.core.utils import BINOPS_TO_OPERATORS as BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS as INPLACE_BINOPS_TO_OPERATORS, OPERATORS_TO_BUILTINS as OPERATORS_TO_BUILTINS, UNARY_BUITINS_TO_OPERATORS as UNARY_BUITINS_TO_OPERATORS

_termcolor: Incomplete

class Loc:
    """Source location

    """
    _defmatcher: Incomplete
    filename: Incomplete
    line: Incomplete
    col: Incomplete
    lines: Incomplete
    maybe_decorator: Incomplete
    def __init__(self, filename, line, col: Incomplete | None = None, maybe_decorator: bool = False) -> None:
        """ Arguments:
        filename - name of the file
        line - line in file
        col - column
        maybe_decorator - Set to True if location is likely a jit decorator
        """
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    @classmethod
    def from_function_id(cls, func_id): ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _find_definition(self): ...
    def _raw_function_name(self): ...
    def get_lines(self): ...
    def _get_path(self): ...
    def strformat(self, nlines_up: int = 2): ...
    def with_lineno(self, line, col: Incomplete | None = None):
        """
        Return a new Loc with this line number.
        """
    def short(self):
        """
        Returns a short string
        """

unknown_loc: Incomplete

class SlotEqualityCheckMixin:
    __slots__: Incomplete
    def __eq__(self, other): ...
    def __le__(self, other): ...
    def __hash__(self): ...

class EqualityCheckMixin:
    """ Mixin for basic equality checking """
    def __eq__(self, other): ...
    def __le__(self, other): ...
    def __hash__(self): ...

class VarMap:
    _con: Incomplete
    def __init__(self) -> None: ...
    def define(self, name, var) -> None: ...
    def get(self, name): ...
    def __contains__(self, name) -> bool: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __hash__(self): ...
    def __iter__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...

class AbstractRHS:
    """Abstract base class for anything that can be the RHS of an assignment.
    This class **does not** define any methods.
    """

class Inst(EqualityCheckMixin, AbstractRHS):
    """
    Base class for all IR instructions.
    """
    def list_vars(self) -> None:
        """
        List the variables used (read or written) by the instruction.
        """
    def _rec_list_vars(self, val):
        """
        A recursive helper used to implement list_vars() in subclasses.
        """

class Stmt(Inst):
    """
    Base class for IR statements (instructions which can appear on their
    own in a Block).
    """
    is_terminator: bool
    is_exit: bool
    def list_vars(self): ...

class Terminator(Stmt):
    """
    IR statements that are terminators: the last statement in a block.
    A terminator must either:
    - exit the function
    - jump to a block

    All subclass of Terminator must override `.get_targets()` to return a list
    of jump targets.
    """
    is_terminator: bool
    def get_targets(self) -> None: ...

class Expr(Inst):
    """
    An IR expression (an instruction which can only be part of a larger
    statement).
    """
    op: Incomplete
    loc: Incomplete
    _kws: Incomplete
    def __init__(self, op, loc, **kws) -> None: ...
    def __getattr__(self, name): ...
    def __setattr__(self, name, value) -> None: ...
    @classmethod
    def binop(cls, fn, lhs, rhs, loc): ...
    @classmethod
    def inplace_binop(cls, fn, immutable_fn, lhs, rhs, loc): ...
    @classmethod
    def unary(cls, fn, value, loc): ...
    @classmethod
    def call(cls, func, args, kws, loc, vararg: Incomplete | None = None, varkwarg: Incomplete | None = None, target: Incomplete | None = None): ...
    @classmethod
    def build_tuple(cls, items, loc): ...
    @classmethod
    def build_list(cls, items, loc): ...
    @classmethod
    def build_set(cls, items, loc): ...
    @classmethod
    def build_map(cls, items, size, literal_value, value_indexes, loc): ...
    @classmethod
    def pair_first(cls, value, loc): ...
    @classmethod
    def pair_second(cls, value, loc): ...
    @classmethod
    def getiter(cls, value, loc): ...
    @classmethod
    def iternext(cls, value, loc): ...
    @classmethod
    def exhaust_iter(cls, value, count, loc): ...
    @classmethod
    def getattr(cls, value, attr, loc): ...
    @classmethod
    def getitem(cls, value, index, loc): ...
    @classmethod
    def typed_getitem(cls, value, dtype, index, loc): ...
    @classmethod
    def static_getitem(cls, value, index, index_var, loc): ...
    @classmethod
    def cast(cls, value, loc):
        """
        A node for implicit casting at the return statement
        """
    @classmethod
    def phi(cls, loc):
        """Phi node
        """
    @classmethod
    def make_function(cls, name, code, closure, defaults, loc):
        """
        A node for making a function object.
        """
    @classmethod
    def null(cls, loc):
        """
        A node for null value.

        This node is not handled by type inference. It is only added by
        post-typing passes.
        """
    @classmethod
    def undef(cls, loc):
        """
        A node for undefined value specifically from LOAD_FAST_AND_CLEAR opcode.
        """
    @classmethod
    def dummy(cls, op, info, loc):
        """
        A node for a dummy value.

        This node is a place holder for carrying information through to a point
        where it is rewritten into something valid. This node is not handled
        by type inference or lowering. It's presence outside of the interpreter
        renders IR as illegal.
        """
    def __repr__(self) -> str: ...
    def list_vars(self): ...
    def infer_constant(self) -> None: ...

class SetItem(Stmt):
    """
    target[index] = value
    """
    target: Incomplete
    index: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, target, index, value, loc) -> None: ...
    def __repr__(self) -> str: ...

class StaticSetItem(Stmt):
    """
    target[constant index] = value
    """
    target: Incomplete
    index: Incomplete
    index_var: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, target, index, index_var, value, loc) -> None: ...
    def __repr__(self) -> str: ...

class DelItem(Stmt):
    """
    del target[index]
    """
    target: Incomplete
    index: Incomplete
    loc: Incomplete
    def __init__(self, target, index, loc) -> None: ...
    def __repr__(self) -> str: ...

class SetAttr(Stmt):
    target: Incomplete
    attr: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, target, attr, value, loc) -> None: ...
    def __repr__(self) -> str: ...

class DelAttr(Stmt):
    target: Incomplete
    attr: Incomplete
    loc: Incomplete
    def __init__(self, target, attr, loc) -> None: ...
    def __repr__(self) -> str: ...

class StoreMap(Stmt):
    dct: Incomplete
    key: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, dct, key, value, loc) -> None: ...
    def __repr__(self) -> str: ...

class Del(Stmt):
    value: Incomplete
    loc: Incomplete
    def __init__(self, value, loc) -> None: ...
    def __str__(self) -> str: ...

class Raise(Terminator):
    is_exit: bool
    exception: Incomplete
    loc: Incomplete
    def __init__(self, exception, loc) -> None: ...
    def __str__(self) -> str: ...
    def get_targets(self): ...

class StaticRaise(Terminator):
    '''
    Raise an exception class and arguments known at compile-time.
    Note that if *exc_class* is None, a bare "raise" statement is implied
    (i.e. re-raise the current exception).
    '''
    is_exit: bool
    exc_class: Incomplete
    exc_args: Incomplete
    loc: Incomplete
    def __init__(self, exc_class, exc_args, loc) -> None: ...
    def __str__(self) -> str: ...
    def get_targets(self): ...

class DynamicRaise(Terminator):
    '''
    Raise an exception class and some argument *values* unknown at compile-time.
    Note that if *exc_class* is None, a bare "raise" statement is implied
    (i.e. re-raise the current exception).
    '''
    is_exit: bool
    exc_class: Incomplete
    exc_args: Incomplete
    loc: Incomplete
    def __init__(self, exc_class, exc_args, loc) -> None: ...
    def __str__(self) -> str: ...
    def get_targets(self): ...

class TryRaise(Stmt):
    """A raise statement inside a try-block
    Similar to ``Raise`` but does not terminate.
    """
    exception: Incomplete
    loc: Incomplete
    def __init__(self, exception, loc) -> None: ...
    def __str__(self) -> str: ...

class StaticTryRaise(Stmt):
    """A raise statement inside a try-block.
    Similar to ``StaticRaise`` but does not terminate.
    """
    exc_class: Incomplete
    exc_args: Incomplete
    loc: Incomplete
    def __init__(self, exc_class, exc_args, loc) -> None: ...
    def __str__(self) -> str: ...

class DynamicTryRaise(Stmt):
    """A raise statement inside a try-block.
    Similar to ``DynamicRaise`` but does not terminate.
    """
    exc_class: Incomplete
    exc_args: Incomplete
    loc: Incomplete
    def __init__(self, exc_class, exc_args, loc) -> None: ...
    def __str__(self) -> str: ...

class Return(Terminator):
    """
    Return to caller.
    """
    is_exit: bool
    value: Incomplete
    loc: Incomplete
    def __init__(self, value, loc) -> None: ...
    def __str__(self) -> str: ...
    def get_targets(self): ...

class Jump(Terminator):
    """
    Unconditional branch.
    """
    target: Incomplete
    loc: Incomplete
    def __init__(self, target, loc) -> None: ...
    def __str__(self) -> str: ...
    def get_targets(self): ...

class Branch(Terminator):
    """
    Conditional branch.
    """
    cond: Incomplete
    truebr: Incomplete
    falsebr: Incomplete
    loc: Incomplete
    def __init__(self, cond, truebr, falsebr, loc) -> None: ...
    def __str__(self) -> str: ...
    def get_targets(self): ...

class Assign(Stmt):
    """
    Assign to a variable.
    """
    value: Incomplete
    target: Incomplete
    loc: Incomplete
    def __init__(self, value, target, loc) -> None: ...
    def __str__(self) -> str: ...

class Print(Stmt):
    """
    Print some values.
    """
    args: Incomplete
    vararg: Incomplete
    consts: Incomplete
    loc: Incomplete
    def __init__(self, args, vararg, loc) -> None: ...
    def __str__(self) -> str: ...

class Yield(Inst):
    value: Incomplete
    loc: Incomplete
    index: Incomplete
    def __init__(self, value, loc, index) -> None: ...
    def __str__(self) -> str: ...
    def list_vars(self): ...

class EnterWith(Stmt):
    '''Enter a "with" context
    '''
    contextmanager: Incomplete
    begin: Incomplete
    end: Incomplete
    loc: Incomplete
    def __init__(self, contextmanager, begin, end, loc) -> None:
        """
        Parameters
        ----------
        contextmanager : IR value
        begin, end : int
            The beginning and the ending offset of the with-body.
        loc : ir.Loc instance
            Source location
        """
    def __str__(self) -> str: ...
    def list_vars(self): ...

class PopBlock(Stmt):
    """Marker statement for a pop block op code"""
    loc: Incomplete
    def __init__(self, loc) -> None: ...
    def __str__(self) -> str: ...

class Arg(EqualityCheckMixin, AbstractRHS):
    name: Incomplete
    index: Incomplete
    loc: Incomplete
    def __init__(self, name, index, loc) -> None: ...
    def __repr__(self) -> str: ...
    def infer_constant(self) -> None: ...

class Const(EqualityCheckMixin, AbstractRHS):
    value: Incomplete
    loc: Incomplete
    use_literal_type: Incomplete
    def __init__(self, value, loc, use_literal_type: bool = True) -> None: ...
    def __repr__(self) -> str: ...
    def infer_constant(self): ...
    def __deepcopy__(self, memo): ...

class Global(EqualityCheckMixin, AbstractRHS):
    name: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, name, value, loc) -> None: ...
    def __str__(self) -> str: ...
    def infer_constant(self): ...
    def __deepcopy__(self, memo): ...

class FreeVar(EqualityCheckMixin, AbstractRHS):
    """
    A freevar, as loaded by LOAD_DECREF.
    (i.e. a variable defined in an enclosing non-global scope)
    """
    index: Incomplete
    name: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, index, name, value, loc) -> None: ...
    def __str__(self) -> str: ...
    def infer_constant(self): ...
    def __deepcopy__(self, memo): ...

class Var(EqualityCheckMixin, AbstractRHS):
    """
    Attributes
    -----------
    - scope: Scope

    - name: str

    - loc: Loc
        Definition location
    """
    scope: Incomplete
    name: Incomplete
    loc: Incomplete
    def __init__(self, scope, name, loc) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def is_temp(self): ...
    @property
    def unversioned_name(self):
        """The unversioned name of this variable, i.e. SSA renaming removed
        """
    @property
    def versioned_names(self):
        """Known versioned names for this variable, i.e. known variable names in
        the scope that have been formed from applying SSA to this variable
        """
    @property
    def all_names(self):
        """All known versioned and unversioned names for this variable
        """
    def __deepcopy__(self, memo): ...

class Scope(EqualityCheckMixin):
    """
    Attributes
    -----------
    - parent: Scope
        Parent scope

    - localvars: VarMap
        Scope-local variable map

    - loc: Loc
        Start of scope location

    """
    parent: Incomplete
    localvars: Incomplete
    loc: Incomplete
    redefined: Incomplete
    var_redefinitions: Incomplete
    def __init__(self, parent, loc) -> None: ...
    def define(self, name, loc):
        """
        Define a variable
        """
    def get(self, name):
        """
        Refer to a variable.  Returns the latest version.
        """
    def get_exact(self, name):
        """
        Refer to a variable.  The returned variable has the exact
        name (exact variable version).
        """
    def get_or_define(self, name, loc): ...
    def redefine(self, name, loc, rename: bool = True):
        """
        Redefine if the name is already defined
        """
    def get_versions_of(self, name):
        """
        Gets all known versions of a given name
        """
    def make_temp(self, loc): ...
    @property
    def has_parent(self): ...
    def __repr__(self) -> str: ...

class Block(EqualityCheckMixin):
    """A code block

    """
    scope: Incomplete
    body: Incomplete
    loc: Incomplete
    def __init__(self, scope, loc) -> None: ...
    def copy(self): ...
    def find_exprs(self, op: Incomplete | None = None) -> Generator[Incomplete]:
        """
        Iterate over exprs of the given *op* in this block.
        """
    def find_insts(self, cls: Incomplete | None = None) -> Generator[Incomplete]:
        """
        Iterate over insts of the given class in this block.
        """
    def find_variable_assignment(self, name):
        '''
        Returns the assignment inst associated with variable "name", None if
        it cannot be found.
        '''
    def prepend(self, inst) -> None: ...
    def append(self, inst) -> None: ...
    def remove(self, inst) -> None: ...
    def clear(self) -> None: ...
    def dump(self, file: Incomplete | None = None) -> None: ...
    @property
    def terminator(self): ...
    @property
    def is_terminated(self): ...
    def verify(self) -> None: ...
    def insert_after(self, stmt, other) -> None:
        """
        Insert *stmt* after *other*.
        """
    def insert_before_terminator(self, stmt) -> None: ...
    def __repr__(self) -> str: ...

class Loop(SlotEqualityCheckMixin):
    """Describes a loop-block
    """
    __slots__: Incomplete
    entry: Incomplete
    exit: Incomplete
    def __init__(self, entry, exit) -> None: ...
    def __repr__(self) -> str: ...

class With(SlotEqualityCheckMixin):
    """Describes a with-block
    """
    __slots__: Incomplete
    entry: Incomplete
    exit: Incomplete
    def __init__(self, entry, exit) -> None: ...
    def __repr__(self) -> str: ...

class FunctionIR:
    blocks: Incomplete
    is_generator: Incomplete
    func_id: Incomplete
    loc: Incomplete
    arg_count: Incomplete
    arg_names: Incomplete
    _definitions: Incomplete
    def __init__(self, blocks, is_generator, func_id, loc, definitions, arg_count, arg_names) -> None: ...
    def equal_ir(self, other):
        """ Checks that the IR contained within is equal to the IR in other.
        Equality is defined by being equal in fundamental structure (blocks,
        labels, IR node type and the order in which they are defined) and the
        IR nodes being equal. IR node equality essentially comes down to
        ensuring a node's `.__dict__` or `.__slots__` is equal, with the
        exception of ignoring 'loc' and 'scope' entries. The upshot is that the
        comparison is essentially location and scope invariant, but otherwise
        behaves as unsurprisingly as possible.
        """
    def diff_str(self, other):
        """
        Compute a human readable difference in the IR, returns a formatted
        string ready for printing.
        """
    _consts: Incomplete
    generator_info: Incomplete
    variable_lifetime: Incomplete
    block_entry_vars: Incomplete
    def _reset_analysis_variables(self) -> None: ...
    def derive(self, blocks, arg_count: Incomplete | None = None, arg_names: Incomplete | None = None, force_non_generator: bool = False, loc: Incomplete | None = None):
        """
        Derive a new function IR from this one, using the given blocks,
        and possibly modifying the argument count and generator flag.

        Post-processing will have to be run again on the new IR.
        """
    def copy(self): ...
    def get_block_entry_vars(self, block):
        """
        Return a set of variable names possibly alive at the beginning of
        the block.
        """
    def infer_constant(self, name):
        """
        Try to infer the constant value of a given variable.
        """
    def get_definition(self, value, lhs_only: bool = False):
        """
        Get the definition site for the given variable name or instance.
        A Expr instance is returned by default, but if lhs_only is set
        to True, the left-hand-side variable is returned instead.
        """
    def get_assignee(self, rhs_value, in_blocks: Incomplete | None = None):
        """
        Finds the assignee for a given RHS value. If in_blocks is given the
        search will be limited to the specified blocks.
        """
    def dump(self, file: Incomplete | None = None) -> None: ...
    def dump_to_string(self): ...
    def dump_generator_info(self, file: Incomplete | None = None) -> None: ...
    def render_dot(self, filename_prefix: str = 'numba_ir', include_ir: bool = True):
        """Render the CFG of the IR with GraphViz DOT via the
        ``graphviz`` python binding.

        Returns
        -------
        g : graphviz.Digraph
            Use `g.view()` to open the graph in the default PDF application.
        """

class UndefinedType(EqualityCheckMixin):
    _singleton: Incomplete
    def __new__(cls): ...
    def __repr__(self) -> str: ...

UNDEFINED: Incomplete
