from _typeshed import Incomplete
from functools import cached_property as cached_property
from numba.core import cgutils as cgutils, generators as generators, ir as ir, types as types, utils as utils
from numba.core.errors import ForbiddenConstruct as ForbiddenConstruct, LoweringError as LoweringError, NumbaNotImplementedError as NumbaNotImplementedError
from numba.core.lowering import BaseLower as BaseLower

_unsupported_builtins: Incomplete

class _Undefined:
    """
    A sentinel value for undefined variable created by Expr.undef.
    """
    def __repr__(self) -> str: ...

_UNDEFINED: Incomplete
PYTHON_BINOPMAP: Incomplete
PYTHON_COMPAREOPMAP: Incomplete

class PyLower(BaseLower):
    GeneratorLower = generators.PyGeneratorLower
    _frozen_strings: Incomplete
    _live_vars: Incomplete
    def init(self) -> None: ...
    def pre_lower(self) -> None: ...
    def post_lower(self) -> None: ...
    def pre_block(self, block) -> None: ...
    def lower_inst(self, inst) -> None: ...
    @cached_property
    def _omitted_typobj(self):
        """Return a `OmittedArg` type instance as a LLVM value suitable for
        testing at runtime.
        """
    def lower_assign(self, inst):
        """
        The returned object must have a new reference
        """
    def lower_yield(self, inst): ...
    def lower_binop(self, expr, op, inplace: bool = False): ...
    def lower_expr(self, expr): ...
    def lower_const(self, const): ...
    def lower_global(self, name, value):
        """
        1) Check global scope dictionary.
        2) Check __builtins__.
            2a) is it a dictionary (for non __main__ module)
            2b) is it a module (for __main__ module)
        """
    def get_module_dict(self): ...
    def get_builtin_obj(self, name): ...
    def builtin_lookup(self, mod, name):
        """
        Args
        ----
        mod:
            The __builtins__ dictionary or module, as looked up in
            a module's globals.
        name: str
            The object to lookup
        """
    def check_occurred(self) -> None:
        """
        Return if an exception occurred.
        """
    def check_error(self, obj):
        """
        Return if *obj* is NULL.
        """
    def check_int_status(self, num, ok_value: int = 0) -> None:
        """
        Raise an exception if *num* is smaller than *ok_value*.
        """
    def is_null(self, obj): ...
    def return_exception_raised(self) -> None:
        """
        Return with the currently raised exception.
        """
    def init_vars(self, block) -> None:
        """
        Initialize live variables for *block*.
        """
    def _getvar(self, name, ltype: Incomplete | None = None): ...
    def loadvar(self, name):
        """
        Load the llvm value of the variable named *name*.
        """
    def delvar(self, name) -> None:
        """
        Delete the variable slot with the given name. This will decref
        the corresponding Python object.
        """
    def storevar(self, value, name, clobber: bool = False) -> None:
        """
        Stores a llvm value and allocate stack slot if necessary.
        The llvm value can be of arbitrary type.
        """
    def cleanup_vars(self) -> None:
        """
        Cleanup live variables.
        """
    def alloca(self, name, ltype: Incomplete | None = None):
        """
        Allocate a stack slot and initialize it to NULL.
        The default is to allocate a pyobject pointer.
        Use ``ltype`` to override.
        """
    def _alloca_var(self, name, fetype): ...
    def incref(self, value) -> None: ...
    def decref(self, value) -> None:
        """
        This is allow to be called on non pyobject pointer, in which case
        no code is inserted.
        """
    def _freeze_string(self, string):
        """
        Freeze a Python string object into the code.
        """
