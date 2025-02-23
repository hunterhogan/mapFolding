from .typeof import Purpose as Purpose, typeof as typeof
from _typeshed import Incomplete
from collections.abc import Generator, Sequence
from numba.core import errors as errors, types as types, utils as utils
from numba.core.typeconv import Conversion as Conversion, rules as rules
from numba.core.typing import templates as templates
from numba.core.utils import order_by_target_specificity as order_by_target_specificity

class Rating:
    __slots__: Incomplete
    promote: int
    safe_convert: int
    unsafe_convert: int
    def __init__(self) -> None: ...
    def astuple(self):
        """Returns a tuple suitable for comparing with the worse situation
        start first.
        """
    def __add__(self, other): ...

class CallStack(Sequence):
    """
    A compile-time call stack
    """
    _stack: Incomplete
    _lock: Incomplete
    def __init__(self) -> None: ...
    def __getitem__(self, index):
        """
        Returns item in the stack where index=0 is the top and index=1 is
        the second item from the top.
        """
    def __len__(self) -> int: ...
    def register(self, target, typeinfer, func_id, args) -> Generator[None]: ...
    def finditer(self, py_func) -> Generator[Incomplete]:
        """
        Yields frame that matches the function object starting from the top
        of stack.
        """
    def findfirst(self, py_func):
        """
        Returns the first result from `.finditer(py_func)`; or None if no match.
        """
    def match(self, py_func, args):
        """
        Returns first function that matches *py_func* and the arguments types in
        *args*; or, None if no match.
        """

class CallFrame:
    """
    A compile-time call frame
    """
    typeinfer: Incomplete
    func_id: Incomplete
    args: Incomplete
    target: Incomplete
    _inferred_retty: Incomplete
    def __init__(self, target, typeinfer, func_id, args) -> None: ...
    def __repr__(self) -> str: ...
    def add_return_type(self, return_type) -> None:
        """Add *return_type* to the list of inferred return-types.
        If there are too many, raise `TypingError`.
        """

class BaseContext:
    """A typing context for storing function typing constrain template.
    """
    _registries: Incomplete
    _functions: Incomplete
    _attributes: Incomplete
    _globals: Incomplete
    tm: Incomplete
    callstack: Incomplete
    def __init__(self) -> None: ...
    def init(self) -> None:
        """
        Initialize the typing context.  Can be overridden by subclasses.
        """
    def refresh(self) -> None:
        """
        Refresh context with new declarations from known registries.
        Useful for third-party extensions.
        """
    def explain_function_type(self, func):
        """
        Returns a string description of the type of a function
        """
    def resolve_function_type(self, func, args, kws):
        """
        Resolve function type *func* for argument types *args* and *kws*.
        A signature is returned.
        """
    def _resolve_builtin_function_type(self, func, args, kws): ...
    def _resolve_user_function_type(self, func, args, kws, literals: Incomplete | None = None): ...
    def _get_attribute_templates(self, typ) -> Generator[Incomplete]:
        """
        Get matching AttributeTemplates for the Numba type.
        """
    def resolve_getattr(self, typ, attr):
        """
        Resolve getting the attribute *attr* (a string) on the Numba type.
        The attribute's type is returned, or None if resolution failed.
        """
    def find_matching_getattr_template(self, typ, attr): ...
    def resolve_setattr(self, target, attr, value):
        """
        Resolve setting the attribute *attr* (a string) on the *target* type
        to the given *value* type.
        A function signature is returned, or None if resolution failed.
        """
    def resolve_static_getitem(self, value, index): ...
    def resolve_static_setitem(self, target, index, value): ...
    def resolve_setitem(self, target, index, value): ...
    def resolve_delitem(self, target, index): ...
    def resolve_module_constants(self, typ, attr):
        """
        Resolve module-level global constants.
        Return None or the attribute type
        """
    def resolve_value_type(self, val):
        """
        Return the numba type of a Python value that is being used
        as a runtime constant.
        ValueError is raised for unsupported types.
        """
    def resolve_value_type_prefer_literal(self, value):
        """Resolve value type and prefer Literal types whenever possible.
        """
    def _get_global_type(self, gv): ...
    def _load_builtins(self) -> None: ...
    def load_additional_registries(self) -> None:
        """
        Load target-specific registries.  Can be overridden by subclasses.
        """
    def install_registry(self, registry):
        """
        Install a *registry* (a templates.Registry instance) of function,
        attribute and global declarations.
        """
    def _lookup_global(self, gv):
        """
        Look up the registered type for global value *gv*.
        """
    def _insert_global(self, gv, gty) -> None:
        """
        Register type *gty* for value *gv*.  Only a weak reference
        to *gv* is kept, if possible.
        """
    def _remove_global(self, gv) -> None:
        """
        Remove the registered type for global value *gv*.
        """
    def insert_global(self, gv, gty) -> None: ...
    def insert_attributes(self, at) -> None: ...
    def insert_function(self, ft) -> None: ...
    def insert_user_function(self, fn, ft) -> None:
        """Insert a user function.

        Args
        ----
        - fn:
            object used as callee
        - ft:
            function template
        """
    def can_convert(self, fromty, toty):
        """
        Check whether conversion is possible from *fromty* to *toty*.
        If successful, return a numba.typeconv.Conversion instance;
        otherwise None is returned.
        """
    def _rate_arguments(self, actualargs, formalargs, unsafe_casting: bool = True, exact_match_required: bool = False):
        """
        Rate the actual arguments for compatibility against the formal
        arguments.  A Rating instance is returned, or None if incompatible.
        """
    def install_possible_conversions(self, actualargs, formalargs):
        """
        Install possible conversions from the actual argument types to
        the formal argument types in the C++ type manager.
        Return True if all arguments can be converted.
        """
    def resolve_overload(self, key, cases, args, kws, allow_ambiguous: bool = True, unsafe_casting: bool = True, exact_match_required: bool = False):
        """
        Given actual *args* and *kws*, find the best matching
        signature in *cases*, or None if none matches.
        *key* is used for error reporting purposes.
        If *allow_ambiguous* is False, a tie in the best matches
        will raise an error.
        If *unsafe_casting* is False, unsafe casting is forbidden.
        """
    def unify_types(self, *typelist): ...
    def unify_pairs(self, first, second):
        """
        Try to unify the two given types.  A third type is returned,
        or None in case of failure.
        """

class Context(BaseContext):
    def load_additional_registries(self) -> None: ...
