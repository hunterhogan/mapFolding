from _typeshed import Incomplete
from numba.core import errors as errors, types as types
from numba.core.typing.typeof import typeof as typeof

class AsNumbaTypeRegistry:
    """
    A registry for python typing declarations.  This registry stores a lookup
    table for simple cases (e.g. int) and a list of functions for more
    complicated cases (e.g. generics like List[int]).

    The as_numba_type registry is meant to work statically on type annotations
    at compile type, not dynamically on instances at runtime. To check the type
    of an object at runtime, see numba.typeof.
    """
    lookup: Incomplete
    functions: Incomplete
    def __init__(self) -> None: ...
    def _numba_type_infer(self, py_type): ...
    def _builtin_infer(self, py_type): ...
    def register(self, func_or_py_type, numba_type: Incomplete | None = None) -> None:
        """
        Extend AsNumbaType to support new python types (e.g. a user defined
        JitClass).  For a simple pair of a python type and a numba type, can
        use as a function register(py_type, numba_type).  If more complex logic
        is required (e.g. for generic types), register can also be used as a
        decorator for a function that takes a python type as input and returns
        a numba type or None.
        """
    def try_infer(self, py_type):
        """
        Try to determine the numba type of a given python type.
        We first consider the lookup dictionary.  If py_type is not there, we
        iterate through the registered functions until one returns a numba type.
        If type inference fails, return None.
        """
    def infer(self, py_type): ...
    def __call__(self, py_type): ...

as_numba_type: Incomplete
