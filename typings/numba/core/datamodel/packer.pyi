from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types

class DataPacker:
    """
    A helper to pack a number of typed arguments into a data structure.
    Omitted arguments (i.e. values with the type `Omitted`) are automatically
    skipped.
    """
    _dmm: Incomplete
    _fe_types: Incomplete
    _models: Incomplete
    _pack_map: Incomplete
    _be_types: Incomplete
    def __init__(self, dmm, fe_types) -> None: ...
    def as_data(self, builder, values):
        """
        Return the given values packed as a data structure.
        """
    def _do_load(self, builder, ptr, formal_list: Incomplete | None = None): ...
    def load(self, builder, ptr):
        """
        Load the packed values and return a (type, value) tuples.
        """
    def load_into(self, builder, ptr, formal_list) -> None:
        """
        Load the packed values into a sequence indexed by formal
        argument number (skipping any Omitted position).
        """

class ArgPacker:
    """
    Compute the position for each high-level typed argument.
    It flattens every composite argument into primitive types.
    It maintains a position map for unflattening the arguments.

    Since struct (esp. nested struct) have specific ABI requirements (e.g.
    alignment, pointer address-space, ...) in different architecture (e.g.
    OpenCL, CUDA), flattening composite argument types simplifes the call
    setup from the Python side.  Functions are receiving simple primitive
    types and there are only a handful of these.
    """
    _dmm: Incomplete
    _fe_args: Incomplete
    _nargs: Incomplete
    _dm_args: Incomplete
    _unflattener: Incomplete
    _be_args: Incomplete
    def __init__(self, dmm, fe_args) -> None: ...
    def as_arguments(self, builder, values):
        """Flatten all argument values
        """
    def from_arguments(self, builder, args):
        """Unflatten all argument values
        """
    def assign_names(self, args, names) -> None:
        """Assign names for each flattened argument values.
        """
    def _assign_names(self, val_or_nested, name, depth=()) -> None: ...
    @property
    def argument_types(self):
        """Return a list of LLVM types that are results of flattening
        composite types.
        """

def _flatten(iterable):
    """
    Flatten nested iterable of (tuple, list).
    """

_PUSH_LIST: int
_APPEND_NEXT_VALUE: int
_APPEND_EMPTY_TUPLE: int
_POP: int

class _Unflattener:
    """
    An object used to unflatten nested sequences after a given pattern
    (an arbitrarily nested sequence).
    The pattern shows the nested sequence shape desired when unflattening;
    the values it contains are irrelevant.
    """
    _code: Incomplete
    def __init__(self, pattern) -> None: ...
    def _build_unflatten_code(self, iterable):
        """Build the unflatten opcode sequence for the given *iterable* structure
        (an iterable of nested sequences).
        """
    def unflatten(self, flatiter):
        """Rebuild a nested tuple structure.
        """
