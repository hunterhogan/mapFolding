import enum
from _typeshed import Incomplete
from functools import total_ordering as total_ordering

class Conversion(enum.IntEnum):
    """
    A conversion kind from one type to the other.  The enum members
    are ordered from stricter to looser.
    """
    exact = 1
    promote = 2
    safe = 3
    unsafe = 4
    nil = 99

class CastSet:
    """A set of casting rules.

    There is at most one rule per target type.
    """
    _rels: Incomplete
    def __init__(self) -> None: ...
    def insert(self, to, rel): ...
    def items(self): ...
    def get(self, item): ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __contains__(self, item) -> bool: ...
    def __iter__(self): ...
    def __getitem__(self, item): ...

class TypeGraph:
    """A graph that maintains the casting relationship of all types.

    This simplifies the definition of casting rules by automatically
    propagating the rules.
    """
    _forwards: Incomplete
    _backwards: Incomplete
    _callback: Incomplete
    def __init__(self, callback: Incomplete | None = None) -> None:
        """
        Args
        ----
        - callback: callable or None
            It is called for each new casting rule with
            (from_type, to_type, castrel).
        """
    def get(self, ty): ...
    def propagate(self, a, b, baserel) -> None: ...
    def insert_rule(self, a, b, rel) -> None: ...
    def promote(self, a, b) -> None: ...
    def safe(self, a, b) -> None: ...
    def unsafe(self, a, b) -> None: ...
