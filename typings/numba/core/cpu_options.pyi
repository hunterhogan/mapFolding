from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod

class AbstractOptionValue(metaclass=ABCMeta):
    """Abstract base class for custom option values.
    """
    @abstractmethod
    def encode(self) -> str:
        """Returns an encoding of the values
        """
    def __repr__(self) -> str: ...

class FastMathOptions(AbstractOptionValue):
    """
    Options for controlling fast math optimization.
    """
    flags: Incomplete
    def __init__(self, value) -> None: ...
    def __bool__(self) -> bool: ...
    __nonzero__ = __bool__
    def encode(self) -> str: ...
    def __eq__(self, other): ...

class ParallelOptions(AbstractOptionValue):
    """
    Options for controlling auto parallelization.
    """
    __slots__: Incomplete
    enabled: Incomplete
    comprehension: Incomplete
    reduction: Incomplete
    inplace_binop: Incomplete
    setitem: Incomplete
    numpy: Incomplete
    stencil: Incomplete
    fusion: Incomplete
    prange: Incomplete
    def __init__(self, value) -> None: ...
    def _get_values(self):
        """Get values as dictionary.
        """
    def __eq__(self, other): ...
    def encode(self) -> str: ...

class InlineOptions(AbstractOptionValue):
    """
    Options for controlling inlining
    """
    _inline: Incomplete
    def __init__(self, value) -> None: ...
    @property
    def is_never_inline(self):
        """
        True if never inline
        """
    @property
    def is_always_inline(self):
        """
        True if always inline
        """
    @property
    def has_cost_model(self):
        """
        True if a cost model is provided
        """
    @property
    def value(self):
        """
        The raw value
        """
    def __eq__(self, other): ...
    def encode(self) -> str: ...
