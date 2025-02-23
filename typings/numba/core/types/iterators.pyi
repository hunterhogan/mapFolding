from ..errors import TypingError as TypingError
from .common import SimpleIterableType as SimpleIterableType, SimpleIteratorType as SimpleIteratorType
from _typeshed import Incomplete

class RangeType(SimpleIterableType):
    dtype: Incomplete
    _iterator_type: Incomplete
    def __init__(self, dtype) -> None: ...
    def unify(self, typingctx, other): ...

class RangeIteratorType(SimpleIteratorType):
    _yield_type: Incomplete
    def __init__(self, dtype) -> None: ...
    def unify(self, typingctx, other): ...

class Generator(SimpleIteratorType):
    """
    Type class for Numba-compiled generator objects.
    """
    gen_func: Incomplete
    arg_types: Incomplete
    state_types: Incomplete
    has_finalizer: Incomplete
    def __init__(self, gen_func, yield_type, arg_types, state_types, has_finalizer) -> None: ...
    @property
    def key(self): ...

class EnumerateType(SimpleIteratorType):
    """
    Type class for `enumerate` objects.
    Type instances are parametered with the underlying source type.
    """
    source_type: Incomplete
    def __init__(self, iterable_type) -> None: ...
    @property
    def key(self): ...

class ZipType(SimpleIteratorType):
    """
    Type class for `zip` objects.
    Type instances are parametered with the underlying source types.
    """
    source_types: Incomplete
    def __init__(self, iterable_types) -> None: ...
    @property
    def key(self): ...

class ArrayIterator(SimpleIteratorType):
    """
    Type class for iterators of array and buffer objects.
    """
    array_type: Incomplete
    def __init__(self, array_type) -> None: ...
