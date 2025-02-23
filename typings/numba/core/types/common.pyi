from .abstract import ArrayCompatible as ArrayCompatible, Dummy as Dummy, IterableType as IterableType, IteratorType as IteratorType
from _typeshed import Incomplete
from numba.core.errors import NumbaTypeError as NumbaTypeError, NumbaValueError as NumbaValueError

class Opaque(Dummy):
    """
    A type that is a opaque pointer.
    """

class SimpleIterableType(IterableType):
    _iterator_type: Incomplete
    def __init__(self, name, iterator_type) -> None: ...
    @property
    def iterator_type(self): ...

class SimpleIteratorType(IteratorType):
    _yield_type: Incomplete
    def __init__(self, name, yield_type) -> None: ...
    @property
    def yield_type(self): ...

class Buffer(IterableType, ArrayCompatible):
    """
    Type class for objects providing the buffer protocol.
    Derived classes exist for more specific cases.
    """
    mutable: bool
    slice_is_copy: bool
    aligned: bool
    LAYOUTS: Incomplete
    dtype: Incomplete
    ndim: Incomplete
    layout: Incomplete
    def __init__(self, dtype, ndim, layout, readonly: bool = False, name: Incomplete | None = None) -> None: ...
    @property
    def iterator_type(self): ...
    @property
    def as_array(self): ...
    def copy(self, dtype: Incomplete | None = None, ndim: Incomplete | None = None, layout: Incomplete | None = None): ...
    @property
    def key(self): ...
    @property
    def is_c_contig(self): ...
    @property
    def is_f_contig(self): ...
    @property
    def is_contig(self): ...
