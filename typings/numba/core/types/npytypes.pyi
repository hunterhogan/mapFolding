from .abstract import DTypeSpec as DTypeSpec, IteratorType as IteratorType, MutableSequence as MutableSequence, Number as Number, Type as Type
from .common import Buffer as Buffer, Opaque as Opaque, SimpleIteratorType as SimpleIteratorType
from .containers import Bytes as Bytes
from .misc import UnicodeType as UnicodeType
from _typeshed import Incomplete
from functools import cached_property as cached_property
from numba.core import utils as utils
from numba.core.typeconv import Conversion as Conversion
from typing import NamedTuple

class CharSeq(Type):
    """
    A fixed-length 8-bit character sequence.
    """
    mutable: bool
    count: Incomplete
    def __init__(self, count) -> None: ...
    @property
    def key(self): ...
    def can_convert_from(self, typingctx, other): ...

class UnicodeCharSeq(Type):
    """
    A fixed-length unicode character sequence.
    """
    mutable: bool
    count: Incomplete
    def __init__(self, count) -> None: ...
    @property
    def key(self): ...
    def can_convert_to(self, typingctx, other): ...
    def can_convert_from(self, typingctx, other): ...
    def __repr__(self) -> str: ...

class _RecordField(NamedTuple):
    type: Incomplete
    offset: Incomplete
    alignment: Incomplete
    title: Incomplete

class Record(Type):
    """
    A Record datatype can be mapped to a NumPy structured dtype.
    A record is very flexible since it is laid out as a list of bytes.
    Fields can be mapped to arbitrary points inside it, even if they overlap.

    *fields* is a list of `(name:str, data:dict)`.
        Where `data` is `{ type: Type, offset: int }`
    *size* is an int; the record size
    *aligned* is a boolean; whether the record is ABI aligned.
    """
    mutable: bool
    @classmethod
    def make_c_struct(cls, name_types):
        """Construct a Record type from a list of (name:str, type:Types).
        The layout of the structure will follow C.

        Note: only scalar types are supported currently.
        """
    fields: Incomplete
    size: Incomplete
    aligned: Incomplete
    bitwidth: Incomplete
    def __init__(self, fields, size, aligned) -> None: ...
    @classmethod
    def _normalize_fields(cls, fields):
        """
        fields:
            [name: str,
             value: {
                 type: Type,
                 offset: int,
                 [ alignment: int ],
                 [ title : str],
             }]
        """
    @property
    def key(self): ...
    @property
    def mangling_args(self): ...
    def __len__(self) -> int:
        """Returns the number of fields
        """
    def offset(self, key):
        """Get the byte offset of a field from the start of the structure.
        """
    def typeof(self, key):
        """Get the type of a field.
        """
    def alignof(self, key):
        """Get the specified alignment of the field.

        Since field alignment is optional, this may return None.
        """
    def has_titles(self):
        """Returns True the record uses titles.
        """
    def is_title(self, key):
        """Returns True if the field named *key* is a title.
        """
    @property
    def members(self):
        """An ordered list of (name, type) for the fields.
        """
    @property
    def dtype(self): ...
    def can_convert_to(self, typingctx, other):
        """
        Convert this Record to the *other*.

        This method only implements width subtyping for records.
        """
    def __repr__(self) -> str: ...

class DType(DTypeSpec, Opaque):
    """
    Type class associated with the `np.dtype`.

    i.e. :code:`assert type(np.dtype('int32')) == np.dtype`

    np.dtype('int32')
    """
    _dtype: Incomplete
    def __init__(self, dtype) -> None: ...
    @property
    def key(self): ...
    @property
    def dtype(self): ...
    def __getitem__(self, arg): ...

class NumpyFlatType(SimpleIteratorType, MutableSequence):
    """
    Type class for `ndarray.flat()` objects.
    """
    array_type: Incomplete
    dtype: Incomplete
    def __init__(self, arrty) -> None: ...
    @property
    def key(self): ...

class NumpyNdEnumerateType(SimpleIteratorType):
    """
    Type class for `np.ndenumerate()` objects.
    """
    array_type: Incomplete
    def __init__(self, arrty) -> None: ...
    @property
    def key(self): ...

class NumpyNdIterType(IteratorType):
    '''
    Type class for `np.nditer()` objects.

    The layout denotes in which order the logical shape is iterated on.
    "C" means logical order (corresponding to in-memory order in C arrays),
    "F" means reverse logical order (corresponding to in-memory order in
    F arrays).
    '''
    arrays: Incomplete
    layout: Incomplete
    dtypes: Incomplete
    ndim: Incomplete
    def __init__(self, arrays) -> None: ...
    @classmethod
    def _compute_layout(cls, arrays): ...
    @property
    def key(self): ...
    @property
    def views(self):
        """
        The views yielded by the iterator.
        """
    @property
    def yield_type(self): ...
    @cached_property
    def indexers(self):
        '''
        A list of (kind, start_dim, end_dim, indices) where:
        - `kind` is either "flat", "indexed", "0d" or "scalar"
        - `start_dim` and `end_dim` are the dimension numbers at which
          this indexing takes place
        - `indices` is the indices of the indexed arrays in self.arrays
        '''
    @cached_property
    def need_shaped_indexing(self):
        """
        Whether iterating on this iterator requires keeping track of
        individual indices inside the shape.  If False, only a single index
        over the equivalent flat shape is required, which can make the
        iterator more efficient.
        """

class NumpyNdIndexType(SimpleIteratorType):
    """
    Type class for `np.ndindex()` objects.
    """
    ndim: Incomplete
    def __init__(self, ndim) -> None: ...
    @property
    def key(self): ...

class Array(Buffer):
    """
    Type class for Numpy arrays.
    """
    mutable: bool
    aligned: bool
    def __init__(self, dtype, ndim, layout, readonly: bool = False, name: Incomplete | None = None, aligned: bool = True) -> None: ...
    @property
    def mangling_args(self): ...
    def copy(self, dtype: Incomplete | None = None, ndim: Incomplete | None = None, layout: Incomplete | None = None, readonly: Incomplete | None = None): ...
    @property
    def key(self): ...
    def unify(self, typingctx, other):
        """
        Unify this with the *other* Array.
        """
    def can_convert_to(self, typingctx, other):
        """
        Convert this Array to the *other*.
        """
    def is_precise(self): ...
    @property
    def box_type(self):
        """Returns the Python type to box to.
        """
    def __repr__(self) -> str: ...

class ArrayCTypes(Type):
    """
    This is the type for `np.ndarray.ctypes`.
    """
    dtype: Incomplete
    ndim: Incomplete
    def __init__(self, arytype) -> None: ...
    @property
    def key(self): ...
    def can_convert_to(self, typingctx, other):
        """
        Convert this type to the corresponding pointer type.
        This allows passing a array.ctypes object to a C function taking
        a raw pointer.

        Note that in pure Python, the array.ctypes object can only be
        passed to a ctypes function accepting a c_void_p, not a typed
        pointer.
        """

class ArrayFlags(Type):
    """
    This is the type for `np.ndarray.flags`.
    """
    array_type: Incomplete
    def __init__(self, arytype) -> None: ...
    @property
    def key(self): ...

class NestedArray(Array):
    '''
    A NestedArray is an array nested within a structured type (which are "void"
    type in NumPy parlance). Unlike an Array, the shape, and not just the number
    of dimensions is part of the type of a NestedArray.
    '''
    _shape: Incomplete
    def __init__(self, dtype, shape) -> None: ...
    @property
    def shape(self): ...
    @property
    def nitems(self): ...
    @property
    def size(self): ...
    @property
    def strides(self): ...
    @property
    def key(self): ...
    def __repr__(self) -> str: ...

class NumPyRandomBitGeneratorType(Type):
    name: str
    def __init__(self, *args, **kwargs) -> None: ...

class NumPyRandomGeneratorType(Type):
    name: str
    def __init__(self, *args, **kwargs) -> None: ...

class PolynomialType(Type):
    coef: Incomplete
    domain: Incomplete
    window: Incomplete
    n_args: Incomplete
    def __init__(self, coef, domain: Incomplete | None = None, window: Incomplete | None = None, n_args: int = 1) -> None: ...
