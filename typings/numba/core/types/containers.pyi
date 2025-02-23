import abc
from .. import utils as utils
from ..errors import TypingError as TypingError
from ..typeconv import Conversion as Conversion
from .abstract import ConstSized as ConstSized, Container as Container, Hashable as Hashable, InitialValue as InitialValue, Literal as Literal, MutableSequence as MutableSequence, Poison as Poison, Sequence as Sequence, Type as Type, TypeRef as TypeRef
from .common import Buffer as Buffer, IterableType as IterableType, SimpleIterableType as SimpleIterableType, SimpleIteratorType as SimpleIteratorType
from .misc import NoneType as NoneType, Optional as Optional, Undefined as Undefined, unliteral as unliteral
from _typeshed import Incomplete
from collections.abc import Sequence as pySequence

class Pair(Type):
    """
    A heterogeneous pair.
    """
    first_type: Incomplete
    second_type: Incomplete
    def __init__(self, first_type, second_type) -> None: ...
    @property
    def key(self): ...
    def unify(self, typingctx, other): ...

class BaseContainerIterator(SimpleIteratorType):
    """
    Convenience base class for some container iterators.

    Derived classes must implement the *container_class* attribute.
    """
    container: Incomplete
    def __init__(self, container) -> None: ...
    def unify(self, typingctx, other): ...
    @property
    def key(self): ...

class BaseContainerPayload(Type):
    """
    Convenience base class for some container payloads.

    Derived classes must implement the *container_class* attribute.
    """
    container: Incomplete
    def __init__(self, container) -> None: ...
    @property
    def key(self): ...

class Bytes(Buffer):
    """
    Type class for Python 3.x bytes objects.
    """
    mutable: bool
    slice_is_copy: bool

class ByteArray(Buffer):
    """
    Type class for bytearray objects.
    """
    slice_is_copy: bool

class PyArray(Buffer):
    """
    Type class for array.array objects.
    """
    slice_is_copy: bool

class MemoryView(Buffer):
    """
    Type class for memoryview objects.
    """

def is_homogeneous(*tys):
    """Are the types homogeneous?
    """

class BaseTuple(ConstSized, Hashable, metaclass=abc.ABCMeta):
    """
    The base class for all tuple types (with a known size).
    """
    @classmethod
    def from_types(cls, tys, pyclass: Incomplete | None = None):
        """
        Instantiate the right tuple type for the given element types.
        """
    @classmethod
    def _make_homogeneous_tuple(cls, dtype, count): ...
    @classmethod
    def _make_heterogeneous_tuple(cls, tys): ...

class BaseAnonymousTuple(BaseTuple, metaclass=abc.ABCMeta):
    """
    Mixin for non-named tuples.
    """
    def can_convert_to(self, typingctx, other):
        """
        Convert this tuple to another one.  Note named tuples are rejected.
        """
    def __unliteral__(self): ...

class _HomogeneousTuple(Sequence, BaseTuple):
    @property
    def iterator_type(self): ...
    def __getitem__(self, i):
        """
        Return element at position i
        """
    def __iter__(self): ...
    def __len__(self) -> int: ...
    @property
    def types(self): ...

class UniTuple(BaseAnonymousTuple, _HomogeneousTuple, Sequence):
    """
    Type class for homogeneous tuples.
    """
    dtype: Incomplete
    count: Incomplete
    def __init__(self, dtype, count) -> None: ...
    @property
    def mangling_args(self): ...
    @property
    def key(self): ...
    def unify(self, typingctx, other):
        """
        Unify UniTuples with their dtype
        """
    def __unliteral__(self): ...
    def __repr__(self) -> str: ...

class UniTupleIter(BaseContainerIterator):
    """
    Type class for homogeneous tuple iterators.
    """
    container_class = _HomogeneousTuple

class _HeterogeneousTuple(BaseTuple):
    def __getitem__(self, i):
        """
        Return element at position i
        """
    def __len__(self) -> int: ...
    def __iter__(self): ...
    @staticmethod
    def is_types_iterable(types) -> None: ...

class UnionType(Type):
    types: Incomplete
    def __init__(self, types) -> None: ...
    def get_type_tag(self, typ): ...

class Tuple(BaseAnonymousTuple, _HeterogeneousTuple):
    def __new__(cls, types): ...
    types: Incomplete
    count: Incomplete
    dtype: Incomplete
    def __init__(self, types) -> None: ...
    @property
    def mangling_args(self): ...
    @property
    def key(self): ...
    def unify(self, typingctx, other):
        """
        Unify elements of Tuples/UniTuples
        """
    def __repr__(self) -> str: ...

class _StarArgTupleMixin:
    @classmethod
    def _make_homogeneous_tuple(cls, dtype, count): ...
    @classmethod
    def _make_heterogeneous_tuple(cls, tys): ...

class StarArgTuple(_StarArgTupleMixin, Tuple):
    """To distinguish from Tuple() used as argument to a `*args`.
    """
    def __new__(cls, types): ...

class StarArgUniTuple(_StarArgTupleMixin, UniTuple):
    """To distinguish from UniTuple() used as argument to a `*args`.
    """
class BaseNamedTuple(BaseTuple, metaclass=abc.ABCMeta): ...

class NamedUniTuple(_HomogeneousTuple, BaseNamedTuple):
    dtype: Incomplete
    count: Incomplete
    fields: Incomplete
    instance_class: Incomplete
    def __init__(self, dtype, count, cls) -> None: ...
    @property
    def iterator_type(self): ...
    @property
    def key(self): ...

class NamedTuple(_HeterogeneousTuple, BaseNamedTuple):
    types: Incomplete
    count: Incomplete
    fields: Incomplete
    instance_class: Incomplete
    def __init__(self, types, cls) -> None: ...
    @property
    def key(self): ...

class List(MutableSequence, InitialValue):
    """
    Type class for (arbitrary-sized) homogeneous lists.
    """
    dtype: Incomplete
    reflected: Incomplete
    def __init__(self, dtype, reflected: bool = False, initial_value: Incomplete | None = None) -> None: ...
    def copy(self, dtype: Incomplete | None = None, reflected: Incomplete | None = None): ...
    def unify(self, typingctx, other): ...
    @property
    def key(self): ...
    @property
    def iterator_type(self): ...
    def is_precise(self): ...
    def __getitem__(self, args):
        """
        Overrides the default __getitem__ from Type.
        """
    def __unliteral__(self): ...
    def __repr__(self) -> str: ...

class LiteralList(Literal, ConstSized, Hashable):
    """A heterogeneous immutable list (basically a tuple with list semantics).
    """
    mutable: bool
    types: Incomplete
    count: Incomplete
    name: Incomplete
    def __init__(self, literal_value) -> None: ...
    def __getitem__(self, i):
        """
        Return element at position i
        """
    def __len__(self) -> int: ...
    def __iter__(self): ...
    @classmethod
    def from_types(cls, tys): ...
    @staticmethod
    def is_types_iterable(types) -> None: ...
    @property
    def iterator_type(self): ...
    def __unliteral__(self): ...
    def unify(self, typingctx, other):
        """
        Unify this with the *other* one.
        """

class ListIter(BaseContainerIterator):
    """
    Type class for list iterators.
    """
    container_class = List

class ListPayload(BaseContainerPayload):
    """
    Internal type class for the dynamically-allocated payload of a list.
    """
    container_class = List

class Set(Container):
    """
    Type class for homogeneous sets.
    """
    mutable: bool
    dtype: Incomplete
    reflected: Incomplete
    def __init__(self, dtype, reflected: bool = False) -> None: ...
    @property
    def key(self): ...
    @property
    def iterator_type(self): ...
    def is_precise(self): ...
    def copy(self, dtype: Incomplete | None = None, reflected: Incomplete | None = None): ...
    def unify(self, typingctx, other): ...
    def __repr__(self) -> str: ...

class SetIter(BaseContainerIterator):
    """
    Type class for set iterators.
    """
    container_class = Set

class SetPayload(BaseContainerPayload):
    """
    Internal type class for the dynamically-allocated payload of a set.
    """
    container_class = Set

class SetEntry(Type):
    """
    Internal type class for the entries of a Set's hash table.
    """
    set_type: Incomplete
    def __init__(self, set_type) -> None: ...
    @property
    def key(self): ...

class ListType(IterableType):
    """List type
    """
    mutable: bool
    item_type: Incomplete
    dtype: Incomplete
    def __init__(self, itemty) -> None: ...
    @property
    def key(self): ...
    def is_precise(self): ...
    @property
    def iterator_type(self): ...
    @classmethod
    def refine(cls, itemty):
        """Refine to a precise list type
        """
    def unify(self, typingctx, other):
        """
        Unify this with the *other* list.
        """
    def __repr__(self) -> str: ...

class ListTypeIterableType(SimpleIterableType):
    """List iterable type
    """
    parent: Incomplete
    yield_type: Incomplete
    def __init__(self, parent) -> None: ...

class ListTypeIteratorType(SimpleIteratorType):
    parent: Incomplete
    iterable: Incomplete
    def __init__(self, iterable) -> None: ...

def _sentry_forbidden_types(key, value) -> None: ...

class DictType(IterableType, InitialValue):
    """Dictionary type
    """
    key_type: Incomplete
    value_type: Incomplete
    keyvalue_type: Incomplete
    def __init__(self, keyty, valty, initial_value: Incomplete | None = None) -> None: ...
    def is_precise(self): ...
    @property
    def iterator_type(self): ...
    @classmethod
    def refine(cls, keyty, valty):
        """Refine to a precise dictionary type
        """
    def unify(self, typingctx, other):
        """
        Unify this with the *other* dictionary.
        """
    @property
    def key(self): ...
    def __unliteral__(self): ...
    def __repr__(self) -> str: ...

class LiteralStrKeyDict(Literal, ConstSized, Hashable):
    """A Dictionary of string keys to heterogeneous values (basically a
    namedtuple with dict semantics).
    """
    class FakeNamedTuple(pySequence):
        __name__: Incomplete
        _fields: Incomplete
        def __init__(self, name, keys) -> None: ...
        def __len__(self) -> int: ...
        def __getitem__(self, key): ...
    mutable: bool
    value_index: Incomplete
    tuple_ty: Incomplete
    types: Incomplete
    count: Incomplete
    fields: Incomplete
    instance_class: Incomplete
    name: Incomplete
    def __init__(self, literal_value, value_index: Incomplete | None = None) -> None: ...
    def __unliteral__(self): ...
    def unify(self, typingctx, other):
        """
        Unify this with the *other* one.
        """
    def __len__(self) -> int: ...
    def __iter__(self): ...
    @property
    def key(self): ...

class DictItemsIterableType(SimpleIterableType):
    """Dictionary iterable type for .items()
    """
    parent: Incomplete
    yield_type: Incomplete
    name: Incomplete
    def __init__(self, parent) -> None: ...

class DictKeysIterableType(SimpleIterableType):
    """Dictionary iterable type for .keys()
    """
    parent: Incomplete
    yield_type: Incomplete
    name: Incomplete
    def __init__(self, parent) -> None: ...

class DictValuesIterableType(SimpleIterableType):
    """Dictionary iterable type for .values()
    """
    parent: Incomplete
    yield_type: Incomplete
    name: Incomplete
    def __init__(self, parent) -> None: ...

class DictIteratorType(SimpleIteratorType):
    parent: Incomplete
    iterable: Incomplete
    def __init__(self, iterable) -> None: ...

class StructRef(Type):
    """A mutable struct.
    """
    _fields: Incomplete
    _typename: Incomplete
    def __init__(self, fields) -> None:
        """
        Parameters
        ----------
        fields : Sequence
            A sequence of field descriptions, which is a 2-tuple-like object
            containing `(name, type)`, where `name` is a `str` for the field
            name, and `type` is a numba type for the field type.
        """
    def preprocess_fields(self, fields):
        """Subclasses can override this to do additional clean up on fields.

        The default is an identity function.

        Parameters:
        -----------
        fields : Sequence[Tuple[str, Type]]
        """
    @property
    def field_dict(self):
        """Return an immutable mapping for the field names and their
        corresponding types.
        """
    def get_data_type(self):
        """Get the payload type for the actual underlying structure referred
        to by this struct reference.

        See also: `ClassInstanceType.get_data_type`
        """

class StructRefPayload(Type):
    """The type of the payload of a mutable struct.
    """
    mutable: bool
    _typename: Incomplete
    _fields: Incomplete
    def __init__(self, typename, fields) -> None: ...
    @property
    def field_dict(self): ...
