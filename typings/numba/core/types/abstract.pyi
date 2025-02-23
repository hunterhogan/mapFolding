import abc
import weakref
from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod
from functools import cached_property as cached_property
from numba.core.utils import get_hashable_key as get_hashable_key

_typecodes: Incomplete

def _autoincr(): ...

_typecache: dict[weakref.ref, weakref.ref]

def _on_type_disposal(wr, _pop=...) -> None: ...

class _TypeMetaclass(ABCMeta):
    """
    A metaclass that will intern instances after they are created.
    This is done by first creating a new instance (including calling
    __init__, which sets up the required attributes for equality
    and hashing), then looking it up in the _typecache registry.
    """
    def __init__(cls, name, bases, orig_vars) -> None: ...
    def _intern(cls, inst): ...
    def __call__(cls, *args, **kwargs):
        """
        Instantiate *cls* (a Type subclass, presumably) and intern it.
        If an interned instance already exists, it is returned, otherwise
        the new instance is returned.
        """

def _type_reconstructor(reconstructor, reconstructor_args, state):
    """
    Rebuild function for unpickling types.
    """

class Type(metaclass=_TypeMetaclass):
    '''
    The base class for all Numba types.
    It is essential that proper equality comparison is implemented.  The
    default implementation uses the "key" property (overridable in subclasses)
    for both comparison and hashing, to ensure sane behaviour.
    '''
    mutable: bool
    reflected: bool
    name: Incomplete
    def __init__(self, name) -> None: ...
    @property
    def key(self):
        """
        A property used for __eq__, __ne__ and __hash__.  Can be overridden
        in subclasses.
        """
    @property
    def mangling_args(self):
        """
        Returns `(basename, args)` where `basename` is the name of the type
        and `args` is a sequence of parameters of the type.

        Subclass should override to specialize the behavior.
        By default, this returns `(self.name, ())`.
        """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __reduce__(self): ...
    def unify(self, typingctx, other) -> None:
        """
        Try to unify this type with the *other*.  A third type must
        be returned, or None if unification is not possible.
        Only override this if the coercion logic cannot be expressed
        as simple casting rules.
        """
    def can_convert_to(self, typingctx, other) -> None:
        '''
        Check whether this type can be converted to the *other*.
        If successful, must return a string describing the conversion, e.g.
        "exact", "promote", "unsafe", "safe"; otherwise None is returned.
        '''
    def can_convert_from(self, typingctx, other) -> None:
        """
        Similar to *can_convert_to*, but in reverse.  Only needed if
        the type provides conversion from other types.
        """
    def is_precise(self):
        """
        Whether this type is precise, i.e. can be part of a successful
        type inference.  Default implementation returns True.
        """
    def augment(self, other) -> None:
        """
        Augment this type with the *other*.  Return the augmented type,
        or None if not supported.
        """
    def __call__(self, *args): ...
    def __getitem__(self, args):
        """
        Return an array of this type.
        """
    def _determine_array_spec(self, args): ...
    def cast_python_value(self, args) -> None: ...
    @property
    def is_internal(self):
        """ Returns True if this class is an internally defined Numba type by
        virtue of the module in which it is instantiated, False else."""
    def dump(self, tab: str = '') -> None: ...

class Dummy(Type):
    """
    Base class for types that do not really have a representation and are
    compatible with a void*.
    """
class Hashable(Type):
    """
    Base class for hashable types.
    """

class Number(Hashable):
    """
    Base class for number types.
    """
    def unify(self, typingctx, other):
        """
        Unify the two number types using Numpy's rules.
        """

class Callable(Type, metaclass=abc.ABCMeta):
    """
    Base class for callables.
    """
    @abstractmethod
    def get_call_type(self, context, args, kws):
        """
        Using the typing *context*, resolve the callable's signature for
        the given arguments.  A signature object is returned, or None.
        """
    @abstractmethod
    def get_call_signatures(self):
        """
        Returns a tuple of (list of signatures, parameterized)
        """
    @abstractmethod
    def get_impl_key(self, sig):
        """
        Returns the impl key for the given signature
        """

class DTypeSpec(Type, metaclass=abc.ABCMeta):
    '''
    Base class for types usable as "dtype" arguments to various Numpy APIs
    (e.g. np.empty()).
    '''
    @property
    @abc.abstractmethod
    def dtype(self):
        """
        The actual dtype denoted by this dtype spec (a Type instance).
        """

class IterableType(Type, metaclass=abc.ABCMeta):
    """
    Base class for iterable types.
    """
    @property
    @abc.abstractmethod
    def iterator_type(self):
        """
        The iterator type obtained when calling iter() (explicitly or implicitly).
        """

class Sized(Type):
    """
    Base class for objects that support len()
    """

class ConstSized(Sized, metaclass=abc.ABCMeta):
    """
    For types that have a constant size
    """
    @abstractmethod
    def __len__(self): ...

class IteratorType(IterableType, metaclass=abc.ABCMeta):
    """
    Base class for all iterator types.
    Derived classes should implement the *yield_type* attribute.
    """
    def __init__(self, name, **kwargs) -> None: ...
    @property
    @abc.abstractmethod
    def yield_type(self):
        """
        The type of values yielded by the iterator.
        """
    @property
    def iterator_type(self): ...

class Container(Sized, IterableType, metaclass=abc.ABCMeta):
    """
    Base class for container types.
    """
class Sequence(Container, metaclass=abc.ABCMeta):
    """
    Base class for 1d sequence types.  Instances should have the *dtype*
    attribute.
    """

class MutableSequence(Sequence, metaclass=abc.ABCMeta):
    """
    Base class for 1d mutable sequence types.  Instances should have the
    *dtype* attribute.
    """
    mutable: bool

class ArrayCompatible(Type, metaclass=abc.ABCMeta):
    """
    Type class for Numpy array-compatible objects (typically, objects
    exposing an __array__ method).
    Derived classes should implement the *as_array* attribute.
    """
    array_priority: float
    @property
    @abc.abstractmethod
    def as_array(self):
        """
        The equivalent array type, for operations supporting array-compatible
        objects (such as ufuncs).
        """
    @cached_property
    def ndim(self): ...
    @cached_property
    def layout(self): ...
    @cached_property
    def dtype(self): ...

class Literal(Type):
    """Base class for Literal types.
    Literal types contain the original Python value in the type.

    A literal type should always be constructed from the `literal(val)`
    function.
    """
    ctor_map: dict[type, type['Literal']]
    _literal_type_cache: Incomplete
    def __init__(self, value) -> None: ...
    _literal_value: Incomplete
    _key: Incomplete
    def _literal_init(self, value) -> None: ...
    @property
    def literal_value(self): ...
    @property
    def literal_type(self): ...

class TypeRef(Dummy):
    """Reference to a type.

    Used when a type is passed as a value.
    """
    instance_type: Incomplete
    def __init__(self, instance_type) -> None: ...
    @property
    def key(self): ...

class InitialValue:
    """
    Used as a mixin for a type will potentially have an initial value that will
    be carried in the .initial_value attribute.
    """
    _initial_value: Incomplete
    def __init__(self, initial_value) -> None: ...
    @property
    def initial_value(self): ...

class Poison(Type):
    '''
    This is the "bottom" type in the type system. It won\'t unify and it\'s
    unliteral version is Poison of itself. It\'s advisable for debugging purposes
    to call the constructor with the type that\'s being poisoned (for whatever
    reason) but this isn\'t strictly required.
    '''
    ty: Incomplete
    def __init__(self, ty) -> None: ...
    def __unliteral__(self): ...
    def unify(self, typingctx, other) -> None: ...
