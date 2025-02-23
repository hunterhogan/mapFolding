from _typeshed import Incomplete
from numba.core.errors import LiteralTypingError as LiteralTypingError, TypingError as TypingError
from numba.core.ir import UndefinedType as UndefinedType
from numba.core.typeconv import Conversion as Conversion
from numba.core.types.abstract import Callable as Callable, Hashable as Hashable, Literal as Literal, Type as Type
from numba.core.types.common import Dummy as Dummy, IterableType as IterableType, Opaque as Opaque, SimpleIteratorType as SimpleIteratorType
from numba.core.utils import get_hashable_key as get_hashable_key

class PyObject(Dummy):
    """
    A generic CPython object.
    """
    def is_precise(self): ...

class Phantom(Dummy):
    """
    A type that cannot be materialized.  A Phantom cannot be used as
    argument or return type.
    """

class Undefined(Dummy):
    """
    A type that is left imprecise.  This is used as a temporaray placeholder
    during type inference in the hope that the type can be later refined.
    """
    def is_precise(self): ...

class UndefVar(Dummy):
    """
    A type that is created by Expr.undef to represent an undefined variable.
    This type can be promoted to any other type.
    This is introduced to handle Python 3.12 LOAD_FAST_AND_CLEAR.
    """
    def can_convert_to(self, typingctx, other): ...

class RawPointer(Opaque):
    """
    A raw pointer without any specific meaning.
    """

class StringLiteral(Literal, Dummy):
    def can_convert_to(self, typingctx, other): ...

def unliteral(lit_type):
    """
    Get base type from Literal type.
    """
def literal(value):
    """Returns a Literal instance or raise LiteralTypingError
    """
def maybe_literal(value):
    """Get a Literal type for the value or None.
    """

class Omitted(Opaque):
    """
    An omitted function argument with a default value.
    """
    _value: Incomplete
    _value_key: Incomplete
    def __init__(self, value) -> None: ...
    @property
    def key(self): ...
    @property
    def value(self): ...

class VarArg(Type):
    """
    Special type representing a variable number of arguments at the
    end of a function's signature.  Only used for signature matching,
    not for actual values.
    """
    dtype: Incomplete
    def __init__(self, dtype) -> None: ...
    @property
    def key(self): ...

class Module(Dummy):
    pymod: Incomplete
    def __init__(self, pymod) -> None: ...
    @property
    def key(self): ...

class MemInfoPointer(Type):
    '''
    Pointer to a Numba "meminfo" (i.e. the information for a managed
    piece of memory).
    '''
    mutable: bool
    dtype: Incomplete
    def __init__(self, dtype) -> None: ...
    @property
    def key(self): ...

class CPointer(Type):
    """
    Type class for pointers to other types.

    Attributes
    ----------
        dtype : The pointee type
        addrspace : int
            The address space pointee belongs to.
    """
    mutable: bool
    dtype: Incomplete
    addrspace: Incomplete
    def __init__(self, dtype, addrspace: Incomplete | None = None) -> None: ...
    @property
    def key(self): ...

class EphemeralPointer(CPointer):
    """
    Type class for pointers which aren't guaranteed to last long - e.g.
    stack-allocated slots.  The data model serializes such pointers
    by copying the data pointed to.
    """

class EphemeralArray(Type):
    """
    Similar to EphemeralPointer, but pointing to an array of elements,
    rather than a single one.  The array size must be known at compile-time.
    """
    dtype: Incomplete
    count: Incomplete
    def __init__(self, dtype, count) -> None: ...
    @property
    def key(self): ...

class Object(Type):
    mutable: bool
    cls: Incomplete
    def __init__(self, clsobj) -> None: ...
    @property
    def key(self): ...

class Optional(Type):
    """
    Type class for optional types, i.e. union { some type, None }
    """
    type: Incomplete
    def __init__(self, typ) -> None: ...
    @property
    def key(self): ...
    def can_convert_to(self, typingctx, other): ...
    def can_convert_from(self, typingctx, other): ...
    def unify(self, typingctx, other): ...

class NoneType(Opaque):
    """
    The type for None.
    """
    def unify(self, typingctx, other):
        """
        Turn anything to a Optional type;
        """

class EllipsisType(Opaque):
    """
    The type for the Ellipsis singleton.
    """

class ExceptionClass(Callable, Phantom):
    """
    The type of exception classes (not instances).
    """
    exc_class: Incomplete
    def __init__(self, exc_class) -> None: ...
    def get_call_type(self, context, args, kws): ...
    def get_call_signatures(self): ...
    def get_impl_key(self, sig): ...
    @property
    def key(self): ...

class ExceptionInstance(Phantom):
    """
    The type of exception instances.  *exc_class* should be the
    exception class.
    """
    exc_class: Incomplete
    def __init__(self, exc_class) -> None: ...
    @property
    def key(self): ...

class SliceType(Type):
    members: Incomplete
    has_step: Incomplete
    def __init__(self, name, members) -> None: ...
    @property
    def key(self): ...

class SliceLiteral(Literal, SliceType):
    def __init__(self, value) -> None: ...
    @property
    def key(self): ...

class ClassInstanceType(Type):
    """
    The type of a jitted class *instance*.  It will be the return-type
    of the constructor of the class.
    """
    mutable: bool
    name_prefix: str
    class_type: Incomplete
    def __init__(self, class_type) -> None: ...
    def get_data_type(self): ...
    def get_reference_type(self): ...
    @property
    def key(self): ...
    @property
    def classname(self): ...
    @property
    def jit_props(self): ...
    @property
    def jit_static_methods(self): ...
    @property
    def jit_methods(self): ...
    @property
    def struct(self): ...
    @property
    def methods(self): ...
    @property
    def static_methods(self): ...

class ClassType(Callable, Opaque):
    """
    The type of the jitted class (not instance).  When the type of a class
    is called, its constructor is invoked.
    """
    mutable: bool
    name_prefix: str
    instance_type_class = ClassInstanceType
    class_name: Incomplete
    class_doc: Incomplete
    _ctor_template_class: Incomplete
    jit_methods: Incomplete
    jit_props: Incomplete
    jit_static_methods: Incomplete
    struct: Incomplete
    def __init__(self, class_def, ctor_template_cls, struct, jit_methods, jit_props, jit_static_methods) -> None: ...
    def get_call_type(self, context, args, kws): ...
    def get_call_signatures(self): ...
    def get_impl_key(self, sig): ...
    @property
    def methods(self): ...
    @property
    def static_methods(self): ...
    @property
    def instance_type(self): ...
    @property
    def ctor_template(self): ...
    def _specialize_template(self, basecls): ...

class DeferredType(Type):
    """
    Represents a type that will be defined later.  It must be defined
    before it is materialized (used in the compiler).  Once defined, it
    behaves exactly as the type it is defining.
    """
    _define: Incomplete
    def __init__(self) -> None: ...
    def get(self): ...
    def define(self, typ) -> None: ...
    def unify(self, typingctx, other): ...

class ClassDataType(Type):
    """
    Internal only.
    Represents the data of the instance.  The representation of
    ClassInstanceType contains a pointer to a ClassDataType which represents
    a C structure that contains all the data fields of the class instance.
    """
    class_type: Incomplete
    def __init__(self, classtyp) -> None: ...

class ContextManager(Callable, Phantom):
    """
    An overly-simple ContextManager type that cannot be materialized.
    """
    cm: Incomplete
    def __init__(self, cm) -> None: ...
    def get_call_signatures(self): ...
    def get_call_type(self, context, args, kws): ...
    def get_impl_key(self, sig): ...

class UnicodeType(IterableType, Hashable):
    def __init__(self, name) -> None: ...
    @property
    def iterator_type(self): ...

class UnicodeIteratorType(SimpleIteratorType):
    data: Incomplete
    def __init__(self, dtype) -> None: ...
