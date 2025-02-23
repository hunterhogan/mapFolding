from _typeshed import Incomplete
from llvmlite import opaque_pointers_enabled as opaque_pointers_enabled
from llvmlite.ir._utils import _StrCaching as _StrCaching

def _wrapname(x): ...

class Type(_StrCaching):
    """
    The base class for all LLVM types.
    """
    is_pointer: bool
    null: str
    def __repr__(self) -> str: ...
    def _to_string(self) -> None: ...
    def as_pointer(self, addrspace: int = 0): ...
    def __ne__(self, other): ...
    def _get_ll_global_value_type(self, target_data, context: Incomplete | None = None):
        """
        Convert this type object to an LLVM type.
        """
    def get_abi_size(self, target_data, context: Incomplete | None = None):
        """
        Get the ABI size of this type according to data layout *target_data*.
        """
    def get_element_offset(self, target_data, ndx, context: Incomplete | None = None): ...
    def get_abi_alignment(self, target_data, context: Incomplete | None = None):
        """
        Get the minimum ABI alignment of this type according to data layout
        *target_data*.
        """
    def format_constant(self, value):
        """
        Format constant *value* of this type.  This method may be overriden
        by subclasses.
        """
    def wrap_constant_value(self, value):
        """
        Wrap constant *value* if necessary.  This method may be overriden
        by subclasses (especially aggregate types).
        """
    def __call__(self, value):
        """
        Create a LLVM constant of this type with the given Python value.
        """

class MetaDataType(Type):
    def _to_string(self): ...
    def as_pointer(self) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class LabelType(Type):
    """
    The label type is the type of e.g. basic blocks.
    """
    def _to_string(self): ...

class PointerType(Type):
    """
    The type of all pointer values.
    By default (without specialisation) represents an opaque pointer.
    """
    is_opaque: bool
    is_pointer: bool
    null: str
    def __new__(cls, pointee: Incomplete | None = None, addrspace: int = 0): ...
    addrspace: Incomplete
    def __init__(self, addrspace: int = 0) -> None: ...
    def _to_string(self): ...
    def __hash__(self): ...
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

class _TypedPointerType(PointerType):
    """
    The type of typed pointer values. To be removed eventually.
    """
    pointee: Incomplete
    is_opaque: bool
    def __init__(self, pointee, addrspace: int = 0) -> None: ...
    def _to_string(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def gep(self, i):
        """
        Resolve the type of the i-th element (for getelementptr lookups).
        """
    @property
    def intrinsic_name(self): ...
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

class VoidType(Type):
    """
    The type for empty values (e.g. a function returning no value).
    """
    def _to_string(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

class FunctionType(Type):
    """
    The type for functions.
    """
    return_type: Incomplete
    args: Incomplete
    var_arg: Incomplete
    def __init__(self, return_type, args, var_arg: bool = False) -> None: ...
    def _to_string(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

class IntType(Type):
    """
    The type for integers.
    """
    null: str
    _instance_cache: Incomplete
    width: int
    def __new__(cls, bits): ...
    @classmethod
    def __new(cls, bits): ...
    def __getnewargs__(self): ...
    def __copy__(self): ...
    def _to_string(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def format_constant(self, val): ...
    def wrap_constant_value(self, val): ...
    @property
    def intrinsic_name(self): ...
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

def _as_float(value):
    """
    Truncate to single-precision float.
    """
def _as_half(value):
    """
    Truncate to half-precision float.
    """
def _format_float_as_hex(value, packfmt, unpackfmt, numdigits): ...
def _format_double(value):
    """
    Format *value* as a hexadecimal string of its IEEE double precision
    representation.
    """

class _BaseFloatType(Type):
    def __new__(cls): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    @classmethod
    def _create_instance(cls) -> None: ...
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

class HalfType(_BaseFloatType):
    """
    The type for single-precision floats.
    """
    null: str
    intrinsic_name: str
    def __str__(self) -> str: ...
    def format_constant(self, value): ...

class FloatType(_BaseFloatType):
    """
    The type for single-precision floats.
    """
    null: str
    intrinsic_name: str
    def __str__(self) -> str: ...
    def format_constant(self, value): ...

class DoubleType(_BaseFloatType):
    """
    The type for double-precision floats.
    """
    null: str
    intrinsic_name: str
    def __str__(self) -> str: ...
    def format_constant(self, value): ...

class _Repeat:
    value: Incomplete
    size: Incomplete
    def __init__(self, value, size) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, item): ...

class VectorType(Type):
    '''
    The type for vectors of primitive data items (e.g. "<f32 x 4>").
    '''
    element: Incomplete
    count: Incomplete
    def __init__(self, element, count) -> None: ...
    @property
    def elements(self): ...
    def __len__(self) -> int: ...
    def _to_string(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __copy__(self): ...
    def format_constant(self, value): ...
    def wrap_constant_value(self, values): ...
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

class Aggregate(Type):
    """
    Base class for aggregate types.
    See http://llvm.org/docs/LangRef.html#t-aggregate
    """
    def wrap_constant_value(self, values): ...

class ArrayType(Aggregate):
    '''
    The type for fixed-size homogenous arrays (e.g. "[f32 x 3]").
    '''
    element: Incomplete
    count: Incomplete
    def __init__(self, element, count) -> None: ...
    @property
    def elements(self): ...
    def __len__(self) -> int: ...
    def _to_string(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def gep(self, i):
        """
        Resolve the type of the i-th element (for getelementptr lookups).
        """
    def format_constant(self, value): ...
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

class BaseStructType(Aggregate):
    """
    The base type for heterogenous struct types.
    """
    _packed: bool
    @property
    def packed(self):
        """
        A boolean attribute that indicates whether the structure uses
        packed layout.
        """
    @packed.setter
    def packed(self, val) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    @property
    def is_opaque(self): ...
    def structure_repr(self):
        """
        Return the LLVM IR for the structure representation
        """
    def format_constant(self, value): ...
    def gep(self, i):
        """
        Resolve the type of the i-th element (for getelementptr lookups).

        *i* needs to be a LLVM constant, so that the type can be determined
        at compile-time.
        """
    def _wrap_packed(self, textrepr):
        """
        Internal helper to wrap textual repr of struct type into packed struct
        """
    @classmethod
    def from_llvm(cls, typeref, ir_ctx):
        """
        Create from a llvmlite.binding.TypeRef
        """

class LiteralStructType(BaseStructType):
    '''
    The type of "literal" structs, i.e. structs with a literally-defined
    type (by contrast with IdentifiedStructType).
    '''
    null: str
    elements: Incomplete
    packed: Incomplete
    def __init__(self, elems, packed: bool = False) -> None:
        """
        *elems* is a sequence of types to be used as members.
        *packed* controls the use of packed layout.
        """
    def _to_string(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class IdentifiedStructType(BaseStructType):
    """
    A type which is a named alias for another struct type, akin to a typedef.
    While literal struct types can be structurally equal (see
    LiteralStructType), identified struct types are compared by name.

    Do not use this directly.
    """
    null: str
    context: Incomplete
    name: Incomplete
    elements: Incomplete
    packed: Incomplete
    def __init__(self, context, name, packed: bool = False) -> None:
        """
        *context* is a llvmlite.ir.Context.
        *name* is the identifier for the new struct type.
        *packed* controls the use of packed layout.
        """
    def _to_string(self): ...
    def get_declaration(self):
        """
        Returns the string for the declaration of the type
        """
    def __eq__(self, other): ...
    def __hash__(self): ...
    def set_body(self, *elems) -> None: ...
