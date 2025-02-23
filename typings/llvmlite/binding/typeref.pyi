import enum
from _typeshed import Incomplete
from llvmlite import ir as ir, opaque_pointers_enabled as opaque_pointers_enabled
from llvmlite.binding import ffi as ffi

class TypeKind(enum.IntEnum):
    void = 0
    half = 1
    float = 2
    double = 3
    x86_fp80 = 4
    fp128 = 5
    ppc_fp128 = 6
    label = 7
    integer = 8
    function = 9
    struct = 10
    array = 11
    pointer = 12
    vector = 13
    metadata = 14
    x86_mmx = 15
    token = 16
    scalable_vector = 17
    bfloat = 18
    x86_amx = 19

_TypeKindToIRType: Incomplete

class TypeRef(ffi.ObjectRef):
    """A weak reference to a LLVM type
    """
    @property
    def name(self):
        """
        Get type name
        """
    @property
    def is_struct(self):
        """
        Returns true if the type is a struct type.
        """
    @property
    def is_pointer(self):
        """
        Returns true if the type is a pointer type.
        """
    @property
    def is_array(self):
        """
        Returns true if the type is an array type.
        """
    @property
    def is_vector(self):
        """
        Returns true if the type is a vector type.
        """
    @property
    def is_function(self):
        """
        Returns true if the type is a function type.
        """
    @property
    def is_function_vararg(self):
        """
        Returns true if a function type accepts a variable number of arguments.
        When the type is not a function, raises exception.
        """
    @property
    def elements(self):
        """
        Returns iterator over enclosing types
        """
    @property
    def element_type(self):
        """
        Returns the pointed-to type. When the type is not a pointer,
        raises exception.
        """
    @property
    def element_count(self):
        """
        Returns the number of elements in an array or a vector. For scalable
        vectors, returns minimum number of elements. When the type is neither
        an array nor a vector, raises exception.
        """
    @property
    def type_width(self):
        """
        Return the basic size of this type if it is a primitive type. These are
        fixed by LLVM and are not target-dependent.
        This will return zero if the type does not have a size or is not a
        primitive type.

        If this is a scalable vector type, the scalable property will be set and
        the runtime size will be a positive integer multiple of the base size.

        Note that this may not reflect the size of memory allocated for an
        instance of the type or the number of bytes that are written when an
        instance of the type is stored to memory.
        """
    @property
    def type_kind(self):
        """
        Returns the LLVMTypeKind enumeration of this type.
        """
    @property
    def is_packed_struct(self): ...
    @property
    def is_literal_struct(self): ...
    @property
    def is_opaque_struct(self): ...
    def get_function_parameters(self) -> tuple['TypeRef']: ...
    def get_function_return(self) -> TypeRef: ...
    def as_ir(self, ir_ctx: ir.Context) -> ir.Type:
        """Convert into a ``llvmlite.ir.Type``.
        """
    def __str__(self) -> str: ...

class _TypeIterator(ffi.ObjectRef):
    def __next__(self): ...
    next = __next__
    def __iter__(self): ...

class _TypeListIterator(_TypeIterator):
    def _dispose(self) -> None: ...
    def _next(self): ...
