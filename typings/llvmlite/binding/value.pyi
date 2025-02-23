import enum
from _typeshed import Incomplete
from llvmlite.binding import ffi as ffi
from llvmlite.binding.common import _decode_string as _decode_string, _encode_string as _encode_string
from llvmlite.binding.typeref import TypeRef as TypeRef

class Linkage(enum.IntEnum):
    external = 0
    available_externally = 1
    linkonce_any = 2
    linkonce_odr = 3
    linkonce_odr_autohide = 4
    weak_any = 5
    weak_odr = 6
    appending = 7
    internal = 8
    private = 9
    dllimport = 10
    dllexport = 11
    external_weak = 12
    ghost = 13
    common = 14
    linker_private = 15
    linker_private_weak = 16

class Visibility(enum.IntEnum):
    default = 0
    hidden = 1
    protected = 2

class StorageClass(enum.IntEnum):
    default = 0
    dllimport = 1
    dllexport = 2

class ValueKind(enum.IntEnum):
    argument = 0
    basic_block = 1
    memory_use = 2
    memory_def = 3
    memory_phi = 4
    function = 5
    global_alias = 6
    global_ifunc = 7
    global_variable = 8
    block_address = 9
    constant_expr = 10
    constant_array = 11
    constant_struct = 12
    constant_vector = 13
    undef_value = 14
    constant_aggregate_zero = 15
    constant_data_array = 16
    constant_data_vector = 17
    constant_int = 18
    constant_fp = 19
    constant_pointer_null = 20
    constant_token_none = 21
    metadata_as_value = 22
    inline_asm = 23
    instruction = 24
    poison_value = 25

class ValueRef(ffi.ObjectRef):
    """A weak reference to a LLVM value.
    """
    _kind: Incomplete
    _parents: Incomplete
    def __init__(self, ptr, kind, parents) -> None: ...
    def __str__(self) -> str: ...
    @property
    def module(self):
        """
        The module this function or global variable value was obtained from.
        """
    @property
    def function(self):
        """
        The function this argument or basic block value was obtained from.
        """
    @property
    def block(self):
        """
        The block this instruction value was obtained from.
        """
    @property
    def instruction(self):
        """
        The instruction this operand value was obtained from.
        """
    @property
    def is_global(self): ...
    @property
    def is_function(self): ...
    @property
    def is_block(self): ...
    @property
    def is_argument(self): ...
    @property
    def is_instruction(self): ...
    @property
    def is_operand(self): ...
    @property
    def is_constant(self): ...
    @property
    def value_kind(self): ...
    @property
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def linkage(self): ...
    @linkage.setter
    def linkage(self, value) -> None: ...
    @property
    def visibility(self): ...
    @visibility.setter
    def visibility(self, value) -> None: ...
    @property
    def storage_class(self): ...
    @storage_class.setter
    def storage_class(self, value) -> None: ...
    def add_function_attribute(self, attr) -> None:
        """Only works on function value

        Parameters
        -----------
        attr : str
            attribute name
        """
    @property
    def type(self):
        """
        This value's LLVM type.
        """
    @property
    def global_value_type(self):
        """
        Uses ``LLVMGlobalGetValueType()``.
        Needed for opaque pointers in globals.
        > For globals, use getValueType().
        See https://llvm.org/docs/OpaquePointers.html#migration-instructions
        """
    @property
    def is_declaration(self):
        """
        Whether this value (presumably global) is defined in the current
        module.
        """
    @property
    def attributes(self):
        """
        Return an iterator over this value's attributes.
        The iterator will yield a string for each attribute.
        """
    @property
    def blocks(self):
        """
        Return an iterator over this function's blocks.
        The iterator will yield a ValueRef for each block.
        """
    @property
    def arguments(self):
        """
        Return an iterator over this function's arguments.
        The iterator will yield a ValueRef for each argument.
        """
    @property
    def instructions(self):
        """
        Return an iterator over this block's instructions.
        The iterator will yield a ValueRef for each instruction.
        """
    @property
    def operands(self):
        """
        Return an iterator over this instruction's operands.
        The iterator will yield a ValueRef for each operand.
        """
    @property
    def opcode(self): ...
    @property
    def incoming_blocks(self):
        """
        Return an iterator over this phi instruction's incoming blocks.
        The iterator will yield a ValueRef for each block.
        """
    def get_constant_value(self, signed_int: bool = False, round_fp: bool = False):
        """
        Return the constant value, either as a literal (when supported)
        or as a string.

        Parameters
        -----------
        signed_int : bool
            if True and the constant is an integer, returns a signed version
        round_fp : bool
            if True and the constant is a floating point value, rounds the
            result upon accuracy loss (e.g., when querying an fp128 value).
            By default, raises an exception on accuracy loss
        """

class _ValueIterator(ffi.ObjectRef):
    kind: Incomplete
    _parents: Incomplete
    def __init__(self, ptr, parents) -> None: ...
    def __next__(self): ...
    next = __next__
    def __iter__(self): ...

class _AttributeIterator(ffi.ObjectRef):
    def __next__(self): ...
    next = __next__
    def __iter__(self): ...

class _AttributeListIterator(_AttributeIterator):
    def _dispose(self) -> None: ...
    def _next(self): ...

class _AttributeSetIterator(_AttributeIterator):
    def _dispose(self) -> None: ...
    def _next(self): ...

class _BlocksIterator(_ValueIterator):
    kind: str
    def _dispose(self) -> None: ...
    def _next(self): ...

class _ArgumentsIterator(_ValueIterator):
    kind: str
    def _dispose(self) -> None: ...
    def _next(self): ...

class _InstructionsIterator(_ValueIterator):
    kind: str
    def _dispose(self) -> None: ...
    def _next(self): ...

class _OperandsIterator(_ValueIterator):
    kind: str
    def _dispose(self) -> None: ...
    def _next(self): ...

class _IncomingBlocksIterator(_ValueIterator):
    kind: str
    def _dispose(self) -> None: ...
    def _next(self): ...
