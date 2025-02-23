from _typeshed import Incomplete
from llvmlite.ir import _utils as _utils, types as types, values as values
from llvmlite.ir._utils import _HasMetadata as _HasMetadata, _StrCaching as _StrCaching, _StringReferenceCaching as _StringReferenceCaching

_VALID_CHARS: Incomplete
_SIMPLE_IDENTIFIER_RE: Incomplete
_CMP_MAP: Incomplete

def _escape_string(text, _map={}):
    """
    Escape the given bytestring for safe use as a LLVM array constant.
    Any unicode string input is first encoded with utf8 into bytes.
    """
def _binop(opname): ...
def _castop(opname): ...

class _ConstOpMixin:
    """
    A mixin defining constant operations, for use in constant-like classes.
    """
    def shl(self, other) -> None:
        """
        Left integer shift:
            lhs << rhs
        """
    def lshr(self, other) -> None:
        """
        Logical (unsigned) right integer shift:
            lhs >> rhs
        """
    def ashr(self, other) -> None:
        """
        Arithmetic (signed) right integer shift:
            lhs >> rhs
        """
    def add(self, other) -> None:
        """
        Integer addition:
            lhs + rhs
        """
    def fadd(self, other) -> None:
        """
        Floating-point addition:
            lhs + rhs
        """
    def sub(self, other) -> None:
        """
        Integer subtraction:
            lhs - rhs
        """
    def fsub(self, other) -> None:
        """
        Floating-point subtraction:
            lhs - rhs
        """
    def mul(self, other) -> None:
        """
        Integer multiplication:
            lhs * rhs
        """
    def fmul(self, other) -> None:
        """
        Floating-point multiplication:
            lhs * rhs
        """
    def udiv(self, other) -> None:
        """
        Unsigned integer division:
            lhs / rhs
        """
    def sdiv(self, other) -> None:
        """
        Signed integer division:
            lhs / rhs
        """
    def fdiv(self, other) -> None:
        """
        Floating-point division:
            lhs / rhs
        """
    def urem(self, other) -> None:
        """
        Unsigned integer remainder:
            lhs % rhs
        """
    def srem(self, other) -> None:
        """
        Signed integer remainder:
            lhs % rhs
        """
    def frem(self, other) -> None:
        """
        Floating-point remainder:
            lhs % rhs
        """
    def or_(self, other) -> None:
        """
        Bitwise integer OR:
            lhs | rhs
        """
    def and_(self, other) -> None:
        """
        Bitwise integer AND:
            lhs & rhs
        """
    def xor(self, other) -> None:
        """
        Bitwise integer XOR:
            lhs ^ rhs
        """
    def _cmp(self, prefix, sign, cmpop, other): ...
    def icmp_signed(self, cmpop, other):
        """
        Signed integer comparison:
            lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>='
        """
    def icmp_unsigned(self, cmpop, other):
        """
        Unsigned integer (or pointer) comparison:
            lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>='
        """
    def fcmp_ordered(self, cmpop, other):
        """
        Floating-point ordered comparison:
            lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>=', 'ord', 'uno'
        """
    def fcmp_unordered(self, cmpop, other):
        """
        Floating-point unordered comparison:
            lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>=', 'ord', 'uno'
        """
    def not_(self):
        """
        Bitwise integer complement:
            ~value
        """
    def neg(self):
        """
        Integer negative:
            -value
        """
    def fneg(self):
        """
        Floating-point negative:
            -value
        """
    def trunc(self, typ) -> None:
        """
        Truncating integer downcast to a smaller type.
        """
    def zext(self, typ) -> None:
        """
        Zero-extending integer upcast to a larger type
        """
    def sext(self, typ) -> None:
        """
        Sign-extending integer upcast to a larger type.
        """
    def fptrunc(self, typ) -> None:
        """
        Floating-point downcast to a less precise type.
        """
    def fpext(self, typ) -> None:
        """
        Floating-point upcast to a more precise type.
        """
    def bitcast(self, typ) -> None:
        """
        Pointer cast to a different pointer type.
        """
    def fptoui(self, typ) -> None:
        """
        Convert floating-point to unsigned integer.
        """
    def uitofp(self, typ) -> None:
        """
        Convert unsigned integer to floating-point.
        """
    def fptosi(self, typ) -> None:
        """
        Convert floating-point to signed integer.
        """
    def sitofp(self, typ) -> None:
        """
        Convert signed integer to floating-point.
        """
    def ptrtoint(self, typ) -> None:
        """
        Cast pointer to integer.
        """
    def inttoptr(self, typ) -> None:
        """
        Cast integer to pointer.
        """
    def gep(self, indices):
        """
        Call getelementptr on this pointer constant.
        """

class Value:
    """
    The base class for all values.
    """
    def __repr__(self) -> str: ...

class _Undefined:
    """
    'undef': a value for undefined values.
    """
    def __new__(cls): ...

Undefined: Incomplete

class Constant(_StrCaching, _StringReferenceCaching, _ConstOpMixin, Value):
    """
    A constant LLVM value.
    """
    type: Incomplete
    constant: Incomplete
    def __init__(self, typ, constant) -> None: ...
    def _to_string(self): ...
    def _get_reference(self): ...
    @classmethod
    def literal_array(cls, elems):
        """
        Construct a literal array constant made of the given members.
        """
    @classmethod
    def literal_struct(cls, elems, packed: bool = False):
        """
        Construct a literal structure constant made of the given members.
        """
    @property
    def addrspace(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __hash__(self): ...
    def __repr__(self) -> str: ...

class FormattedConstant(Constant):
    """
    A constant with an already formatted IR representation.
    """
    def __init__(self, typ, constant) -> None: ...
    def _to_string(self): ...
    def _get_reference(self): ...

class NamedValue(_StrCaching, _StringReferenceCaching, Value):
    """
    The base class for named values.
    """
    name_prefix: str
    deduplicate_name: bool
    parent: Incomplete
    type: Incomplete
    def __init__(self, parent, type, name) -> None: ...
    def _to_string(self): ...
    def descr(self, buf) -> None: ...
    def _get_name(self): ...
    _name: Incomplete
    def _set_name(self, name) -> None: ...
    name: Incomplete
    def _get_reference(self): ...
    def __repr__(self) -> str: ...
    @property
    def function_type(self): ...

class MetaDataString(NamedValue):
    """
    A metadata string, i.e. a constant string used as a value in a metadata
    node.
    """
    string: Incomplete
    def __init__(self, parent, string) -> None: ...
    def descr(self, buf) -> None: ...
    def _get_reference(self): ...
    _to_string = _get_reference
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __hash__(self): ...

class MetaDataArgument(_StrCaching, _StringReferenceCaching, Value):
    """
    An argument value to a function taking metadata arguments.
    This can wrap any other kind of LLVM value.

    Do not instantiate directly, Builder.call() will create these
    automatically.
    """
    type: Incomplete
    wrapped_value: Incomplete
    def __init__(self, value) -> None: ...
    def _get_reference(self): ...
    _to_string = _get_reference

class NamedMetaData:
    """
    A named metadata node.

    Do not instantiate directly, use Module.add_named_metadata() instead.
    """
    parent: Incomplete
    operands: Incomplete
    def __init__(self, parent) -> None: ...
    def add(self, md) -> None: ...

class MDValue(NamedValue):
    '''
    A metadata node\'s value, consisting of a sequence of elements ("operands").

    Do not instantiate directly, use Module.add_metadata() instead.
    '''
    name_prefix: str
    operands: Incomplete
    def __init__(self, parent, values, name) -> None: ...
    def descr(self, buf) -> None: ...
    def _get_reference(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __hash__(self): ...

class DIToken:
    """
    A debug information enumeration value that should appear bare in
    the emitted metadata.

    Use this to wrap known constants, e.g. the DW_* enumerations.
    """
    value: Incomplete
    def __init__(self, value) -> None: ...

class DIValue(NamedValue):
    """
    A debug information descriptor, containing key-value pairs.

    Do not instantiate directly, use Module.add_debug_info() instead.
    """
    name_prefix: str
    is_distinct: Incomplete
    kind: Incomplete
    operands: Incomplete
    def __init__(self, parent, is_distinct, kind, operands, name) -> None: ...
    def descr(self, buf) -> None: ...
    def _get_reference(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __hash__(self): ...

class GlobalValue(NamedValue, _ConstOpMixin, _HasMetadata):
    """
    A global value.
    """
    name_prefix: str
    deduplicate_name: bool
    linkage: str
    storage_class: str
    section: str
    metadata: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

class GlobalVariable(GlobalValue):
    """
    A global variable.
    """
    value_type: Incomplete
    initializer: Incomplete
    unnamed_addr: bool
    global_constant: bool
    addrspace: Incomplete
    align: Incomplete
    def __init__(self, module, typ, name, addrspace: int = 0) -> None: ...
    def descr(self, buf) -> None: ...

class AttributeSet(set):
    """A set of string attribute.
    Only accept items listed in *_known*.

    Properties:
    * Iterate in sorted order
    """
    _known: Incomplete
    def __init__(self, args=()) -> None: ...
    def _expand(self, name, typ): ...
    def add(self, name): ...
    def _to_list(self, typ): ...

class FunctionAttributes(AttributeSet):
    _known: Incomplete
    _alignstack: int
    _personality: Incomplete
    def __init__(self, args=()) -> None: ...
    def add(self, name) -> None: ...
    @property
    def alignstack(self): ...
    @alignstack.setter
    def alignstack(self, val) -> None: ...
    @property
    def personality(self): ...
    @personality.setter
    def personality(self, val) -> None: ...
    def _to_list(self, ret_type): ...

class Function(GlobalValue):
    """Represent a LLVM Function but does uses a Module as parent.
    Global Values are stored as a set of dependencies (attribute `depends`).
    """
    ftype: Incomplete
    scope: Incomplete
    blocks: Incomplete
    attributes: Incomplete
    args: Incomplete
    return_value: Incomplete
    calling_convention: str
    def __init__(self, module, ftype, name) -> None: ...
    @property
    def module(self): ...
    @property
    def entry_basic_block(self): ...
    @property
    def basic_blocks(self): ...
    def append_basic_block(self, name: str = ''): ...
    def insert_basic_block(self, before, name: str = ''):
        """Insert block before
        """
    def descr_prototype(self, buf) -> None:
        '''
        Describe the prototype ("head") of the function.
        '''
    def descr_body(self, buf) -> None:
        """
        Describe of the body of the function.
        """
    def descr(self, buf) -> None: ...
    def __str__(self) -> str: ...
    @property
    def is_declaration(self): ...

class ArgumentAttributes(AttributeSet):
    _known: Incomplete
    _align: int
    _dereferenceable: int
    _dereferenceable_or_null: int
    def __init__(self, args=()) -> None: ...
    def _expand(self, name, typ): ...
    @property
    def align(self): ...
    @align.setter
    def align(self, val) -> None: ...
    @property
    def dereferenceable(self): ...
    @dereferenceable.setter
    def dereferenceable(self, val) -> None: ...
    @property
    def dereferenceable_or_null(self): ...
    @dereferenceable_or_null.setter
    def dereferenceable_or_null(self, val) -> None: ...
    def _to_list(self, typ): ...

class _BaseArgument(NamedValue):
    parent: Incomplete
    attributes: Incomplete
    def __init__(self, parent, typ, name: str = '') -> None: ...
    def __repr__(self) -> str: ...
    def add_attribute(self, attr) -> None: ...

class Argument(_BaseArgument):
    """
    The specification of a function argument.
    """
    def __str__(self) -> str: ...

class ReturnValue(_BaseArgument):
    """
    The specification of a function's return value.
    """
    def __str__(self) -> str: ...

class Block(NamedValue):
    """
    A LLVM IR basic block. A basic block is a sequence of
    instructions whose execution always goes from start to end.  That
    is, a control flow instruction (branch) can only appear as the
    last instruction, and incoming branches can only jump to the first
    instruction.
    """
    scope: Incomplete
    instructions: Incomplete
    terminator: Incomplete
    def __init__(self, parent, name: str = '') -> None: ...
    @property
    def is_terminated(self): ...
    @property
    def function(self): ...
    @property
    def module(self): ...
    def descr(self, buf) -> None: ...
    def replace(self, old, new) -> None:
        """Replace an instruction"""
    def _format_name(self): ...

class BlockAddress(Value):
    """
    The address of a basic block.
    """
    type: Incomplete
    function: Incomplete
    basic_block: Incomplete
    def __init__(self, function, basic_block) -> None: ...
    def __str__(self) -> str: ...
    def get_reference(self): ...
