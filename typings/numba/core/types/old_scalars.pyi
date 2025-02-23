from .abstract import Dummy as Dummy, Hashable as Hashable, Literal as Literal, Number as Number, Type as Type
from _typeshed import Incomplete
from functools import cached_property as cached_property
from numba.core import utils as utils
from numba.core.typeconv import Conversion as Conversion
from numba.np import npdatetime_helpers as npdatetime_helpers

class Boolean(Hashable):
    def cast_python_value(self, value): ...

def parse_integer_bitwidth(name): ...
def parse_integer_signed(name): ...

class Integer(Number):
    bitwidth: Incomplete
    signed: Incomplete
    def __init__(self, name, bitwidth: Incomplete | None = None, signed: Incomplete | None = None) -> None: ...
    @classmethod
    def from_bitwidth(cls, bitwidth, signed: bool = True): ...
    def cast_python_value(self, value): ...
    def __lt__(self, other): ...
    @property
    def maxval(self):
        """
        The maximum value representable by this type.
        """
    @property
    def minval(self):
        """
        The minimal value representable by this type.
        """

class IntegerLiteral(Literal, Integer):
    def __init__(self, value) -> None: ...
    def can_convert_to(self, typingctx, other): ...

class BooleanLiteral(Literal, Boolean):
    def __init__(self, value) -> None: ...
    def can_convert_to(self, typingctx, other): ...

class Float(Number):
    bitwidth: Incomplete
    def __init__(self, *args, **kws) -> None: ...
    def cast_python_value(self, value): ...
    def __lt__(self, other): ...

class Complex(Number):
    underlying_float: Incomplete
    bitwidth: Incomplete
    def __init__(self, name, underlying_float, **kwargs) -> None: ...
    def cast_python_value(self, value): ...
    def __lt__(self, other): ...

class _NPDatetimeBase(Type):
    """
    Common base class for np.datetime64 and np.timedelta64.
    """
    unit: Incomplete
    unit_code: Incomplete
    def __init__(self, unit, *args, **kws) -> None: ...
    def __lt__(self, other): ...
    def cast_python_value(self, value): ...

class NPTimedelta(_NPDatetimeBase):
    type_name: str

class NPDatetime(_NPDatetimeBase):
    type_name: str

class EnumClass(Dummy):
    """
    Type class for Enum classes.
    """
    basename: str
    instance_class: Incomplete
    dtype: Incomplete
    def __init__(self, cls, dtype) -> None: ...
    @property
    def key(self): ...
    @cached_property
    def member_type(self):
        """
        The type of this class' members.
        """

class IntEnumClass(EnumClass):
    """
    Type class for IntEnum classes.
    """
    basename: str
    @cached_property
    def member_type(self):
        """
        The type of this class' members.
        """

class EnumMember(Type):
    """
    Type class for Enum members.
    """
    basename: str
    class_type_class = EnumClass
    instance_class: Incomplete
    dtype: Incomplete
    def __init__(self, cls, dtype) -> None: ...
    @property
    def key(self): ...
    @property
    def class_type(self):
        """
        The type of this member's class.
        """

class IntEnumMember(EnumMember):
    """
    Type class for IntEnum members.
    """
    basename: str
    class_type_class = IntEnumClass
    def can_convert_to(self, typingctx, other):
        """
        Convert IntEnum members to plain integers.
        """
