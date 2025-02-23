from _typeshed import Incomplete
from numba.core.typeconv import Conversion as Conversion
from numba.core.types.new_scalars.scalars import Boolean as Boolean, BooleanLiteral as BooleanLiteral, Complex as Complex, Float as Float, Integer as Integer, IntegerLiteral as IntegerLiteral, parse_integer_bitwidth as parse_integer_bitwidth, parse_integer_signed as parse_integer_signed

class MachineInteger(Integer):
    bitwidth: Incomplete
    signed: Incomplete
    def __init__(self, name, bitwidth: Incomplete | None = None, signed: Incomplete | None = None) -> None: ...
    @classmethod
    def from_bitwidth(cls, bitwidth, signed: bool = True): ...
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

class MachineIntegerLiteral(IntegerLiteral, MachineInteger):
    def __init__(self, value) -> None: ...
    def can_convert_to(self, typingctx, other): ...

class MachineBoolean(Boolean): ...

class MachineBooleanLiteral(BooleanLiteral, MachineBoolean):
    def __init__(self, value) -> None: ...
    def can_convert_to(self, typingctx, other): ...

class MachineFloat(Float):
    bitwidth: Incomplete
    def __init__(self, *args, **kws) -> None: ...
    def __lt__(self, other): ...

class MachineComplex(Complex):
    underlying_float: Incomplete
    bitwidth: Incomplete
    def __init__(self, name, underlying_float, **kwargs) -> None: ...
    def __lt__(self, other): ...
