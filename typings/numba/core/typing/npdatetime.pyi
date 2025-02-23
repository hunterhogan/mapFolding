import operator
from itertools import product as product
from numba.core import errors as errors, types as types
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, AttributeTemplate as AttributeTemplate, ConcreteTemplate as ConcreteTemplate, infer as infer, infer_getattr as infer_getattr, infer_global as infer_global, signature as signature
from numba.np import npdatetime_helpers as npdatetime_helpers
from numba.np.numpy_support import numpy_version as numpy_version

class TimedeltaUnaryOp(AbstractTemplate):
    def generic(self, args, kws): ...

class TimedeltaBinOp(AbstractTemplate):
    def generic(self, args, kws): ...

class TimedeltaCmpOp(AbstractTemplate):
    def generic(self, args, kws): ...

class TimedeltaOrderedCmpOp(AbstractTemplate):
    def generic(self, args, kws): ...

class TimedeltaMixOp(AbstractTemplate):
    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        ({int, float}, timedelta64) -> timedelta64
        """

class TimedeltaDivOp(AbstractTemplate):
    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        (timedelta64, timedelta64) -> float
        """

class TimedeltaUnaryPos(TimedeltaUnaryOp):
    key = operator.pos

class TimedeltaUnaryNeg(TimedeltaUnaryOp):
    key = operator.neg

class TimedeltaBinAdd(TimedeltaBinOp):
    key = operator.add

class TimedeltaBinSub(TimedeltaBinOp):
    key = operator.sub

class TimedeltaBinMult(TimedeltaMixOp):
    key = operator.mul

class TimedeltaTrueDiv(TimedeltaDivOp):
    key = operator.truediv

class TimedeltaFloorDiv(TimedeltaDivOp):
    key = operator.floordiv

class TimedeltaCmpEq(TimedeltaOrderedCmpOp):
    key = operator.eq

class TimedeltaCmpNe(TimedeltaOrderedCmpOp):
    key = operator.ne

class TimedeltaCmpEq(TimedeltaCmpOp):
    key = operator.eq

class TimedeltaCmpNe(TimedeltaCmpOp):
    key = operator.ne

class TimedeltaCmpLt(TimedeltaOrderedCmpOp):
    key = operator.lt

class TimedeltaCmpLE(TimedeltaOrderedCmpOp):
    key = operator.le

class TimedeltaCmpGt(TimedeltaOrderedCmpOp):
    key = operator.gt

class TimedeltaCmpGE(TimedeltaOrderedCmpOp):
    key = operator.ge

class TimedeltaAbs(TimedeltaUnaryOp): ...

class DatetimePlusTimedelta(AbstractTemplate):
    key = operator.add
    def generic(self, args, kws): ...

class DatetimeMinusTimedelta(AbstractTemplate):
    key = operator.sub
    def generic(self, args, kws): ...

class DatetimeMinusDatetime(AbstractTemplate):
    key = operator.sub
    def generic(self, args, kws): ...

class DatetimeCmpOp(AbstractTemplate):
    def generic(self, args, kws): ...

class DatetimeCmpEq(DatetimeCmpOp):
    key = operator.eq

class DatetimeCmpNe(DatetimeCmpOp):
    key = operator.ne

class DatetimeCmpLt(DatetimeCmpOp):
    key = operator.lt

class DatetimeCmpLE(DatetimeCmpOp):
    key = operator.le

class DatetimeCmpGt(DatetimeCmpOp):
    key = operator.gt

class DatetimeCmpGE(DatetimeCmpOp):
    key = operator.ge

class DatetimeMinMax(AbstractTemplate):
    def generic(self, args, kws): ...
