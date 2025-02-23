import math
from _typeshed import Incomplete
from numba.core import types as types
from numba.core.typing.templates import ConcreteTemplate as ConcreteTemplate, Registry as Registry, signature as signature

registry: Incomplete
infer_global: Incomplete

class Math_unary(ConcreteTemplate):
    cases: Incomplete

class Math_unary_with_fp16(ConcreteTemplate):
    cases: Incomplete

class Math_atan2(ConcreteTemplate):
    key = math.atan2
    cases: Incomplete

class Math_hypot(ConcreteTemplate):
    key = math.hypot
    cases: Incomplete

class Math_binary(ConcreteTemplate):
    cases: Incomplete

class Math_remainder(ConcreteTemplate):
    cases: Incomplete

class Math_pow(ConcreteTemplate):
    cases: Incomplete

class Math_frexp(ConcreteTemplate):
    cases: Incomplete

class Math_ldexp(ConcreteTemplate):
    cases: Incomplete

class Math_isnan(ConcreteTemplate):
    cases: Incomplete

class Math_modf(ConcreteTemplate):
    cases: Incomplete
