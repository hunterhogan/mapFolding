from _typeshed import Incomplete
from numba.core import types as types, utils as utils
from numba.core.typing.templates import AttributeTemplate as AttributeTemplate, ConcreteTemplate as ConcreteTemplate, Registry as Registry, signature as signature

registry: Incomplete
infer_global: Incomplete

class Math_unary(ConcreteTemplate):
    cases: Incomplete

class Math_atan2(ConcreteTemplate):
    cases: Incomplete

class Math_converter(ConcreteTemplate):
    cases: Incomplete

class Math_floor_ceil(Math_converter): ...

class Math_copysign(ConcreteTemplate):
    cases: Incomplete

class Math_hypot(ConcreteTemplate):
    cases: Incomplete

class Math_nextafter(ConcreteTemplate):
    cases: Incomplete

class Math_predicate(ConcreteTemplate):
    cases: Incomplete

class Math_isfinite(Math_predicate): ...

class Math_pow(ConcreteTemplate):
    cases: Incomplete

class Math_gcd(ConcreteTemplate):
    cases: Incomplete

class Math_frexp(ConcreteTemplate):
    cases: Incomplete

class Math_ldexp(ConcreteTemplate):
    cases: Incomplete
