from _typeshed import Incomplete
from numba.core import types as types, utils as utils
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, ConcreteTemplate as ConcreteTemplate, Registry as Registry, signature as signature

registry: Incomplete
infer_global: Incomplete

class CMath_unary(ConcreteTemplate):
    cases: Incomplete

class CMath_predicate(ConcreteTemplate):
    cases: Incomplete

class CMath_isfinite(CMath_predicate): ...

class Cmath_log(ConcreteTemplate):
    cases: Incomplete
