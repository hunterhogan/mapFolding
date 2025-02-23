from _typeshed import Incomplete
from numba.core import types as types
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, AttributeTemplate as AttributeTemplate, Registry as Registry, signature as signature

registry: Incomplete
infer: Incomplete
infer_global: Incomplete
infer_getattr: Incomplete

class EnumAttribute(AttributeTemplate):
    key = types.EnumMember
    def resolve_value(self, ty): ...

class EnumClassAttribute(AttributeTemplate):
    key = types.EnumClass
    def generic_resolve(self, ty, attr):
        """
        Resolve attributes of an enum class as enum members.
        """

class EnumClassStaticGetItem(AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class EnumCompare(AbstractTemplate):
    def generic(self, args, kws): ...

class EnumEq(EnumCompare): ...
class EnumNe(EnumCompare): ...
