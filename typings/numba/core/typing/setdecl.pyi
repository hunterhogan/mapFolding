from .templates import AbstractTemplate as AbstractTemplate, AttributeTemplate as AttributeTemplate, CallableTemplate as CallableTemplate, ConcreteTemplate as ConcreteTemplate, Registry as Registry, bound_function as bound_function, make_callable_template as make_callable_template, signature as signature
from _typeshed import Incomplete
from numba.core import types as types
from numba.core.typing import collections as collections

registry: Incomplete
infer: Incomplete
infer_global: Incomplete
infer_getattr: Incomplete

class SetBuiltin(AbstractTemplate):
    def generic(self, args, kws): ...

class SetAttribute(AttributeTemplate):
    key = types.Set
    def resolve_add(self, set, args, kws): ...
    def resolve_update(self, set, args, kws): ...
    def _resolve_operator(self, set, args, kws): ...
    def _resolve_comparator(self, set, args, kws): ...

class SetOperator(AbstractTemplate):
    def generic(self, args, kws): ...

class SetComparison(AbstractTemplate):
    def generic(self, args, kws): ...

class ConcreteSetOperator(SetOperator):
    key = op_key

class ConcreteInplaceSetOperator(SetOperator):
    key = op_key
