import operator
from .. import errors as errors, types as types, utils as utils
from .builtins import normalize_1d_index as normalize_1d_index
from .templates import AbstractTemplate as AbstractTemplate, AttributeTemplate as AttributeTemplate, ConcreteTemplate as ConcreteTemplate, bound_function as bound_function, infer as infer, infer_getattr as infer_getattr, infer_global as infer_global, make_callable_template as make_callable_template, signature as signature
from _typeshed import Incomplete

class InContainer(AbstractTemplate):
    key = operator.contains
    def generic(self, args, kws): ...

class ContainerLen(AbstractTemplate):
    def generic(self, args, kws): ...

class SequenceBool(AbstractTemplate):
    key = operator.truth
    def generic(self, args, kws): ...

class GetItemSequence(AbstractTemplate):
    key: Incomplete
    def generic(self, args, kws): ...

class SetItemSequence(AbstractTemplate):
    def generic(self, args, kws): ...

class DelItemSequence(AbstractTemplate):
    def generic(self, args, kws): ...

class NamedTupleAttribute(AttributeTemplate):
    key = types.BaseNamedTuple
    def resolve___class__(self, tup): ...
    def generic_resolve(self, tup, attr): ...

class NamedTupleClassAttribute(AttributeTemplate):
    key = types.NamedTupleClass
    def resolve___call__(self, classty):
        """
        Resolve the named tuple constructor, aka the class's __call__ method.
        """
