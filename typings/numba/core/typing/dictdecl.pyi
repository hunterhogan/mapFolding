from .. import errors as errors, types as types
from .templates import AbstractTemplate as AbstractTemplate, Registry as Registry, signature as signature
from _typeshed import Incomplete

registry: Incomplete
infer: Incomplete
infer_global: Incomplete
infer_getattr: Incomplete
_message_dict_support: Incomplete

class DictBuiltin(AbstractTemplate):
    def generic(self, args, kws): ...
