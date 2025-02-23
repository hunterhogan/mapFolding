from _typeshed import Incomplete
from llvmlite.ir import _utils as _utils, types as types

class Context:
    scope: Incomplete
    identified_types: Incomplete
    def __init__(self) -> None: ...
    def get_identified_type(self, name, packed: bool = False): ...

global_context: Incomplete
