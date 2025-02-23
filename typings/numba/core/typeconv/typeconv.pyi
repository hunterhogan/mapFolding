from _typeshed import Incomplete
from numba.core import types as types
from numba.core.typeconv import Conversion as Conversion, _typeconv as _typeconv, castgraph as castgraph

base_url: str
dev_url: Incomplete
user_url: Incomplete
dashes: Incomplete
msg: Incomplete

class TypeManager:
    _conversion_codes: Incomplete
    _ptr: Incomplete
    _types: Incomplete
    def __init__(self) -> None: ...
    def select_overload(self, sig, overloads, allow_unsafe, exact_match_required): ...
    def check_compatible(self, fromty, toty): ...
    def set_compatible(self, fromty, toty, by) -> None: ...
    def set_promote(self, fromty, toty) -> None: ...
    def set_unsafe_convert(self, fromty, toty) -> None: ...
    def set_safe_convert(self, fromty, toty) -> None: ...
    def get_pointer(self): ...

class TypeCastingRules:
    """
    A helper for establishing type casting rules.
    """
    _tm: Incomplete
    _tg: Incomplete
    def __init__(self, tm) -> None: ...
    def promote(self, a, b) -> None:
        """
        Set `a` can promote to `b`
        """
    def unsafe(self, a, b) -> None:
        """
        Set `a` can unsafe convert to `b`
        """
    def safe(self, a, b) -> None:
        """
        Set `a` can safe convert to `b`
        """
    def promote_unsafe(self, a, b) -> None:
        """
        Set `a` can promote to `b` and `b` can unsafe convert to `a`
        """
    def safe_unsafe(self, a, b) -> None:
        """
        Set `a` can safe convert to `b` and `b` can unsafe convert to `a`
        """
    def unsafe_unsafe(self, a, b) -> None:
        """
        Set `a` can unsafe convert to `b` and `b` can unsafe convert to `a`
        """
    def _cb_update(self, a, b, rel) -> None:
        """
        Callback for updating.
        """
