from _typeshed import Incomplete
from numba.core import types as types
from numba.core.imputils import lower_builtin as lower_builtin

_message_dict_support: Incomplete

def dict_constructor(context, builder, sig, args): ...
def impl_dict(context, builder, sig, args):
    """
    The `dict()` implementation simply forwards the work to `Dict.empty()`.
    """
