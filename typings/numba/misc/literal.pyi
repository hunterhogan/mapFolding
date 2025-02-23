from numba.core import types as types
from numba.core.errors import TypingError as TypingError
from numba.core.extending import overload as overload
from numba.misc.special import literal_unroll as literal_unroll, literally as literally

def _ov_literally(obj): ...
def literal_unroll_impl(container): ...
