from _typeshed import Incomplete
from numba.core import config as config, types as types
from numba.core.errors import NumbaValueError as NumbaValueError

_pep3118_int_types: Incomplete
_pep3118_scalar_map: Incomplete
_type_map: Incomplete

def decode_pep3118_format(fmt, itemsize):
    """
    Return the Numba type for an item with format string *fmt* and size
    *itemsize* (in bytes).
    """
def get_type_class(typ):
    """
    Get the Numba type class for buffer-compatible Python *typ*.
    """
def infer_layout(val):
    """
    Infer layout of the given memoryview *val*.
    """
