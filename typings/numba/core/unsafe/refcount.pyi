from numba.core import cgutils as cgutils, types as types
from numba.core.extending import intrinsic as intrinsic
from numba.core.runtime.nrtdynmod import _meminfo_struct_type as _meminfo_struct_type

def dump_refcount(typingctx, obj):
    """Dump the refcount of an object to stdout.

    Returns True if and only if object is reference-counted and NRT is enabled.
    """
def get_refcount(typingctx, obj):
    """Get the current refcount of an object.

    FIXME: only handles the first object
    """
