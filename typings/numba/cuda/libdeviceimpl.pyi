from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types
from numba.core.imputils import Registry as Registry
from numba.cuda import libdevice as libdevice, libdevicefuncs as libdevicefuncs

registry: Incomplete
lower: Incomplete

def libdevice_implement(func, retty, nbargs): ...
def libdevice_implement_multiple_returns(func, retty, prototype_args): ...
