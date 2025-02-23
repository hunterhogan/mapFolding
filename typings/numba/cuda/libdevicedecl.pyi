from _typeshed import Incomplete
from numba.core.typing.templates import ConcreteTemplate as ConcreteTemplate, Registry as Registry
from numba.cuda import libdevice as libdevice, libdevicefuncs as libdevicefuncs

registry: Incomplete
register_global: Incomplete

def libdevice_declare(func, retty, args) -> None: ...
