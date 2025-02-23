from llvmlite import opaque_pointers_enabled as opaque_pointers_enabled
from llvmlite.binding import ffi as ffi

def create_context(): ...
def get_global_context(): ...

class ContextRef(ffi.ObjectRef):
    def __init__(self, context_ptr) -> None: ...
    def _dispose(self) -> None: ...

class GlobalContextRef(ContextRef):
    def _dispose(self) -> None: ...
