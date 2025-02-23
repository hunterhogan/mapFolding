from _typeshed import Incomplete
from ctypes import Structure
from llvmlite.binding import ffi as ffi, object_file as object_file, targets as targets

def create_mcjit_compiler(module, target_machine, use_lmm: Incomplete | None = None):
    """
    Create a MCJIT ExecutionEngine from the given *module* and
    *target_machine*.

    *lmm* controls whether the llvmlite memory manager is used. If not supplied,
    the default choice for the platform will be used (``True`` on 64-bit ARM
    systems, ``False`` otherwise).
    """
def check_jit_execution() -> None:
    """
    Check the system allows execution of in-memory JITted functions.
    An exception is raised otherwise.
    """

class ExecutionEngine(ffi.ObjectRef):
    """An ExecutionEngine owns all Modules associated with it.
    Deleting the engine will remove all associated modules.
    It is an error to delete the associated modules.
    """
    _object_cache: Incomplete
    _modules: Incomplete
    _td: Incomplete
    def __init__(self, ptr, module) -> None:
        """
        Module ownership is transferred to the EE
        """
    def get_function_address(self, name):
        """
        Return the address of the function named *name* as an integer.

        It's a fatal error in LLVM if the symbol of *name* doesn't exist.
        """
    def get_global_value_address(self, name):
        """
        Return the address of the global value named *name* as an integer.

        It's a fatal error in LLVM if the symbol of *name* doesn't exist.
        """
    def add_global_mapping(self, gv, addr) -> None: ...
    def add_module(self, module) -> None:
        """
        Ownership of module is transferred to the execution engine
        """
    def finalize_object(self) -> None:
        '''
        Make sure all modules owned by the execution engine are fully processed
        and "usable" for execution.
        '''
    def run_static_constructors(self) -> None:
        """
        Run static constructors which initialize module-level static objects.
        """
    def run_static_destructors(self) -> None:
        """
        Run static destructors which perform module-level cleanup of static
        resources.
        """
    def remove_module(self, module) -> None:
        """
        Ownership of module is returned
        """
    @property
    def target_data(self):
        """
        The TargetData for this execution engine.
        """
    def enable_jit_events(self):
        """
        Enable JIT events for profiling of generated code.
        Return value indicates whether connection to profiling tool
        was successful.
        """
    def _find_module_ptr(self, module_ptr):
        """
        Find the ModuleRef corresponding to the given pointer.
        """
    def add_object_file(self, obj_file) -> None:
        """
        Add object file to the jit. object_file can be instance of
        :class:ObjectFile or a string representing file system path
        """
    _object_cache_notify: Incomplete
    _object_cache_getbuffer: Incomplete
    def set_object_cache(self, notify_func: Incomplete | None = None, getbuffer_func: Incomplete | None = None) -> None:
        '''
        Set the object cache "notifyObjectCompiled" and "getBuffer"
        callbacks to the given Python functions.
        '''
    def _raw_object_cache_notify(self, data) -> None:
        """
        Low-level notify hook.
        """
    def _raw_object_cache_getbuffer(self, data) -> None:
        """
        Low-level getbuffer hook.
        """
    def _dispose(self) -> None: ...

class _ObjectCacheRef(ffi.ObjectRef):
    """
    Internal: an ObjectCache instance for use within an ExecutionEngine.
    """
    def __init__(self, obj) -> None: ...
    def _dispose(self) -> None: ...

class _ObjectCacheData(Structure):
    _fields_: Incomplete

_ObjectCacheNotifyFunc: Incomplete
_ObjectCacheGetBufferFunc: Incomplete
_notify_c_hook: Incomplete
_getbuffer_c_hook: Incomplete
