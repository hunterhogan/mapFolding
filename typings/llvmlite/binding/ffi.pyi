from _typeshed import Incomplete
from llvmlite.binding.common import _decode_string as _decode_string, _is_shutting_down as _is_shutting_down
from llvmlite.utils import get_library_name as get_library_name

def _make_opaque_ref(name): ...

LLVMContextRef: Incomplete
LLVMModuleRef: Incomplete
LLVMValueRef: Incomplete
LLVMTypeRef: Incomplete
LLVMExecutionEngineRef: Incomplete
LLVMPassManagerBuilderRef: Incomplete
LLVMPassManagerRef: Incomplete
LLVMTargetDataRef: Incomplete
LLVMTargetLibraryInfoRef: Incomplete
LLVMTargetRef: Incomplete
LLVMTargetMachineRef: Incomplete
LLVMMemoryBufferRef: Incomplete
LLVMAttributeListIterator: Incomplete
LLVMElementIterator: Incomplete
LLVMAttributeSetIterator: Incomplete
LLVMGlobalsIterator: Incomplete
LLVMFunctionsIterator: Incomplete
LLVMBlocksIterator: Incomplete
LLVMArgumentsIterator: Incomplete
LLVMInstructionsIterator: Incomplete
LLVMOperandsIterator: Incomplete
LLVMIncomingBlocksIterator: Incomplete
LLVMTypesIterator: Incomplete
LLVMObjectCacheRef: Incomplete
LLVMObjectFileRef: Incomplete
LLVMSectionIteratorRef: Incomplete
LLVMOrcLLJITRef: Incomplete
LLVMOrcDylibTrackerRef: Incomplete
LLVMPipelineTuningOptionsRef: Incomplete
LLVMModulePassManagerRef: Incomplete
LLVMFunctionPassManagerRef: Incomplete
LLVMPassBuilderRef: Incomplete

class _LLVMLock:
    """A Lock to guarantee thread-safety for the LLVM C-API.

    This class implements __enter__ and __exit__ for acquiring and releasing
    the lock as a context manager.

    Also, callbacks can be attached so that every time the lock is acquired
    and released the corresponding callbacks will be invoked.
    """
    _lock: Incomplete
    _cblist: Incomplete
    def __init__(self) -> None: ...
    def register(self, acq_fn, rel_fn) -> None:
        """Register callbacks that are invoked immediately after the lock is
        acquired (``acq_fn()``) and immediately before the lock is released
        (``rel_fn()``).
        """
    def unregister(self, acq_fn, rel_fn) -> None:
        """Remove the registered callbacks.
        """
    def __enter__(self) -> None: ...
    def __exit__(self, *exc_details) -> None: ...

class _suppress_cleanup_errors:
    _context: Incomplete
    def __init__(self, context) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None): ...

class _lib_wrapper:
    """Wrap libllvmlite with a lock such that only one thread may access it at
    a time.

    This class duck-types a CDLL.
    """
    __slots__: Incomplete
    _lib_handle: Incomplete
    _fntab: Incomplete
    _lock: Incomplete
    def __init__(self) -> None: ...
    def _load_lib(self) -> None: ...
    @property
    def _lib(self): ...
    def __getattr__(self, name): ...
    @property
    def _name(self):
        """The name of the library passed in the CDLL constructor.

        For duck-typing a ctypes.CDLL
        """
    @property
    def _handle(self):
        """The system handle used to access the library.

        For duck-typing a ctypes.CDLL
        """

class _lib_fn_wrapper:
    """Wraps and duck-types a ctypes.CFUNCTYPE to provide
    automatic locking when the wrapped function is called.

    TODO: we can add methods to mark the function as threadsafe
          and remove the locking-step on call when marked.
    """
    __slots__: Incomplete
    _lock: Incomplete
    _cfn: Incomplete
    def __init__(self, lock, cfn) -> None: ...
    @property
    def argtypes(self): ...
    @argtypes.setter
    def argtypes(self, argtypes) -> None: ...
    @property
    def restype(self): ...
    @restype.setter
    def restype(self, restype) -> None: ...
    def __call__(self, *args, **kwargs): ...

def _importlib_resources_path_repl(package, resource):
    """Replacement implementation of `import.resources.path` to avoid
    deprecation warning following code at importlib_resources/_legacy.py
    as suggested by https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy

    Notes on differences from importlib.resources implementation:

    The `_common.normalize_path(resource)` call is skipped because it is an
    internal API and it is unnecessary for the use here. What it does is
    ensuring `resource` is a str and that it does not contain path separators.
    """

_importlib_resources_path: Incomplete
lib: Incomplete

def register_lock_callback(acq_fn, rel_fn) -> None:
    """Register callback functions for lock acquire and release.
    *acq_fn* and *rel_fn* are callables that take no arguments.
    """
def unregister_lock_callback(acq_fn, rel_fn) -> None:
    """Remove the registered callback functions for lock acquire and release.
    The arguments are the same as used in `register_lock_callback()`.
    """

class _DeadPointer:
    """
    Dummy class to make error messages more helpful.
    """

class OutputString:
    """
    Object for managing the char* output of LLVM APIs.
    """
    _as_parameter_: Incomplete
    @classmethod
    def from_return(cls, ptr):
        """Constructing from a pointer returned from the C-API.
        The pointer must be allocated with LLVMPY_CreateString.

        Note
        ----
        Because ctypes auto-converts *restype* of *c_char_p* into a python
        string, we must use *c_void_p* to obtain the raw pointer.
        """
    _ptr: Incomplete
    _owned: Incomplete
    def __init__(self, owned: bool = True, init: Incomplete | None = None) -> None: ...
    def close(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def __del__(self, _is_shutting_down=...) -> None: ...
    def __str__(self) -> str: ...
    def __bool__(self) -> bool: ...
    __nonzero__ = __bool__
    @property
    def bytes(self):
        """Get the raw bytes of content of the char pointer.
        """

def ret_string(ptr):
    """To wrap string return-value from C-API.
    """
def ret_bytes(ptr):
    """To wrap bytes return-value from C-API.
    """

class ObjectRef:
    '''
    A wrapper around a ctypes pointer to a LLVM object ("resource").
    '''
    _closed: bool
    _as_parameter_: Incomplete
    _owned: bool
    _ptr: Incomplete
    _capi: Incomplete
    def __init__(self, ptr) -> None: ...
    def close(self) -> None:
        """
        Close this object and do any required clean-up actions.
        """
    def detach(self) -> None:
        """
        Detach the underlying LLVM resource without disposing of it.
        """
    def _dispose(self) -> None:
        """
        Dispose of the underlying LLVM resource.  Should be overriden
        by subclasses.  Automatically called by close(), __del__() and
        __exit__() (unless the resource has been detached).
        """
    @property
    def closed(self):
        """
        Whether this object has been closed.  A closed object can't
        be used anymore.
        """
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def __del__(self, _is_shutting_down=...) -> None: ...
    def __bool__(self) -> bool: ...
    def __eq__(self, other): ...
    __nonzero__ = __bool__
    def __hash__(self): ...
