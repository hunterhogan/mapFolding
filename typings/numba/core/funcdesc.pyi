from _typeshed import Incomplete
from numba.core import itanium_mangler as itanium_mangler, types as types
from numba.core.utils import _dynamic_modname as _dynamic_modname, _dynamic_module as _dynamic_module

def default_mangler(name, argtypes, *, abi_tags=(), uid: Incomplete | None = None): ...
def qualifying_prefix(modname, qualname):
    """
    Returns a new string that is used for the first half of the mangled name.
    """

class FunctionDescriptor:
    '''
    Base class for function descriptors: an object used to carry
    useful metadata about a natively callable function.

    Note that while `FunctionIdentity` denotes a Python function
    which is being concretely compiled by Numba, `FunctionDescriptor`
    may be more "abstract".
    '''
    __slots__: Incomplete
    native: Incomplete
    modname: Incomplete
    global_dict: Incomplete
    qualname: Incomplete
    unique_name: Incomplete
    doc: Incomplete
    typemap: Incomplete
    calltypes: Incomplete
    args: Incomplete
    kws: Incomplete
    restype: Incomplete
    argtypes: Incomplete
    uid: Incomplete
    mangled_name: Incomplete
    env_name: Incomplete
    inline: Incomplete
    noalias: Incomplete
    abi_tags: Incomplete
    def __init__(self, native, modname, qualname, unique_name, doc, typemap, restype, calltypes, args, kws, mangler: Incomplete | None = None, argtypes: Incomplete | None = None, inline: bool = False, noalias: bool = False, env_name: Incomplete | None = None, global_dict: Incomplete | None = None, abi_tags=(), uid: Incomplete | None = None) -> None: ...
    def lookup_globals(self):
        """
        Return the global dictionary of the function.
        It may not match the Module's globals if the function is created
        dynamically (i.e. exec)
        """
    def lookup_module(self):
        """
        Return the module in which this function is supposed to exist.
        This may be a dummy module if the function was dynamically
        generated or the module can't be found.
        """
    def lookup_function(self):
        """
        Return the original function object described by this object.
        """
    @property
    def llvm_func_name(self):
        """
        The LLVM-registered name for the raw function.
        """
    @property
    def llvm_cpython_wrapper_name(self):
        """
        The LLVM-registered name for a CPython-compatible wrapper of the
        raw function (i.e. a PyCFunctionWithKeywords).
        """
    @property
    def llvm_cfunc_wrapper_name(self):
        """
        The LLVM-registered name for a C-compatible wrapper of the
        raw function.
        """
    def __repr__(self) -> str: ...
    @classmethod
    def _get_function_info(cls, func_ir):
        """
        Returns
        -------
        qualname, unique_name, modname, doc, args, kws, globals

        ``unique_name`` must be a unique name.
        """
    @classmethod
    def _from_python_function(cls, func_ir, typemap, restype, calltypes, native, mangler: Incomplete | None = None, inline: bool = False, noalias: bool = False, abi_tags=()): ...

class PythonFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for Numba-compiled functions.
    """
    __slots__: Incomplete
    @classmethod
    def from_specialized_function(cls, func_ir, typemap, restype, calltypes, mangler, inline, noalias, abi_tags):
        """
        Build a FunctionDescriptor for a given specialization of a Python
        function (in nopython mode).
        """
    @classmethod
    def from_object_mode_function(cls, func_ir):
        """
        Build a FunctionDescriptor for an object mode variant of a Python
        function.
        """

class ExternalFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for opaque external functions
    (e.g. raw C functions).
    """
    __slots__: Incomplete
    def __init__(self, name, restype, argtypes) -> None: ...
