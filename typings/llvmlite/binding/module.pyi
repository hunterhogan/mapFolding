from _typeshed import Incomplete
from llvmlite.binding import ffi as ffi
from llvmlite.binding.common import _decode_string as _decode_string, _encode_string as _encode_string
from llvmlite.binding.context import get_global_context as get_global_context
from llvmlite.binding.linker import link_modules as link_modules
from llvmlite.binding.value import TypeRef as TypeRef, ValueRef as ValueRef

def parse_assembly(llvmir: str, context: Incomplete | None = None) -> ModuleRef:
    """
    Create Module from a LLVM IR string
    """
def parse_bitcode(bitcode, context: Incomplete | None = None):
    """
    Create Module from a LLVM *bitcode* (a bytes object).
    """

class ModuleRef(ffi.ObjectRef):
    """
    A reference to a LLVM module.
    """
    _context: Incomplete
    def __init__(self, module_ptr, context) -> None: ...
    def __str__(self) -> str: ...
    def as_bitcode(self):
        """
        Return the module's LLVM bitcode, as a bytes object.
        """
    def _dispose(self) -> None: ...
    def get_function(self, name):
        """
        Get a ValueRef pointing to the function named *name*.
        NameError is raised if the symbol isn't found.
        """
    def get_global_variable(self, name):
        """
        Get a ValueRef pointing to the global variable named *name*.
        NameError is raised if the symbol isn't found.
        """
    def get_struct_type(self, name):
        """
        Get a TypeRef pointing to a structure type named *name*.
        NameError is raised if the struct type isn't found.
        """
    def verify(self) -> None:
        """
        Verify the module IR's correctness.  RuntimeError is raised on error.
        """
    @property
    def name(self):
        """
        The module's identifier.
        """
    @name.setter
    def name(self, value) -> None: ...
    @property
    def source_file(self):
        """
        The module's original source file name
        """
    @property
    def data_layout(self):
        """
        This module's data layout specification, as a string.
        """
    @data_layout.setter
    def data_layout(self, strrep) -> None: ...
    @property
    def triple(self):
        '''
        This module\'s target "triple" specification, as a string.
        '''
    @triple.setter
    def triple(self, strrep) -> None: ...
    def link_in(self, other, preserve: bool = False) -> None:
        """
        Link the *other* module into this one.  The *other* module will
        be destroyed unless *preserve* is true.
        """
    @property
    def global_variables(self):
        '''
        Return an iterator over this module\'s global variables.
        The iterator will yield a ValueRef for each global variable.

        Note that global variables don\'t include functions
        (a function is a "global value" but not a "global variable" in
         LLVM parlance)
        '''
    @property
    def functions(self):
        """
        Return an iterator over this module's functions.
        The iterator will yield a ValueRef for each function.
        """
    @property
    def struct_types(self):
        """
        Return an iterator over the struct types defined in
        the module. The iterator will yield a TypeRef.
        """
    def clone(self): ...

class _Iterator(ffi.ObjectRef):
    kind: Incomplete
    _parents: Incomplete
    def __init__(self, ptr, parents) -> None: ...
    def __next__(self): ...
    next = __next__
    def __iter__(self): ...

class _GlobalsIterator(_Iterator):
    kind: str
    def _dispose(self) -> None: ...
    def _next(self): ...

class _FunctionsIterator(_Iterator):
    kind: str
    def _dispose(self) -> None: ...
    def _next(self): ...

class _TypesIterator(_Iterator):
    kind: str
    def _dispose(self) -> None: ...
    def __next__(self): ...
    def _next(self): ...
    next = __next__
