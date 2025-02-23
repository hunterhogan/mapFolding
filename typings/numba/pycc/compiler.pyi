from _typeshed import Incomplete

__all__ = ['Compiler']

class ExportEntry:
    """
    A simple record for exporting symbols.
    """
    symbol: Incomplete
    signature: Incomplete
    function: Incomplete
    def __init__(self, symbol, signature, function) -> None: ...
    def __repr__(self) -> str: ...

class _ModuleCompiler:
    """A base class to compile Python modules to a single shared library or
    extension module.

    :param export_entries: a list of ExportEntry instances.
    :param module_name: the name of the exported module.
    """
    method_def_ty: Incomplete
    method_def_ptr: Incomplete
    env_def_ty: Incomplete
    env_def_ptr: Incomplete
    module_name: Incomplete
    export_python_wrap: bool
    dll_exports: Incomplete
    export_entries: Incomplete
    external_init_function: Incomplete
    use_nrt: Incomplete
    typing_context: Incomplete
    context: Incomplete
    def __init__(self, export_entries, module_name, use_nrt: bool = False, **aot_options) -> None: ...
    def _mangle_method_symbol(self, func_name): ...
    def _emit_python_wrapper(self, llvm_module) -> None:
        """Emit generated Python wrapper and extension module code.
        """
    exported_function_types: Incomplete
    function_environments: Incomplete
    environment_gvs: Incomplete
    def _cull_exports(self):
        """Read all the exported functions/modules in the translator
        environment, and join them into a single LLVM module.
        """
    def write_llvm_bitcode(self, output, wrap: bool = False, **kws) -> None: ...
    def write_native_object(self, output, wrap: bool = False, **kws) -> None: ...
    def emit_type(self, tyobj): ...
    def emit_header(self, output) -> None: ...
    def _emit_method_array(self, llvm_module):
        """
        Collect exported methods and emit a PyMethodDef array.

        :returns: a pointer to the PyMethodDef array.
        """
    def _emit_environment_array(self, llvm_module, builder, pyapi):
        """
        Emit an array of env_def_t structures (see modulemixin.c)
        storing the pickled environment constants for each of the
        exported functions.
        """
    def _emit_envgvs_array(self, llvm_module, builder, pyapi):
        """
        Emit an array of Environment pointers that needs to be filled at
        initialization.
        """
    def _emit_module_init_code(self, llvm_module, builder, modobj, method_array, env_array, envgv_array):
        '''
        Emit call to "external" init function, if any.
        '''

class ModuleCompiler(_ModuleCompiler):
    _ptr_fun: Incomplete
    visitproc_ty: Incomplete
    inquiry_ty: Incomplete
    traverseproc_ty: Incomplete
    freefunc_ty: Incomplete
    m_init_ty: Incomplete
    _char_star: Incomplete
    module_def_base_ty: Incomplete
    module_def_ty: Incomplete
    @property
    def module_create_definition(self):
        """
        Return the signature and name of the Python C API function to
        initialize the module.
        """
    @property
    def module_init_definition(self):
        """
        Return the name and signature of the module's initialization function.
        """
    def _emit_python_wrapper(self, llvm_module) -> None: ...

# Names in __all__ with no definition:
#   Compiler
