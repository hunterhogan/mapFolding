import abc
from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from numba.core import cgutils as cgutils, config as config, utils as utils
from numba.core.compiler_lock import require_global_compiler_lock as require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning as NumbaInvalidConfigWarning
from numba.core.llvm_bindings import create_pass_manager_builder as create_pass_manager_builder
from numba.core.runtime import rtsys as rtsys
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct as remove_redundant_nrt_refct
from numba.misc.inspection import disassemble_elf_to_cfg as disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection as PassTimingsCollection

_x86arch: Incomplete

def _is_x86(triple): ...
def _parse_refprune_flags():
    """Parse refprune flags from the `config`.

    Invalid values are ignored an warn via a `NumbaInvalidConfigWarning`
    category.

    Returns
    -------
    flags : llvmlite.binding.RefPruneSubpasses
    """
def dump(header, body, lang) -> None: ...

class _CFG:
    """
    Wraps the CFG graph for different display method.

    Instance of the class can be stringified (``__repr__`` is defined) to get
    the graph in DOT format.  The ``.display()`` method plots the graph in
    PDF.  If in IPython notebook, the returned image can be inlined.
    """
    cres: Incomplete
    name: Incomplete
    py_func: Incomplete
    dot: Incomplete
    kwargs: Incomplete
    def __init__(self, cres, name, py_func, **kwargs) -> None: ...
    def pretty_printer(self, filename: Incomplete | None = None, view: Incomplete | None = None, render_format: Incomplete | None = None, highlight: bool = True, interleave: bool = False, strip_ir: bool = False, show_key: bool = True, fontsize: int = 10):
        '''
        "Pretty" prints the DOT graph of the CFG.
        For explanation of the parameters see the docstring for
        numba.core.dispatcher::inspect_cfg.
        '''
    def display(self, filename: Incomplete | None = None, format: str = 'pdf', view: bool = False):
        """
        Plot the CFG.  In IPython notebook, the return image object can be
        inlined.

        The *filename* option can be set to a specific path for the rendered
        output to write to.  If *view* option is True, the plot is opened by
        the system default application for the image format (PDF). *format* can
        be any valid format string accepted by graphviz, default is 'pdf'.
        """
    def _repr_svg_(self): ...
    def __repr__(self) -> str: ...

class CodeLibrary(metaclass=ABCMeta):
    """
    An interface for bundling LLVM code together and compiling it.
    It is tied to a *codegen* instance (e.g. JITCPUCodegen) that will
    determine how the LLVM code is transformed and linked together.
    """
    _finalized: bool
    _object_caching_enabled: bool
    _disable_inspection: bool
    _codegen: Incomplete
    _name: Incomplete
    _recorded_timings: Incomplete
    _dynamic_globals: Incomplete
    def __init__(self, codegen: CPUCodegen, name: str) -> None: ...
    @property
    def has_dynamic_globals(self): ...
    @property
    def recorded_timings(self): ...
    @property
    def codegen(self):
        """
        The codegen object owning this library.
        """
    @property
    def name(self): ...
    def __repr__(self) -> str: ...
    def _raise_if_finalized(self) -> None: ...
    def _ensure_finalized(self) -> None: ...
    def create_ir_module(self, name):
        """
        Create an LLVM IR module for use by this library.
        """
    @abstractmethod
    def add_linking_library(self, library):
        """
        Add a library for linking into this library, without losing
        the original library.
        """
    @abstractmethod
    def add_ir_module(self, ir_module):
        """
        Add an LLVM IR module's contents to this library.
        """
    @abstractmethod
    def finalize(self):
        """
        Finalize the library.  After this call, nothing can be added anymore.
        Finalization involves various stages of code optimization and
        linking.
        """
    @abstractmethod
    def get_function(self, name):
        """
        Return the function named ``name``.
        """
    @abstractmethod
    def get_llvm_str(self):
        """
        Get the human-readable form of the LLVM module.
        """
    @abstractmethod
    def get_asm_str(self):
        """
        Get the human-readable assembly.
        """
    _compiled_object: Incomplete
    _compiled: bool
    def enable_object_caching(self) -> None: ...
    def _get_compiled_object(self): ...
    def _set_compiled_object(self, value) -> None: ...

class CPUCodeLibrary(CodeLibrary):
    _linking_libraries: Incomplete
    _final_module: Incomplete
    _shared_module: Incomplete
    def __init__(self, codegen, name) -> None: ...
    def _optimize_functions(self, ll_module) -> None:
        """
        Internal: run function-level optimizations inside *ll_module*.
        """
    def _optimize_final_module(self) -> None:
        """
        Internal: optimize this library's final module.
        """
    def _get_module_for_linking(self):
        '''
        Internal: get a LLVM module suitable for linking multiple times
        into another library.  Exported functions are made "linkonce_odr"
        to allow for multiple definitions, inlining, and removal of
        unused exports.

        See discussion in https://github.com/numba/numba/pull/890
        '''
    def add_linking_library(self, library) -> None: ...
    def add_ir_module(self, ir_module) -> None: ...
    def add_llvm_module(self, ll_module) -> None: ...
    def finalize(self) -> None: ...
    def _finalize_dynamic_globals(self) -> None: ...
    def _verify_declare_only_symbols(self) -> None: ...
    _finalized: bool
    def _finalize_final_module(self) -> None:
        """
        Make the underlying LLVM module ready to use.
        """
    def get_defined_functions(self) -> Generator[Incomplete]:
        """
        Get all functions defined in the library.  The library must have
        been finalized.
        """
    def get_function(self, name): ...
    def _sentry_cache_disable_inspection(self) -> None: ...
    def get_llvm_str(self): ...
    def get_asm_str(self): ...
    def get_function_cfg(self, name, py_func: Incomplete | None = None, **kwargs):
        """
        Get control-flow graph of the LLVM function
        """
    def get_disasm_cfg(self, mangled_name):
        """
        Get the CFG of the disassembly of the ELF object at symbol mangled_name.

        Requires python package: r2pipe
        Requires radare2 binary on $PATH.
        Notebook rendering requires python package: graphviz
        Optionally requires a compiler toolchain (via pycc) to link the ELF to
        get better disassembly results.
        """
    @classmethod
    def _dump_elf(cls, buf):
        """
        Dump the symbol table of an ELF file.
        Needs pyelftools (https://github.com/eliben/pyelftools)
        """
    _compiled: bool
    _compiled_object: Incomplete
    @classmethod
    def _object_compiled_hook(cls, ll_module, buf) -> None:
        """
        `ll_module` was compiled into object code `buf`.
        """
    @classmethod
    def _object_getbuffer_hook(cls, ll_module):
        """
        Return a cached object code for `ll_module`.
        """
    def serialize_using_bitcode(self):
        """
        Serialize this library using its bitcode as the cached representation.
        """
    def serialize_using_object_code(self):
        """
        Serialize this library using its object code as the cached
        representation.  We also include its bitcode for further inlining
        with other libraries.
        """
    @classmethod
    def _unserialize(cls, codegen, state): ...

class AOTCodeLibrary(CPUCodeLibrary):
    def emit_native_object(self):
        """
        Return this library as a native object (a bytestring) -- for example
        ELF under Linux.

        This function implicitly calls .finalize().
        """
    def emit_bitcode(self):
        """
        Return this library as LLVM bitcode (a bytestring).

        This function implicitly calls .finalize().
        """
    def _finalize_specific(self) -> None: ...

class JITCodeLibrary(CPUCodeLibrary):
    def get_pointer_to_function(self, name):
        """
        Generate native code for function named *name* and return a pointer
        to the start of the function (as an integer).

        This function implicitly calls .finalize().

        Returns
        -------
        pointer : int
            - zero (null) if no symbol of *name* is defined by this code
              library.
            - non-zero if the symbol is defined.
        """
    def _finalize_specific(self) -> None: ...

class RuntimeLinker:
    """
    For tracking unresolved symbols generated at runtime due to recursion.
    """
    PREFIX: str
    _unresolved: Incomplete
    _defined: Incomplete
    _resolved: Incomplete
    def __init__(self) -> None: ...
    def scan_unresolved_symbols(self, module, engine) -> None:
        """
        Scan and track all unresolved external symbols in the module and
        allocate memory for it.
        """
    def scan_defined_symbols(self, module) -> None:
        """
        Scan and track all defined symbols.
        """
    def resolve(self, engine) -> None:
        """
        Fix unresolved symbols if they are defined.
        """

def _proxy(old): ...

class JitEngine:
    """Wraps an ExecutionEngine to provide custom symbol tracking.
    Since the symbol tracking is incomplete  (doesn't consider
    loaded code object), we are not putting it in llvmlite.
    """
    _ee: Incomplete
    _defined_symbols: Incomplete
    def __init__(self, ee) -> None: ...
    def is_symbol_defined(self, name):
        """Is the symbol defined in this session?
        """
    def _load_defined_symbols(self, mod) -> None:
        """Extract symbols from the module
        """
    def add_module(self, module):
        """Override ExecutionEngine.add_module
        to keep info about defined symbols.
        """
    def add_global_mapping(self, gv, addr):
        """Override ExecutionEngine.add_global_mapping
        to keep info about defined symbols.
        """
    set_object_cache: Incomplete
    finalize_object: Incomplete
    get_function_address: Incomplete
    get_global_value_address: Incomplete

class Codegen(metaclass=ABCMeta):
    """
    Base Codegen class. It is expected that subclasses set the class attribute
    ``_library_class``, indicating the CodeLibrary class for the target.

    Subclasses should also initialize:

    ``self._data_layout``: the data layout for the target.
    ``self._target_data``: the binding layer ``TargetData`` for the target.
    """
    @abstractmethod
    def _create_empty_module(self, name):
        """
        Create a new empty module suitable for the target.
        """
    @abstractmethod
    def _add_module(self, module):
        """
        Add a module to the execution engine. Ownership of the module is
        transferred to the engine.
        """
    @property
    def target_data(self):
        '''
        The LLVM "target data" object for this codegen instance.
        '''
    def create_library(self, name, **kwargs):
        """
        Create a :class:`CodeLibrary` object for use with this codegen
        instance.
        """
    def unserialize_library(self, serialized): ...

class CPUCodegen(Codegen, metaclass=abc.ABCMeta):
    _data_layout: Incomplete
    _llvm_module: Incomplete
    _rtlinker: Incomplete
    def __init__(self, module_name) -> None: ...
    _tm_features: Incomplete
    _tm: Incomplete
    _engine: Incomplete
    _target_data: Incomplete
    _loopvect: bool
    _opt_level: int
    def _init(self, llvm_module) -> None: ...
    def _create_empty_module(self, name): ...
    def _module_pass_manager(self, **kwargs): ...
    def _function_pass_manager(self, llvm_module, **kwargs): ...
    def _pass_manager_builder(self, **kwargs):
        """
        Create a PassManagerBuilder.

        Note: a PassManagerBuilder seems good only for one use, so you
        should call this method each time you want to populate a module
        or function pass manager.  Otherwise some optimizations will be
        missed...
        """
    def _check_llvm_bugs(self) -> None:
        """
        Guard against some well-known LLVM bug(s).
        """
    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
    def _scan_and_fix_unresolved_refs(self, module) -> None: ...
    def insert_unresolved_ref(self, builder, fnty, name): ...
    def _get_host_cpu_name(self): ...
    def _get_host_cpu_features(self): ...

class AOTCPUCodegen(CPUCodegen):
    """
    A codegen implementation suitable for Ahead-Of-Time compilation
    (e.g. generation of object files).
    """
    _library_class = AOTCodeLibrary
    _cpu_name: Incomplete
    def __init__(self, module_name, cpu_name: Incomplete | None = None) -> None: ...
    def _customize_tm_options(self, options) -> None: ...
    def _customize_tm_features(self): ...
    def _add_module(self, module) -> None: ...

class JITCPUCodegen(CPUCodegen):
    """
    A codegen implementation suitable for Just-In-Time compilation.
    """
    _library_class = JITCodeLibrary
    def _customize_tm_options(self, options) -> None: ...
    def _customize_tm_features(self): ...
    def _add_module(self, module) -> None: ...
    def set_env(self, env_name, env) -> None:
        """Set the environment address.

        Update the GlobalVariable named *env_name* to the address of *env*.
        """

def initialize_llvm() -> None:
    """Safe to use multiple times.
    """
def get_host_cpu_features():
    """Get host CPU features using LLVM.

    The features may be modified due to user setting.
    See numba.config.ENABLE_AVX.
    """
