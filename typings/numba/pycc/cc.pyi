from _typeshed import Incomplete
from numba import cext as cext
from numba.core import sigutils as sigutils, typing as typing
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.pycc.compiler import ExportEntry as ExportEntry, ModuleCompiler as ModuleCompiler
from numba.pycc.platform import Toolchain as Toolchain
from setuptools.extension import Extension

dir_util: Incomplete
log: Incomplete
extension_libs: Incomplete

class CC:
    """
    An ahead-of-time compiler to create extension modules that don't
    depend on Numba.
    """
    _mixin_sources: Incomplete
    _extra_cflags: Incomplete
    _extra_ldflags: Incomplete
    _basename: Incomplete
    _init_function: Incomplete
    _exported_functions: Incomplete
    _source_path: Incomplete
    _source_module: Incomplete
    _toolchain: Incomplete
    _verbose: bool
    _output_dir: Incomplete
    _output_file: Incomplete
    _use_nrt: bool
    _target_cpu: str
    def __init__(self, extension_name, source_module: Incomplete | None = None) -> None: ...
    @property
    def name(self):
        """
        The name of the extension module to create.
        """
    @property
    def output_file(self):
        """
        The specific output file (a DLL) that will be generated.
        """
    @output_file.setter
    def output_file(self, value) -> None: ...
    @property
    def output_dir(self):
        """
        The directory the output file will be put in.
        """
    @output_dir.setter
    def output_dir(self, value) -> None: ...
    @property
    def use_nrt(self): ...
    @use_nrt.setter
    def use_nrt(self, value) -> None: ...
    @property
    def target_cpu(self):
        """
        The target CPU model for code generation.
        """
    @target_cpu.setter
    def target_cpu(self, value) -> None: ...
    @property
    def verbose(self):
        """
        Whether to display detailed information when compiling.
        """
    @verbose.setter
    def verbose(self, value) -> None: ...
    def export(self, exported_name, sig):
        """
        Mark a function for exporting in the extension module.
        """
    @property
    def _export_entries(self): ...
    def _get_mixin_sources(self): ...
    def _get_mixin_defines(self): ...
    def _get_extra_cflags(self): ...
    def _get_extra_ldflags(self): ...
    def _compile_mixins(self, build_dir): ...
    def _compile_object_files(self, build_dir): ...
    def compile(self) -> None:
        """
        Compile the extension module.
        """
    def distutils_extension(self, **kwargs):
        """
        Create a distutils extension object that can be used in your
        setup.py.
        """

class _CCExtension(Extension):
    """
    A Numba-specific Extension subclass to LLVM-compile pure Python code
    to an extension module.
    """
    _cc: Incomplete
    _distutils_monkey_patched: bool
    extra_objects: Incomplete
    def _prepare_object_files(self, build_ext) -> None: ...
    @classmethod
    def monkey_patch_distutils(cls) -> None:
        """
        Monkey-patch distutils with our own build_ext class knowing
        about pycc-compiled extensions modules.
        """
