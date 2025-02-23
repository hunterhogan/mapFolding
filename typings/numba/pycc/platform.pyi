from _typeshed import Incomplete
from collections.abc import Generator

CCompiler: Incomplete
new_compiler: Incomplete
customize_compiler: Incomplete
log: Incomplete
_configs: Incomplete

def get_configs(arg): ...

find_shared_ending: Incomplete
find_pyext_ending: Incomplete

def _gentmpfile(suffix) -> Generator[Incomplete]: ...
def external_compiler_works():
    '''
    Returns True if the "external compiler" bound in numpy.distutil is present
    and working, False otherwise.
    '''

class _DummyExtension:
    libraries: Incomplete

class Toolchain:
    _verbose: bool
    _compiler: Incomplete
    _build_ext: Incomplete
    _py_lib_dirs: Incomplete
    _py_include_dirs: Incomplete
    _math_info: Incomplete
    def __init__(self) -> None: ...
    @property
    def verbose(self): ...
    @verbose.setter
    def verbose(self, value) -> None: ...
    def _raise_external_compiler_error(self) -> None: ...
    def compile_objects(self, sources, output_dir, include_dirs=(), depends=(), macros=(), extra_cflags: Incomplete | None = None):
        """
        Compile the given source files into a separate object file each,
        all beneath the *output_dir*.  A list of paths to object files
        is returned.

        *macros* has the same format as in distutils: a list of 1- or 2-tuples.
        If a 1-tuple (name,), the given name is considered undefined by
        the C preprocessor.
        If a 2-tuple (name, value), the given name is expanded into the
        given value by the C preprocessor.
        """
    def link_shared(self, output, objects, libraries=(), library_dirs=(), export_symbols=(), extra_ldflags: Incomplete | None = None) -> None:
        """
        Create a shared library *output* linking the given *objects*
        and *libraries* (all strings).
        """
    def get_python_libraries(self):
        """
        Get the library arguments necessary to link with Python.
        """
    def get_python_library_dirs(self):
        """
        Get the library directories necessary to link with Python.
        """
    def get_python_include_dirs(self):
        """
        Get the include directories necessary to compile against the Python
        and Numpy C APIs.
        """
    def get_ext_filename(self, ext_name):
        """
        Given a C extension's module name, return its intended filename.
        """

def _quote_arg(arg):
    """
    Quote the argument for safe use in a shell command line.
    """
def _is_sequence(arg): ...
